"""Dev console: the static console at /ui, the /v1/dev API behind it, the ranker lab,
offline jobs, reports, and demo traffic (memory backend)."""

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tests.helpers import NOW
from theraflo_ads.api import build_dev_container, create_app
from theraflo_ads.api.container import (
    DEV_ADMIN_TOKENS,
    DEV_ADVERTISER_KEYS,
    DEV_SERVICE_TOKENS,
    DEV_USER_TOKENS,
)
from theraflo_ads.api.jobs import JobError, JobRunner, list_reports, read_report
from theraflo_ads.api.lab import POLICIES
from theraflo_ads.devtools.traffic import Demo, TrafficRunner
from theraflo_ads.experiments import HoldoutConfig
from theraflo_ads.ml.bundle import save_bundle
from theraflo_ads.ml.features import FEATURE_SCHEMA_VERSION
from theraflo_ads.ml.models import CalibratedModel, PriorModel
from theraflo_ads.ml.registry import ModelRegistry, Stage
from theraflo_ads.observability.metrics import _quantiles

ADMIN = {"Authorization": f"Bearer {next(iter(DEV_ADMIN_TOKENS))}"}
SERVICE = {"Authorization": f"Bearer {next(iter(DEV_SERVICE_TOKENS))}"}
ALICE_TOKEN = next(t for t, u in DEV_USER_TOKENS.items() if u == "user-alice")
ALICE = {"Authorization": f"Bearer {ALICE_TOKEN}"}
KEYS = {a: k for k, a in DEV_ADVERTISER_KEYS.items()}


class FakeClock:
    def __init__(self):
        self.now = NOW

    def __call__(self):
        return self.now


def write_registry(models: Path) -> None:
    """A tiny CTR model (predicts the category prior) registered as offline-validated."""
    bundle = save_bundle(
        models / "bundle",
        name="ctr",
        version="prior-test",
        model=CalibratedModel(PriorModel(("prior_ctr", "log_price"))),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )
    registry = ModelRegistry(models / "registry")
    version = registry.register("ctr", bundle)
    registry.transition("ctr", version, Stage.OFFLINE_VALIDATED, "test")


@pytest.fixture
def make_console(tmp_path, monkeypatch):
    def make(*, registry: bool = False, session_key: str | None = None):
        models = tmp_path / "models"
        if registry:
            write_registry(models)
        monkeypatch.setenv("THERAFLO_ADS_MODEL_DIR", str(models))
        monkeypatch.setenv("THERAFLO_ADS_REPORTS_DIR", str(tmp_path / "reports"))
        if session_key:
            monkeypatch.setenv("THERAFLO_ADS_SESSION_KEY", session_key)
        else:
            monkeypatch.delenv("THERAFLO_ADS_SESSION_KEY", raising=False)
        clock = FakeClock()
        container = build_dev_container(
            clock=clock,
            user_hash_key=b"u" * 32,
            token_key=b"t" * 32,
            holdouts=HoldoutConfig(uniform_exploration_share=0.0, no_ads_share=0.0),
            seed=3,
        )
        return TestClient(create_app(container)), container

    return make


@pytest.fixture
def console(make_console):
    return make_console()


def rank(client, headers=ALICE, placement="home_feed"):
    response = client.post(
        "/v1/ads/rank", json={"session_id": "s1", "placement": placement}, headers=headers
    )
    assert response.status_code == 200, response.text
    return response.json()


def served(client, headers=ALICE):
    """Rank until an ad is served (ε-exploration can sample NO_AD)."""
    for _ in range(6):
        body = rank(client, headers)
        if body["ads"]:
            return body
    raise AssertionError("no ad served")


# ----- the console itself -----


def test_the_console_is_served_with_a_strict_csp(console):
    client, _ = console
    home = client.get("/", follow_redirects=False)
    assert home.status_code == 307
    assert home.headers["location"] == "/ui/"
    page = client.get("/ui/")
    assert page.status_code == 200
    # Relative asset links: a path-rewriting proxy in front (LocalStack's Container Apps
    # gateway) would otherwise send the browser to URLs the app doesn't serve.
    assert '<script type="module" src="js/main.js">' in page.text
    assert 'href="/' not in page.text and 'src="/' not in page.text
    # They resolve against /ui/, so /ui must redirect there rather than serve the page.
    bare = client.get("/ui", follow_redirects=False)
    assert bare.status_code in (307, 308) and bare.headers["location"].endswith("/ui/")
    assert client.get("/ui/js/main.js").status_code == 200
    csp = page.headers["content-security-policy"]
    assert "script-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    script = client.get("/ui/js/pages/lab.js")
    assert script.status_code == 200
    # Browsers refuse module scripts served as text/plain (the Windows registry default).
    assert script.headers["content-type"].startswith("text/javascript")


def test_the_console_is_not_mounted_without_dev_credentials(console):
    _, container = console
    container.dev_mode = False
    client = TestClient(create_app(container))
    assert client.get("/ui/").status_code == 404
    assert client.get("/v1/dev/identities").status_code == 404
    assert client.get("/", follow_redirects=False).status_code == 404


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/v1/dev/overview"),
        ("POST", "/v1/dev/rank-preview"),
        ("PUT", "/v1/dev/serving-policy"),
        ("GET", "/v1/dev/users/user-alice"),
        ("POST", "/v1/dev/users"),
        ("GET", "/v1/dev/jobs"),
        ("GET", "/v1/dev/reports"),
        ("POST", "/v1/dev/traffic"),
        ("GET", "/v1/admin/audit"),
    ],
)
def test_console_routes_need_the_admin_token(console, method, path):
    client, _ = console
    assert client.request(method, path, json={}).status_code == 401
    assert client.request(method, path, json={}, headers=ALICE).status_code == 401


def test_identities_bootstrap_the_console(console):
    client, _ = console
    body = client.get("/v1/dev/identities").json()
    assert body["backend"] == "memory"
    assert {u["user_id"] for u in body["users"]} == set(DEV_USER_TOKENS.values())
    assert {a["advertiser_id"] for a in body["advertisers"]} == set(DEV_ADVERTISER_KEYS.values())
    assert "gambling" in body["taxonomy"]["restricted"]
    assert "prompt_injection" in body["violations"]
    assert "journal" in body["placements"]["ad_free"]
    assert body["test_user_tokens"] == "process"


# ----- test users -----


def test_test_users_get_working_tokens_and_profiles(console):
    client, _ = console
    made = client.post("/v1/dev/users", json={"age_band": "teen"}, headers=ADMIN)
    assert made.status_code == 201
    user = made.json()
    assert user["token_kind"] == "process"
    body = rank(client, {"Authorization": f"Bearer {user['token']}"})
    profile = client.get(f"/v1/dev/users/{user['user_id']}", headers=ADMIN).json()
    assert profile["age_band"] == "teen"
    # Minors are always contextual-only, whatever they consent to.
    assert profile["consent"] == "personalized"
    assert profile["effective_consent"] == "contextual_only"
    assert profile["exposures_24h"]["total"] == len(body["ads"])
    assert profile["caps"]["per_user_24h"] == 12
    built_in = client.post("/v1/dev/users", json={"user_id": "user-alice"}, headers=ADMIN)
    assert built_in.status_code == 409


def test_test_users_get_session_tokens_with_a_session_key(make_console):
    client, _ = make_console(session_key="ab" * 32)
    user = client.post("/v1/dev/users", json={}, headers=ADMIN).json()
    assert user["token_kind"] == "session"
    rank(client, {"Authorization": f"Bearer {user['token']}"})


def test_suppression_can_be_lifted(console):
    client, _ = console
    suppress = {"user_id": "user-alice", "hours": 24}
    assert (
        client.post("/v1/internal/suppressions", json=suppress, headers=SERVICE).status_code == 204
    )
    assert client.get("/v1/dev/users/user-alice", headers=ADMIN).json()["suppressed"] is True
    assert rank(client)["ads"] == []
    assert client.delete("/v1/dev/users/user-alice/suppression", headers=ADMIN).status_code == 200
    assert client.get("/v1/dev/users/user-alice", headers=ADMIN).json()["suppressed"] is False
    assert served(client)["ads"]


# ----- ranker lab -----


def test_preview_scores_every_policy_and_serves_nothing(console):
    client, container = console
    decisions = len(container.platform.decision_log())
    body = client.post("/v1/dev/rank-preview", json={"user_id": "user-alice"}, headers=ADMIN)
    assert body.status_code == 200, body.text
    result = body.json()
    assert result["blocked"] is None
    assert {p["name"] for p in result["policies"]} == set(POLICIES)
    eligible = {c["ad_id"] for c in result["candidates"] if c["eligible"]}
    assert eligible == {"ad_bookly", "ad_fitco", "ad_sleepwell", "ad_trailco"}
    rejections = {c["ad_id"]: c["rejection"] for c in result["candidates"]}
    assert rejections["ad_luckyspin"] == "restricted_category"
    for policy in result["policies"]:
        if not policy["available"]:
            continue
        assert set(policy["probabilities"]) <= eligible | {"NO_AD"}
        assert sum(policy["probabilities"].values()) == pytest.approx(1.0)
        assert policy["expected_revenue"] >= 0
    ranker = next(p for p in result["policies"] if p["name"] == "ranker")
    assert ranker["available"] is False
    assert "no CTR model" in ranker["error"]
    heuristic = next(p for p in result["policies"] if p["name"] == "heuristic")
    assert heuristic["serving"] is True
    assert heuristic["tv_from_serving"] == pytest.approx(0.0)
    assert {s["ad_id"] for s in result["scores"]} == eligible
    assert all(s["ml"] is None for s in result["scores"])
    # Read-only: no decision logged, no impression counted, no budget reserved.
    assert len(container.platform.decision_log()) == decisions
    profile = client.get("/v1/dev/users/user-alice", headers=ADMIN).json()
    assert profile["exposures_24h"]["total"] == 0
    budget = container.platform.budget
    assert budget.reserved("cmp_bookly", NOW.date()) == 0


@pytest.mark.parametrize(
    ("what_if", "blocked"),
    [
        ({"placement": "journal"}, "placement_not_allowed"),
        ({"age_band": "under_13"}, "under_13"),
        ({"suppressed": True}, "suppression_window"),
    ],
)
def test_preview_blocks_before_any_policy(console, what_if, blocked):
    client, _ = console
    result = client.post("/v1/dev/rank-preview", json=what_if, headers=ADMIN).json()
    assert result["blocked"] == blocked
    assert result["policies"] == []


def test_preview_what_ifs_change_eligibility(console):
    client, _ = console
    hidden = client.post(
        "/v1/dev/rank-preview",
        json={"user_id": "user-alice", "hidden_ads": ["ad_bookly"], "hour": 23},
        headers=ADMIN,
    ).json()
    rejections = {c["ad_id"]: c["rejection"] for c in hidden["candidates"]}
    assert rejections["ad_bookly"] == "hidden_by_user"
    assert hidden["daypart"] == "evening"
    teen = client.post("/v1/dev/rank-preview", json={"age_band": "teen"}, headers=ADMIN).json()
    assert teen["effective_consent"] == "contextual_only"
    library = client.post(
        "/v1/dev/rank-preview", json={"placement": "content_library"}, headers=ADMIN
    ).json()
    # FitCo targets the home feed only.
    rejections = {c["ad_id"]: c["rejection"] for c in library["candidates"]}
    assert rejections["ad_fitco"] == "placement_targeting"
    unknown = client.post("/v1/dev/rank-preview", json={"policies": ["rl"]}, headers=ADMIN)
    assert unknown.status_code == 422


def test_the_serving_policy_switches_at_runtime(console):
    client, _ = console
    switched = client.put("/v1/dev/serving-policy", json={"policy": "beta_ts"}, headers=ADMIN)
    assert switched.status_code == 200
    chain = switched.json()["chain"]
    assert [level["version"] for level in chain] == [
        "beta_ts_v1",
        "heuristic_eps_v1",
        "popularity_v1",
    ]
    assert served(client)["served_by"] == "bandit"
    # The uniform policy only serves the exploration holdout; a non-production model never.
    uniform = client.put("/v1/dev/serving-policy", json={"policy": "uniform"}, headers=ADMIN)
    assert uniform.status_code == 422
    ranker = client.put("/v1/dev/serving-policy", json={"policy": "ranker"}, headers=ADMIN)
    assert ranker.status_code == 409
    assert "production" in ranker.json()["detail"]
    audit = client.get("/v1/admin/audit", headers=ADMIN).json()["entries"]
    assert audit[0]["event"] == "serving_policy_changed"
    assert audit[0]["policy"] == "beta_ts"


def test_bandits_learn_from_every_served_impression(console):
    client, _ = console
    before = client.get("/v1/dev/bandit", headers=ADMIN).json()
    body = served(client)
    after = client.get("/v1/dev/bandit", headers=ADMIN).json()
    assert after["lin_ts"]["updates"] == before["lin_ts"]["updates"] + 1
    assert after["lin_ts"]["learns_online"] is True
    impressions = {a["ad_id"]: a["impressions"] for a in after["beta_ts"]["arms"]}
    assert impressions[body["ads"][0]["ad_id"]] == 1
    arm = next(a for a in after["beta_ts"]["arms"] if a["ad_id"] == "ad_bookly")
    assert arm["ci95"][0] < arm["posterior_mean"] < arm["ci95"][1]


def test_a_model_is_promoted_before_the_ranker_serves(make_console):
    client, _ = make_console(registry=True)
    models = client.get("/v1/dev/models", headers=ADMIN).json()
    assert models["loaded"]["source"] == "registry ctr/v1 (offline_validated)"
    assert models["loaded"]["production"] is False
    assert models["loaded"]["missing_features"] == []
    preview = client.post("/v1/dev/rank-preview", json={}, headers=ADMIN).json()
    ranker = next(p for p in preview["policies"] if p["name"] == "ranker")
    assert ranker["available"] is True
    assert all(s["ml"] is not None for s in preview["scores"])
    assert preview["reference"]["source"] == "ML ranker estimates"
    refused = client.put("/v1/dev/serving-policy", json={"policy": "ranker"}, headers=ADMIN)
    assert refused.status_code == 409

    promoted = client.post(
        "/v1/dev/models/promote", json={"name": "ctr", "version": "v1"}, headers=ADMIN
    )
    assert promoted.status_code == 200, promoted.text
    assert promoted.json()["registry"]["ctr"]["production"] == "v1"
    stages = [h["to"] for h in promoted.json()["registry"]["ctr"]["versions"][0]["history"]]
    assert stages[-3:] == ["shadow", "canary", "production"]
    assert client.put("/v1/dev/serving-policy", json={"policy": "ranker"}, headers=ADMIN).is_success
    assert served(client)["served_by"] == "ranker"
    # Nothing to roll back to: the request is refused and production keeps serving.
    rollback = client.post("/v1/dev/models/rollback", json={"name": "ctr"}, headers=ADMIN)
    assert rollback.status_code == 409
    missing = client.post(
        "/v1/dev/models/promote", json={"name": "ctr", "version": "v9"}, headers=ADMIN
    )
    assert missing.status_code == 409


def test_registry_paths_written_on_windows_load_everywhere(tmp_path):
    write_registry(tmp_path)
    index = tmp_path / "registry" / "registry.json"
    index.write_text(index.read_text().replace('"ctr/v1"', '"ctr\\\\v1"'))
    info = ModelRegistry(tmp_path / "registry").info("ctr", "v1")
    assert info.path == tmp_path / "registry" / "ctr" / "v1"
    assert (info.path / "manifest.json").exists()


# ----- decisions, labels, overview -----


def test_decisions_and_labels_are_inspectable(console):
    client, _ = console
    body = served(client)
    ad = body["ads"][0]
    detail = client.get(f"/v1/dev/decisions/{body['request_id']}", headers=ADMIN).json()
    record = detail["decision"]
    assert record["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert record["features"]  # the dev platform now builds and logs features
    chosen = [a for a in detail["actions"] if a["chosen"]]
    assert [a["action"] for a in chosen] == [ad["ad_id"]]
    assert detail["label"]["state"] == "pending"
    assert client.get(ad["click_url"], follow_redirects=False).status_code == 302
    label = client.get(f"/v1/dev/impressions/{ad['impression_id']}", headers=ADMIN).json()
    assert label["click_id"]
    recent = client.get("/v1/dev/decisions", params={"limit": 3}, headers=ADMIN).json()
    assert recent["decisions"][0]["request_id"] == body["request_id"]
    assert client.get("/v1/dev/decisions/missing", headers=ADMIN).status_code == 404
    assert client.get("/v1/dev/impressions/missing", headers=ADMIN).status_code == 404


def test_overview_reports_health_safety_and_fill(console):
    client, _ = console
    served(client)
    rank(client, placement="journal")
    overview = client.get("/v1/dev/overview", headers=ADMIN).json()
    assert all(overview["health"].values())
    assert overview["serving"]["policy"] == "heuristic"
    totals = overview["metrics"]["totals"]
    assert set(totals["safety_violations"].values()) == {0.0}
    assert totals["blocked"]["placement_not_allowed"] == 1
    assert overview["fill"]["served"] >= 1
    assert overview["fill"]["window"] >= overview["fill"]["served"] + 1
    assert overview["metrics"]["rank_latency_ms"]["count"] >= 2
    catalog = client.get("/v1/dev/catalog", headers=ADMIN).json()["campaigns"]
    bookly = next(c for c in catalog if c["campaign_id"] == "cmp_bookly")
    assert bookly["ads"][0]["ad_id"] == "ad_bookly"


def test_latency_quantiles_interpolate_inside_buckets():
    buckets = [(0.005, 0.0), (0.01, 50.0), (0.025, 100.0), (float("inf"), 100.0)]
    q = _quantiles(buckets)
    assert q["count"] == 100
    assert q["p50"] == pytest.approx(10.0)
    assert q["p95"] == pytest.approx(23.5)
    assert _quantiles([])["p50"] is None


# ----- advertiser and admin reads -----


def test_advertisers_list_only_their_own_campaigns_and_creatives(console):
    client, _ = console
    bookly = {"X-Advertiser-Key": KEYS["adv_bookly"]}
    trailco = {"X-Advertiser-Key": KEYS["adv_trailco"]}
    creative = {
        "ad_id": "ad_bookly_console",
        "campaign_id": "cmp_bookly",
        "declared_category": "books",
        "headline": "Cozy mysteries",
        "body": "Cozy mysteries for rainy evenings, delivered.",
        "landing_url": "https://bookly.example/new",
    }
    assert client.post("/v1/advertiser/creatives", json=creative, headers=bookly).status_code == 202
    mine = client.get("/v1/advertiser/creatives", headers=bookly).json()["creatives"]
    assert [c["ad_id"] for c in mine] == ["ad_bookly_console"]
    assert mine[0]["headline"] == "Cozy mysteries"
    assert client.get("/v1/advertiser/creatives", headers=trailco).json()["creatives"] == []
    campaigns = client.get("/v1/advertiser/campaigns", headers=bookly).json()["campaigns"]
    assert [c["campaign_id"] for c in campaigns] == ["cmp_bookly"]
    assert campaigns[0]["billed_today"] == 0
    assert {a["ad_id"] for a in campaigns[0]["ads"]} == {"ad_bookly", "ad_bookly_console"}


def test_admins_read_the_kill_switch_audit_log_and_claims(console):
    client, _ = console
    assert client.get("/v1/admin/kill-switch", headers=ADMIN).json()["kill_switch"] is False
    body = {"enabled": True, "reason": "console test"}
    assert client.put("/v1/admin/kill-switch", json=body, headers=ADMIN).status_code == 200
    assert client.get("/v1/admin/kill-switch", headers=ADMIN).json()["kill_switch"] is True
    entries = client.get("/v1/admin/audit", params={"limit": 1}, headers=ADMIN).json()["entries"]
    assert entries[0]["event"] == "kill_switch"
    assert entries[0]["reason"] == "console test"
    claims = client.get(
        "/v1/admin/claims", params={"advertiser_id": "adv_sleepwell"}, headers=ADMIN
    ).json()["claims"]
    assert [c["claim_id"] for c in claims] == ["clm_sleepwell_1"]


# ----- jobs -----


class FakeProcess:
    def __init__(self, lines, release=None):
        self._lines = lines
        self._release = release
        self.terminated = False
        self.stdout = self._stream()

    def _stream(self):
        yield from self._lines
        if self._release is not None:
            self._release.wait(5)

    def wait(self):
        return -15 if self.terminated else 0

    def terminate(self):
        self.terminated = True
        if self._release is not None:
            self._release.set()

    kill = terminate


def wait_done(runner: JobRunner, job_id: str, timeout: float = 60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = runner.get(job_id)
        if job is not None and job.status != "running":
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_job_parameters_are_validated(tmp_path):
    runner = JobRunner(tmp_path, backend="memory")
    with pytest.raises(JobError, match="unknown job"):
        runner.submit("rm", {})
    with pytest.raises(JobError, match="unknown parameters"):
        runner.submit("simulate", {"users": 100, "shell": "echo"})
    with pytest.raises(JobError, match="between"):
        runner.submit("simulate", {"users": 10**7})
    with pytest.raises(JobError, match="invalid value"):
        runner.submit("simulate", {"days": "3; rm -rf /"})
    with pytest.raises(JobError, match="invalid value"):
        runner.submit("review_eval", {"reviewer": "anthropic"})
    with pytest.raises(JobError, match="infra backend"):
        runner.submit("labels_finalize", {})
    with pytest.raises(JobError, match="missing"):
        runner.submit("rl_fqi", {})  # no data/raw in this working directory


def test_jobs_run_an_argument_list_without_billed_credentials(tmp_path, monkeypatch):
    calls = []

    def popen(argv, **kwargs):
        calls.append(SimpleNamespace(argv=argv, kwargs=kwargs))
        return FakeProcess(["eval done\n"])

    monkeypatch.setattr("theraflo_ads.api.jobs.subprocess.Popen", popen)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    runner = JobRunner(tmp_path, backend="memory")
    job = runner.submit("review_eval", {"reviewer": "rules"})
    done = wait_done(runner, job.job_id)
    assert done.status == "succeeded"
    assert "".join(done.output) == "eval done\n"
    call = calls[0]
    assert call.argv[:3] == [sys.executable, "-m", "theraflo_ads.cli"]
    assert call.argv[3:7] == ["review", "eval", "--reviewer", "rules"]
    assert "shell" not in call.kwargs
    assert "ANTHROPIC_API_KEY" not in call.kwargs["env"]
    assert call.kwargs["env"]["THERAFLO_ADS_REVIEWER"] == "none"


def test_one_job_runs_at_a_time_and_can_be_cancelled(tmp_path, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(
        "theraflo_ads.api.jobs.subprocess.Popen",
        lambda argv, **kwargs: FakeProcess(["working\n"], release),
    )
    runner = JobRunner(tmp_path, backend="memory")
    first = runner.submit("review_compare", {})
    with pytest.raises(JobError, match="another job"):
        runner.submit("experiment_mde", {})
    runner.cancel(first.job_id)
    assert wait_done(runner, first.job_id).status == "cancelled"
    second = runner.submit("experiment_mde", {})
    release.set()
    assert wait_done(runner, second.job_id).status in {"succeeded", "cancelled"}


def test_a_real_job_runs_the_cli(tmp_path):
    runner = JobRunner(tmp_path, backend="memory")
    job = runner.submit("experiment_mde", {"baseline_rate": 0.001})
    done = wait_done(runner, job.job_id, timeout=120)
    output = "".join(done.output)
    assert done.status == "succeeded", output
    assert "baseline rate 0.001" in output


def test_the_jobs_api(console):
    client, _ = console
    catalog = client.get("/v1/dev/jobs", headers=ADMIN).json()["catalog"]
    assert {"simulate", "review_eval", "rl_fqi"} <= {spec["name"] for spec in catalog}
    finalize = next(spec for spec in catalog if spec["name"] == "labels_finalize")
    assert finalize["unavailable"] == "needs the infra backend"
    bad = client.post(
        "/v1/dev/jobs", json={"name": "simulate", "params": {"users": -1}}, headers=ADMIN
    )
    assert bad.status_code == 409
    assert client.get("/v1/dev/jobs/nope", headers=ADMIN).status_code == 404


# ----- reports -----


def test_reports_are_listed_and_read_without_traversal(tmp_path):
    root = tmp_path / "reports"
    (root / "research" / "bundles").mkdir(parents=True)
    (root / "a.json").write_text('{"ok": true}')
    (root / "research" / "b.md").write_text("# B")
    (root / "research" / "bundles" / "manifest.json").write_text("{}")
    (root / "model.txt.bin").write_text("x")
    (tmp_path / "secret.json").write_text('{"key": 1}')
    assert [r["path"] for r in list_reports(root)] == ["a.json", "research/b.md"]
    assert read_report(root, "research/b.md") == ("# B", "md")
    for path in ("../secret.json", str(tmp_path / "secret.json"), "model.txt.bin", "nope.json"):
        with pytest.raises(KeyError):
            read_report(root, path)


def test_the_reports_api(console, tmp_path):
    client, _ = console
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "ope.json").write_text('{"estimate": 1}')
    listing = client.get("/v1/dev/reports", headers=ADMIN).json()["reports"]
    assert [r["path"] for r in listing] == ["ope.json"]
    report = client.get("/v1/dev/reports/ope.json", headers=ADMIN).json()
    assert report == {"path": "ope.json", "kind": "json", "content": '{"estimate": 1}'}
    assert client.get("/v1/dev/reports/..%2Fsecret.json", headers=ADMIN).status_code == 404


# ----- demo traffic -----


def test_demo_traffic_exercises_the_api(console):
    client, container = console
    users = [(ALICE_TOKEN, "user-alice")]
    demo = Demo("http://testserver", users, seed=5, client=client)
    for _ in range(12):
        demo.visit()
    demo.submit_creative()
    demo.crisis()
    stats = demo.snapshot()
    assert stats["rank"] == 12
    assert stats.get("ad_served", 0) + stats.get("no_ad", 0) == 12
    assert stats["creative"] == 1
    assert stats["suppression"] == 1
    assert len(container.platform.decision_log()) == 12


def test_the_traffic_runner_starts_and_stops(console):
    client, _ = console
    runner = TrafficRunner()
    demo = Demo("http://testserver", [(ALICE_TOKEN, "user-alice")], seed=5, client=client)
    state = runner.start_demo(demo, rate=40, minutes=0, population=1, floods=False)
    assert state["running"] is True
    with pytest.raises(RuntimeError, match="already running"):
        runner.start_demo(demo, rate=1, minutes=1, population=1, floods=False)
    deadline = time.monotonic() + 10
    while runner.status()["stats"].get("rank", 0) < 3 and time.monotonic() < deadline:
        time.sleep(0.05)
    stopped = runner.stop()
    assert stopped["running"] is False
    assert stopped["stats"]["rank"] >= 3


def test_the_traffic_api(console, monkeypatch):
    client, _ = console
    # A closed port: requests fail fast, and are counted rather than raised.
    monkeypatch.setenv("THERAFLO_ADS_CONSOLE_SELF_URL", "http://127.0.0.1:9")
    body = {"rate": 5, "minutes": 1, "population": 3, "floods": False}
    started = client.post("/v1/dev/traffic", json=body, headers=ADMIN)
    assert started.status_code == 200, started.text
    assert started.json()["running"] is True
    assert client.post("/v1/dev/traffic", json=body, headers=ADMIN).status_code == 409
    assert client.delete("/v1/dev/traffic", headers=ADMIN).json()["running"] is False
    too_big = {**body, "population": 10_000}
    assert client.post("/v1/dev/traffic", json=too_big, headers=ADMIN).status_code == 422
