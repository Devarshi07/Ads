"""Dev console: the `/v1/dev/*` API behind the browser console at `/ui`.

Mounted only with dev credentials: the memory backend, or the infra backend outside
production. Every route needs the dev admin token except `GET /v1/dev/identities`, which
hands the console the dev credentials; they are in the source code anyway, and refused
when THERAFLO_ADS_ENV=production.

The console uses the public API for everything a real client does (ranking, events,
clicks, postbacks, creatives, reports, admin review). These routes add what only a test
bench needs: test users, decision records, a read-only ranking preview, the serving-policy
switch, bandit and model state, demo traffic, offline jobs, and reports.

No `from __future__ import annotations`: FastAPI resolves the dependency aliases defined
inside dev_router() at definition time.
"""

import secrets
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

from theraflo_ads import __version__
from theraflo_ads.api.container import (
    DEV_ADMIN_TOKENS,
    DEV_ADVERTISER_KEYS,
    DEV_SERVICE_TOKENS,
    DEV_USER_TOKENS,
    Container,
)
from theraflo_ads.api.jobs import JobError, JobRunner, list_reports, read_report
from theraflo_ads.api.lab import POLICIES, SERVABLE, LabError
from theraflo_ads.counters.frequency import CounterUnavailable
from theraflo_ads.domain import AdRequest, AgeBand, ConsentScope, UserContext
from theraflo_ads.policy.decision import Decision
from theraflo_ads.review.creative import Violation
from theraflo_ads.review.store import QueueKind
from theraflo_ads.safety.engine import effective_consent
from theraflo_ads.safety.policy import AD_FREE_PLACEMENTS

_ID = r"^[A-Za-z0-9_-]+$"
Consent = Literal["personalized", "contextual_only"]
Age = Literal["adult", "teen", "under_13", "unknown"]


class NewUser(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str | None = Field(default=None, min_length=3, max_length=48, pattern=_ID)
    consent: Consent = "personalized"
    age_band: Age = "adult"
    region: str | None = Field(default=None, pattern=r"^[A-Z]{2,3}$")


class PreviewRequest(BaseModel):
    """A what-if request. With `user_id`, the user's real profile and exposures, with any
    field given here overriding the profile; without, a fresh synthetic user."""

    model_config = ConfigDict(extra="forbid")

    user_id: str | None = Field(default=None, min_length=1, max_length=128)
    placement: str = Field(default="home_feed", min_length=1, max_length=64, pattern=_ID)
    consent: Consent | None = None
    age_band: Age | None = None
    device: Literal["mobile", "web"] | None = None
    hour: int | None = Field(default=None, ge=0, le=23)
    suppressed: bool | None = None
    hidden_ads: list[str] | None = Field(default=None, max_length=50)
    policies: list[str] | None = Field(default=None, max_length=len(POLICIES))


class ServingRequest(BaseModel):
    policy: Literal["heuristic", "ranker", "beta_ts", "lin_ts", "popularity"]


class ModelAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["ctr", "cvr"]
    version: str | None = Field(default=None, pattern=r"^v[0-9]{1,4}$")


class TrafficRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rate: float = Field(default=8.0, gt=0, le=40)
    minutes: float = Field(default=10.0, ge=0, le=240)
    population: int = Field(default=300, ge=1, le=2000)
    seed: int = Field(default=7, ge=0, le=1_000_000)
    floods: bool = True


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64, pattern=_ID)
    params: dict[str, Any] = Field(default_factory=dict)


# Population mix for console traffic: (share, consent, age band).
TRAFFIC_MIX = (
    (0.55, ConsentScope.PERSONALIZED, AgeBand.ADULT),
    (0.30, ConsentScope.CONTEXTUAL_ONLY, AgeBand.ADULT),
    (0.10, ConsentScope.PERSONALIZED, AgeBand.TEEN),
    (0.05, ConsentScope.CONTEXTUAL_ONLY, AgeBand.UNKNOWN),
)


def dev_router(c: Container, admin_dependency, *, reports_root: Path, jobs: JobRunner):
    from theraflo_ads.devtools.traffic import TrafficRunner

    router = APIRouter(prefix="/v1/dev", tags=["dev console"])
    Admin = Annotated[str, Depends(admin_dependency)]
    traffic = TrafficRunner()

    def lab():
        if c.lab is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "ranker lab unavailable")
        return c.lab

    @router.get("/identities")
    def identities() -> dict:
        """Dev credentials for the console's identity switcher (never valid in production)."""
        catalog = c.catalog
        campaigns: dict[str, list[str]] = {}
        for campaign in catalog.campaigns.values():
            campaigns.setdefault(campaign.advertiser_id, []).append(campaign.campaign_id)
        taxonomy = c.taxonomy
        policy = c.platform.safety.current()
        return {
            "backend": c.backend,
            "version": __version__,
            "test_user_tokens": c.dev_users.kind if c.dev_users else None,
            "taxonomy": {
                "allowed": sorted(taxonomy.allowed),
                "restricted": sorted(taxonomy.restricted),
                "minor_restricted": sorted(taxonomy.minor_restricted),
                "health_adjacent": sorted(taxonomy.health_adjacent),
                "high_risk": sorted(taxonomy.high_risk),
            },
            "violations": [v.value for v in Violation],
            "placements": {
                "ads": sorted(policy.ad_placements),
                "ad_free": sorted(AD_FREE_PLACEMENTS),
            },
            "users": [{"token": t, "user_id": u} for t, u in DEV_USER_TOKENS.items()],
            "advertisers": [
                {"key": k, "advertiser_id": a, "campaigns": sorted(campaigns.get(a, []))}
                for k, a in DEV_ADVERTISER_KEYS.items()
            ],
            "admin_token": next(iter(DEV_ADMIN_TOKENS)),
            "service_token": next(iter(DEV_SERVICE_TOKENS)),
        }

    # ----- test users -----

    @router.post("/users", status_code=status.HTTP_201_CREATED)
    def create_user(body: NewUser, actor: Admin) -> dict:
        if c.dev_users is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "test users unavailable")
        user_id = body.user_id or f"test-{secrets.token_hex(3)}"
        if user_id in DEV_USER_TOKENS.values():
            raise HTTPException(status.HTTP_409_CONFLICT, "that is a built-in dev user")
        with c.lock:
            c.profiles.upsert(
                user_id,
                consent=ConsentScope(body.consent),
                age_band=AgeBand(body.age_band),
                region=body.region,
            )
        token = c.dev_users.issue(user_id)
        c.record_audit("dev_user_created", actor, consent=body.consent, age_band=body.age_band)
        return {"user_id": user_id, "token": token, "token_kind": c.dev_users.kind}

    @router.get("/users/{user_id}")
    def user_profile(user_id: str, actor: Admin) -> dict:
        now = c.clock()
        with c.lock:
            user = c.profiles.context(user_id)
            try:
                snapshot = c.platform.frequency.snapshot(user.user_hash, now)
            except CounterUnavailable:
                snapshot = None
        caps = c.platform.frequency.caps
        holdout = c.platform.decisions.holdouts.assign(user.user_hash)
        return {
            "user_id": user_id,
            "user_hash": user.user_hash,
            "consent": user.consent.value,
            "effective_consent": effective_consent(user.consent, user.age_band).value,
            "age_band": user.age_band.value,
            "region": user.region,
            "suppressed_until": _iso(user.suppressed_until),
            "suppressed": user.suppressed_until is not None and now < user.suppressed_until,
            "hidden_ads": sorted(user.hidden_ads),
            "hidden_advertisers": sorted(user.hidden_advertisers),
            "holdout": holdout.value,
            "exposures_24h": (
                {"total": snapshot.user_count, "by_ad": dict(snapshot.ad_counts)}
                if snapshot is not None
                else None
            ),
            "caps": {
                "per_user_24h": caps.per_user_24h,
                "per_ad_24h": caps.per_ad_24h,
                "per_advertiser_24h": caps.per_advertiser_24h,
                "min_gap_same_ad_s": caps.min_gap_same_ad.total_seconds(),
            },
        }

    @router.delete("/users/{user_id}/suppression")
    def lift_suppression(user_id: str, actor: Admin) -> dict:
        """Ends a suppression window now (the window is otherwise 24h)."""
        with c.lock:
            c.profiles.suppress(user_id, c.clock())
        c.record_audit("dev_suppression_lifted", actor)
        return {"suppressed": False}

    # ----- overview -----

    @router.get("/overview")
    def overview(actor: Admin) -> dict:
        policy = c.platform.safety.current()
        queue = {}
        if c.review is not None:
            queue = {kind.value: len(c.review.queue(kind)) for kind in QueueKind}
        with c.lock:
            recent = c.platform.recent_decisions(200)
        return {
            # Ad-free screens, caps, holdouts, and NO_AD samples all count as unfilled.
            "fill": {
                "window": len(recent),
                "served": sum(1 for d in recent if d.impression_id is not None),
            },
            "backend": c.backend,
            "version": __version__,
            "health": dict(c.platform.health()),
            "kill_switch": policy.kill_switch,
            "safety_policy_version": policy.version,
            "ad_placements": sorted(policy.ad_placements),
            "ad_free_placements": sorted(AD_FREE_PLACEMENTS),
            "serving": c.lab.serving_state() if c.lab else None,
            "review_queue": queue,
            "reviewer": _reviewer(c),
            "metrics": c.metrics.summary(),
            "traffic": traffic.status(),
        }

    @router.get("/catalog")
    def catalog(actor: Admin) -> dict:
        now = c.clock()
        catalog = c.catalog
        campaigns = []
        for campaign in catalog.campaigns.values():
            campaigns.append(
                {
                    "campaign_id": campaign.campaign_id,
                    "advertiser_id": campaign.advertiser_id,
                    "pricing_model": campaign.pricing_model.value,
                    "price": campaign.price,
                    "daily_budget": campaign.daily_budget,
                    "billed_today": _safe(
                        lambda cid=campaign.campaign_id: c.platform.budget.billed(cid, now.date())
                    ),
                    "active": campaign.active,
                    "placements": sorted(campaign.placements) if campaign.placements else None,
                    "priority": campaign.priority,
                    "ads": [
                        {
                            "ad_id": ad.ad_id,
                            "category": ad.category,
                            "review_status": ad.review_status.value,
                            "landing_url": ad.landing_url,
                        }
                        for ad in catalog.ads.values()
                        if ad.campaign_id == campaign.campaign_id
                    ],
                }
            )
        return {"campaigns": sorted(campaigns, key=lambda x: x["campaign_id"])}

    # ----- decisions -----

    @router.get("/decisions")
    def recent_decisions(actor: Admin, limit: Annotated[int, Query(ge=1, le=200)] = 25) -> dict:
        with c.lock:
            decisions = c.platform.recent_decisions(limit)
        return {"decisions": [_summary(d) for d in decisions]}

    @router.get("/decisions/{request_id}")
    def decision(request_id: str, actor: Admin) -> dict:
        with c.lock:
            found = c.platform.find_request(request_id)
        if found is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such decision")
        catalog = c.catalog
        actions = []
        for action, p in sorted(found.action_probabilities.items(), key=lambda x: -x[1]):
            ad = catalog.ads.get(action)
            actions.append(
                {
                    "action": action,
                    "probability": p,
                    "category": ad.category if ad else None,
                    "advertiser_id": catalog.campaign_for(ad).advertiser_id if ad else None,
                    "chosen": action == found.chosen_action,
                }
            )
        return {
            "decision": found.to_log(),
            "actions": actions,
            "label": _label(c, found.impression_id),
        }

    @router.get("/impressions/{impression_id}")
    def impression(impression_id: str, actor: Admin) -> dict:
        """The attribution label: viewable, click, conversions, and what Theraflo earned."""
        label = _label(c, impression_id)
        if label is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such impression")
        return label

    # ----- ranker lab -----

    @router.get("/policies")
    def policies(actor: Admin) -> dict:
        return {
            "policies": [
                {"name": name, "description": text, "servable": name in SERVABLE}
                for name, text in POLICIES.items()
            ],
            "serving": lab().serving_state(),
        }

    @router.post("/rank-preview")
    def rank_preview(body: PreviewRequest, actor: Admin) -> dict:
        """Every policy scores one request; nothing is served, spent, or counted."""
        the_lab = lab()
        now = c.clock()
        if body.hour is not None:
            now = now.replace(hour=body.hour, minute=0, second=0, microsecond=0)
        with c.lock:
            if body.user_id:
                user = c.profiles.context(body.user_id, device=body.device)
            else:
                user = UserContext(
                    user_hash=f"preview-{uuid4().hex[:16]}",
                    consent=ConsentScope.PERSONALIZED,
                    age_band=AgeBand.ADULT,
                    device=body.device,
                )
            overrides: dict[str, Any] = {}
            if body.consent is not None:
                overrides["consent"] = ConsentScope(body.consent)
            if body.age_band is not None:
                overrides["age_band"] = AgeBand(body.age_band)
            if body.suppressed is not None:
                overrides["suppressed_until"] = (
                    now + timedelta(hours=1) if body.suppressed else None
                )
            if body.hidden_ads is not None:
                overrides["hidden_ads"] = frozenset(body.hidden_ads)
            user = replace(user, **overrides)
            request = AdRequest(
                request_id=f"preview-{uuid4().hex[:12]}",
                session_id="console-preview",
                placement_id=body.placement,
                user=user,
                now=now,
            )
            try:
                result = the_lab.preview(request, body.policies or tuple(POLICIES))
            except LabError as exc:
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from None
        result["user"] = {
            "user_id": body.user_id,
            "synthetic": not body.user_id,
            "consent": user.consent.value,
            "age_band": user.age_band.value,
            "suppressed": user.suppressed_until is not None and now < user.suppressed_until,
            "hidden_ads": sorted(user.hidden_ads),
            "holdout": c.platform.decisions.holdouts.assign(user.user_hash).value,
        }
        result["placement"] = body.placement
        return result

    @router.put("/serving-policy")
    def set_serving_policy(body: ServingRequest, actor: Admin) -> dict:
        try:
            with c.lock:
                state = lab().serve(body.policy)
        except LabError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from None
        c.record_audit("serving_policy_changed", actor, policy=body.policy)
        return state

    @router.get("/bandit")
    def bandit(actor: Admin) -> dict:
        with c.lock:
            return lab().bandit_state()

    @router.get("/models")
    def models(actor: Admin) -> dict:
        return lab().model_report()

    @router.post("/models/promote")
    def promote(body: ModelAction, actor: Admin) -> dict:
        if body.version is None:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "version is required")
        try:
            with c.lock:
                report = lab().promote(body.name, body.version)
        except LabError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from None
        c.record_audit("model_promoted", actor, model=body.name, version=body.version)
        return report

    @router.post("/models/rollback")
    def rollback(body: ModelAction, actor: Admin) -> dict:
        try:
            with c.lock:
                report = lab().rollback(body.name)
        except LabError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from None
        c.record_audit("model_rolled_back", actor, model=body.name)
        return report

    # ----- demo traffic -----

    @router.get("/traffic")
    def traffic_status(actor: Admin) -> dict:
        return traffic.status()

    @router.post("/traffic")
    def start_traffic(body: TrafficRequest, request: Request, actor: Admin) -> dict:
        from theraflo_ads.devtools.traffic import Demo

        if traffic.status()["running"]:
            raise HTTPException(status.HTTP_409_CONFLICT, "traffic is already running")
        if c.dev_users is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "test users unavailable")
        users = _traffic_users(c, body.population, body.seed)
        catalog = c.catalog
        advertiser_of = {
            ad.ad_id: catalog.campaign_for(ad).advertiser_id for ad in catalog.ads.values()
        }
        demo = Demo(_self_url(request), users, body.seed, advertiser_of=advertiser_of)
        try:
            state = traffic.start_demo(
                demo,
                rate=body.rate,
                minutes=body.minutes,
                population=body.population,
                floods=body.floods,
            )
        except RuntimeError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from None
        c.record_audit("demo_traffic_started", actor, rate=body.rate, minutes=body.minutes)
        return state

    @router.delete("/traffic")
    def stop_traffic(actor: Admin) -> dict:
        return traffic.stop()

    # ----- jobs and reports -----

    @router.get("/jobs")
    def list_jobs(actor: Admin) -> dict:
        return {"catalog": jobs.catalog(), "jobs": [j.to_dict() for j in jobs.jobs()]}

    @router.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
    def run_job(body: JobRequest, actor: Admin) -> dict:
        try:
            job = jobs.submit(body.name, body.params)
        except JobError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from None
        c.record_audit("job_started", actor, job=body.name)
        return job.to_dict(with_output=True)

    @router.get("/jobs/{job_id}")
    def job(job_id: str, actor: Admin) -> dict:
        found = jobs.get(job_id)
        if found is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such job")
        return found.to_dict(with_output=True)

    @router.delete("/jobs/{job_id}")
    def cancel_job(job_id: str, actor: Admin) -> dict:
        try:
            return jobs.cancel(job_id).to_dict(with_output=True)
        except KeyError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such job") from None

    @router.get("/reports")
    def reports(actor: Admin) -> dict:
        return {"root": str(reports_root), "reports": list_reports(reports_root)}

    @router.get("/reports/{path:path}")
    def report(path: str, actor: Admin) -> dict:
        try:
            content, kind = read_report(reports_root, path)
        except KeyError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such report") from None
        except JobError as exc:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, str(exc)) from None
        return {"path": path, "kind": kind, "content": content}

    return router


def _traffic_users(c: Container, population: int, seed: int) -> list[tuple[str, str]]:
    """Fresh test users with a realistic consent and age mix (yesterday's are capped)."""
    import random

    rng = random.Random(seed)
    run = secrets.token_hex(3)
    users = []
    for i in range(population):
        point, cumulative = rng.random(), 0.0
        consent, age = TRAFFIC_MIX[-1][1], TRAFFIC_MIX[-1][2]
        for share, mix_consent, mix_age in TRAFFIC_MIX:
            cumulative += share
            if point < cumulative:
                consent, age = mix_consent, mix_age
                break
        user_id = f"traffic-{run}-{i}"
        with c.lock:
            c.profiles.upsert(user_id, consent=consent, age_band=age)
        users.append((c.dev_users.issue(user_id), user_id))
    return users


def _self_url(request: Request) -> str:
    """This API, from inside its own process: the port it was reached on, over loopback."""
    import os

    configured = os.environ.get("THERAFLO_ADS_CONSOLE_SELF_URL")
    if configured:
        return configured.rstrip("/")
    server = request.scope.get("server") or ("127.0.0.1", 8000)
    return f"http://127.0.0.1:{server[1]}"


def _reviewer(c: Container) -> str | None:
    if c.review is None:
        return None
    model = getattr(c.review.pipeline, "model", None)
    if model is None:
        return "rules and human queue (no model)" if c.backend == "memory" else "review worker"
    return getattr(model, "name", type(model).__name__)


def _summary(d: Decision) -> dict[str, Any]:
    return {
        "request_id": d.request_id,
        "timestamp": d.timestamp.isoformat(),
        "placement": d.placement,
        "served_by": d.served_by.value,
        "chosen_action": d.chosen_action,
        "propensity": d.action_probabilities.get(d.chosen_action),
        "eligible": len([a for a in d.eligible_actions if a != "NO_AD"]),
        "blocked_reason": d.blocked_reason,
        "degraded_reasons": list(d.degraded_reasons),
        "holdout": d.holdout.value,
        "consent_scope": d.consent_scope.value,
        "policy_version": d.policy_version,
        "ope_eligible": d.ope_eligible,
        "impression_id": d.impression_id,
    }


def _label(c: Container, impression_id: str | None) -> dict[str, Any] | None:
    if impression_id is None:
        return None
    with c.lock, c.platform.read_attribution() as attribution:
        label = attribution.label(impression_id)
    return _plain(asdict(label)) if label is not None else None


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_plain(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _safe(read) -> float | None:
    try:
        return float(read())
    except Exception:
        return None
