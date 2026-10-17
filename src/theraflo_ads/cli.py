"""Command line entry point: `theraflo-ads <command>`.

migrate                 apply database migrations (Alembic)
gen-keys [--out .env]   fresh signing keys, printed or written to a .env file
seed-demo               load the demo catalog, priors, and profiles into Postgres
serve                   run the API (THERAFLO_ADS_BACKEND=memory|infra)
relay                   move committed outbox rows to RabbitMQ
worker <queue>          consume a queue: accounting | analytics | features
simulate [...]          run the simulator (see `theraflo-ads simulate --help`)
monitor drift           drift job over the decision log (reports/drift.json)
retrain                 validation + drift + one retraining-controller step
labels finalize         close label windows whose attribution has matured (Postgres)
rl fqi | rl ppo         offline FQI (OPE-gated) / simulator PPO, with safety evals
experiment mde [...]    minimum detectable effect table for a baseline rate and traffic
experiment simulate     simulated A/B test: heuristic vs the trained ML ranker
ml <stage>              training pipeline: generate | build | train | evaluate | register | all
                        | promote <name> <version> | rollback <name>   (used by dvc.yaml)
review eval             creative-review eval suite
                        (--reviewer rules|simulated|naive|ollama|anthropic)
review compare          rules vs simulated vs injection-following reviewer, with ablations
review export           SFT / preference data from labeled reviews, split by advertiser
review gate             post-training / prompt-change gate: candidate vs baseline eval reports
review worker           run the machine review for pending creatives (Postgres)
privacy delete-pending  run queued deletions: shred user keys, erase operational state (cron)
research scaling        learning curves, model size, feature ablation, revenue vs cost
research obd            OPE estimators validated on the Open Bandit Dataset (ZOZO, CC BY 4.0)
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from importlib import resources
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    from theraflo_ads.observability.logs import configure_logging

    # THERAFLO_ADS_LOG_FORMAT=json for one JSON object per line; secrets are always redacted.
    configure_logging()
    parser = argparse.ArgumentParser(
        prog="theraflo-ads",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("migrate", help="apply database migrations")
    keys = sub.add_parser("gen-keys", help="fresh signing keys for a .env file")
    keys.add_argument("--out", default=None, help="write them to this file (ASCII)")
    keys.add_argument("--force", action="store_true", help="replace an existing file")
    sub.add_parser("seed-demo", help="load demo inventory into Postgres")
    serve = sub.add_parser("serve", help="run the API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--workers", type=int, default=1)
    sub.add_parser("relay", help="run the outbox relay")
    worker = sub.add_parser("worker", help="run a queue consumer")
    worker.add_argument("queue", choices=["accounting", "analytics", "features"])
    sub.add_parser("simulate", help="run the simulator", add_help=False)
    monitor = sub.add_parser("monitor", help="drift monitoring")
    monitor.add_argument("action", choices=["drift"])
    monitor.add_argument("--raw", default="data/raw")
    monitor.add_argument(
        "--baseline-days",
        type=int,
        default=None,
        help="fixed reference window ending this many days after the data starts",
    )
    retrain = sub.add_parser("retrain", help="one retraining-controller step")
    retrain.add_argument("--raw", default="data/raw")
    retrain.add_argument("--state", default="models/retraining_state.json")
    retrain.add_argument("--baseline-days", type=int, default=None)
    labels = sub.add_parser("labels", help="label maintenance")
    labels.add_argument("action", choices=["finalize"])
    rl = sub.add_parser("rl", help="reinforcement learning experiments")
    rl.add_argument("algorithm", choices=["fqi", "ppo"])
    rl.add_argument("--raw", default="data/raw")
    rl.add_argument("--rounds", type=int, default=10)
    rl.add_argument("--out", default="reports/rl.json")
    exp = sub.add_parser("experiment", help="experiment design and simulated A/B tests")
    exp.add_argument("action", choices=["mde", "simulate"])
    exp.add_argument("--baseline-rate", type=float, default=0.0003)
    exp.add_argument("--daily-units", default="500000,112000")
    exp.add_argument("--lifts", default="0.02,0.05,0.10,0.20")
    exp.add_argument("--users", type=int, default=4000)
    exp.add_argument("--days", type=int, default=28)
    exp.add_argument("--pre-days", type=int, default=7)
    exp.add_argument("--ctr-model", default="models/candidates/ctr")
    exp.add_argument("--out", default="reports/experiment.json")
    ml = sub.add_parser("ml", help="training pipeline stages")
    ml.add_argument(
        "stage",
        choices=[
            "generate",
            "build",
            "train",
            "evaluate",
            "register",
            "all",
            "promote",
            "rollback",
        ],
    )
    ml.add_argument("name", nargs="?")
    ml.add_argument("version", nargs="?")
    ml.add_argument("--params", default="params.yaml")
    review = sub.add_parser("review", help="LLM-assisted creative review")
    review.add_argument("action", choices=["eval", "compare", "export", "gate", "worker"])
    review.add_argument(
        "--reviewer",
        default="simulated",
        choices=["rules", "simulated", "naive", "ollama", "anthropic"],
    )
    review.add_argument(
        "--ollama-model", default=None, help="with --reviewer ollama (default qwen3.5:9b)"
    )
    review.add_argument(
        "--think", action="store_true", help="with --reviewer ollama: reason before answering"
    )
    review.add_argument(
        "--live", action="store_true", help="required with --reviewer anthropic (billed calls)"
    )
    review.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    review.add_argument("--limit", type=int, default=None, help="evaluate at most N cases")
    review.add_argument("--out", default=None)
    review.add_argument("--candidate", default=None, help="candidate eval report (gate)")
    review.add_argument("--baseline", default=None, help="baseline eval report (gate)")
    review.add_argument("--registry", default="models/registry")
    review.add_argument("--interval", type=float, default=5.0)
    privacy = sub.add_parser("privacy", help="right-to-deletion job")
    privacy.add_argument("action", choices=["delete-pending"])
    research = sub.add_parser("research", help="learning-curve / scaling study")
    research.add_argument("study", choices=["scaling", "obd"])
    research.add_argument("--zip", default="data/external/obd/open_bandit_dataset.zip")
    research.add_argument(
        "--download", action="store_true", help="fetch the 413 MB dataset zip if missing"
    )
    research.add_argument("--repetitions", type=int, default=30)
    research.add_argument("--params", default="research/params.yaml")
    research.add_argument("--data", default="data/research")
    research.add_argument("--out", default="reports/research")
    research.add_argument("--sim-users", type=int, default=3000)
    research.add_argument("--sim-days", type=int, default=14)

    args, rest = parser.parse_known_args(argv)
    match args.command:
        case "migrate":
            migrate()
        case "gen-keys":
            gen_keys(args.out, force=args.force)
        case "seed-demo":
            seed_demo()
        case "serve":
            import uvicorn

            if args.workers > 1:
                # Found running the compose stack: with 2 workers, a scrape saw 17 of 20 requests.
                logging.getLogger(__name__).warning(
                    "--workers %d: each worker keeps its own Prometheus metrics, so /metrics "
                    "shows one worker's share; run one worker per container to monitor it",
                    args.workers,
                )
            uvicorn.run(
                "theraflo_ads.api.app:create_app",
                factory=True,
                host=args.host,
                port=args.port,
                workers=args.workers,
                log_config=None,  # keep the redacting / JSON handlers configured above
                access_log=False,  # the app writes its own access log without raw paths
            )
        case "relay":
            run_relay()
        case "worker":
            run_worker(args.queue)
        case "simulate":
            from theraflo_ads.simulator.__main__ import main as simulate

            simulate(rest)
        case "monitor":
            print_json(drift_report(Path(args.raw), args.baseline_days).to_dict())
        case "retrain":
            run_retrain(Path(args.raw), Path(args.state), args.baseline_days)
        case "labels":
            finalize_labels()
        case "rl":
            run_rl(args)
        case "experiment":
            run_experiment(args)
        case "ml":
            run_ml(args.stage, args.name, args.version, Path(args.params))
        case "review":
            run_review(args)
        case "privacy":
            run_deletions()
        case "research":
            run_research(args)


def migrate() -> None:
    from alembic import command
    from alembic.config import Config

    ini = resources.files("theraflo_ads.infra") / "migrations" / "alembic.ini"
    config = Config(str(ini))
    config.set_main_option("script_location", str(Path(str(ini)).parent))
    command.upgrade(config, "head")


def gen_keys(out: str | None = None, *, force: bool = False) -> None:
    """Writing the file here, instead of `gen-keys > .env`, keeps it ASCII in every shell:
    Windows PowerShell's `>` writes UTF-16, which Docker Compose cannot read."""
    import secrets

    lines = [
        f"THERAFLO_ADS_{name}={secrets.token_hex(32)}"
        for name in ("USER_HASH_KEY", "TOKEN_KEY", "SESSION_KEY")
    ]
    if out is None:
        print("\n".join(lines))
        return
    path = Path(out)
    if path.exists() and not force:
        raise SystemExit(
            f"{path} exists; --force replaces it (tokens and hashes made with its keys stop "
            "matching)"
        )
    path.write_bytes(("\n".join(lines) + "\n").encode("ascii"))
    print(f"wrote {len(lines)} keys to {path}")


def seed_demo() -> None:
    from theraflo_ads.api.container import DEMO_PRIORS, DEMO_PROFILES, demo_catalog, key_from_env
    from theraflo_ads.infra.postgres import make_engine
    from theraflo_ads.infra.settings import Settings
    from theraflo_ads.infra.stores import PgProfileStore, save_catalog
    from theraflo_ads.security.identifiers import UserHasher

    engine = make_engine(Settings())
    save_catalog(engine, demo_catalog(), DEMO_PRIORS)
    # Profiles are keyed by the user hash, so seeding needs the same key as the API.
    key = key_from_env("THERAFLO_ADS_USER_HASH_KEY", required=True)
    hasher = UserHasher("ads", {"k1": key}, "k1")
    profiles = PgProfileStore(engine, hasher)
    for user_id, profile in DEMO_PROFILES.items():
        profiles.upsert(
            user_id, consent=profile.consent, age_band=profile.age_band, region=profile.region
        )
    print("seeded demo catalog, priors, and profiles")


def _stop_event() -> threading.Event:
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    return stop


def run_relay() -> None:
    from theraflo_ads.infra.postgres import make_engine
    from theraflo_ads.infra.relay import OutboxRelay
    from theraflo_ads.infra.settings import Settings

    settings = Settings()
    OutboxRelay(make_engine(settings), settings.amqp_url).run_forever(_stop_event())


def run_worker(queue: str) -> None:
    from theraflo_ads.infra.settings import Settings
    from theraflo_ads.infra.worker import QueueWorker

    settings = Settings()
    if queue == "accounting":
        from theraflo_ads.events.accounting import AccountingHandler
        from theraflo_ads.infra.platform import make_redis
        from theraflo_ads.infra.postgres import PgBillingLedger, make_engine, pg_uow_factory
        from theraflo_ads.infra.redis_stores import RedisBudgetStore
        from theraflo_ads.infra.worker import accounting_dispatcher

        engine = make_engine(settings)
        budget = RedisBudgetStore(make_redis(settings), PgBillingLedger(engine))
        handler = accounting_dispatcher(
            AccountingHandler(uow_factory=pg_uow_factory(engine), budget=budget)
        )
    elif queue == "features":
        from theraflo_ads.infra.platform import make_redis
        from theraflo_ads.infra.postgres import make_engine
        from theraflo_ads.infra.redis_features import (
            FeatureAggregatorWorker,
            RedisClickCounts,
            RedisInteractionStore,
        )
        from theraflo_ads.infra.stores import load_catalog

        client = make_redis(settings)
        handler = FeatureAggregatorWorker(
            client,
            load_catalog(make_engine(settings)),
            RedisInteractionStore(client),
            RedisClickCounts(client),
        )
    else:
        from theraflo_ads.infra.lake import LakeWriter
        from theraflo_ads.infra.postgres import make_engine
        from theraflo_ads.infra.privacy_store import PgUserKeyStore
        from theraflo_ads.privacy.shredding import Pseudonymizer

        # Snapshots carry per-user-key pseudonyms, never user hashes (crypto-shredding).
        pseudonymizer = Pseudonymizer(PgUserKeyStore(make_engine(settings)))
        handler = LakeWriter(settings.lake_path, pseudonymizer)
    QueueWorker(
        settings.amqp_url, queue, handler, max_attempts=settings.worker_max_attempts
    ).run_forever(_stop_event())


def print_json(data) -> None:
    import json

    print(json.dumps(data, indent=2, default=str)[:6000])


def _raw_inputs(raw: Path):
    import json
    from datetime import datetime, timedelta

    from theraflo_ads.ml import pipeline

    meta = json.loads((raw / "meta.json").read_text(encoding="utf-8"))
    records = pipeline.load_records(raw)
    labels = pipeline.load_labels(raw)
    catalog = pipeline.load_catalog_json(raw / "catalog.json")
    now = max(datetime.fromisoformat(r["timestamp"]) for r in records) + timedelta(seconds=1)
    start = min(datetime.fromisoformat(r["timestamp"]) for r in records)
    return meta, records, labels, catalog, now, start


def drift_report(raw: Path, baseline_days: int | None):
    import json
    from datetime import timedelta

    from theraflo_ads.monitoring.jobs import run_drift_job

    meta, records, labels, catalog, now, start = _raw_inputs(raw)
    report = run_drift_job(
        records,
        labels,
        now=now,
        maturity=timedelta(seconds=meta["maturity_delay_s"]),
        categories={a.ad_id: a.category for a in catalog.ads.values()},
        baseline=start + timedelta(days=baseline_days) if baseline_days else None,
    )
    out = Path("reports/drift.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    return report


def run_retrain(raw: Path, state: Path, baseline_days: int | None) -> None:
    from theraflo_ads.ml import pipeline
    from theraflo_ads.ml.validation import validate
    from theraflo_ads.orchestration.controller import RetrainingController

    params = pipeline.load_params()
    features, models, reports = Path("data/features"), Path("models/candidates"), Path("reports")
    _, records, labels, _, now, _ = _raw_inputs(raw)
    quality = validate(records, labels, now=now)
    report = drift_report(raw, baseline_days)

    def retrain():
        pipeline.build(params, raw, features)
        pipeline.train(params, features, models)
        gates = pipeline.evaluate(params, features, models, raw, reports)["gates"]
        pipeline.register(models, reports, Path("models/registry"))
        return {"passed": gates["ctr"]["passed"], "reasons": gates["ctr"]["reasons"]}

    controller = RetrainingController(
        state, retrain=retrain, recalibrate=lambda: pipeline.recalibrate(params, features, models)
    )
    print_json(
        {
            "validation": {"passed": quality.passed, "failures": list(quality.failures)},
            "drift_detected": report.drift_detected,
            "step": controller.step(report, quality).to_dict(),
        }
    )


def finalize_labels() -> None:
    from theraflo_ads.attribution.service import AttributionService
    from theraflo_ads.domain import utc_now
    from theraflo_ads.infra.postgres import PgUnitOfWork, make_engine
    from theraflo_ads.infra.settings import Settings

    with PgUnitOfWork(make_engine(Settings())) as uow:
        count = AttributionService(uow.attribution).finalize(utc_now())
    print(f"finalized {count} labels")


def run_deletions() -> None:
    from theraflo_ads.domain import utc_now
    from theraflo_ads.infra.platform import make_redis
    from theraflo_ads.infra.postgres import make_engine
    from theraflo_ads.infra.privacy_store import PgDeletionRequests, PgUserKeyStore, pg_erasers
    from theraflo_ads.infra.settings import Settings
    from theraflo_ads.infra.stores import record_audit
    from theraflo_ads.privacy.shredding import DeletionService

    settings = Settings()
    engine = make_engine(settings)

    def audit(event: str, details: dict) -> None:
        record_audit(
            engine, event, "deletion_job", {name: str(value) for name, value in details.items()}
        )

    service = DeletionService(
        requests=PgDeletionRequests(engine),
        keys=PgUserKeyStore(engine),
        erasers=pg_erasers(engine, make_redis(settings)),
        clock=utc_now,
        on_event=audit,
    )
    done = service.run_pending()
    print(f"completed {len(done)} deletion requests")


def run_research(args: argparse.Namespace) -> None:
    """scaling: generates (once) a larger simulated log than the training pipeline's, then
    runs the study. obd: extracts (once) the Open Bandit Dataset columns, then validates."""
    if args.study == "obd":
        run_obd(args)
        return
    import yaml

    from theraflo_ads.ml import pipeline
    from theraflo_ads.research import scaling

    params = yaml.safe_load(Path(args.params).read_text(encoding="utf-8"))
    raw, features = Path(args.data) / "raw", Path(args.data) / "features"
    if not (raw / "meta.json").exists():
        pipeline.generate(params, raw)
    if not (features / "dataset.json").exists():
        pipeline.build(params, raw, features)
    scaling.run(
        features,
        Path(args.out),
        sim_users=args.sim_users,
        sim_days=args.sim_days,
        progress=lambda message: logging.getLogger("research").info(message),
    )
    print((Path(args.out) / "scaling.md").read_text(encoding="utf-8"))


def run_obd(args: argparse.Namespace) -> None:
    from theraflo_ads.research import obd

    archive = Path(args.zip)
    if not archive.exists():
        if not args.download:
            raise SystemExit(f"{archive} not found: pass --download to fetch it from {obd.URL}")
        import urllib.request

        archive.parent.mkdir(parents=True, exist_ok=True)
        # A fixed https URL (the dataset's official release), only with --download.
        urllib.request.urlretrieve(obd.URL, archive)  # nosec B310
    say = logging.getLogger("research").info
    data = archive.parent / "parquet"
    obd.extract(archive, data, progress=say)
    out = Path(args.out)
    obd.run(data, out, repetitions=args.repetitions, progress=say)
    print((out / "obd_ope.md").read_text(encoding="utf-8"))


def run_rl(args: argparse.Namespace) -> None:
    import json

    from theraflo_ads.rl import experiment as rl
    from theraflo_ads.rl.fqi import ActionFeaturizer, QPolicy
    from theraflo_ads.rl.ppo import PPOConfig
    from theraflo_ads.simulator.world import SimConfig

    config = rl.simulation_config()
    if args.algorithm == "fqi":
        trained = rl.train_fqi(Path(args.raw))
        q = trained["q"]

        def make(world):
            q.featurizer = ActionFeaturizer(world.catalog, world.features)
            return QPolicy(q, epsilon=0.1)

        extra = {"ope": trained["ope"], "fqi_history": trained["history"]}
    else:
        policy, history = rl.train_ppo(
            SimConfig(seed=7), rounds=args.rounds, config=PPOConfig(learning_rate=0.5)
        )

        def make(world):
            policy.featurizer = ActionFeaturizer(world.catalog, world.features)
            policy.buffer.clear()
            return policy

        extra = {"ppo_history": history}
    results = {
        "heuristic": rl.evaluate_in_simulator(None, config),
        args.algorithm: rl.evaluate_in_simulator(make, config),
    }
    report = {"simulator": rl.summarize(results), **extra}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print_json(report["simulator"])


def run_experiment(args: argparse.Namespace) -> None:
    import json

    if args.action == "mde":
        from theraflo_ads.experiments.power import mde_table

        daily = [float(x) for x in args.daily_units.split(",")]
        lifts = [float(x) for x in args.lifts.split(",")]
        print(f"baseline rate {args.baseline_rate}, 50/50 split, alpha 0.05, power 0.8")
        print(
            f"{'lift':>6} {'units/arm':>14} " + " ".join(f"{f'days@{d:,.0f}':>14}" for d in daily)
        )
        for row in mde_table(args.baseline_rate, lifts, daily):
            days = " ".join(f"{d:14,.0f}" for d in row.days)
            print(f"{row.relative_lift:6.0%} {row.units_per_arm:14,} {days}")
        return

    from theraflo_ads.simulator.ab import ABConfig, ml_arm, run_ab_test
    from theraflo_ads.simulator.world import SimConfig

    config = ABConfig(
        SimConfig(n_users=args.users, n_days=args.days, seed=7, run_seed=101),
        pre_days=args.pre_days,
    )
    report, context = run_ab_test(config, ml_arm(Path(args.ctr_model)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"report": report.to_dict(), "context": context}, indent=2, default=str),
        encoding="utf-8",
    )
    decision = report.decision
    print(f"promote: {decision.promote}")
    for reason in decision.reasons:
        print(f"  - {reason}")
    print(f"report written to {out}")


def run_ml(stage: str, name: str | None, version: str | None, params_path: Path) -> None:
    import json

    from theraflo_ads.ml import pipeline
    from theraflo_ads.ml.registry import ModelRegistry

    params = pipeline.load_params(params_path)
    raw, features, models = Path("data/raw"), Path("data/features"), Path("models/candidates")
    reports, registry = Path("reports"), Path("models/registry")
    stages = {
        "generate": lambda: pipeline.generate(params, raw),
        "build": lambda: pipeline.build(params, raw, features),
        "train": lambda: pipeline.train(params, features, models),
        "evaluate": lambda: pipeline.evaluate(params, features, models, raw, reports),
        "register": lambda: pipeline.register(models, reports, registry),
    }
    if stage == "all":
        for step in stages.values():
            step()
        result = json.loads((reports / "gates.json").read_text(encoding="utf-8"))
    elif stage == "promote":
        ModelRegistry(registry).promote_to_production(name, version, "promoted via CLI")
        result = {"promoted": f"{name}/{version}"}
    elif stage == "rollback":
        result = {"restored": ModelRegistry(registry).rollback(name, "rollback via CLI")}
    else:
        result = stages[stage]()
    print(json.dumps(result, indent=2, default=str)[:4000])


def _review_model(name: str, live: bool, ollama_model: str | None = None, think: bool = False):
    from theraflo_ads.review.simulated import SimulatedReviewerConfig, SimulatedReviewModel

    if name == "rules":
        return None
    if name == "simulated":
        return SimulatedReviewModel()
    if name == "naive":
        return SimulatedReviewModel(
            SimulatedReviewerConfig(name="simulated-naive-v1", follows_injections=True)
        )
    if name == "ollama":
        # A local model: free, no --live needed. Ollama must be running with the model pulled.
        from dataclasses import replace

        from theraflo_ads.review.local import (
            OllamaNarratorModel,
            OllamaReviewModel,
            OllamaSettings,
        )

        settings = OllamaSettings.from_env()
        settings = replace(
            settings, model=ollama_model or settings.model, think=think or settings.think
        )
        return OllamaReviewModel(settings=settings), OllamaNarratorModel(settings=settings)
    if not live:
        raise SystemExit("--reviewer anthropic calls Claude and is billed: add --live to confirm")
    from theraflo_ads.review.llm import AnthropicNarratorModel, AnthropicReviewModel

    return AnthropicReviewModel(), AnthropicNarratorModel()


def _write_json(path: Path, data) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def run_review(args: argparse.Namespace) -> None:
    from dataclasses import replace

    from theraflo_ads.review.evals import run_review_eval
    from theraflo_ads.review.evalset import build_evalset

    evalset = build_evalset()
    if args.limit:
        evalset = replace(evalset, cases=evalset.cases[: args.limit])
    if args.action == "eval":
        model = _review_model(args.reviewer, args.live, args.ollama_model, args.think)
        narrator = None
        if isinstance(model, tuple):
            model, narrator = model
        report = run_review_eval(
            evalset,
            model,
            split=args.split,
            narrator=narrator,
            narrator_reports=5 if narrator is not None else 40,
            progress=lambda i, n: print(f"\r{i}/{n}", end="", file=sys.stderr),
        )
        print(file=sys.stderr)
        out = Path(args.out or f"reports/review_eval_{args.reviewer}.json")
        _write_json(out, report.to_dict())
        print_json(_review_summary(report.to_dict()))
        print(f"report written to {out}")
    elif args.action == "compare":
        rows = {}
        for name in ("rules", "simulated", "naive"):
            report = run_review_eval(evalset, _review_model(name, False), split=args.split)
            rows[name] = report.to_dict()
        out = Path(args.out or "reports/review_compare.json")
        _write_json(out, rows)
        print_json({name: _review_summary(r) for name, r in rows.items()})
        print(f"report written to {out}")
    elif args.action == "export":
        from theraflo_ads.review.pipeline import ReviewConfig, ReviewPipeline
        from theraflo_ads.review.post_training import export_post_training, labels_from_evalset

        model = _review_model(args.reviewer, args.live)
        model = model[0] if isinstance(model, tuple) else model
        if model is None:
            raise SystemExit("export needs a model reviewer (its mistakes become preference pairs)")
        pipeline = ReviewPipeline(
            model=model,
            claims=evalset.claims_for,
            config=ReviewConfig(audit_rate=0.0, always_consult_model=True),
        )
        judgements = {}
        for case in evalset.cases:
            primary = [
                j for j in pipeline.review(case.creative).judgements if j.variant == "primary"
            ]
            if primary:
                judgements[case.creative.ad_id] = primary[0]
        result = export_post_training(
            labels_from_evalset(evalset.cases),
            judgements,
            Path(args.out or "data/post_training"),
            prompt_version=model.prompt_version,
            base_model=model.name,
        )
        print_json(result.manifest)
    elif args.action == "gate":
        run_review_gate(args)
    else:
        run_review_worker(args.interval, args.reviewer, args.live)


def _review_summary(report: dict) -> dict:
    keys = (
        "reviewer",
        "n",
        "false_approval",
        "false_rejection",
        "abstention",
        "automation_rate",
        "claim_grounding_rate",
        "injection_approvals",
        "health_adjacent_auto",
        "taxonomy_accuracy",
        "hallucinated_claims",
        "kappa",
        "passed",
        "gate_reasons",
    )
    summary = {k: report[k] for k in keys}
    summary["model_only"] = report["model_only"]
    summary["numeric_faithfulness"] = report["numeric_faithfulness"]
    return summary


def run_review_gate(args: argparse.Namespace) -> None:
    """Without --candidate/--baseline: demonstrates the flow with the simulated reviewer as the
    baseline and a candidate that misses fewer violations (a stand-in for a few-shot prompt or
    an adapter), both evaluated on the same set."""
    import json

    from theraflo_ads.ml.registry import ModelRegistry
    from theraflo_ads.review.evals import run_review_eval
    from theraflo_ads.review.evalset import build_evalset
    from theraflo_ads.review.post_training import post_training_gate, register_reviewer
    from theraflo_ads.review.simulated import SimulatedReviewerConfig, SimulatedReviewModel

    if args.candidate and args.baseline:
        candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        lineage = candidate.get("lineage", {})
    else:
        evalset = build_evalset()
        baseline = run_review_eval(evalset, SimulatedReviewModel(), split=args.split).to_dict()
        better = SimulatedReviewerConfig(name="simulated-reviewer-v2", miss_rate=0.04)
        candidate = run_review_eval(
            evalset, SimulatedReviewModel(better), split=args.split
        ).to_dict()
        lineage = {
            "base_model": "simulated-reviewer",
            "adapter_version": "v2-demo",
            "training_data_hash": "see data/post_training/manifest.json",
            "prompt_version": "simulated_prompt_v1",
            "eval_report": "reports/review_gate.json",
        }
    gate = post_training_gate(candidate, baseline, lineage)
    registry = ModelRegistry(Path(args.registry))
    version = register_reviewer(
        registry,
        spec={"reviewer": candidate["reviewer"], "prompt_version": candidate["prompt_version"]},
        report=candidate,
        lineage=lineage,
        gate=gate,
    )
    result = {
        "passed": gate.passed,
        "reasons": list(gate.reasons),
        "checks": gate.checks,
        "registered": f"creative_reviewer/{version}",
        "stage": registry.info("creative_reviewer", version).stage.value,
    }
    _write_json(Path(args.out or "reports/review_gate.json"), result)
    print_json(result)


def run_review_worker(interval: float, reviewer: str, live: bool) -> None:
    """Machine review for pending creatives. THERAFLO_ADS_REVIEWER selects the reviewer when
    --reviewer is left at its default."""
    import os

    from prometheus_client import start_http_server

    from theraflo_ads.domain import utc_now
    from theraflo_ads.infra.postgres import make_engine
    from theraflo_ads.infra.review_store import PgReviewRepository, ad_writer
    from theraflo_ads.infra.settings import Settings
    from theraflo_ads.infra.stores import load_catalog, record_audit
    from theraflo_ads.observability.metrics import Metrics
    from theraflo_ads.review.wiring import build_review_service, reviewer_from_env

    engine = make_engine(Settings())
    catalog = load_catalog(engine)
    if "THERAFLO_ADS_REVIEWER" in os.environ:
        model = reviewer_from_env()
    else:
        model = _review_model(reviewer, live)
        model = model[0] if isinstance(model, tuple) else model
    # Reviews happen here, not in the API, so this process serves its own /metrics
    # (scraped by Prometheus as theraflo-ads-review-worker).
    metrics = Metrics()
    port = int(os.environ.get("THERAFLO_ADS_METRICS_PORT", "9100"))
    start_http_server(port, registry=metrics.registry)

    def on_event(event: str, details: dict) -> None:
        if event == "creative_reviewed":
            metrics.review_outcomes.labels(details["outcome"]).inc()
        record_audit(engine, event, "review_worker", {k: str(v) for k, v in details.items()})

    service = build_review_service(
        catalog=catalog,
        repository=PgReviewRepository(engine),
        clock=utc_now,
        lock=threading.Lock(),
        model=model,
        persist_ad=ad_writer(engine),
        on_event=on_event,
    )
    stop = _stop_event()
    logging.getLogger(__name__).info(
        "review worker started (reviewer=%s, metrics on :%d)",
        getattr(model, "name", "rules only"),
        port,
    )
    while not stop.is_set():
        fresh = load_catalog(engine)  # campaigns created since the last pass
        catalog.campaigns.update(fresh.campaigns)
        for ad_id, ad in fresh.ads.items():
            catalog.ads.setdefault(ad_id, ad)
        if not service.process_pending(limit=50):
            stop.wait(interval)


if __name__ == "__main__":
    main(sys.argv[1:])
