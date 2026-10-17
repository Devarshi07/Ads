"""Training pipeline stages (design doc: Training Pipeline, DVC Strategy, Post-Training, Evals).

    generate   simulated traffic → data/raw (decision log, labels, catalog, priors)
    build      label-mature examples + time split → data/features
    train      CTR + post-click CVR models, calibrated → models/candidates
    evaluate   test metrics vs baseline, slices, gates, export parity, OPE → reports/
    register   gated: register both models as offline-validated

Each stage is a pure function of its inputs and params.yaml, so `dvc repro`
reruns only what changed. In production, `generate` is replaced by an export
of the lake partitions written by the analytics worker.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

# Only `git rev-parse HEAD` for lineage: fixed argv, no user input.
import subprocess  # nosec B404
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from theraflo_ads.attribution.service import ImpressionLabel, LabelState
from theraflo_ads.domain import Ad, AdCatalog, Campaign, PricingModel, ReviewStatus
from theraflo_ads.ml import dataset
from theraflo_ads.ml.bundle import load_bundle, save_bundle
from theraflo_ads.ml.evaluation import (
    GateConfig,
    classification_metrics,
    evaluate_gates,
    slice_metrics,
    slice_regressions,
)
from theraflo_ads.ml.features import FEATURE_SCHEMA_VERSION, FeatureBuilder
from theraflo_ads.ml.models import CalibratedModel, LightGBMModel, LogisticModel, PriorModel
from theraflo_ads.ml.post_training import export_parity, fit_calibrator
from theraflo_ads.ml.registry import ModelRegistry, Stage, log_mlflow_run
from theraflo_ads.ml.scorer import MLScorer
from theraflo_ads.ml.validation import validate
from theraflo_ads.ope.estimators import (
    doubly_robust,
    ips,
    samples_from_decisions,
    snips,
    user_clusters,
)
from theraflo_ads.ope.targets import category_reward_model, target_from_policy
from theraflo_ads.policy.decision import decision_from_record
from theraflo_ads.policy.policies import EpsilonGreedyPolicy
from theraflo_ads.policy.scoring import HeuristicScorer
from theraflo_ads.pricing import CategoryPriors
from theraflo_ads.safety.policy import default_safety_policy

REWARD_VERSION = "platform_revenue_v1"
SLICES = ("placement", "pricing_model", "consent_scope", "category")

DEFAULT_PARAMS: dict[str, Any] = {
    "simulation": {
        "users": 4000,
        "days": 28,
        "seed": 7,
        "epsilon": 0.1,
        "personalization_exclusions": True,
        "drift_day": None,
        "drift_ctr": {},
    },
    "split": {"validation_days": 4, "test_days": 6},
    "model": {
        "algorithm": "lightgbm",
        "num_boost_round": 400,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 50,
    },
    "calibration": {"ctr": "isotonic", "cvr": "platt"},
    "gates": {
        "max_ece": 0.02,
        "max_slice_log_loss_regression": 0.02,
        "max_log_loss_regression": 0.0,
    },
    "mlflow": {"tracking_uri": "sqlite:///mlflow.db", "experiment": "theraflo-ads-ranking"},
}


def load_params(path: Path = Path("params.yaml")) -> dict[str, Any]:
    params = json.loads(json.dumps(DEFAULT_PARAMS))
    if path.exists():
        import yaml

        for section, values in (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).items():
            params.setdefault(section, {}).update(values or {})
    return params


# ----- generate -----


def generate(params: dict[str, Any], out: Path) -> dict[str, Any]:
    from theraflo_ads.simulator import SimConfig, run_simulation

    sim = params["simulation"]
    result = run_simulation(
        SimConfig(
            n_users=sim["users"],
            n_days=sim["days"],
            seed=sim["seed"],
            epsilon=sim["epsilon"],
            personalization_exclusions=sim["personalization_exclusions"],
            drift_day=sim.get("drift_day"),
            drift_ctr=tuple((sim.get("drift_ctr") or {}).items()),
        )
    )
    out.mkdir(parents=True, exist_ok=True)
    records = pd.DataFrame(
        {
            "request_id": [d.request_id for d in result.decisions],
            "timestamp": [d.timestamp for d in result.decisions],
            "record": [json.dumps(d.to_log()) for d in result.decisions],
        }
    )
    records.to_parquet(out / "decisions.parquet", index=False)
    labels = pd.DataFrame(
        [_label_row(label) for label in result.platform.store.attribution.labels()]
    )
    labels.to_parquet(out / "labels.parquet", index=False)
    save_catalog_json(out / "catalog.json", result.world.catalog)
    (out / "priors.json").write_text(
        json.dumps({"ctr": dict(result.world.priors.ctr), "cvr": dict(result.world.priors.cvr)}),
        encoding="utf-8",
    )
    meta = {
        "cutoff": result.final_time.isoformat(),
        "maturity_delay_s": result.platform.attribution_policy.maturity_delay.total_seconds(),
        "attribution_version": result.platform.attribution_policy.version,
        "decisions": len(result.decisions),
        "impressions": int(len(labels)),
        "safety_violations": dict(result.violations),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# ----- build -----


def build(params: dict[str, Any], raw: Path, out: Path) -> dict[str, Any]:
    meta = json.loads((raw / "meta.json").read_text(encoding="utf-8"))
    catalog = load_catalog_json(raw / "catalog.json")
    records, labels = load_records(raw), load_labels(raw)
    quality = validate(records, labels, now=datetime.fromisoformat(meta["cutoff"]))
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation.json").write_text(
        json.dumps(dataclasses.asdict(quality), indent=2, default=str), encoding="utf-8"
    )
    if not quality.passed:
        raise RuntimeError(f"data validation failed: {'; '.join(quality.failures)}")
    frame = dataset.examples(
        records,
        labels,
        categories={a.ad_id: a.category for a in catalog.ads.values()},
        pricing={
            a.ad_id: catalog.campaign_for(a).pricing_model.value for a in catalog.ads.values()
        },
        cutoff=datetime.fromisoformat(meta["cutoff"]),
        maturity=timedelta(seconds=meta["maturity_delay_s"]),
    )
    split = params["split"]
    train, validation, test = dataset.time_split(
        frame, validation_days=split["validation_days"], test_days=split["test_days"]
    )
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("validation", validation), ("test", test)):
        part.to_parquet(out / f"{name}.parquet", index=False)
    info = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": dataset.feature_names(frame),
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "clicks": {
            "train": int(train.clicked.sum()),
            "validation": int(validation.clicked.sum()),
            "test": int(test.clicked.sum()),
        },
        "days": {
            "train": [train.day.min(), train.day.max()],
            "validation": [validation.day.min(), validation.day.max()],
            "test": [test.day.min(), test.day.max()],
        },
        "attribution_version": meta["attribution_version"],
    }
    (out / "dataset.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


# ----- train -----


def train(params: dict[str, Any], features: Path, out: Path) -> dict[str, Any]:
    info = json.loads((features / "dataset.json").read_text(encoding="utf-8"))
    names = info["feature_names"]
    columns = [f"{dataset.FEATURE_PREFIX}{n}" for n in names]
    train_df = pd.read_parquet(features / "train.parquet")
    validation = pd.read_parquet(features / "validation.parquet")
    lineage = {
        "git_commit": _git_commit(),
        "data_hash": _hash_dir(features),
        "training_window": info["days"]["train"],
        "feature_schema_version": info["feature_schema_version"],
        "attribution_version": info["attribution_version"],
        "label_dataset_version": _hash_dir(features)[:12],
        "reward_version": REWARD_VERSION,
        "params": {"model": params["model"], "calibration": params["calibration"]},
    }
    results = {}
    for name, label, rows, kind in (
        ("ctr", "clicked", lambda f: f, params["calibration"]["ctr"]),
        ("cvr", "converted", lambda f: f[f.clicked == 1], params["calibration"]["cvr"]),
    ):
        fit_rows, val_rows = rows(train_df), rows(validation)
        x, y = fit_rows[columns].to_numpy(float), fit_rows[label].to_numpy(float)
        xv, yv = val_rows[columns].to_numpy(float), val_rows[label].to_numpy(float)
        model = _fit(params["model"], x, y, names, (xv, yv), small=name == "cvr")
        calibrated = fit_calibrator(CalibratedModel(model), xv, yv, kind=kind)
        metrics = classification_metrics(yv, calibrated.predict(xv))
        run_id = log_mlflow_run(
            tracking_uri=params["mlflow"]["tracking_uri"],
            experiment=params["mlflow"]["experiment"],
            run_name=f"{name}-{model.algorithm}",
            params={**params["model"], "calibration": kind, "rows": len(fit_rows)},
            metrics={f"validation_{k}": v for k, v in metrics.items()},
            artifacts=[],
            tags={"model": name, "data_hash": lineage["data_hash"]},
        )
        save_bundle(
            out / name,
            name=name,
            version=lineage["data_hash"][:12],
            model=calibrated,
            feature_schema_version=info["feature_schema_version"],
            metrics={"validation": metrics},
            lineage={**lineage, "mlflow_run_id": run_id},
        )
        results[name] = {"rows": len(fit_rows), "validation": metrics, "mlflow_run_id": run_id}
    (out / "training.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def _fit(model_params: dict, x, y, names, valid, *, small: bool):
    if model_params["algorithm"] == "logistic":
        return LogisticModel.fit(x, y, names, c=0.5)
    lgb_params = {
        "learning_rate": model_params["learning_rate"],
        "num_leaves": 7 if small else model_params["num_leaves"],
        "min_data_in_leaf": 20 if small else model_params["min_data_in_leaf"],
    }
    return LightGBMModel.fit(
        x,
        y,
        names,
        valid=valid,
        params=lgb_params,
        num_boost_round=model_params["num_boost_round"],
    )


# ----- evaluate -----


def evaluate(params: dict[str, Any], features: Path, models: Path, raw: Path, out: Path) -> dict:
    info = json.loads((features / "dataset.json").read_text(encoding="utf-8"))
    names = info["feature_names"]
    columns = [f"{dataset.FEATURE_PREFIX}{n}" for n in names]
    test = pd.read_parquet(features / "test.parquet")
    schema = info["feature_schema_version"]
    ctr = load_bundle(models / "ctr", feature_schema_version=schema)
    cvr = load_bundle(models / "cvr", feature_schema_version=schema)
    gates_config = GateConfig(**params["gates"])
    report: dict[str, Any] = {}
    gate_report: dict[str, Any] = {}
    for bundle, label, rows, prior_column in (
        (ctr, "clicked", test, "prior_ctr"),
        (cvr, "converted", test[test.clicked == 1], "prior_cvr"),
    ):
        x = rows[columns].to_numpy(float)
        frame = rows.assign(
            candidate=bundle.model.predict(x),
            baseline=PriorModel(tuple(names), prior_column).predict_raw(x),
        )
        candidate = classification_metrics(frame[label].to_numpy(), frame.candidate.to_numpy())
        baseline = classification_metrics(frame[label].to_numpy(), frame.baseline.to_numpy())
        regressions = slice_regressions(
            frame, label, SLICES, min_positives=gates_config.min_slice_positives
        )
        gate = evaluate_gates(candidate, baseline, regressions, gates_config)
        report[bundle.name] = {
            "candidate": candidate,
            "baseline": baseline,
            "slices": slice_metrics(frame, label, "candidate", SLICES),
            "slice_regressions": [dataclasses.asdict(r) for r in regressions],
        }
        gate_report[bundle.name] = {"passed": gate.passed, "reasons": list(gate.reasons)}

    parity = {}
    golden = test[columns].head(200).to_numpy(float)
    for bundle, label in ((ctr, "clicked"), (cvr, "converted")):
        result = export_parity(
            bundle.model,
            golden,
            test[columns].to_numpy(float),
            test[label].to_numpy(float),
            feature_schema_version=schema,
        )
        parity[bundle.name] = dataclasses.asdict(result)
        gate_report[bundle.name]["parity_passed"] = result.passed
        if not result.passed:
            gate_report[bundle.name]["reasons"] += list(result.reasons)
            gate_report[bundle.name]["passed"] = False

    ope = evaluate_policies_offline(
        raw, ctr, cvr, test_start=info["days"]["test"][0], epsilon=params["simulation"]["epsilon"]
    )
    report["ope"] = ope
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(_flat_metrics(report), indent=2), encoding="utf-8")
    (out / "evaluation.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    (out / "parity.json").write_text(json.dumps(parity, indent=2), encoding="utf-8")
    (out / "ope.json").write_text(json.dumps(ope, indent=2), encoding="utf-8")
    (out / "gates.json").write_text(json.dumps(gate_report, indent=2), encoding="utf-8")
    return {"gates": gate_report, "ope": ope}


def evaluate_policies_offline(
    raw: Path, ctr, cvr, *, test_start: str, epsilon: float
) -> dict[str, Any]:
    """OPE on the test window's logs with platform revenue as the reward (design doc: Evals →
    OPE Evals). Targets explore at the logging policy's epsilon, which keeps importance
    weights bounded; a greedy target that rarely agrees with the logs is inconclusive."""
    catalog = load_catalog_json(raw / "catalog.json")
    priors = load_priors_json(raw / "priors.json")
    labels = load_labels(raw)
    decisions = [
        decision_from_record(r) for r in load_records(raw) if r["timestamp"][:10] >= test_start
    ]
    rewards = {label.request_id: label.platform_revenue for label in labels.values()}
    samples = samples_from_decisions(decisions, rewards)
    users = user_clusters(samples)
    reward_model = category_reward_model(samples, catalog)
    excluded = default_safety_policy().personalization_excluded_categories
    features = FeatureBuilder(catalog, priors, excluded_categories=excluded)
    policies = {
        "ml_ranker": MLScorer(catalog, features, ctr, cvr, priors=priors),
        "ml_ctr_prior_cvr": MLScorer(catalog, features, ctr, None, priors=priors),
        "heuristic": HeuristicScorer(catalog, priors),
    }
    logged = float(np.mean([s.reward for s in samples])) if samples else 0.0
    out: dict[str, Any] = {
        "samples": len(samples),
        "logging_policy_value": logged,
        "target_epsilon": epsilon,
    }
    for name, scorer in policies.items():
        target = target_from_policy(EpsilonGreedyPolicy(scorer, epsilon=epsilon), catalog)
        out[name] = {
            e.estimator: {
                "value": e.value,
                "ci": [e.ci_low, e.ci_high],
                "ess": e.ess,
                "conclusive": e.conclusive,
                "reason": e.reason,
                "top_weight_events": e.top_weight_events,
            }
            for e in (
                ips(samples, target, clusters=users),
                snips(samples, target, clusters=users),
                doubly_robust(samples, target, reward_model, clusters=users),
            )
        }
    return out


# ----- recalibrate -----


def recalibrate(params: dict[str, Any], features: Path, models: Path) -> dict[str, Any]:
    """The cheap response to calibration drift: refit the CTR calibrator on the most recent
    label-mature window, gate it on a later, held-out slice of that window, and write the
    result as a new candidate bundle."""
    info = json.loads((features / "dataset.json").read_text(encoding="utf-8"))
    columns = [f"{dataset.FEATURE_PREFIX}{n}" for n in info["feature_names"]]
    recent = pd.read_parquet(features / "test.parquet")
    days = sorted(recent.day.unique())
    fit_rows = recent[recent.day < days[len(days) // 2]]
    check_rows = recent[recent.day >= days[len(days) // 2]]
    current = load_bundle(models / "ctr", feature_schema_version=info["feature_schema_version"])
    refit = fit_calibrator(
        CalibratedModel(current.model.model),
        fit_rows[columns].to_numpy(float),
        fit_rows.clicked.to_numpy(float),
        kind=params["calibration"]["ctr"],
    )
    x, y = check_rows[columns].to_numpy(float), check_rows.clicked.to_numpy(float)
    before = classification_metrics(y, current.model.predict(x))
    after = classification_metrics(y, refit.predict(x))
    max_ece = params["gates"]["max_ece"]
    reasons = []
    if after["ece"] > max_ece:
        reasons.append(f"ECE {after['ece']:.4f} > {max_ece}")
    if after["log_loss"] > before["log_loss"]:
        reasons.append("recalibration made log loss worse")
    if not reasons:
        save_bundle(
            models / "ctr_recalibrated",
            name="ctr",
            version=f"{current.version}-recal",
            model=refit,
            feature_schema_version=info["feature_schema_version"],
            metrics={"recalibration_check": after, "before": before},
            lineage={**current.manifest["lineage"], "recalibrated_on": [days[0], days[-1]]},
        )
    return {
        "passed": not reasons,
        "reasons": reasons,
        "ece_before": before["ece"],
        "ece_after": after["ece"],
    }


# ----- register -----


def register(models: Path, reports: Path, registry_root: Path) -> dict[str, Any]:
    """Each component is promoted on its own evidence. A CVR model that loses to the priors is
    not registered; serving then uses the prior CVR with the ML CTR model."""
    gates = json.loads((reports / "gates.json").read_text(encoding="utf-8"))
    registry = ModelRegistry(registry_root)
    out: dict[str, Any] = {}
    for name in ("ctr", "cvr"):
        if not gates[name]["passed"]:
            out[name] = {"registered": False, "reasons": gates[name]["reasons"]}
            continue
        manifest = json.loads((models / name / "manifest.json").read_text(encoding="utf-8"))
        version = registry.register(name, models / name, metrics=manifest["metrics"])
        registry.transition(name, version, Stage.OFFLINE_VALIDATED, "offline gates passed")
        out[name] = {"registered": True, "version": version}
    if out["ctr"]["registered"] and not out["cvr"]["registered"]:
        out["cvr"]["serving_fallback"] = "category priors"
    (reports / "registered.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# ----- io -----


def load_records(raw: Path) -> list[dict]:
    frame = pd.read_parquet(raw / "decisions.parquet")
    return [json.loads(r) for r in frame["record"]]


def load_labels(raw: Path) -> dict[str, ImpressionLabel]:
    frame = pd.read_parquet(raw / "labels.parquet")
    labels = {}
    for row in frame.to_dict("records"):
        row["pricing_model"] = PricingModel(row["pricing_model"])
        row["state"] = LabelState(row["state"])
        for key in ("served_at", "clicked_at"):
            value = row[key]
            row[key] = None if pd.isna(value) else pd.Timestamp(value).to_pydatetime()
        for key in ("click_id", "reservation_id"):
            row[key] = None if row[key] is None or pd.isna(row[key]) else row[key]
        labels[row["impression_id"]] = ImpressionLabel(**row)
    return labels


def _label_row(label: ImpressionLabel) -> dict[str, Any]:
    row = dataclasses.asdict(label)
    row["pricing_model"] = label.pricing_model.value
    row["state"] = label.state.value
    return row


def save_catalog_json(path: Path, catalog: AdCatalog) -> None:
    data = {
        "campaigns": [
            {
                **dataclasses.asdict(c),
                "pricing_model": c.pricing_model.value,
                "placements": sorted(c.placements) if c.placements is not None else None,
            }
            for c in catalog.campaigns.values()
        ],
        "ads": [
            {**dataclasses.asdict(a), "review_status": a.review_status.value}
            for a in catalog.ads.values()
        ],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_catalog_json(path: Path) -> AdCatalog:
    data = json.loads(path.read_text(encoding="utf-8"))
    campaigns = {
        c["campaign_id"]: Campaign(
            **{
                **c,
                "pricing_model": PricingModel(c["pricing_model"]),
                "placements": frozenset(c["placements"]) if c["placements"] is not None else None,
            }
        )
        for c in data["campaigns"]
    }
    ads = {
        a["ad_id"]: Ad(**{**a, "review_status": ReviewStatus(a["review_status"])})
        for a in data["ads"]
    }
    return AdCatalog(campaigns, ads)


def load_priors_json(path: Path) -> CategoryPriors:
    data = json.loads(path.read_text(encoding="utf-8"))
    return CategoryPriors(ctr=data["ctr"], cvr=data["cvr"])


def _flat_metrics(report: dict[str, Any]) -> dict[str, float]:
    flat = {}
    for name in ("ctr", "cvr"):
        for which in ("candidate", "baseline"):
            for key in ("auc", "log_loss", "ece", "brier"):
                flat[f"{name}_{which}_{key}"] = report[name][which][key]
    for policy in ("ml_ranker", "ml_ctr_prior_cvr", "heuristic"):
        flat[f"ope_{policy}_snips"] = report["ope"][policy]["snips"]["value"]
    flat["ope_logging_policy_value"] = report["ope"]["logging_policy_value"]
    return flat


def _hash_dir(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*")):
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(  # nosec B603 B607
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "uncommitted"
