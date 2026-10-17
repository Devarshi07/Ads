"""Post-training, Track B (design doc: Post-Training → Track B — LLM Components, Gates).

Start with prompting; post-train only when evals show prompting misses the targets.
This module prepares the data and enforces the gates; the fine-tuning job
itself (e.g. a LoRA adapter on a self-hosted reviewer) runs elsewhere.

- SFT: human-labeled policy decisions on advertiser creatives.
- Preference pairs (DPO-style): where a human corrected the model; corrections of
  false approvals are weighted more than corrections of false rejections.
- Inputs are advertiser creatives only. Theraflo user data (journals,
  conversations, wellness activity, anything keyed by a user) can never enter:
  records are built from `Creative` objects and checked field by field.
- Splits are by advertiser, so evals never test on an advertiser seen in training.
- The gate compares a candidate's eval report with the current reviewer's:
  no regression on adversarial / safety slices, no forgetting on general tasks,
  and complete lineage. The previous version stays registered for rollback.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from theraflo_ads.ml.registry import ModelRegistry, Stage
from theraflo_ads.review.creative import Confidence, Creative, Judgement, Verdict, Violation
from theraflo_ads.review.evalset import EvalCase, split_of
from theraflo_ads.review.grounding import extract_claims
from theraflo_ads.review.injection import render_untrusted
from theraflo_ads.review.llm import MODEL, system_prompt
from theraflo_ads.review.taxonomy import Taxonomy, default_taxonomy

DATASET_VERSION = "review_post_training_v1"
FALSE_APPROVAL_WEIGHT = 3.0
ALLOWED_FIELDS = frozenset(
    {
        "id",
        "advertiser_id",
        "split",
        "system",
        "input",
        "target",
        "chosen",
        "rejected",
        "weight",
        "correction",
        "source",
    }
)
FORBIDDEN_MARKERS = ("user_hash", "user_id", "session_id", "impression_id", "journal", "mood_log")
LINEAGE_KEYS = (
    "base_model",
    "adapter_version",
    "training_data_hash",
    "prompt_version",
    "eval_report",
)


@dataclass(frozen=True)
class HumanLabel:
    """A policy decision by a human reviewer: the training target."""

    creative: Creative
    verdict: Verdict  # approve | reject
    category: str
    violations: tuple[Violation, ...]
    source: str = "human_review"


def labels_from_evalset(cases: Iterable[EvalCase]) -> list[HumanLabel]:
    return [
        HumanLabel(
            c.creative,
            Verdict.APPROVE if c.expected == "approve" else Verdict.REJECT,
            c.true_category,
            (c.violation,) if c.violation else (),
            source="eval_label",
        )
        for c in cases
    ]


def target_payload(label: HumanLabel) -> dict[str, Any]:
    violations = [v.value for v in label.violations]
    rationale = (
        "Policy violation: " + ", ".join(violations) + "."
        if violations
        else "Complies with the ad policy."
    )
    return {
        "verdict": label.verdict.value,
        "category": label.category,
        "violations": violations,
        "claims": extract_claims(label.creative),
        "addresses_reviewer": Violation.PROMPT_INJECTION in label.violations,
        "confidence": Confidence.HIGH.value,
        "rationale": rationale,
    }


def correction_kind(label: HumanLabel, judgement: Judgement) -> str | None:
    if judgement.verdict is Verdict.APPROVE and label.verdict is Verdict.REJECT:
        return "false_approval"
    if judgement.verdict is Verdict.REJECT and label.verdict is Verdict.APPROVE:
        return "false_rejection"
    if judgement.category != label.category:
        return "category"
    if set(judgement.violations) != set(label.violations) and label.verdict is Verdict.REJECT:
        return "violation_label"
    return None


def _keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, inner in value.items():
            yield str(key)
            yield from _keys(inner)
    elif isinstance(value, list):
        for inner in value:
            yield from _keys(inner)


def assert_no_user_data(record: Mapping[str, Any]) -> None:
    """Records carry creative-derived fields only. A field naming user data anywhere in the
    record (top level or nested) aborts the export."""
    extra = set(record) - ALLOWED_FIELDS
    if extra:
        raise ValueError(f"unexpected fields in a training record: {sorted(extra)}")
    for key in _keys(record):
        if any(marker in key.lower() for marker in FORBIDDEN_MARKERS):
            raise ValueError(f"training record has a user-data field {key!r}")


@dataclass
class ExportResult:
    out_dir: Path
    counts: dict[str, dict[str, int]]
    manifest: dict[str, Any]
    corrections: dict[str, int] = field(default_factory=dict)


def export_post_training(
    labels: Sequence[HumanLabel],
    model_judgements: Mapping[str, Judgement],
    out_dir: Path,
    *,
    taxonomy: Taxonomy | None = None,
    prompt_version: str = "review_prompt_v1",
    salt: str = "split_v1",
    base_model: str = MODEL,
) -> ExportResult:
    """Writes {train,val,test}.sft.jsonl and {train,val,test}.dpo.jsonl plus a manifest.
    `model_judgements` maps ad_id → the model's primary judgement (for preference pairs)."""
    taxonomy = taxonomy or default_taxonomy()
    system = system_prompt(taxonomy, "primary")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    corrections: Counter = Counter()
    advertisers_by_split: dict[str, set[str]] = {}
    for label in labels:
        creative = label.creative
        split = split_of(creative.advertiser_id, salt)
        advertisers_by_split.setdefault(split, set()).add(creative.advertiser_id)
        target = target_payload(label)
        prompt = f"{render_untrusted(creative)}\n\nReview the creative above."
        sft = {
            "id": creative.ad_id,
            "advertiser_id": creative.advertiser_id,
            "split": split,
            "system": system,
            "input": prompt,
            "target": target,
            "source": label.source,
        }
        assert_no_user_data(sft)
        rows.setdefault(split, {"sft": [], "dpo": []})["sft"].append(sft)
        judgement = model_judgements.get(creative.ad_id)
        kind = correction_kind(label, judgement) if judgement is not None else None
        if kind is not None:
            corrections[kind] += 1
            pair = {
                "id": creative.ad_id,
                "advertiser_id": creative.advertiser_id,
                "split": split,
                "system": system,
                "input": prompt,
                "chosen": target,
                "rejected": judgement.payload(),
                "weight": FALSE_APPROVAL_WEIGHT if kind == "false_approval" else 1.0,
                "correction": kind,
            }
            assert_no_user_data(pair)
            rows[split]["dpo"].append(pair)

    overlap = [
        (a, b)
        for a in advertisers_by_split
        for b in advertisers_by_split
        if a < b and advertisers_by_split[a] & advertisers_by_split[b]
    ]
    if overlap:
        raise RuntimeError(f"advertisers leak across splits: {overlap}")

    hashes: dict[str, str] = {}
    counts: dict[str, dict[str, int]] = {}
    for split, kinds in sorted(rows.items()):
        counts[split] = {}
        for kind, records in kinds.items():
            path = out_dir / f"{split}.{kind}.jsonl"
            data = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
            path.write_bytes(data.encode("utf-8"))
            hashes[path.name] = hashlib.sha256(data.encode("utf-8")).hexdigest()
            counts[split][kind] = len(records)
    combined = hashlib.sha256(
        "".join(f"{k}:{v}" for k, v in sorted(hashes.items())).encode()
    ).hexdigest()
    manifest = {
        "version": DATASET_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "base_model": base_model,  # the reviewer whose judgements became preference pairs
        "prompt_version": prompt_version,
        "taxonomy_version": taxonomy.version,
        "split_salt": salt,
        "counts": counts,
        "corrections": dict(corrections),
        "false_approval_weight": FALSE_APPROVAL_WEIGHT,
        "files": hashes,
        "training_data_hash": combined,
        "contains_user_data": False,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return ExportResult(out_dir, counts, manifest, dict(corrections))


# ----- gates -----


ADVERSARIAL_SLICES = ("injection", "restricted_euphemism")
SAFETY_SLICES = ("restricted_explicit", "vulnerability", "personal_data", "unsubstantiated_claim")


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: tuple[str, ...]
    checks: Mapping[str, Any]


def _slice_auto_approved(report: Mapping[str, Any], slices: Sequence[str]) -> int:
    return sum(report["by_slice"].get(s, {}).get("auto_approved", 0) for s in slices)


def post_training_gate(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    lineage: Mapping[str, Any],
    *,
    noise_margin: float = 0.01,
    forgetting_margin: float = 0.01,
) -> GateResult:
    """Candidate and baseline are EvalReport dicts on the same eval set and split."""
    reasons = []
    if candidate.get("evalset_version") != baseline.get("evalset_version") or candidate.get(
        "split"
    ) != baseline.get("split"):
        reasons.append("candidate and baseline were evaluated on different sets")
    if not candidate.get("passed"):
        reasons.append(
            "candidate fails the absolute eval gates: "
            + "; ".join(candidate.get("gate_reasons", []))
        )
    adv_c = _slice_auto_approved(candidate, ADVERSARIAL_SLICES)
    adv_b = _slice_auto_approved(baseline, ADVERSARIAL_SLICES)
    if adv_c > adv_b:
        reasons.append(f"adversarial-set approvals {adv_c} > baseline {adv_b}")
    safe_c = _slice_auto_approved(candidate, SAFETY_SLICES)
    safe_b = _slice_auto_approved(baseline, SAFETY_SLICES)
    if safe_c > safe_b:
        reasons.append(f"safety-set approvals {safe_c} > baseline {safe_b}")
    fa_c, fa_b = candidate["false_approval"], baseline["false_approval"]
    if fa_c["count"] > fa_b["count"]:
        reasons.append(f"false approvals {fa_c['count']} > baseline {fa_b['count']}")
    if fa_c["ci95"][1] > fa_b["ci95"][1] + noise_margin:
        reasons.append("false-approval upper bound regressed")
    if candidate["taxonomy_accuracy"] < baseline["taxonomy_accuracy"] - forgetting_margin:
        reasons.append(
            f"taxonomy accuracy {candidate['taxonomy_accuracy']:.3f} < baseline "
            f"{baseline['taxonomy_accuracy']:.3f} (forgetting)"
        )
    fr_c, fr_b = candidate["false_rejection"]["rate"] or 0, baseline["false_rejection"]["rate"] or 0
    if fr_c > fr_b + 0.02:
        reasons.append(f"false-rejection rate {fr_c:.3f} > baseline {fr_b:.3f} + 0.02")
    missing = [k for k in LINEAGE_KEYS if not lineage.get(k)]
    if missing:
        reasons.append(f"lineage incomplete: {missing}")
    checks = {
        "adversarial_approvals": [adv_c, adv_b],
        "safety_approvals": [safe_c, safe_b],
        "false_approvals": [fa_c["count"], fa_b["count"]],
        "taxonomy_accuracy": [candidate["taxonomy_accuracy"], baseline["taxonomy_accuracy"]],
        "abstention": [candidate["abstention"]["rate"], baseline["abstention"]["rate"]],
    }
    return GateResult(not reasons, tuple(reasons), checks)


REVIEWER_COMPONENT = "creative_reviewer"


def register_reviewer(
    registry: ModelRegistry,
    *,
    spec: Mapping[str, Any],
    report: Mapping[str, Any],
    lineage: Mapping[str, Any],
    gate: GateResult | None = None,
) -> str:
    """Prompt and adapter versions are deployments: registered with their eval report and
    lineage, validated only through the gate, and rolled back like any model."""
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp)
        (bundle / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        (bundle / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (bundle / "lineage.json").write_text(json.dumps(lineage, indent=2), encoding="utf-8")
        metrics = {
            "false_approvals": report["false_approval"]["count"],
            "false_approval_upper": report["false_approval"]["ci95"][1],
            "abstention": report["abstention"]["rate"],
            "automation_rate": report["automation_rate"],
            "eval_passed": report["passed"],
        }
        version = registry.register(REVIEWER_COMPONENT, bundle, metrics=metrics)
    if report.get("passed") and (gate is None or gate.passed):
        registry.transition(
            REVIEWER_COMPONENT,
            version,
            Stage.OFFLINE_VALIDATED,
            "eval suite and post-training gate passed",
        )
    return version
