"""Model registry (design doc: Model Registry, Model Rollback, DVC + MLflow Responsibilities).

A filesystem registry with an explicit lifecycle. Rollback never retrains: the
previous production version and a known-safe baseline are always kept.
MLflow tracks runs (params, metrics, artifacts); this registry decides what
serves.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class Stage(StrEnum):
    TRAINED = "trained"
    OFFLINE_VALIDATED = "offline_validated"
    SHADOW = "shadow"
    CANARY = "canary"
    PRODUCTION = "production"
    RETIRED = "retired"


ALLOWED = {
    Stage.TRAINED: {Stage.OFFLINE_VALIDATED, Stage.RETIRED},
    Stage.OFFLINE_VALIDATED: {Stage.SHADOW, Stage.RETIRED},
    Stage.SHADOW: {Stage.CANARY, Stage.RETIRED},
    Stage.CANARY: {Stage.PRODUCTION, Stage.RETIRED},
    Stage.PRODUCTION: {Stage.RETIRED},
    Stage.RETIRED: set(),
}


class RegistryError(Exception):
    pass


@dataclass(frozen=True)
class VersionInfo:
    name: str
    version: str
    stage: Stage
    path: Path
    metrics: dict[str, Any]


class ModelRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._index = root / "registry.json"

    def register(self, name: str, bundle_dir: Path, *, metrics: dict | None = None) -> str:
        data = self._load()
        model = data["models"].setdefault(
            name,
            {"versions": {}, "production": None, "previous_production": None, "known_safe": None},
        )
        version = f"v{len(model['versions']) + 1}"
        target = self.root / name / version
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(bundle_dir, target)
        model["versions"][version] = {
            "stage": Stage.TRAINED.value,
            # POSIX separators: a registry written on Windows must load in the Linux image.
            "path": target.relative_to(self.root).as_posix(),
            "metrics": metrics or {},
            "history": [self._event(None, Stage.TRAINED, "registered")],
        }
        self._save(data)
        return version

    def transition(self, name: str, version: str, stage: Stage, reason: str) -> None:
        data = self._load()
        model = self._model(data, name)
        entry = model["versions"].get(version)
        if entry is None:
            raise RegistryError(f"unknown version {name}/{version}")
        current = Stage(entry["stage"])
        if stage not in ALLOWED[current]:
            raise RegistryError(f"{name}/{version}: {current} -> {stage} is not allowed")
        if stage is Stage.PRODUCTION:
            previous = model["production"]
            if previous and previous != version:
                prev_entry = model["versions"][previous]
                prev_entry["history"].append(
                    self._event(Stage.PRODUCTION, Stage.RETIRED, "replaced")
                )
                prev_entry["stage"] = Stage.RETIRED.value
                model["previous_production"] = previous
            model["production"] = version
        entry["history"].append(self._event(current, stage, reason))
        entry["stage"] = stage.value
        if stage is Stage.RETIRED and model["production"] == version:
            model["production"] = None
        self._save(data)

    def promote_to_production(self, name: str, version: str, reason: str) -> None:
        """Walk a validated model through shadow and canary into production (each step is
        recorded; in a live system each step is gated on its own evidence)."""
        for stage in (Stage.SHADOW, Stage.CANARY, Stage.PRODUCTION):
            current = self.info(name, version).stage
            if current is Stage.PRODUCTION:
                return
            if stage in ALLOWED[current]:
                self.transition(name, version, stage, reason)

    def mark_known_safe(self, name: str, version: str) -> None:
        data = self._load()
        self._model(data, name)["known_safe"] = version
        self._save(data)

    def rollback(self, name: str, reason: str) -> str:
        """Point production at the previous production version (or the known-safe one)."""
        data = self._load()
        model = self._model(data, name)
        target = model["previous_production"] or model["known_safe"]
        if not target or target == model["production"]:
            raise RegistryError(f"{name}: no version to roll back to")
        current = model["production"]
        if current:
            entry = model["versions"][current]
            entry["history"].append(
                self._event(Stage.PRODUCTION, Stage.RETIRED, f"rollback: {reason}")
            )
            entry["stage"] = Stage.RETIRED.value
        restored = model["versions"][target]
        restored["history"].append(
            self._event(Stage(restored["stage"]), Stage.PRODUCTION, f"rollback: {reason}")
        )
        restored["stage"] = Stage.PRODUCTION.value
        model["production"] = target
        model["previous_production"] = None
        self._save(data)
        return target

    def production(self, name: str) -> VersionInfo | None:
        data = self._load()
        model = data["models"].get(name)
        if not model or not model["production"]:
            return None
        return self.info(name, model["production"])

    def info(self, name: str, version: str) -> VersionInfo:
        entry = self._model(self._load(), name)["versions"][version]
        # Indexes written on Windows before the fix hold backslash paths.
        path = self.root.joinpath(*entry["path"].replace("\\", "/").split("/"))
        return VersionInfo(name, version, Stage(entry["stage"]), path, entry["metrics"])

    def versions(self, name: str) -> dict[str, Stage]:
        model = self._load()["models"].get(name, {"versions": {}})
        return {v: Stage(e["stage"]) for v, e in model["versions"].items()}

    def history(self, name: str, version: str) -> list[dict[str, Any]]:
        return self._model(self._load(), name)["versions"][version]["history"]

    def _model(self, data: dict, name: str) -> dict:
        if name not in data["models"]:
            raise RegistryError(f"unknown model {name}")
        return data["models"][name]

    def _load(self) -> dict:
        if not self._index.exists():
            return {"models": {}}
        return json.loads(self._index.read_text(encoding="utf-8"))

    def _save(self, data: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self._index.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self._index)  # atomic swap: a crash never leaves a half-written index

    @staticmethod
    def _event(before: Stage | None, after: Stage, reason: str) -> dict[str, Any]:
        return {
            "from": before.value if before else None,
            "to": after.value,
            "reason": reason,
            "at": datetime.now(UTC).isoformat(),
        }


def log_mlflow_run(
    *,
    tracking_uri: str,
    experiment: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: list[Path],
    tags: dict[str, str] | None = None,
) -> str | None:
    """Best effort: tracking must never break training. Returns the run ID, or None."""
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})
            mlflow.set_tags(tags or {})
            for path in artifacts:
                if path.is_dir():
                    mlflow.log_artifacts(str(path), artifact_path=path.name)
                elif path.exists():
                    mlflow.log_artifact(str(path))
            return run.info.run_id
    except Exception:
        return None
