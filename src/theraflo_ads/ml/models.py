"""CTR / CVR models and calibrators (design doc: Prediction Models, Post-Training → Track A).

Every artifact serializes to a non-executable format (JSON, LightGBM text), never
pickle, so loading a model cannot run code.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

_EPS = 1e-6


class Model(Protocol):
    algorithm: str
    feature_names: tuple[str, ...]

    def predict_raw(self, x: np.ndarray) -> np.ndarray: ...

    def to_files(self) -> dict[str, str]: ...


# ----- calibrators -----


@dataclass(frozen=True)
class IsotonicCalibrator:
    """Piecewise-linear isotonic map fitted on a held-out, label-mature window."""

    x: tuple[float, ...]
    y: tuple[float, ...]
    kind: str = "isotonic"

    @classmethod
    def fit(cls, scores: np.ndarray, labels: np.ndarray) -> IsotonicCalibrator:
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip", y_min=_EPS, y_max=1 - _EPS)
        iso.fit(scores, labels)
        return cls(tuple(map(float, iso.X_thresholds_)), tuple(map(float, iso.y_thresholds_)))

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        if len(self.x) < 2:
            return np.full(len(scores), self.y[0] if self.y else 0.0)
        return np.interp(scores, self.x, self.y)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "x": list(self.x), "y": list(self.y)}


@dataclass(frozen=True)
class PlattCalibrator:
    a: float
    b: float
    kind: str = "platt"

    @classmethod
    def fit(cls, scores: np.ndarray, labels: np.ndarray) -> PlattCalibrator:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=1e6, max_iter=1000)
        model.fit(_logit(scores).reshape(-1, 1), labels)
        return cls(float(model.coef_[0, 0]), float(model.intercept_[0]))

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        return _sigmoid(self.a * _logit(scores) + self.b)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "a": self.a, "b": self.b}


@dataclass(frozen=True)
class IdentityCalibrator:
    kind: str = "identity"

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind}


Calibrator = IsotonicCalibrator | PlattCalibrator | IdentityCalibrator


def calibrator_from_dict(data: dict[str, Any]) -> Calibrator:
    match data["kind"]:
        case "isotonic":
            return IsotonicCalibrator(tuple(data["x"]), tuple(data["y"]))
        case "platt":
            return PlattCalibrator(data["a"], data["b"])
        case "identity":
            return IdentityCalibrator()
    raise ValueError(f"unknown calibrator {data['kind']!r}")


# ----- models -----


@dataclass
class LogisticModel:
    feature_names: tuple[str, ...]
    weights: np.ndarray
    bias: float
    mean: np.ndarray
    scale: np.ndarray
    algorithm: str = "logistic"

    @classmethod
    def fit(
        cls, x: np.ndarray, y: np.ndarray, feature_names: Sequence[str], *, c: float = 1.0
    ) -> LogisticModel:
        from sklearn.linear_model import LogisticRegression

        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale == 0] = 1.0
        model = LogisticRegression(C=c, max_iter=2000)
        model.fit((x - mean) / scale, y)
        return cls(
            tuple(feature_names), model.coef_[0].copy(), float(model.intercept_[0]), mean, scale
        )

    def predict_raw(self, x: np.ndarray) -> np.ndarray:
        return _sigmoid(((x - self.mean) / self.scale) @ self.weights + self.bias)

    def to_files(self) -> dict[str, str]:
        payload = {
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }
        return {"model.json": json.dumps(payload)}

    @classmethod
    def from_files(cls, files: dict[str, str], feature_names: Sequence[str]) -> LogisticModel:
        data = json.loads(files["model.json"])
        return cls(
            tuple(feature_names),
            np.array(data["weights"]),
            data["bias"],
            np.array(data["mean"]),
            np.array(data["scale"]),
        )


@dataclass
class LightGBMModel:
    feature_names: tuple[str, ...]
    booster: Any
    algorithm: str = "lightgbm"
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str],
        *,
        valid: tuple[np.ndarray, np.ndarray] | None = None,
        params: dict[str, Any] | None = None,
        num_boost_round: int = 300,
    ) -> LightGBMModel:
        import lightgbm as lgb

        params = {
            "objective": "binary",
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "verbose": -1,
            "seed": 7,
            "deterministic": True,
            "num_threads": 1,
            **(params or {}),
        }
        names = list(feature_names)
        train = lgb.Dataset(x, y, feature_name=names, free_raw_data=False)
        callbacks = []
        valid_sets = []
        if valid is not None:
            valid_sets = [lgb.Dataset(valid[0], valid[1], feature_name=names, reference=train)]
            callbacks = [lgb.early_stopping(30, verbose=False)]
        booster = lgb.train(
            params, train, num_boost_round, valid_sets=valid_sets, callbacks=callbacks
        )
        return cls(tuple(feature_names), booster, params=params)

    def predict_raw(self, x: np.ndarray) -> np.ndarray:
        return self.booster.predict(x, num_iteration=self.booster.best_iteration or None)

    def to_files(self) -> dict[str, str]:
        return {"model.txt": self.booster.model_to_string()}

    @classmethod
    def from_files(cls, files: dict[str, str], feature_names: Sequence[str]) -> LightGBMModel:
        import lightgbm as lgb

        return cls(tuple(feature_names), lgb.Booster(model_str=files["model.txt"]))


@dataclass
class PriorModel:
    """The heuristic baseline as a model: predicts the category prior already in the features."""

    feature_names: tuple[str, ...]
    column: str = "prior_ctr"
    algorithm: str = "prior"

    def predict_raw(self, x: np.ndarray) -> np.ndarray:
        return x[:, self.feature_names.index(self.column)]

    def to_files(self) -> dict[str, str]:
        return {"model.json": json.dumps({"column": self.column})}


@dataclass
class CalibratedModel:
    model: Model
    calibrator: Calibrator = field(default_factory=IdentityCalibrator)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.model.feature_names

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.clip(self.calibrator(self.model.predict_raw(x)), _EPS, 1 - _EPS)


def model_from_files(algorithm: str, files: dict[str, str], feature_names: Sequence[str]) -> Model:
    match algorithm:
        case "logistic":
            return LogisticModel.from_files(files, feature_names)
        case "lightgbm":
            return LightGBMModel.from_files(files, feature_names)
        case "prior":
            return PriorModel(tuple(feature_names), json.loads(files["model.json"])["column"])
    raise ValueError(f"unknown algorithm {algorithm!r}")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def logit(p: float) -> float:
    p = min(max(p, _EPS), 1 - _EPS)
    return math.log(p / (1 - p))
