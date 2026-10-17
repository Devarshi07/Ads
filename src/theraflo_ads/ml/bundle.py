"""Model bundles (design doc: Model Registry, Model-Service Reliability, Security).

A bundle is a directory: model file(s), calibrator.json, manifest.json. The
manifest records lineage (git commit, DVC data hash, training window,
attribution/label/reward versions, MLflow run), metrics, and a SHA-256 for
every file. With a signing key it also carries an HMAC signature; loading
verifies checksums, the signature, and the feature schema before anything is
served.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from theraflo_ads.ml.models import CalibratedModel, calibrator_from_dict, model_from_files


class BundleError(Exception):
    """A bundle failed verification. Serving falls back to the previous policy level."""


@dataclass
class ModelBundle:
    name: str
    version: str
    model: CalibratedModel
    manifest: dict[str, Any]
    path: Path


def save_bundle(
    directory: Path,
    *,
    name: str,
    version: str,
    model: CalibratedModel,
    feature_schema_version: str,
    metrics: dict[str, Any] | None = None,
    lineage: dict[str, Any] | None = None,
    signing_key: bytes | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    files = dict(model.model.to_files())
    files["calibrator.json"] = json.dumps(model.calibrator.to_dict())
    for filename, content in files.items():
        # Explicit LF line endings: artifacts must be byte-identical on every OS
        # (LightGBM's text format rejects CRLF, and checksums must match everywhere).
        (directory / filename).write_text(content, encoding="utf-8", newline="\n")
    manifest = {
        "name": name,
        "version": version,
        "algorithm": model.model.algorithm,
        "feature_schema_version": feature_schema_version,
        "feature_names": list(model.feature_names),
        "created_at": datetime.now(UTC).isoformat(),
        "metrics": metrics or {},
        "lineage": lineage or {},
        "files": {f: _sha256((directory / f).read_bytes()) for f in sorted(files)},
    }
    if signing_key is not None:
        manifest["signature"] = _sign(manifest, signing_key)
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8", newline="\n"
    )
    return directory


def load_bundle(
    directory: Path,
    *,
    signing_key: bytes | None = None,
    require_signature: bool = False,
    feature_schema_version: str | None = None,
) -> ModelBundle:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise BundleError(f"no manifest in {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    signature = manifest.get("signature")
    if require_signature and signature is None:
        raise BundleError("unsigned bundle")
    if signature is not None and signing_key is not None:
        unsigned = {k: v for k, v in manifest.items() if k != "signature"}
        if not hmac.compare_digest(signature, _sign(unsigned, signing_key)):
            raise BundleError("bad signature")
    files: dict[str, str] = {}
    for filename, expected in manifest["files"].items():
        data = (directory / filename).read_bytes()
        if _sha256(data) != expected:
            raise BundleError(f"checksum mismatch for {filename}")
        files[filename] = data.decode("utf-8")
    if feature_schema_version and manifest["feature_schema_version"] != feature_schema_version:
        raise BundleError(
            f"feature schema {manifest['feature_schema_version']} "
            f"!= serving {feature_schema_version}"
        )
    model = model_from_files(manifest["algorithm"], files, manifest["feature_names"])
    calibrator = calibrator_from_dict(json.loads(files["calibrator.json"]))
    return ModelBundle(
        manifest["name"],
        manifest["version"],
        CalibratedModel(model, calibrator),
        manifest,
        directory,
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sign(manifest: dict[str, Any], key: bytes) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(key, canonical, hashlib.sha256).hexdigest()
