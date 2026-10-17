"""Offline jobs and reports for the dev console.

Jobs are an allowlist of `theraflo-ads` commands with bounded parameters. Each runs as a
subprocess with an argument list (never a shell), one at a time, with a timeout; its output
is kept for the console. Nothing billed can start here: the live Claude reviewer is not on
the list, and ANTHROPIC_API_KEY is removed from the job environment.

Reports are files under reports/, read with path traversal refused.
"""

from __future__ import annotations

import os
import subprocess  # nosec B404
import sys
import threading
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JobError(ValueError):
    pass


@dataclass(frozen=True)
class Param:
    kind: str  # int | float | choice | bool
    default: Any
    low: float | None = None
    high: float | None = None
    choices: tuple[str, ...] = ()
    help: str = ""

    def parse(self, name: str, value: Any) -> Any:
        if value is None:
            return self.default
        try:
            match self.kind:
                case "int":
                    parsed: Any = int(value)
                case "float":
                    parsed = float(value)
                case "bool":
                    if not isinstance(value, bool):
                        raise ValueError
                    return value
                case "choice":
                    if value not in self.choices:
                        raise ValueError
                    return value
                case _:
                    raise ValueError
        except (TypeError, ValueError):
            raise JobError(f"{name}: invalid value {value!r}") from None
        if (self.low is not None and parsed < self.low) or (
            self.high is not None and parsed > self.high
        ):
            raise JobError(f"{name} must be between {self.low} and {self.high}")
        return parsed

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "default": self.default,
            "low": self.low,
            "high": self.high,
            "choices": list(self.choices),
            "help": self.help,
        }


@dataclass(frozen=True)
class JobSpec:
    name: str
    title: str
    description: str
    argv: Callable[[Mapping[str, Any]], list[str]]
    params: Mapping[str, Param] = field(default_factory=dict)
    needs: tuple[str, ...] = ()  # files that must exist (relative to the working directory)
    outputs: tuple[str, ...] = ()  # report files it writes
    backend: str | None = None  # "infra": works only against Postgres
    timeout_s: float | None = None  # longer than the runner's default (local-model evals)

    def to_dict(self, cwd: Path, backend: str) -> dict[str, Any]:
        missing = [n for n in self.needs if not (cwd / n).exists()]
        unavailable = None
        if self.backend is not None and self.backend != backend:
            unavailable = f"needs the {self.backend} backend"
        elif missing:
            unavailable = f"missing {', '.join(missing)}"
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "params": {k: p.to_dict() for k, p in self.params.items()},
            "outputs": list(self.outputs),
            "unavailable": unavailable,
        }


def _flag(name: str, value: Any) -> list[str]:
    return [f"--{name}", str(value)]


def _eval_out(p: Mapping[str, Any]) -> str:
    think = "_think" if p["think"] and p["reviewer"] == "ollama" else ""
    return f"reports/console/review_eval_{p['reviewer']}{think}.json"


JOBS: dict[str, JobSpec] = {
    spec.name: spec
    for spec in (
        JobSpec(
            "simulate",
            "Simulator run",
            "Synthetic users and advertisers through the real decision path: revenue, "
            "safety violations (must be 0), OPE of the logged policy, fatigue and churn.",
            lambda p: [
                "simulate",
                *_flag("users", p["users"]),
                *_flag("days", p["days"]),
                *_flag("seed", p["seed"]),
                *_flag("epsilon", p["epsilon"]),
                *(["--stateless"] if p["stateless"] else []),
            ],
            {
                "users": Param("int", 800, 50, 5000),
                "days": Param("int", 3, 1, 14),
                "seed": Param("int", 7, 0, 1_000_000),
                "epsilon": Param("float", 0.05, 0.0, 0.5, help="exploration share"),
                "stateless": Param(
                    "bool", False, help="no fatigue, caps, or budgets (OPE is unbiased here)"
                ),
            },
        ),
        JobSpec(
            "review_eval",
            "Creative-review eval",
            "The labeled eval set through a reviewer: false approvals (with an upper bound), "
            "abstention, automation rate, injection cases. `ollama` is a local model (free; "
            "Ollama must be running, model in THERAFLO_ADS_OLLAMA_MODEL; a full run takes "
            "minutes). The live Claude reviewer is billed and not offered here; run "
            "`theraflo-ads review eval --reviewer anthropic --live`.",
            lambda p: [
                "review",
                "eval",
                *_flag("reviewer", p["reviewer"]),
                *_flag("split", p["split"]),
                *(["--think"] if p["think"] else []),
                *_flag("out", _eval_out(p)),
            ],
            {
                "reviewer": Param(
                    "choice", "simulated", choices=("rules", "simulated", "naive", "ollama")
                ),
                "split": Param("choice", "all", choices=("all", "train", "val", "test")),
                "think": Param("bool", False, help="ollama only: reason before answering"),
            },
            outputs=("reports/console/review_eval_<reviewer>.json",),
            timeout_s=3600.0,
        ),
        JobSpec(
            "review_compare",
            "Reviewer comparison",
            "Rules vs simulated vs an injection-following reviewer, with ablations of each "
            "control (grounding, dual check, injection scan).",
            lambda p: ["review", "compare", *_flag("out", "reports/console/review_compare.json")],
            outputs=("reports/console/review_compare.json",),
        ),
        JobSpec(
            "experiment_mde",
            "Experiment power (MDE)",
            "Minimum detectable effect for a baseline conversion rate at both traffic "
            "scenarios of the design doc.",
            lambda p: ["experiment", "mde", *_flag("baseline-rate", p["baseline_rate"])],
            {"baseline_rate": Param("float", 0.0003, 0.000001, 0.5, help="per-impression rate")},
        ),
        JobSpec(
            "experiment_simulate",
            "Simulated A/B test",
            "Heuristic vs the trained ML ranker in the simulator, with CUPED and sequential "
            "monitoring.",
            lambda p: [
                "experiment",
                "simulate",
                *_flag("users", p["users"]),
                *_flag("days", p["days"]),
                *_flag("pre-days", p["pre_days"]),
                *_flag("out", "reports/console/experiment.json"),
            ],
            {
                "users": Param("int", 1500, 200, 4000),
                "days": Param("int", 10, 3, 28),
                "pre_days": Param("int", 5, 0, 7),
            },
            needs=("models/candidates/ctr/manifest.json",),
            outputs=("reports/console/experiment.json",),
        ),
        JobSpec(
            "rl_fqi",
            "Offline RL (FQI)",
            "Fitted Q iteration on the logged decisions, gated by off-policy evaluation and "
            "simulator safety evals. RL never serves without passing them.",
            lambda p: [
                "rl",
                "fqi",
                *_flag("rounds", p["rounds"]),
                *_flag("out", "reports/console/rl_fqi.json"),
            ],
            {"rounds": Param("int", 5, 1, 20)},
            needs=("data/raw/meta.json",),
            outputs=("reports/console/rl_fqi.json",),
        ),
        JobSpec(
            "monitor_drift",
            "Drift monitor",
            "Feature, prediction, and label drift over the logged decisions (PSI, KS).",
            lambda p: ["monitor", "drift"],
            needs=("data/raw/meta.json",),
            outputs=("reports/drift.json",),
        ),
        JobSpec(
            "labels_finalize",
            "Finalize labels",
            "Close label windows whose attribution has matured (the nightly job).",
            lambda p: ["labels", "finalize"],
            backend="infra",
        ),
        JobSpec(
            "privacy_delete_pending",
            "Run queued deletions",
            "Right to deletion: shred user keys and erase operational state for queued "
            "requests (a cron job in production).",
            lambda p: ["privacy", "delete-pending"],
            backend="infra",
        ),
    )
}

MAX_OUTPUT_LINES = 4000


@dataclass
class Job:
    job_id: str
    name: str
    params: dict[str, Any]
    argv: list[str]
    status: str = "running"  # running | succeeded | failed | cancelled | timed_out
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    exit_code: int | None = None
    output: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_OUTPUT_LINES))
    process: subprocess.Popen | None = None
    timeout_s: float | None = None

    def to_dict(self, *, with_output: bool = False) -> dict[str, Any]:
        data = {
            "job_id": self.job_id,
            "name": self.name,
            "params": self.params,
            "command": "theraflo-ads " + " ".join(self.argv),
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_s": (
                ((self.finished_at or datetime.now(UTC)) - self.started_at).total_seconds()
            ),
            "exit_code": self.exit_code,
        }
        if with_output:
            data["output"] = "".join(self.output)
        return data


class JobRunner:
    def __init__(
        self,
        cwd: Path,
        *,
        backend: str,
        specs: Mapping[str, JobSpec] = JOBS,
        timeout_s: float = 900.0,
        history: int = 25,
        python: str = sys.executable,
    ) -> None:
        self._cwd = cwd
        self._backend = backend
        self._specs = specs
        self._timeout = timeout_s
        self._history = history
        self._python = python
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}

    def catalog(self) -> list[dict[str, Any]]:
        return [spec.to_dict(self._cwd, self._backend) for spec in self._specs.values()]

    def submit(self, name: str, params: Mapping[str, Any]) -> Job:
        spec = self._specs.get(name)
        if spec is None:
            raise JobError(f"unknown job {name!r}")
        unavailable = spec.to_dict(self._cwd, self._backend)["unavailable"]
        if unavailable:
            raise JobError(f"{name}: {unavailable}")
        unknown = set(params) - set(spec.params)
        if unknown:
            raise JobError(f"unknown parameters: {', '.join(sorted(unknown))}")
        values = {k: p.parse(k, params.get(k)) for k, p in spec.params.items()}
        argv = spec.argv(values)
        with self._lock:
            if any(j.status == "running" for j in self._jobs.values()):
                raise JobError("another job is running; wait for it or cancel it")
            job = Job(uuid.uuid4().hex[:12], name, values, argv, timeout_s=spec.timeout_s)
            env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
            env["THERAFLO_ADS_REVIEWER"] = "none"
            env["PYTHONUNBUFFERED"] = "1"
            (self._cwd / "reports" / "console").mkdir(parents=True, exist_ok=True)
            # An argument list from the allowlist above, with validated values: no shell.
            job.process = subprocess.Popen(  # nosec B603
                [self._python, "-m", "theraflo_ads.cli", *argv],
                cwd=self._cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            self._jobs[job.job_id] = job
            self._trim()
        threading.Thread(target=self._watch, args=(job,), daemon=True).start()
        return job

    def _watch(self, job: Job) -> None:
        process = job.process
        if process is None or process.stdout is None:
            return
        timer = threading.Timer(job.timeout_s or self._timeout, self._expire, args=(job,))
        timer.daemon = True
        timer.start()
        try:
            for line in process.stdout:
                job.output.append(line)
            code = process.wait()
        finally:
            timer.cancel()
        with self._lock:
            job.exit_code = code
            job.finished_at = datetime.now(UTC)
            if job.status == "running":
                job.status = "succeeded" if code == 0 else "failed"

    def _expire(self, job: Job) -> None:
        with self._lock:
            if job.status != "running" or job.process is None:
                return
            job.status = "timed_out"
            job.output.append(f"\n[killed after {job.timeout_s or self._timeout:.0f}s]\n")
        job.process.kill()

    def cancel(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status != "running" or job.process is None:
                return job
            job.status = "cancelled"
            job.output.append("\n[cancelled from the console]\n")
        job.process.terminate()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def jobs(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.started_at, reverse=True)

    def _trim(self) -> None:
        finished = [j for j in self.jobs() if j.status != "running"]
        for job in finished[self._history :]:
            self._jobs.pop(job.job_id, None)


REPORT_SUFFIXES = (".json", ".md", ".txt")
MAX_REPORT_BYTES = 2_000_000


def list_reports(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    reports = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in REPORT_SUFFIXES and "bundles" not in path.parts:
            stat = path.stat()
            reports.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
                }
            )
    return reports


def read_report(root: Path, relative: str) -> tuple[str, str]:
    """(content, kind). Only files below `root` with a report suffix; no traversal."""
    base = root.resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base) or path.suffix not in REPORT_SUFFIXES:
        raise KeyError(relative)
    if not path.is_file():
        raise KeyError(relative)
    if path.stat().st_size > MAX_REPORT_BYTES:
        raise JobError(f"{relative} is larger than {MAX_REPORT_BYTES // 1_000_000} MB")
    return path.read_text(encoding="utf-8", errors="replace"), path.suffix.lstrip(".")
