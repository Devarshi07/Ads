"""Local open-weight models through Ollama, in place of Claude (design doc: LLM Components &
Hallucination Controls). Free, and private: creatives never leave the machine.

The contract is the Claude adapter's (`review.llm`):

- the same policy prompt, and advertiser text escaped and delimited (`render_untrusted`);
- output constrained to `judgement_schema` by Ollama's structured outputs, then validated
  again by `parse_judgement`;
- no tools are passed;
- every failure (Ollama down, model not pulled, timeout, truncation, invalid JSON) raises
  `ReviewerUnavailable` or `InvalidReviewOutput`, and the pipeline queues the creative for a
  human. Nothing is approved because a call failed.

    THERAFLO_ADS_REVIEWER=ollama                       (and THERAFLO_ADS_NARRATOR=ollama)
    THERAFLO_ADS_OLLAMA_URL=http://127.0.0.1:11434     (compose: http://host.docker.internal:11434)
    THERAFLO_ADS_OLLAMA_MODEL=qwen3.5:9b
    THERAFLO_ADS_OLLAMA_THINK=false                    (true: reasoning first, slower)

Decoding is greedy with a fixed seed without thinking, so evals reproduce. With thinking, the
temperature follows the Qwen guidance for reasoning mode (greedy decoding can loop there).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from theraflo_ads.review.creative import (
    Creative,
    InvalidReviewOutput,
    Judgement,
    ReviewerUnavailable,
    Violation,
    judgement_schema,
    parse_judgement,
)
from theraflo_ads.review.injection import render_untrusted
from theraflo_ads.review.llm import NARRATOR_SYSTEM, PROMPT_VERSION, system_prompt
from theraflo_ads.review.taxonomy import Taxonomy, default_taxonomy

DEFAULT_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3.5:9b"
SEED = 7
LOCAL_PROMPT = "local1"

# Appended to the shared policy for local models (Claude's prompt is unchanged). Ollama
# constrains decoding to the schema but, unlike Claude's API, does not show it to the model:
# without the label list, qwen3.5:9b guessed and labeled online courses and travel "other"
# (46% category accuracy over the eval set). Small models also listed plain product
# descriptions as claims, which grounding then sent to a human. The examples here are not
# from the eval set.
_LOCAL_GUIDANCE = """\
Answer with one JSON object and nothing else, with exactly these fields:
- verdict: "approve", "reject", or "escalate".
- category: what is actually being sold, exactly one of: {categories}.
- violations: a list, empty when there are none, drawn only from: {violations}.
- claims: the claims, quoted from the creative; [] when there are none.
- addresses_reviewer: true or false.
- confidence: "high", "medium", or "low".
- rationale: one or two sentences.

What counts as a claim: a statement about an effect or result (on sleep, health, mood, money, \
or skills), about evidence ("clinically proven", "studies show"), a number or statistic, a \
rating, or a comparison ("best", "#1"). Plain descriptions of what is sold are not claims and \
go in no list: "wool sweaters in three weights", "a cookbook of quick weeknight meals", "a \
birdwatching newsletter every Sunday"."""


def local_system_prompt(taxonomy: Taxonomy, variant: str) -> str:
    guidance = _LOCAL_GUIDANCE.format(
        categories=", ".join(sorted(taxonomy.labels)),
        violations=", ".join(v.value for v in Violation),
    )
    return f"{system_prompt(taxonomy, variant)}\n\n{guidance}"


@dataclass(frozen=True)
class OllamaSettings:
    url: str = DEFAULT_URL
    model: str = DEFAULT_MODEL
    think: bool = False

    @classmethod
    def from_env(cls) -> OllamaSettings:
        think = os.environ.get("THERAFLO_ADS_OLLAMA_THINK", "false").strip().lower()
        return cls(
            url=os.environ.get("THERAFLO_ADS_OLLAMA_URL", DEFAULT_URL).rstrip("/"),
            model=os.environ.get("THERAFLO_ADS_OLLAMA_MODEL", DEFAULT_MODEL).strip(),
            think=think in ("1", "true", "yes", "on"),
        )


class OllamaClient:
    """`POST /api/chat`, non-streaming. Transport failures become `ReviewerUnavailable`."""

    def __init__(
        self, url: str, *, timeout: float = 180.0, transport: httpx.BaseTransport | None = None
    ) -> None:
        self._http = httpx.Client(base_url=url, timeout=timeout, transport=transport)

    def chat(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        try:
            response = self._http.post("/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            raise ReviewerUnavailable("timeout") from exc
        except httpx.HTTPError as exc:
            raise ReviewerUnavailable("connection") from exc
        if response.status_code == 404:
            raise ReviewerUnavailable("model_not_found")  # `ollama pull <model>` first
        if response.status_code >= 400:
            raise ReviewerUnavailable(f"http_{response.status_code}")
        try:
            data = response.json()
        except ValueError as exc:
            raise InvalidReviewOutput("invalid_response") from exc
        if not isinstance(data, dict):
            raise InvalidReviewOutput("invalid_response")
        return data


def _content(data: Mapping[str, Any]) -> str:
    if data.get("done_reason") == "length":
        raise InvalidReviewOutput("max_tokens")
    text = str((data.get("message") or {}).get("content") or "")
    if not text.strip():
        raise InvalidReviewOutput("no_text")
    return text


def _usage(data: Mapping[str, Any]) -> dict[str, int]:
    usage = {
        "input_tokens": int(data.get("prompt_eval_count") or 0),
        "output_tokens": int(data.get("eval_count") or 0),
    }
    if data.get("total_duration"):
        usage["duration_ms"] = int(data["total_duration"]) // 1_000_000
    return usage


class OllamaReviewModel:
    def __init__(
        self,
        client: OllamaClient | None = None,
        *,
        settings: OllamaSettings | None = None,
        taxonomy: Taxonomy | None = None,
        timeout: float = 180.0,
    ) -> None:
        self._settings = settings or OllamaSettings.from_env()
        self._client = client or OllamaClient(self._settings.url, timeout=timeout)
        self._taxonomy = taxonomy or default_taxonomy()
        self._schema = judgement_schema(self._taxonomy)
        self.name = f"ollama:{self._settings.model}"
        think = "+think" if self._settings.think else ""
        self.prompt_version = f"{PROMPT_VERSION}+{LOCAL_PROMPT}{think}"

    def request(self, creative: Creative, variant: str = "primary") -> dict[str, Any]:
        think = self._settings.think
        return {
            "model": self._settings.model,
            "messages": [
                {"role": "system", "content": local_system_prompt(self._taxonomy, variant)},
                {
                    "role": "user",
                    "content": f"{render_untrusted(creative)}\n\nReview the creative above.",
                },
            ],
            "format": self._schema,
            "think": think,
            "stream": False,
            "keep_alive": "15m",
            "options": {
                "temperature": 0.6 if think else 0.0,
                "seed": SEED,
                # A review is about 1k tokens of prompt and 150 of answer. A small window
                # keeps a 9B model entirely on an 8 GB GPU; reasoning needs the larger one.
                "num_ctx": 8192 if think else 4096,
                "num_predict": 6144 if think else 1024,
            },
        }

    def judge(self, creative: Creative, variant: str = "primary") -> Judgement:
        data = self._client.chat(self.request(creative, variant))
        try:
            payload = json.loads(_content(data))
        except json.JSONDecodeError as exc:
            raise InvalidReviewOutput("invalid_json") from exc
        return parse_judgement(
            payload,
            self._taxonomy,
            model=self.name,
            prompt_version=f"{self.prompt_version}:{variant}",
            variant=variant,
            usage=_usage(data),
        )


class OllamaNarratorModel:
    """Report summaries. `narrate` checks every number against the report and falls back to
    the template, so a local model's arithmetic slips never reach an advertiser."""

    def __init__(
        self,
        client: OllamaClient | None = None,
        *,
        settings: OllamaSettings | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._settings = settings or OllamaSettings.from_env()
        self._client = client or OllamaClient(self._settings.url, timeout=timeout)
        self.name = f"ollama:{self._settings.model}"

    def write(self, facts: str, guidance: str = "") -> str:
        content = f"<metrics>\n{facts}\n</metrics>"
        if guidance:
            content += f"\n\n{guidance}"
        data = self._client.chat(
            {
                "model": self._settings.model,
                "messages": [
                    {"role": "system", "content": NARRATOR_SYSTEM},
                    {"role": "user", "content": content},
                ],
                "think": False,
                "stream": False,
                "keep_alive": "15m",
                "options": {"temperature": 0.0, "seed": SEED, "num_predict": 512},
            }
        )
        return _content(data).strip()
