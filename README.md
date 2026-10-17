# Theraflo Ads

A safety-first ads decisioning platform for Theraflo, a mental-health and wellness app. The design lives in [`../theraflo_ads_end_to_end_design.md`](../theraflo_ads_end_to_end_design.md); this repository implements every phase of its roadmap as a working, tested prototype.

**Everything here runs on simulated users and synthetic creatives, except the OPE validation, which uses 26.9M real logged impressions from the Open Bandit Dataset.** The simulated numbers show that the machinery works and how to read it. None of them is evidence about production behaviour, and none should be used to promote a model.

- 678 tests (653 unit, 25 integration against real Postgres, Redis, and RabbitMQ), ruff-clean, bandit-clean; alert rules pass `promtool` unit tests.
- A browser [dev console](#dev-console) at `/ui` to test everything by hand: the app as any user, the ranker lab, the advertiser portal, review, and offline jobs.
- About 18k lines of source and 5.6k lines of tests, in Python 3.12.

## Contents

- [Quickstart](#quickstart)
- [Dev console](#dev-console)
- [Architecture](#architecture)
- [Milestones](#milestones)
- [Commands](#commands)
- [Safety rules the code enforces](#safety-rules-the-code-enforces)
- [LLM-assisted creative review (M8)](#llm-assisted-creative-review-m8)
- [Operations (M9)](#operations-m9)
- [Scaling research (M10)](#scaling-research-m10)
- [OPE validation on real logs (Open Bandit Dataset)](#ope-validation-on-real-logs-open-bandit-dataset)
- [Findings](#findings)
- [Bugs the harness caught](#bugs-the-harness-caught)
- [Windows notes](#windows-notes)
- [Limits and next steps](#limits-and-next-steps)

## Quickstart

Commands use forward slashes, so they work in PowerShell and Git Bash alike. Run them from the `theraflo-ads` folder.

**1. In-memory API (no Docker).**

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[infra,ml,mlops,llm,dev]"   # .venv/bin/python on Linux/macOS
.venv/Scripts/theraflo-ads serve                                       # http://127.0.0.1:8000/ui
```

- **Open the console** at http://127.0.0.1:8000/ui (see [Dev console](#dev-console)). The Swagger UI is at `/docs`.
- **Credentials.** In the Swagger UI, click **Authorize** and paste a user token from `DEV_USER_TOKENS` in `src/theraflo_ads/api/container.py`. Admin endpoints take a token from `DEV_ADMIN_TOKENS`. Advertiser endpoints take an `X-Advertiser-Key` header from `DEV_ADVERTISER_KEYS`.
- **Try** `POST /v1/ads/rank` with `{"session_id": "s1", "placement": "home_feed"}`, then the same request with `"placement": "journal"`, which never gets an ad.
- **Behaviour.** State lives in memory and resets on restart. The dev credentials are refused when `THERAFLO_ADS_ENV=production`.

**2. Full stack in Docker.** This brings up Postgres, Redis, RabbitMQ, the API, the outbox relay, four workers, Prometheus, and Grafana.

```bash
.venv/Scripts/theraflo-ads gen-keys --out .env
docker compose --profile app --profile monitoring up -d --build --wait
docker compose --profile app run --rm migrate theraflo-ads seed-demo
```

Then open the console at http://localhost:8000/ui and start demo traffic from its Overview page, or run `.venv/Scripts/python scripts/demo_traffic.py` (10 minutes). The dashboards stay empty until something uses the API. Demo traffic plays users and advertisers at about 8 screens per second:

- ads on the feed, and the ad-free journal and mood screens;
- viewable events, clicks, conversions, and hidden ads;
- crisis suppressions;
- clean and violating creatives;
- one burst a minute that the rate limiter cuts off.

| Where | What to look at |
|---|---|
| **Dev console**, http://localhost:8000/ui | Everything interactive: the app as any user, the ranker lab, the advertiser portal, review, jobs, and reports. See [Dev console](#dev-console). |
| **Grafana**, http://localhost:3000 | Opens on the Theraflo Ads dashboard (last 15 minutes, refreshing every 10 s). Top row: the safety gate (must stay 0), the kill switch, and dependency health. Below: decisions by policy; blocked requests by reason (`placement_not_allowed` is the ad-free screens working); rank latency p50/p95/p99; HTTP status; events; creative review outcomes and the human queue; rate limiting; dead-letter queues. |
| **Prometheus**, http://localhost:9090 | The home page is an empty query box. Try **Alerts** (`/alerts`, the 12 rules, green when quiet) and **Status → Targets** (`/targets`: API, review worker, RabbitMQ). On the graph page, type a query such as `sum by (served_by) (rate(theraflo_ads_decisions_total[1m]))`. |
| **API**, http://localhost:8000/docs | Swagger UI. Authorize with a token from `DEV_USER_TOKENS` (`src/theraflo_ads/api/container.py`) and call the endpoints yourself. |
| **RabbitMQ**, http://localhost:15672 | Queues tab: `accounting`, `features`, `analytics`, their `.retry` and `.dlq` queues, and message rates. Login is in `docker-compose.yml`. |
| **Logs** | `docker compose logs -f api` (JSON, one line per request) or any other service. |

- **To see an alert fire,** run `docker compose stop redis`. Within seconds, ads stop (fail closed) and the dependency panel turns red. After about 2 minutes, `DependencyDown` goes from pending to firing on `/alerts`. Bring it back with `docker compose start redis`.
- **If Grafana says "Unauthorized",** an old session cookie from another app on `localhost:3000` is in your browser. The stack now uses its own cookie name, so a hard refresh fixes it; otherwise use a private window.

- **Ports and reviewer.** Every port is published on 127.0.0.1 only. Add `API_PORT=…`, `GRAFANA_PORT=…`, and so on to `.env` to move a port. Add `THERAFLO_ADS_REVIEWER=simulated` to auto-review submitted creatives; the default is rules plus the human queue.
- **Stop** with `docker compose --profile app --profile monitoring down`; add `-v` to delete the data too.
- **Mixed setup.** To run the Python processes on the host against the containerized Postgres, Redis, and RabbitMQ, start only `docker compose up -d postgres redis rabbitmq`. Set the variables from `.env` plus `THERAFLO_ADS_BACKEND=infra` in each terminal, then run `migrate`, `seed-demo`, `serve`, `relay`, `worker accounting|features|analytics`, and `review worker`.

**3. Tests.**

```bash
.venv/Scripts/python -m pytest -m "not integration"     # 653 tests, no services needed
.venv/Scripts/python -m pytest -m integration           # 25 tests; needs postgres, redis, rabbitmq
```

- **Integration tests need the app's workers stopped.** They purge and consume the RabbitMQ queues, so they skip, with a message, while `--profile app` workers are attached; `docker compose --profile app stop` first.
- **Separate services instead.** Point `THERAFLO_ADS_TEST_DATABASE_URL`, `THERAFLO_ADS_TEST_REDIS_URL`, and `THERAFLO_ADS_TEST_AMQP_URL` at another Postgres, Redis, and RabbitMQ.

**4. Practice a cloud deployment for free.** [`deploy/localstack-azure/GUIDE.md`](deploy/localstack-azure/GUIDE.md) deploys the project to the LocalStack for Azure emulator with the real `az` commands: the console first, then the full architecture (managed Postgres, Redis, RabbitMQ, the API, relay, workers, and jobs). It explains every flag, the emulator's limits, and what changes on real Azure.

## Dev console

http://localhost:8000/ui, on the memory server or the Docker stack. Every action calls the real API with the right role's credentials: a user's token, an advertiser's key, the Theraflo backend's service token, or the admin token. The **⇄** button opens a drawer with each request the console sent and its response.

| Page | What you can do |
|---|---|
| **Overview** | Readiness of every dependency, the kill switch (with a reason, audited), the live serving chain, the safety gate (must stay 0), and the review queue. Live decisions per second, fill rate, and rank latency; counters by policy, block reason, event status, rate limiting, and review outcome. Start and stop demo traffic from a population of test users with mixed consent and age. Recent decisions: click one for its full log record. |
| **App simulator** | Be any user, or create test users with a consent and age band. Open the app's screens: the feed and the library can show ads; the journal, mood check-in, crisis resources, and therapy chat never do. A rendered ad sends `AD_VIEWABLE` as the SDK would. Click it, hide it, report it, or ask "why this ad?", then send the advertiser's conversion postback for the click. Change consent, open crisis resources (ads pause for 24 hours), or delete the user's data. The side panel shows the decision behind each screen: who served it, every propensity, the features, the holdout, and the attribution label. |
| **Ranker lab** | A what-if request (user, screen, consent, age, device, hour, suppression, hidden ads) through every stage: hard-constraint eligibility per ad, ranker scores (category priors next to the calibrated ML model), and the full distribution of every policy side by side: heuristic, ML ranker, Beta and linear Thompson sampling, popularity, and uniform. Nothing is served, spent, or counted. Switch the live serving policy. Inspect bandit posteriors and linear-bandit weights, and the loaded model: its metrics, missing features, and feature importance. Promote a registry version to production, or roll it back. |
| **Advertiser portal** | Per advertiser: campaigns and today's billed spend. Submit creatives, with a preset for each review control (clean, substantiated claim, unsupported statistic, low-mood targeting, restricted category, health-data request, prompt injection, urgency), and watch their review status. Pull the aggregated report and its narrative; send conversion and reversal postbacks. |
| **Review & safety** | The human review queue and the audit sample. Each item shows the creative as untrusted text, next to the machine review record: outcome, violations, rule and injection signals, and claim grounding. Approve or reject; read the measured false-approval rate; add substantiated claims; flip the kill switch; read the audit log. |
| **Jobs & reports** | Allowlisted offline jobs with bounded parameters and live output: the simulator, the creative-review eval and comparison, MDE, a simulated A/B test, offline RL (FQI), drift, and, on the infra backend, label finalization and queued deletions. Every report under `reports/`, with Markdown rendered. |

- **Dev only.** The console and `/v1/dev/*` are mounted only with dev credentials; `THERAFLO_ADS_ENV=production` removes them. `GET /v1/dev/identities` hands the console the dev credentials, which is why compose publishes every port on 127.0.0.1 only.
- **A strict page.** It loads only same-origin files under a CSP with no inline script and no framing. Advertiser text is always inserted as text, never as HTML.
- **Nothing billed.** Jobs run without `ANTHROPIC_API_KEY`, with the reviewer forced to `none`, and the live Claude eval is not offered; run it from the CLI with `--live`.
- **Only a production model serves.** The lab previews whichever model it finds (registry production, else the newest registered version, else `models/candidates`), but the ML ranker serves only after a version is promoted. **Promote** runs the same steps as `theraflo-ads ml promote` and writes `models/registry/registry.json`.
- **The two backends.** On the memory backend, test users last until the API restarts; with `THERAFLO_ADS_SESSION_KEY` set, as on the Docker stack, they get session tokens. On the infra backend, Beta Thompson sampling reads the counts the features worker keeps in Redis, while the linear bandit has only its warm start.
- **The trained CTR model does not fit the demo catalog.** It was fitted on the simulator's catalog, so 11 of its 36 features are absent from the five-ad demo catalog and score as 0. The lab shows this.

## Architecture

```text
                     ┌──────────── serving path (sync, no LLM) ────────────┐
POST /v1/ads/rank →  │ ConstraintEngine → policy chain → budget reservation │ → served-ad safety gate → response
                     │  hard rules,         ranker/bandit/   + 1-event        │   (re-checks every hard
                     │  fail closed         heuristic →      headroom         │    constraint; withholds)
                     │                      popularity → NO_AD                │
                     └──────┬───────────────────────────────────────────────┘
                            │ one transaction: decision + impression + label + outbox row
                            ▼
                   Postgres ──relay──▶ RabbitMQ ──▶ accounting worker (billing, labels)
                                                 ──▶ features worker   (Redis interaction counts)
                                                 ──▶ analytics worker  (lake, pseudonymized)
offline:  lake → DVC pipeline (build → train → evaluate → register) → registry → shadow/canary
          drift job → retraining controller        creative review worker (rules + Claude)
          deletion job (crypto-shredding)          scaling research
```

Redis holds derived state only (frequency counters, budget reservations, features, rate limits), each rebuildable from Postgres; a missing counter is never read as zero. LLMs run offline or asynchronously only.

## Milestones

| | Milestone | What it delivered | Main modules |
|---|---|---|---|
| M1 | Decisioning core | Constraint engine, policy chain, exact propensity logging, budgets with headroom, frequency caps, signed impression tokens, attribution and delayed labels, OPE (IPS/SNIPS/DR), simulator | `safety/`, `policy/`, `counters/`, `attribution/`, `ope/`, `simulator/` |
| M2 | Infrastructure | Postgres schema + Alembic, transactional outbox, relay with publisher confirms, RabbitMQ retry/DLQ topology, idempotent workers, Redis stores with Lua reservations, shared kill switch | `infra/`, `events/`, `storage/` |
| M3 | ML pipeline | Shared online/offline features, LightGBM / logistic CTR+CVR with calibration, non-pickle signed bundles, gates (incl. paired-bootstrap slice regressions), registry lifecycle with rollback, MLflow + DVC | `ml/`, `dvc.yaml`, `params.yaml` |
| M4 | Experiments | Assignment, SRM, CUPED, winsorization, delta-method ratios, pre-registered MDE, guardrails, promotion rule, sequential tests, simulated A/B | `experiments/`, `simulator/ab.py` |
| M5 | Bandits | Beta and linear Thompson sampling with Monte Carlo propensities, guarded fallbacks | `policy/bandits.py`, `policy/guard.py` |
| M6 | Monitoring | Context-feature drift, per-category calibration, concept drift, business signals, retraining controller (recalibrate first), Airflow DAG | `monitoring/`, `orchestration/` |
| M7 | RL | Offline FQI with a support constraint (OPE-gated), simulator-only PPO, vulnerable-cohort safety evals | `rl/` |
| M8 | LLM ad review | Structured-output Claude reviewer, rules, injection defenses, claim grounding, dual check, fail-closed human queue, audit sampling, report narrator with numeric checks, "why this ad", advertiser reports, eval suite, post-training data + gates | `review/` |
| M9 | Hardening | Prometheus metrics + served-ad safety gate, JSON logs with redaction, rate limiting, crypto-shredding deletion job, load and chaos scripts, Dockerfile, compose profiles, alert rules, Grafana dashboard, CI, security scans | `observability/`, `privacy/`, `scripts/`, `monitoring/`, `.github/` |
| M10 | Scaling research | Learning curves with a fitted power law, model size, feature ablation, revenue vs complexity vs serving cost; OPE validated on the Open Bandit Dataset | `research/` |

## Commands

```text
theraflo-ads migrate | gen-keys | seed-demo | serve | relay
theraflo-ads worker accounting|features|analytics
theraflo-ads simulate [--users N --days D --stateless]
theraflo-ads ml generate|build|train|evaluate|register|all|promote|rollback   (dvc repro runs these)
theraflo-ads monitor drift | retrain | labels finalize
theraflo-ads experiment mde | experiment simulate
theraflo-ads rl fqi | rl ppo
theraflo-ads review eval|compare|export|gate|worker [--reviewer rules|simulated|naive|anthropic --live]
theraflo-ads privacy delete-pending
theraflo-ads research scaling
theraflo-ads research obd [--download]                 (413 MB public dataset, CC BY 4.0)
python scripts/loadtest.py --users 20 --duration 30
python scripts/chaos.py
```

## Safety rules the code enforces

| Rule | Where | Design doc |
|---|---|---|
| Anything that can remove an ad runs **before** the policy samples | `policy/service.py` | Online Request Flow |
| Placement allowlist; ad-free placements rejected at config time | `safety/policy.py` | Ad-Free Placements |
| Restricted categories, expanded for teens and unknown age; under-13 gets no ads | `safety/engine.py` | Restricted Categories, Minors |
| Suppression window after crisis resources; the ads service never learns why | `safety/engine.py`, `api/app.py` | Ad-Free Placements |
| Minors and unknown age are always contextual-only | `safety/engine.py` | Consent |
| Counter, budget, or kill-switch state unreadable → NO_AD | `safety/engine.py`, `infra/stores.py` | Failure Hierarchy |
| A served ad that breaks any hard constraint is withheld at the API and pages someone | `observability/metrics.py`, `api/app.py` | Monitoring |
| NO_AD is an action in every distribution; exploration floors on stochastic policies | `policy/` | RL Safety |
| Impulse categories are never personalized | `safety/policy.py`, `ml/features.py` | Why Penalties Are Not Enough |
| Deterministic, degraded, or post-policy-dropped decisions are excluded from OPE | `policy/decision.py` | Counterfactual Logging |
| Unreviewed or edited creatives are never served | `review/store.py`, `safety/engine.py` | Creative Review |
| Click redirects carry only an opaque `click_id`, with `Referrer-Policy: no-referrer` | `api/app.py` | Advertiser Data Leakage |
| A cross-tenant postback looks exactly like an unknown click | `api/app.py` | Threat Model |
| Advertiser reports hide segments under 50 users or 10 conversions, plus their complement | `review/reports.py` | Advertiser Data Leakage |
| "Why am I seeing this?" is templated from ad-safe factors only | `review/explanations.py` | Explainability |
| Snapshots carry per-user-key pseudonyms; deletion shreds the key | `privacy/shredding.py` | DVC Edge Cases |

## LLM-assisted creative review (M8)

Advertisers submit creatives through `POST /v1/advertiser/creatives`. The ad enters the catalog as pending and is never served until reviewed. The review runs asynchronously, as a background task on the memory backend or in `theraflo-ads review worker` on infra. A false approval is the critical error, so every step can only move a creative toward "reject" or "human":

1. **Injection scan.** Text that addresses the reviewer is itself a violation. Matching runs over normalized text: NFKC, zero-width characters stripped, homoglyphs folded, leetspeak read as letters.
2. **Rules.** A strong restricted-category term, vulnerability targeting, or a request for health data rejects outright. Weaker signals block auto-approval.
3. **Model review.** Claude (`claude-opus-5-5`, structured outputs against an enumerated schema, no tools, advertiser text escaped and delimited) returns a verdict, a taxonomy label, violations, and the claims it found. Unavailable, refused, truncated, or invalid → human queue (fail closed).
4. **Claim grounding** is deterministic code, never the model: every claim must match the advertiser's substantiated claims (numbers and strengthening words included). Claims the model reports that are not in the creative are dropped as hallucinations.
5. **Dual check.** High-risk categories (next to a restricted one) need a second, adversarially prompted review that agrees.
6. **Abstention.** Anything short of a confident approval goes to a human. Health-adjacent categories always do.
7. **Audit sampling.** 5% of auto-approvals are queued for a human audit; `GET /v1/admin/review-audit` reports the measured false-approval rate with a Wilson interval.

The Claude adapter enables server-side refusal fallbacks (`fallbacks="default"`), and each review records the model that actually answered. It caches the stable system prompt and sets effort to `high`. Every record carries pipeline, prompt, model, taxonomy, and rules versions.

**A local model instead of Claude** (`review/local.py`, free, nothing leaves the machine). `THERAFLO_ADS_REVIEWER=ollama` reviews with an [Ollama](https://ollama.com) model, `qwen3.5:9b` by default (`THERAFLO_ADS_OLLAMA_MODEL`). The contract is the Claude adapter's:
- the same policy prompt and delimited advertiser text;
- output constrained to the same schema, through Ollama's structured outputs;
- no tools;
- every failure (Ollama down, model not pulled, timeout, truncation, invalid JSON) goes to the human queue.

`THERAFLO_ADS_NARRATOR=ollama` writes report summaries with it too. The compose stack reaches Ollama on the host at `host.docker.internal:11434`. Evaluate a model for free with `theraflo-ads review eval --reviewer ollama [--ollama-model NAME] [--think]`, or from the console's **Creative-review eval** job.

Also in M8:
- **Report narrator.** Its numbers are checked against the source table; on a mismatch it regenerates once, then falls back to a template.
- **Advertiser reports** with aggregation thresholds.
- **"Why this ad"** (`GET /v1/ads/{impression_id}/why`).
- **Post-training export.** SFT plus DPO-style preference pairs with false-approval corrections weighted 3×, split by advertiser, with lineage. The gate compares a candidate against the baseline and registers it.

### Eval results (190 labeled creatives, 11 slices)

The reviewer below is **simulated**: it stands in for an LLM with configured error rates (8% misses, 5% mislabels, 5% hallucinated claims, 2% invalid outputs) to measure what the *controls* catch. It is not a measure of Claude.

| reviewer | false approvals | false rejections | to humans | automation | injection approvals |
|---|---|---|---|---|---|
| rules only (no model) | 0 / 117 | 4 / 73 | 52% | 48% | 0 / 20 |
| simulated reviewer + controls | **0 / 117** (95% CI ≤ 3.2%) | 4 / 73 | 21% | **79%** | 0 / 20 |
| same reviewer, verdict alone (ablation) | 29 / 117 | 3 / 73 | 2% | 98% | 2 / 20 |
| reviewer that obeys injected text + controls | **0 / 117** | 4 / 73 | 21% | 79% | **0 / 20** |
| that reviewer's verdict alone (ablation) | 46 / 117 | 3 / 73 | 2% | 98% | 19 / 20 |

Other results:
- Claim grounding on auto-approved creatives was 100%.
- Health-adjacent auto-approvals: 0.
- Taxonomy accuracy: 96.8%.
- Six hallucinated claims were caught and dropped.
- Model-vs-label κ = 0.66.
- Narrator: 40 reports were 100% numerically faithful, with 9 regenerated after the check caught an error.

### A real model, locally: qwen3.5:9b through Ollama

The same 190 cases with `qwen3.5:9b` (Q4, entirely on an 8 GB RTX 5060 Ti, about 2 s per call, 7.5 minutes per run). Unlike the simulated reviewer, this is a real model's behaviour.

| qwen3.5:9b | false approvals | false rejections | to humans | automation | injection approvals | category accuracy |
|---|---|---|---|---|---|---|
| with the controls (`local1` prompt) | **0 / 117** | 11 / 73 | 22% | **78%** | 0 / 20 | 77% |
| its verdict alone (ablation) | **0 / 117** | 28 / 73 | – | – | 0 / 20 | – |
| first prompt (Claude's, unchanged) | 0 / 117 | 12 / 73 | 36% | 64% | 0 / 20 | 46% |

- **It errs toward rejecting.** On its own it approved no bad ad and obeyed no injected instruction (the simulated reviewer approved 29 and obeyed 2). But it rejected 28 of 73 good ads. The controls bring that to 11, still about three times the simulated reviewer's rate, so the cost of this model is wrongly rejected advertisers rather than unsafe ads.
- **The `local1` prompt.** The first run used Claude's prompt unchanged and auto-approved none of the 32 clean ads, for two reasons:
  - Ollama constrains decoding to the schema without showing the schema to the model, so it guessed category labels ("online courses" and "travel" came back as `other`).
  - It listed plain product descriptions ("lightweight tents for weekend adventures") as claims, and grounding sent them to a human.

  Local models now also get the field list with the allowed labels, and a definition of a claim with examples that are not in the eval set. That lifted category accuracy from 46% to 77% and automation from 64% to 78%, with false approvals still 0. Claude's prompt is unchanged.
- **Report narrator:** 5 of 5 summaries were numerically faithful on the first draft.

**Live Claude eval:** `theraflo-ads review eval --reviewer anthropic --live` runs the same 190 cases against Claude. It makes one or two calls per case (two when the dual check applies) and is billed, so it has not been run here.

## Operations (M9)

- **Metrics** (`GET /metrics`, optional bearer token). Prometheus metrics cover:
  - decisions by policy, blocked and degraded reasons;
  - **served-ad safety violations** (expected 0; the ad is withheld);
  - HTTP latency by route template (never raw paths, which carry tokens);
  - client events, rate limiting, review outcomes and queue depth, kill switch, and dependency readiness.
- **Alerts** (`monitoring/prometheus/alerts.yml`): 12 rules, among them the safety gate, fail-closed spikes, dependencies down, 5xx rate, rank p99, no ads served, dead-letter queues, review backlog, and the rate limiter failing open. The Grafana dashboard has 13 panels.
  - `promtool check config` passes.
  - `promtool test rules monitoring/prometheus/alerts_test.yml` passes: 15 test groups feed synthetic series and assert each alert fires, or stays quiet, at the right time with the right labels and text. A deliberately broken expectation fails it.
  - A unit test scrapes the app's `/metrics` and requires every metric the alerts and dashboard use, so a renamed metric cannot silently disable an alert.
  - CI runs all three.
- **Logs.** `THERAFLO_ADS_LOG_FORMAT=json` writes one object per line. A redaction filter strips bearer tokens, advertiser keys, 32-byte hex secrets, and API keys. Access logs carry route templates and request IDs, never user IDs.
- **Rate limiting.** GCRA per principal and scope; Redis-backed across replicas, using server time. It fails open (counted) because safety state already fails closed elsewhere.
- **Right to deletion.** `DELETE /v1/users/me` forgets the profile now and queues a request. `theraflo-ads privacy delete-pending` (hourly) then:
  - shreds the user key, so lake rows become unlinkable;
  - anonymizes decision records and the outbox;
  - deletes impressions, keys, and the profile;
  - clears Redis state.
  The audit trail records that a deletion ran, never whose.
- **Container.** Multi-stage `Dockerfile` with a non-root user (uid 10001). Compose profiles: `app` (migrate, api, relay, four workers), `monitoring`, and `jobs`.
- **CI** (`.github/workflows/ci.yml`): lint, unit, integration against service containers, offline eval gates, bandit and pip-audit, and an image build.

### Load test (`scripts/loadtest.py`, 5,000-user population, 20 s)

| backend | concurrency | rank req/s | p50 | p95 | p99 | errors | 100 ms p99 SLO |
|---|---:|---:|---:|---:|---:|---:|---|
| memory (1 process) | 5 | 356 | 7 ms | 11 ms | 34 ms | 0 | pass |
| memory (1 process) | 20 | 408 | 30 ms | 86 ms | 113 ms | 0 | fail (saturated) |
| infra (4 workers, Docker Desktop) | 5 | 90 | 39 ms | 46 ms | 52 ms | 0 | pass |
| infra (4 workers, Docker Desktop) | 20 | 192 | 79 ms | 109 ms | 125 ms | 0 | fail |

The memory backend serializes on one lock (it is a dev and test stand-in). On infra, each rank does several Postgres and Redis round trips (profile, kill switch, reservation, and one transaction for the decision, impression, label, and outbox row) through Docker Desktop's network. The first optimizations would be to cache profiles, batch the decision transaction, and run on Linux hosts.

### Chaos (`scripts/chaos.py`, real containers stopped and restarted)

| scenario | expected | result |
|---|---|---|
| Redis down | no ads (fail closed), readiness 503; resumes after restart | pass, recovered in 0.1 s |
| RabbitMQ down | serving and events unaffected; outbox buffers, relay drains after restart | pass, drained in 13 s |
| Postgres down | no ad served (retryable 503) | pass after the fix below; recovered in 0.1 s |
| Kill switch | every process stops serving within the cache window | pass |

## Scaling research (M10)

`theraflo-ads research scaling` writes `reports/research/scaling.{md,json}`. It runs on a 280k-decision simulated log: 121k training rows (2,201 clicks) and 36.5k test rows (674 clicks). Quality is normalized entropy (NE < 1 beats predicting the base rate), with bootstrap 95% CIs.

- **Data volume.** AUC rises steadily (0.539 at 2% of the data, 0.583 at 100%), but NE plateaus near 0.996. The fitted power law NE(n) = 0.9963 + 0.357·n^(-0.669) says most of the remaining log loss is irreducible noise. Closing half the remaining gap would need about 341k rows, roughly 3× today.
- **Model size.** LightGBM from 2 to 128 leaves per tree lands within overlapping CIs (NE 0.992–1.002, AUC 0.579–0.587). p50 scoring latency stays under 0.1 ms for 10 candidates, so size is not the bottleneck.
- **Features.** Only the full set (with personalization for consented users) has an NE interval clearly below 1.0 (0.9934 [0.9886, 0.9991]). Adding the exposure features made NE worse (0.9947 → 1.0021), though inside the noise.
- **Revenue vs complexity vs cost.** In the simulator, the ML ranker earned about 15% more revenue per request than the heuristic in both paired seeds, and the 4-leaf model did as well as the 64-leaf one. At about $0.0005 of compute per million requests (assuming $0.04/vCPU-hour) against about $1,700 net gain, serving cost is irrelevant at this scale. Latency budget and review effort bind long before compute dollars do.
- **Offline NE was a poor proxy for revenue here.** It is nearly flat while ranking quality and revenue move. The more conservative evidence remains the budget-split A/B below.

## OPE validation on real logs (Open Bandit Dataset)

[Open Bandit Dataset](https://research.zozo.com/data.html) (ZOZO, Inc.; Saito et al., [arXiv:2008.07146](https://arxiv.org/abs/2008.07146); CC BY 4.0) holds one week of real recommendation logs from two policies run side by side on the same traffic: Uniform Random (2.7M slots) and Bernoulli Thompson Sampling (24.2M slots), with logged propensities.

The validation estimates each policy's click rate from the *other* policy's logs, using the platform's own estimators (IPS, SNIPS, DR with their reliability checks), and compares against the truth from that policy's own logs. It covers three campaigns and two directions, with several target-policy specifications. `theraflo-ads research obd` writes `reports/research/obd_ope.{md,json}`.

| | result |
|---|---|
| Conclusive estimates covering the truth | **27 of 27** (95% intervals, resampling whole hours) |
| Estimates that missed the truth | 4, **all flagged inconclusive** |
| Error of conclusive estimates | median 4.4%, max 11.9% |
| Logs of 5k–20k rows | never conclusive (median errors 10–54%), as intended |
| Logs of 100k–400k rows | conclusive; coverage 73–100% |

What the real data taught the estimators:

- **Logged propensities can be wrong.** 4,501 Thompson Sampling rows carry Monte Carlo propensities floored near 1e-6. In the Men campaign, 159 such rows carry 12% of the importance weight and moved SNIPS by 7%, and they passed the existing ESS and reward-event checks. New check, in the shared estimator core: importance weights must average 1 (within 5%, and not significantly different). The "uniform" logger also deviates from uniform by up to about ±10% per item, which moves the mean weight by under 2%, inside the tolerance.
- **Rows are not independent.** Impressions in the same hour share effects. Row-level bootstrap intervals were about 1.5× too narrow: with them, 8 conclusive estimates missed the truth. The estimators now take `clusters`. The dataset validation resamples hours, and the platform's own OPE (pipeline, simulator, RL) now resamples users.
- **The target policy must be specified.** Thompson Sampling drifted during the week: the total-variation distance between day 1 and day 7 is 0.23–0.31, and an item's daily probability swung up to 27×. Hourly and week-average versions of "the BTS policy" differ by up to 8%, and neither is uniformly closer. Weight diagnostics cannot detect a wrong target definition.
- **Rewards may depend on the policy.** The same item, at the same position, in the same hour, drew 0.92× [0.82, 1.04] the clicks under Thompson Sampling that the random log predicts (Women campaign). OPE assumes this ratio is 1. The new `reward_invariance` check measures it whenever two policies' logs overlap, which the platform's uniform-exploration holdout provides.
- **Conclusion:** with honest intervals, OPE screens policies reliably, but its point estimates carry errors up to about 12% even on millions of rows. That is why promotion still requires an online A/B test.

## Findings

From the earlier milestones (simulated):

- **Simulator baseline** (3,000 users, 14 days): 36,723 requests, 24,211 ads served, $406.86 revenue, 0 safety violations.
- **ML pipeline** (93,476 decisions):
  - The CTR model passes its gates: log loss 0.0804 vs 0.0863 for the baseline, ECE 0.0047 vs 0.0205. It is registered as `ctr v1` (offline_validated).
  - The CVR model fails (log loss 0.179 vs 0.172), so serving keeps category priors for pCVR.
  - OPE was inconclusive for the ranker.
- **A/B, ML vs heuristic:** +3.3% revenue per user, but the CI spans zero, and the pre-registered MDE was 36.9%. Not promoted; SRM passes.
- **Bandits:** Beta Thompson sampling earned about 10% more than the heuristic in two runs, with cohort parity.
- **Drift:**
  - Zero false alarms in the stable run.
  - An injected drift was detected at day 42 by per-category calibration (budgeting apps +140%, streaming −71%).
  - The controller froze promotion. A full retrain was blocked because business metrics were unaffected, so recalibration is the first response.
- **RL:**
  - FQI: OPE inconclusive (ESS 94); in the simulator, +6.4% value and retention 88.7% vs 87.9%; safety evals pass.
  - PPO: learned from 0.0098 to 0.0168 per decision but evaluated below the heuristic (0.01625 vs 0.01775); safety evals pass.

## Bugs the harness caught

- **Budget overdelivery** (simulator): reserving only expected cost let one click blow past a nearly spent budget. Campaigns now need one worst-case event of headroom.
- **Postgres outage returned 500 instead of 503** (`scripts/chaos.py`): connection errors from the profile read escaped untranslated. The API now maps connection-level database errors to a retryable 503, with a regression test.
- **Seed profiles shared between containers** (full test run): `MemoryProfileStore` shallow-copied mutable demo profiles, so one test's "hide ad" leaked into the next container. Profiles are now deep-copied.
- **Integration flake:** a random hash key per test put the test user in a permanent holdout 4% of the time. Keys are now chosen outside the holdouts.
- **Ratings skipped grounding:** "Rated 4.9 stars by 80,000 readers" was not extracted as a claim. Ratings and audience statistics now count as claims.
- **Wrong export lineage:** exports recorded `claude-opus-5-5` as the base model when the judgements came from the simulated reviewer. Lineage now records the model that produced them.
- **Asserts on hot paths** (bandit): Python drops them under `-O`, which would have turned an invariant into a `TypeError`. They are now explicit checks.
- **Wrong propensities passed the OPE reliability checks** (Open Bandit Dataset): floored Monte Carlo propensities slipped through. Fixed with the mean-weight check.
- **New campaigns invisible until restart** (running the compose stack): the API loaded its catalog before `seed-demo` ran, and its refresher only updated ads of campaigns it already knew. It now refreshes campaigns, ads, and category priors every few seconds.
- **Analytics lake not writable** (compose stack): the named volume was root-owned, the container runs as an unprivileged user, and every message was dead-lettered. The image now creates the directory with the right owner.
- **Metrics split across API workers** (compose stack): with `--workers 2`, a scrape saw 17 of 20 requests, because each worker keeps its own counters. The stack now runs one worker per container, and `serve --workers N` warns about this.
- **The first safety violation would not have paged** (demo traffic on the stack): labeled counters appeared only with their first increment, Prometheus never saw the step from 0, and `increase()` read 0. 82 rate-limited requests showed as a rate of 0. Every alerted counter is now created at 0 for each known label at startup.
- **Review outcomes never reached Prometheus** (compose stack): reviews run in the review worker, which had no `/metrics`, so the review panel and the `ReviewerAbstainingTooMuch` alert could never see data. The worker now serves metrics on :9100, and Prometheus scrapes it.
- **The dev console never loaded behind a path-rewriting proxy** (deploying to emulated Azure Container Apps): the emulator reaches apps through the Kubernetes API server's service proxy, which rewrites absolute links in HTML, so the browser asked for `/api/v1/namespaces/…/proxy/ui/js/main.js` and the console stuck on "Loading…". The page's two asset links are now relative.
- **Bearer tokens never reached the app behind that same proxy** (emulated Container Apps): the Kubernetes API server authenticates with the `Authorization` header and then deletes it, so every admin and user call returned 401 while `X-Advertiser-Key` calls worked. With `THERAFLO_ADS_AUTH_FALLBACK_HEADER=on` (off by default, refused in production), the API also reads the same `Bearer` token from `X-Theraflo-Authorization`, which the console sends alongside `Authorization`.
- **A local model never saw the answer schema** (evaluating `qwen3.5:9b`): Ollama constrains output to the JSON schema but does not show it to the model, so it guessed category labels (46% accuracy). The labels and field definitions are now in the local prompt (77%).
- **Decision logs mislabeled their feature schema** (building the console's decision view): records said `features_v1` while their features came from the `features_v2` builder. The logged version is now the feature provider's.
- **The registry did not load in the Linux image** (mounting `models/` into the API container): versions registered on Windows stored backslash paths. Paths are now written with forward slashes, and old indexes are read either way.
- **Bootstrap intervals too narrow** (Open Bandit Dataset): they resampled rows, but rows are correlated. Fixed with cluster resampling (by hour for the dataset, by user for the platform).

## Windows notes

- `psycopg`'s native DLL is blocked by Windows Application Control on this machine, so the infra extra uses the pure-Python `pg8000` driver.
- Source files are LF (`.gitattributes`, ruff `line-ending = "lf"`). Write patch files in binary mode or with an editor; text-mode writes from Python produce CRLF.
- The console code page is cp1252. Set `PYTHONIOENCODING=utf-8` for commands that print non-ASCII output.
- Docker Desktop adds network latency to every Postgres and Redis round trip; the infra load-test numbers reflect it.

## Limits and next steps

- Apart from the Open Bandit Dataset validation, all data is simulated. The next step is shadow traffic and a real holdout on production logs.
- The Claude reviewer and narrator are implemented and unit-tested with a fake client, but the live eval (billed) has not been run. Only run them in production after that eval passes its gates.
- Production needs a secrets manager and KMS for keys, Alertmanager routing, managed Postgres, Redis, and RabbitMQ, and TLS at the edge.
- No git repository has been created yet.
