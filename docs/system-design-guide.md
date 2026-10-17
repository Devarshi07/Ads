# Theraflo Ads: system design study guide

The goal: understand every part of this system well enough to explain it, defend it, and say
honestly where it would break. Everything here was checked against the code; file paths are
relative to `theraflo-ads/src/theraflo_ads/`. Numbers marked *simulated* come from the simulator,
*estimate* from stated assumptions, and *measured* from a real run.

Companion: `docs/interview-prep.md` has the stories, results and resume-level numbers. This guide is
the "how it works and why" underneath them.

**Contents**
1. The project in plain words
2. Who talks to the system: actors and APIs
3. The big picture: three paths
4. Life of a request, step by step
5. Codebase map: what lives where
6. Where data lives
7. The design patterns, by name
8. Consistency: what is exact, and what can lag
9. Failure modes
10. Scaling: today, 10×, 100×, 1000×
11. Known gaps and open problems
12. Technology choices, in one table
13. Cross-question drill
14. Glossary

---

## 1. The project in plain words

**Theraflo** is a mental-health and wellness app. **Theraflo Ads** is the system that decides, every
time a user opens a screen that may show an ad, **which ad to show, or no ad at all**.

Think of it as a careful shopkeeper with a strict rulebook:
1. **First, the rulebook.** Some screens never get ads: crisis resources, therapy chat, journal, mood
   check-in, assessment and onboarding. Some products are never advertised: gambling, alcohol,
   payday loans and 13 more categories. Minors get stricter rules. Someone who just opened crisis
   resources sees no ads for 24 hours.
2. **Then, the choice.** Among the ads that are allowed, a model picks the one most likely to earn
   money without being clickbait, sometimes exploring (trying less-proven ads) to keep learning.
3. **Then, the bookkeeping.** It records exactly what it chose and with what probability, so that
   later it can measure whether a *different* choice would have been better without risking users.
4. **Afterwards:** it counts views, clicks and conversions, bills advertisers (never above their
   budget), learns from the results, and watches for drift and failures.

**Why it's interesting:** a model that only maximises revenue would learn to target distressed or
impulsive users, because they may click more. So safety isn't a penalty the model can trade away;
it's a **hard filter that runs before the model**. When the system isn't sure it's safe (a
dependency is down, or state can't be read), it shows **no ad**. That's called "fail closed".

**Scale target:** about 100K monthly active users, roughly 112K–500K ad decisions a day (1.3–6
requests/s on average), designed for 100–300 requests/s at peak.

---

## 2. Who talks to the system: actors and APIs

| Actor | How they authenticate | What they call |
|---|---|---|
| **The Theraflo app** (a user's phone) | User bearer token, or a session token for test users | `POST /v1/ads/rank`, `POST /v1/events`, `GET /v1/click/{token}`, `PUT /v1/users/me/consent`, `DELETE /v1/users/me`, `GET /v1/ads/{impression_id}/why` |
| **The Theraflo backend** (server to server) | Service bearer token | `POST /v1/internal/suppressions` (the user opened crisis resources) |
| **Advertisers** | `X-Advertiser-Key` header (keys stored hashed) | `POST/GET /v1/advertiser/creatives`, `GET /v1/advertiser/campaigns`, `GET /v1/advertiser/report`, `POST /v1/postbacks/conversion`, `POST /v1/postbacks/reversal` |
| **Admins / trust and safety** | Admin bearer token | Review queue (`/v1/admin/review-queue…`), audit sampling (`/v1/admin/review-audit`), substantiated claims (`/v1/admin/claims`), kill switch (`/v1/admin/kill-switch`), audit log (`/v1/admin/audit`) |
| **Prometheus** | Optional bearer token | `GET /metrics` |
| **Load balancers / the platform** | none | `GET /healthz` (the process is alive), `GET /readyz` (every dependency is ready) |
| **Developers (dev only)** | Dev credentials | The console at `/ui` and `/v1/dev/*`. Never mounted when `THERAFLO_ADS_ENV=production`. |

**Why the API is shaped this way:**
- **One endpoint decides** (`/rank`), and **one endpoint records what happened** (`/events`). The app
  never tells the server "bill this"; it reports events, and the server decides what's billable.
- **Every event must carry the signed impression token** from the rank response, bound to that
  user. So a client can't invent impressions or clicks for ads it was never shown.
- **The click redirect carries only an opaque `click_id`.** No user identity goes to the advertiser,
  and `Referrer-Policy: no-referrer` stops the browser leaking the page.
- **A postback for another advertiser's click looks exactly like an unknown click.** Advertisers
  can't probe each other's data.

---

## 3. The big picture: three paths

```
 SERVING PATH (synchronous, milliseconds, no LLM, fails closed)
 app ──POST /v1/ads/rank──▶ API ──▶ constraint engine ──▶ policy chain ──▶ budget + frequency
                              │       (hard rules)        (ML / bandit /     (Redis, atomic)
                              │                            heuristic /
                              │                            popularity / NO_AD)
                              ▼
            ONE Postgres transaction: decision + impression + reward label + outbox row
                              │
 EVENT PATH (asynchronous, seconds, at-least-once + idempotent)
                              ▼
              outbox relay ──publisher confirms──▶ RabbitMQ topic exchange "theraflo.events"
                                                    ├─▶ accounting queue  → billing, labels
                                                    ├─▶ features queue    → Redis features, bandit counts
                                                    └─▶ analytics queue   → data lake (pseudonymised)
                                                    (each with .retry and .dlq queues)

 OFFLINE PATH (batch, minutes to days)
   data lake ──▶ DVC pipeline: generate → build → train → evaluate → register ──▶ model registry
   drift job ──▶ retraining controller (recalibrate first)      label finalisation (nightly)
   creative review worker (rules + LLM, asynchronous)            deletion job (hourly, crypto-shredding)
   OPE, experiments, RL, scaling research (offline evaluation only)
```

**The key split:** the serving path must be fast and safe, so it does the minimum synchronously and
pushes everything else (billing, features, analytics) onto the event path. The event path must be
reliable, which is why it uses the outbox, retries and dead-letter queues. The offline path must be
reproducible and gated, which is why it uses DVC, evaluation gates and a registry.

---

## 4. Life of a request, step by step

### 4.1 `POST /v1/ads/rank`: choosing an ad
Code: `api/app.py` → `policy/service.py` (`DecisionService.decide`). The order is deliberate.

1. **Authenticate** the user's bearer token, then check the **rate limit** (GCRA, per user and
   scope, shared through Redis). If the limiter's Redis fails, it fails *open* and counts it.
2. **Load the user context.** The profile is read from Postgres by **keyed user hash**, never the raw
   user ID (`infra/stores.py`, `PgProfileStore`): consent scope, age band, hidden ads, suppression.
3. **Assign holdouts.** A stable hash puts 2% of users in **uniform exploration** and 2% in **no
   ads** (`experiments/assignment.py`).
4. **Apply hard constraints** (`safety/engine.py`). Each ad is removed if:
   - the placement isn't on the allowlist, or the user is under 13;
   - the kill switch is on;
   - the user is inside a crisis suppression window (24 h);
   - the category is restricted (16 categories, plus 3 more for teens and unknown age);
   - the creative isn't approved, or was edited;
   - a frequency cap is reached (per ad 3 per 24 h, per advertiser 8, per user 12, at least 10 min
     between repeats of the same ad);
   - the budget can't absorb **one worst-case event** of headroom.

   If any of this state **can't be read** (Redis timeout of 50 ms, Postgres error), the answer is
   **NO_AD**. A missing counter is never read as zero.
5. **The policy chain samples** over the eligible set only. The chain is the guarded ML ranker →
   heuristic → popularity → NO_AD (`ml/serving.py`, `policy/policies.py`).
   - Each guarded level has a **40 ms latency budget** and a **circuit breaker**: 5 failures open it
     for 30 s, and the next level serves (`policy/guard.py`).
   - The policy returns a full **probability distribution** that always includes **NO_AD**, with
     exploration floors. The system samples from it and logs the **exact probability** of the action
     taken.
   - **Shadow models**, if any, score the same request and are logged, never served.
6. **Reserve the budget, then record the frequency.**
   - The budget reservation is **atomic**: a Redis Lua script checks and reserves the expected cost
     in one step, with headroom, for 15 minutes.
   - The impression is then added to the user's frequency set (a pipelined `ZADD` with a TTL). The
     cap check in step 4 and this record are separate steps (see section 11).
   - If either fails *after* the policy chose, the decision becomes **NO_AD, marked degraded**, so
     it never pollutes OPE or training.
7. **Sign an impression token** (HMAC over impression, ad, user hash and expiry;
   `security/tokens.py`).
8. **Write once, atomically.** One Postgres transaction writes the **decision record** (eligible
   ads, the full probability distribution, the chosen action, holdout, feature schema, versions), the
   **impression**, the **reward label** placeholder, and an **outbox row** with the `DECISION`
   message (`infra/postgres.py`). If Postgres is down, the response is a retryable **503** with
   `Retry-After`.
9. **Re-check before responding.** The served-ad safety gate checks every hard constraint again on
   the final answer. A violation withholds the ad and increments an alerted counter (target 0).
10. **Respond** with the ad (or an empty list, meaning NO_AD, which is normal), the impression token,
    and the click URL.

**Why this order:** anything that can remove an ad runs **before** sampling, so probabilities are
computed over what can actually be shown. That keeps OPE valid and makes it impossible for the
policy to "learn around" a rule.

### 4.2 `POST /v1/events`: the user saw or clicked the ad
Code: `events/ingestion.py`, `events/service.py`.
1. The token must be **valid, unexpired, and bound to this user**. Forged or expired events are kept
   in `rejected_events` for fraud analysis, but never become rewards or billable.
2. **Idempotency:** each event has an `event_id`; a duplicate returns `duplicate`
   (`ON CONFLICT DO NOTHING`).
3. The event and its outbox row commit together. The app gets `accepted` immediately; everything
   else happens asynchronously.

### 4.3 The outbox relay and the queues
Code: `infra/relay.py`, `infra/amqp.py`, `infra/worker.py`.
1. The relay claims unpublished outbox rows with `SELECT … FOR UPDATE SKIP LOCKED`, so several
   relays never take the same row.
2. It publishes to the **topic exchange `theraflo.events`** (routing key = message type) with
   **publisher confirms**, and marks a row `published_at` **only after the broker acknowledges it**.
3. If RabbitMQ is down, it retries with exponential backoff (0.5 s up to 30 s, with jitter) while
   rows accumulate in Postgres. A crash between publishing and marking means a duplicate send, which
   consumers handle.
4. **Queue bindings:**
   - `accounting` ← viewable, clicked, conversion, conversion-reversed;
   - `features` ← decision, clicked;
   - `analytics` ← everything (`#`).
5. **Workers:** consume → handle → ack. A transient failure goes to `<queue>.retry` (a per-message
   TTL, then back to the queue through `theraflo.redeliver`), up to **6 attempts**. A permanent
   failure or exhausted retries go to `<queue>.dlq` with the error attached. The original delivery
   is acked only **after** its retry or dead-letter copy is published, so nothing is lost.
   - **accounting** (`events/accounting.py`): billing and labels.
   - **features** (`infra/redis_features.py`): user features and Beta-bandit impression and click
     counts in Redis.
   - **analytics** (`infra/lake.py`): appends to `lake/<type>/date=YYYY-MM-DD/part-<host>-<pid>.jsonl`.
     Partitions are immutable and daily, and duplicates are possible (at-least-once delivery), so
     dataset builds deduplicate.

### 4.4 Click and conversion: attribution and billing
Code: `attribution/service.py`, `events/accounting.py`, `counters/budget.py`.
- `GET /v1/click/{token}` verifies the token, records the click, and redirects with an opaque
  `click_id`.
- **The advertiser's server** later sends `POST /v1/postbacks/conversion` with that `click_id`.
- **Attribution is last-click within 7 days**, plus 1 day of grace for late clicks, so a label
  **matures after 8 days**. Only mature labels are used for training, so the model never learns
  "no conversion" from a conversion that simply hasn't arrived yet.
- **Billing settles the reservation** made at serve time. Every charge carries an **idempotency
  key**, so a retried event never bills twice. A reservation older than 15 minutes is billed as a
  direct charge. Advertisers are **never billed above budget**; any excess is logged as
  overdelivery, absorbed by Theraflo.
- **A reversal** (refund or chargeback) creates a credit event; billed amounts are corrected, never
  silently edited. Invoices reconcile from the durable ledger, never from Redis.

### 4.5 Creative review: an advertiser submits an ad
Code: `review/pipeline.py`, plus `rules.py`, `injection.py`, `grounding.py`, `llm.py` / `local.py`.
1. `POST /v1/advertiser/creatives` stores it as **pending**. It isn't servable.
2. The **review worker** (`theraflo-ads review worker`) runs the pipeline. Every step can only move a
   creative towards **reject** or **human review**:
   1. **injection scan** on normalised text;
   2. **deterministic rules**;
   3. **model review** (Claude, or a local model through Ollama, or the simulated reviewer for
      evals), with structured output, no tools, and delimited advertiser text;
   4. **claim grounding** in code: each claim must match the advertiser's substantiated claims;
   5. a **dual check** for high-risk categories;
   6. **abstention**: anything short of a confident approval goes to a human;
   7. **5% audit sampling** of auto-approvals.
3. If the model is down, invalid or refusing, the creative goes **to the human queue**, never to
   auto-approval.
4. The API's catalog **refreshes every 5 s** (`THERAFLO_ADS_CATALOG_REFRESH_S`), so an approved ad
   becomes servable within about 5 s.

### 4.6 Safety controls that arrive from outside
- **Crisis suppression:** the Theraflo backend calls `POST /v1/internal/suppressions` with a
  service token. The user gets no ads for 24 hours. The ads service is never told *why*.
- **Consent change:** `PUT /v1/users/me/consent`. Minors and unknown age are always
  contextual-only, whatever they set.
- **Kill switch:** `PUT /v1/admin/kill-switch` writes to `platform_settings` in Postgres, with a
  reason, and audits it. Every process re-reads it at most every **1 s**; if it can't be read, the
  switch is treated as **on**.

### 4.7 Right to deletion: `DELETE /v1/users/me`
Code: `privacy/shredding.py`, `infra/privacy_store.py`.
1. The profile is forgotten immediately, and a deletion request is queued.
2. The hourly job **shreds the user's key** (in `user_keys`, the one table never included in dataset
   snapshots). Data-lake rows were pseudonymised with that key, so they become **unlinkable**: you
   can't delete rows from immutable, versioned snapshots, but you can destroy the key.
3. It anonymises decision records and the outbox, deletes impressions, keys and the profile, and
   clears Redis state. The audit log records that a deletion ran, never whose.
4. **Caveat:** each process caches user keys for up to **60 s**, so a shredded key can be used for
   at most 60 s by a process that already held it.

### 4.8 A model's life: from training to serving
Code: `ml/pipeline.py`, `ml/registry.py`, `ml/serving.py`, `dvc.yaml`.
1. **DVC stages:** `generate` (simulated logs) → `build` (features, from the same code serving uses)
   → `train` (LightGBM CTR/CVR + isotonic calibration) → `evaluate` (gates) → `register`.
2. **Gates:**
   - log loss no worse than the baseline;
   - ECE ≤ 0.02;
   - no slice regresses by more than 0.02 log loss (paired bootstrap);
   - export parity (the exported bundle scores exactly like the trained model).
3. **Registry stages:** `offline_validated` → `shadow` (scores live traffic, logged only) →
   `production`. **Only production serves.** Rollback goes back to the previous production version.
   Bundles are signed and never pickled, and a bundle that fails verification (checksum, signature,
   feature schema) doesn't load, so serving falls back to the heuristic.
4. **The API builds its serving chain at start-up** from the registry. The dev console can swap the
   chain live.
5. **Monitoring:** the drift job checks PSI and KS on features, calibration per category, and
   concept drift. The **retraining controller** prefers **recalibration** over a full retrain, and
   freezes promotion during a drift spike.

---

## 5. Codebase map: what lives where

| Area | Package or file | What it does |
|---|---|---|
| **Entry points** | `cli.py` | Every command: `serve`, `relay`, `worker …`, `migrate`, `seed-demo`, `ml …`, `monitor …`, `experiment …`, `rl …`, `review …`, `privacy …`, `research …`, `simulate` |
| | `api/app.py` | The HTTP API: routes, auth roles, rate limits, metrics middleware, error mapping (database down → 503) |
| | `api/container.py` | Wiring: builds the in-memory or Postgres/Redis "container" of every component from environment variables |
| | `platform.py`, `infra/platform.py` | Assembles the whole decision platform; the same platform wired to memory or to real infrastructure |
| **Serving core** | `policy/service.py` | The decision path (section 4.1) |
| | `policy/policies.py`, `scoring.py` | Heuristic, popularity and uniform policies; sampling; distribution validation; expected-revenue scoring |
| | `policy/bandits.py` | Beta and linear Thompson sampling, with Monte Carlo propensities |
| | `policy/guard.py` | Circuit breaker and latency budget per policy level |
| | `policy/decision.py` | The decision record: what's logged, and `ope_eligible` |
| **Safety** | `safety/policy.py` | The rulebook: ad-free placements, restricted categories, personalisation exclusions, the 24 h suppression |
| | `safety/engine.py` | The constraint engine: per-ad eligibility with block reasons, failing closed |
| **Money** | `pricing.py` | CPM / CPC / CPA expected revenue and worst-case event cost |
| | `counters/budget.py`, `frequency.py` | Budget reservations (15 min TTL, headroom, idempotent settle) and frequency caps |
| **Events** | `events/schemas.py` | Versioned event contracts |
| | `events/ingestion.py`, `service.py` | Token checks, idempotency, recording |
| | `events/bus.py`, `accounting.py` | Message types, the in-process bus, the billing worker |
| | `attribution/service.py` | Last-click 7-day attribution, delayed labels, maturity |
| **Infrastructure** | `infra/schema.py`, `migrations/` | 22 Postgres tables, with Alembic migrations |
| | `infra/postgres.py` | Repositories, the unit of work with the outbox, the impression log, the billing ledger, and `ON CONFLICT` idempotency |
| | `infra/redis_stores.py`, `redis_features.py` | Hot state: frequency, budgets, rate limits, features, bandit counts |
| | `infra/relay.py`, `amqp.py`, `worker.py`, `lake.py` | Outbox relay, queue topology, consumers with retry and DLQ, data-lake writer |
| | `infra/stores.py`, `review_store.py`, `privacy_store.py` | Catalog, priors, profiles, kill switch, audit log, review storage, user keys |
| | `infra/settings.py` | Environment settings: database, Redis and AMQP URLs, Redis timeout 50 ms, DB pool 10 + 10 overflow, worker max attempts 6 |
| **ML** | `ml/features.py` | Features shared by training and serving (no skew) |
| | `ml/dataset.py`, `validation.py` | Training examples from the decision log; data checks |
| | `ml/models.py`, `post_training.py` | LightGBM / logistic models, isotonic calibration, export |
| | `ml/evaluation.py` | Metrics and gates |
| | `ml/bundle.py`, `registry.py`, `serving.py`, `scorer.py` | Signed bundles, the registry lifecycle, building the serving chain, the online scorer |
| | `ml/pipeline.py` | The DVC stages |
| **Evaluation** | `ope/estimators.py`, `targets.py` | IPS, SNIPS, DR, ESS, mean-weight check, cluster bootstrap |
| | `experiments/*` | Assignment and holdouts, stats, CUPED, power and MDE, sequential tests, the promotion rule |
| | `simulator/*` | A synthetic world driving the **real** platform; A/B simulation |
| | `research/scaling.py`, `obd.py` | Learning curves; OPE validation on real logs |
| **Monitoring** | `monitoring/drift.py`, `calibration.py`, `embedding.py`, `jobs.py`, `retraining.py` | Drift signals and the drift job |
| | `orchestration/controller.py`, `airflow_dag.py` | The retraining state machine; scheduled jobs |
| | `observability/metrics.py`, `logs.py`, `ratelimit.py` | Prometheus metrics, JSON logs with redaction, GCRA |
| **RL** | `rl/dataset.py`, `fqi.py`, `ppo.py`, `experiment.py`, `safety.py` | Offline FQI, simulator-only PPO, vulnerable-cohort safety evals |
| **Creative review** | `review/pipeline.py` | The 7 steps in 4.5 |
| | `review/rules.py`, `injection.py`, `grounding.py`, `text.py`, `taxonomy.py` | The deterministic controls |
| | `review/llm.py`, `local.py`, `simulated.py`, `wiring.py` | Reviewer back ends (Claude, Ollama, simulated) and how one is selected |
| | `review/store.py`, `reports.py`, `narrator.py`, `explanations.py` | Queue and records, advertiser reports, LLM narration with number checks, "why this ad" |
| | `review/evals.py`, `evalset.py`, `metrics.py`, `post_training.py` | Eval suite (190 cases), statistics, SFT and DPO export |
| **Privacy and security** | `privacy/shredding.py`, `security/identifiers.py`, `tokens.py`, `sessions.py` | Crypto-shredding; keyed pseudonyms; signed impression tokens; session tokens |
| **Dev only** | `api/dev.py`, `lab.py`, `jobs.py`, `devtools/traffic.py`, `api/static/console/` | The browser console, ranker lab, allowlisted jobs, demo traffic |

**Outside `src/`:**
- `docker-compose.yml`: postgres, redis, rabbitmq, migrate, api, relay, 4 workers, deletions,
  prometheus, grafana.
- `Dockerfile`: multi-stage, non-root.
- `monitoring/`: 12 alert rules plus their tests, and a Grafana dashboard.
- `.github/workflows/ci.yml`: lint, unit, integration, evals, security, alert tests, image build.
- `dvc.yaml`, `params.yaml`: the pipeline and its gate thresholds.
- `scripts/`: load test, chaos test, demo traffic.
- `deploy/localstack-azure/`: the cloud deployment guide.

---

## 6. Where data lives

### 6.1 Postgres: the system of record (22 tables)

| Group | Tables | Notes |
|---|---|---|
| Catalog | `campaigns`, `ads`, `category_priors` | Refreshed into each API process every 5 s |
| Serving log | `decisions`, `impressions`, `impression_labels` | Written in **one transaction per decision** |
| Events | `events`, `rejected_events`, `conversions` | `event_id` is unique, which gives idempotency; rejected events are kept for fraud analysis |
| Reliable messaging | `outbox` | `published_at` is set only after the broker confirms |
| Money | `budget_days`, `billing_ledger` | Invoices reconcile from the ledger, never from Redis |
| Users | `user_profiles`, `user_keys`, `deletion_requests` | Keyed by user hash; `user_keys` is never in dataset snapshots |
| Control | `platform_settings`, `audit_log` | Kill switch; who changed what (never whose data was deleted) |
| Review | `creatives`, `creative_reviews`, `review_queue`, `review_decisions`, `substantiated_claims` | The full audit trail of every review |

### 6.2 Redis: derived, rebuildable, hot state

| Key pattern | Holds |
|---|---|
| `freq:{user_hash}`, `freq:ready:{user_hash}` | A sorted set of the user's recent impressions (timestamps, with a TTL), rebuilt from Postgres when missing, plus a "ready" marker: no marker means "unknown", never zero |
| `bud:res:…`, `bud:billed:…`, `bud:exp:…`, `bud:ready:…`, `bud:rid:{reservation_id}` | Budget reservations, billed amounts, expiries, readiness, reservation lookup |
| `feat:{user_hash}` | Online user features (only with personalisation consent; expires after 30 days) |
| `imp:{category}`, `clk:{category}`, `bandit:beta_ts…` | Bandit and feature counts |
| `seen:{message}`, `pending:…` | Message deduplication for the features worker |
| Rate-limit keys | GCRA state per principal and scope |

**Rule:** if Redis is wiped, nothing is lost, because everything is rebuildable from Postgres. While
it's being rebuilt, the "ready" markers make serving **fail closed** rather than guess.

### 6.3 RabbitMQ: the event fan-out
- **Exchanges:** `theraflo.events` (topic), `theraflo.redeliver` (retry return), `theraflo.dlx`
  (dead letters).
- **Queues:** `accounting`, `features` and `analytics`, each durable, with its own `.retry` and
  `.dlq` queue.

### 6.4 Files
- **Data lake:** `lake/<type>/date=…/part-<host>-<pid>.jsonl`. Each process writes its own file, so
  several analytics replicas never collide.
- **Model registry:** `models/registry/registry.json` plus signed bundles.
- **DVC:** data and pipeline versions.
- **MLflow:** training runs.

---

## 7. The design patterns, by name

Interviewers love names. For each: what it is, and where it's used here.

| Pattern | What it means | Here |
|---|---|---|
| **Transactional outbox** | Write the business row and "message to send" in one DB transaction; a relay publishes later | Every decision and event (`infra/postgres.py`, `relay.py`) |
| **At-least-once + idempotent consumer** | Messages may arrive twice; handlers make duplicates harmless | `ON CONFLICT DO NOTHING` on event and message IDs; billing idempotency keys |
| **Publisher confirms** | The broker acknowledges each message before it's marked sent | The relay |
| **Retry with exponential backoff and jitter** | Retry transient failures with growing, randomised delays | Relay (0.5 s up to 30 s); workers (`.retry` queues, 6 attempts) |
| **Dead-letter queue** | Park messages that can't be processed, with the error, for humans | `<queue>.dlq`, alerted (`DeadLetteredMessages`) |
| **Fail closed vs fail open** | On uncertainty, choose the safe default | Safety and budget state → NO_AD; the rate limiter → open |
| **Graceful degradation / fallback chain** | A chain of decreasingly clever but more reliable options | ML → heuristic → popularity → NO_AD |
| **Circuit breaker** | Stop calling something that keeps failing, and retry after a cooldown | 5 failures → open for 30 s (`policy/guard.py`) |
| **Timeouts and latency budgets** | Never wait forever on a dependency | Redis 50 ms; 40 ms per policy level |
| **Bulkheads** | Isolate workloads so one can't sink the others | Separate worker processes and queues per concern |
| **Source of truth vs derived state** | One durable store; caches can be rebuilt | Postgres is the truth; Redis is derived |
| **Reservation (escrow)** | Hold capacity before consuming it | Budget reserved at serve time, settled later (15 min TTL) |
| **Signed tokens / stateless verification** | Verify without a database lookup | HMAC impression tokens bound to the user, with expiry |
| **Keyed pseudonymisation / crypto-shredding** | Encrypt or pseudonymise per user; delete the key to delete the data | `user_keys` + `privacy/shredding.py` |
| **Shadow deployment** | A new model scores live traffic, but its output is only logged | Registry `shadow` stage |
| **Canary / A-B with holdouts** | Expose a fraction of traffic; keep permanent control groups | 2% exploration + 2% no-ads holdouts; experiments package |
| **Kill switch** | A global brake that needs no deploy | `platform_settings`, 1 s cache, defaults to on if unreadable |
| **Health vs readiness probes** | Alive vs able to serve | `/healthz` vs `/readyz` (per-dependency checks) |
| **Rate limiting (GCRA)** | Smooth, burst-tolerant limits | `observability/ratelimit.py` |
| **Schema-versioned contracts** | Events and features carry versions | `events/schemas.py`; feature schema in every decision log |
| **Tested alerts / metric contracts** | Alerts as code, unit-tested | `promtool` tests; a metric-presence test |

---

## 8. Consistency: what is exact, and what can lag

**Strongly consistent (atomic):**
- the decision, its impression, its label and its outbox row (one transaction);
- an event and its outbox row;
- a budget check-and-reserve (a Redis Lua script).

**Not atomic:** the frequency cap is checked when eligibility is decided, and the impression is
recorded later. Two simultaneous requests for the same user can both pass a cap, so it can be
exceeded by one. That's tolerable for a soft UX limit, but it's listed in section 11.

**Eventually consistent, with known lag:**

| What | Normal lag | Why it's acceptable |
|---|---|---|
| Billing, labels, features, bandit counts, data lake | Seconds (the relay's poll plus the queue) | Budgets are protected by the **reservation** at serve time, not by billing |
| A new or approved ad becoming servable | Up to 5 s (catalog refresh) | An unapproved ad is never servable, so the lag only delays revenue, never safety |
| The kill switch taking effect | Up to 1 s per process | An incident brake measured in seconds |
| A shredded user key | Up to 60 s per process | It only affects pseudonymisation of new rows during that window |
| Labels usable for training | 8 days (maturity) | Otherwise the model learns false negatives |

**Ordering:** no global ordering is assumed across events. Each handler is idempotent and
order-tolerant: a conversion is matched by `click_id`, and billing by idempotency key.

**Clocks:** rate limiting uses the Redis server's time, so app servers' clock skew doesn't matter.
Token expiry and attribution windows use server time.

---

## 9. Failure modes

| What fails | Behaviour | Proven by |
|---|---|---|
| Redis | NO_AD for everyone (fail closed); readiness 503; serving resumes on recovery | Chaos test: recovered in 0.1 s; also observed in the Rootsight test |
| Postgres | NO_AD, retryable 503 with `Retry-After` | Chaos test, after fixing a bug that returned 500 |
| RabbitMQ | Serving unaffected; outbox rows wait; the relay drains them on recovery | Chaos test: drained in 12.6 s |
| A worker | Messages wait in the queue; retries, then DLQ | Worker tests |
| The ML model | Circuit breaker opens; the heuristic serves | Guard tests |
| The LLM reviewer | Creatives go to the human queue | Review tests and evals |
| The rate limiter's Redis | Fails open (counted, alerted) | Alert `RateLimiterFailingOpen` |
| A bad deploy | The platform keeps the last healthy revision | Observed during the cloud deployment |
| A bad model | Gates block it; or rollback in the registry | Registry tests |
| Everything else | Kill switch: every process stops within 1 s | Chaos test |

**The 12 alerts:**
- `ServedAdBrokeHardConstraint`: the safety gate; must be 0.
- `KillSwitchOn`
- `FailClosedSpike`
- `DependencyDown`
- `HighServerErrorRate`
- `RankLatencyP99High`
- `NoAdsServed`
- `DeadLetteredMessages`
- `CreativeReviewBacklog`
- `ReviewerAbstainingTooMuch`
- `RateLimitedSurge`
- `RateLimiterFailingOpen`

---

## 10. Scaling: today, 10×, 100×, 1000×

### Where time and work go today
Each `/rank` makes:
- 1 profile read from Postgres, which **isn't cached**;
- the kill-switch read (cached 1 s);
- several Redis round trips (frequency, budget, rate limit, features);
- the policy, under 0.1 ms for LightGBM scoring;
- one Postgres transaction writing about 4 rows.

**Measured** on a laptop (Docker Desktop, 4 workers, 5 concurrent clients): **90 req/s at p99 52
ms**, saturating near **192 req/s** (p99 125 ms). The in-memory build reaches 356 req/s at p99 34
ms on one process, so the database round trips are most of the cost.

**The API is stateless** on the real backend: its state lives in Postgres and Redis, apart from the
catalog snapshot and the model, which every replica loads for itself. So **horizontal scaling is
just more replicas**, until the shared stores saturate.

### Back-of-envelope sizing
This is an estimate. It assumes peak = 3× average, about 4 rows per decision, and that 66% of
requests serve an ad.

| Scale | MAU | Average / peak rank req/s | API replicas (at about 90 req/s each) | Postgres rows written per second at peak |
|---|---|---|---|---|
| Today | 100K | 1.3–6 / 4–18 (design target 100–300) | 1–2 (one handles about 500K–2.3M MAU) | about 70 |
| 10× | 1M | 13–60 / 40–180 | 2–3 | about 720 |
| 100× | 10M | 130–600 / 400–1,800 | 5–20, or fewer after the optimisations below | about 7,000 |
| 1000× | 100M | 1,300–6,000 / 4,000–18,000 | 50–200 | about 70,000: not one Postgres any more |

### What to change at each step

**Before 10× (cheap wins):**
1. **Cache profiles:** a read-through cache, with invalidation on consent or suppression changes,
   saves one database read per request.
2. **Pipeline the Redis calls** into one round trip, using a single Lua script for
   frequency + budget.
3. **Move Postgres off Burstable B1ms** to General Purpose, and add a read replica for the console,
   reports and dataset builds.
4. **Autoscale API replicas** on CPU and concurrency, keeping minimum replicas for latency.
5. **Prune and partition the outbox and logs** (see section 11).

**Around 100×:**
1. **Separate the decision log from the transactional database.** Keep the outbox atomicity, but
   partition `decisions`, `impressions` and `events` by day, archive old partitions to the lake, and
   batch inserts.
2. **Move from RabbitMQ to Kafka** once replay, long retention and several consumer groups matter.
   The outbox makes that a relay change, not an API change.
3. **Shard the relay:** run several relays (`SKIP LOCKED` already allows it), or use change data
   capture (Debezium) instead of polling.
4. **A Redis cluster with replicas,** keys sharded by user hash. Budgets per campaign are hot keys,
   so split them by slice.
5. **Separate model serving:** a dedicated scorer service or a batch-scoring cache, if models grow.
6. **A retrieval stage** once inventory reaches thousands of ads: retrieve top-K, then rank.

**Around 1000×:**
- **Multi-region:** users pinned to a region, and per-region budget **allocation** (a hard problem:
  a global budget across regions needs pacing quotas, not a single counter).
- **A streaming feature store** (Kafka → Flink) instead of workers writing to Redis.
- **Columnar storage for analytics** (Parquet on a lake), and a proper experiment platform.
- **Serving a separate exploration policy**, with logged propensities, through a policy service.

**What doesn't need to change:** the safety design, the probability logging, and the OPE and A/B
methodology. These scale with traffic and get *better* with more data, because effective sample
sizes grow and MDEs shrink.

---

## 11. Known gaps and open problems

Say these before an interviewer finds them. Each has a fix.

| Gap | Why it matters | Fix |
|---|---|---|
| **The outbox is never pruned** (rows are marked `published_at`, never deleted) | The table grows by about 500K+ rows a day at target scale; vacuum and index costs rise | Delete or partition published rows older than N days (keep a short replay window) |
| **No retention or partitioning** on `decisions`, `impressions`, `events` | The same growth problem; slower queries | Daily partitions, archived to the lake; a retention policy |
| **The profile is read from Postgres on every rank, uncached** | One database round trip per request; the main per-request cost after the write | A read-through cache with short TTL and invalidation on consent and suppression |
| **Redis is a hard serving dependency** (fail closed) | A Redis outage is an **ad outage** (lost revenue), by design | Redis HA (replica + failover) to cut the outage time; the trade-off stays |
| **Single Postgres, single region, Burstable tier** | Limited headroom; a regional outage stops serving | General Purpose + zone-redundant HA, a read replica, tested point-in-time restore |
| **RabbitMQ: one node, no persistent volume** in the practice deployment | A restart loses queued messages (the outbox re-sends only unpublished rows) | Quorum queues, persistent storage, or a managed broker |
| **One relay** | A throughput ceiling, and a single point of slowness (not of loss) | Several relays (`SKIP LOCKED` allows it) or change data capture |
| **Catalog by polling** (every 5 s) | Approvals, pauses and budget changes take up to 5 s to apply, plus a constant database load | Push invalidation through events; keep polling as a backstop |
| **The model registry is a JSON file** on disk; the chain loads at start-up | Replicas need a shared volume; promotion needs a restart or reload | Registry in blob storage or MLflow; hot reload on version change |
| **The data lake is on container disk** in the cloud deployment | Lost on restart | Blob storage / ADLS (Azure Files mounts) |
| **Silent config fallback:** an unknown `THERAFLO_ADS_BACKEND` value quietly runs the in-memory build | A misconfigured deploy looks healthy but isn't (it happened during deployment) | Validate config at start-up and refuse unknown values |
| **The frequency check and record aren't atomic** | Concurrent requests for the same user can exceed a cap by one | Check and record in one Lua script, like the budget reservation |
| **Key cache of 60 s** after deletion; kill-switch cache of 1 s | Short windows of stale state | Acceptable, but documented; lower or invalidate if required |
| **Metrics are per process** | Several server workers in one container split the counters | One worker per container, scale with replicas (done); or a multiprocess metrics mode |
| **Security hardening not done:** TLS to Postgres, a non-admin database user, Key Vault, edge TLS | Required in production | Listed in the README; the app refuses dev credentials in production |
| **Load testing is thin:** 20 s runs on a laptop | Doesn't prove sustained behaviour (leaks, GC, pool exhaustion) | Soak tests on cloud hardware; test autoscaling |
| **Simulated data** (except the Open Bandit Dataset); OPE for the ranker was inconclusive; the A/B MDE is high | Revenue claims aren't production evidence | Shadow traffic, then a real holdout and A/B |
| **Lake duplicates** from at-least-once delivery | Double counting if not handled | Dataset builds deduplicate by event ID (done); keep it that way |

---

## 12. Technology choices, in one table

The long version, with alternatives and when to switch, is in `docs/interview-prep.md` section 10.

| Tool | Role | One-line why |
|---|---|---|
| Python 3.12 + FastAPI | API and everything else | One language for serving, ML, OPE and the simulator; typed request validation |
| PostgreSQL + Alembic | System of record | Transactions make the outbox possible; versioned migrations |
| Redis (+ Lua) | Hot derived state | Sub-millisecond atomic counters across replicas |
| RabbitMQ | Event fan-out | Work queues with retry and DLQ; right-sized for this volume |
| LightGBM (+ isotonic calibration) | CTR model | Strong on tabular data, fast, calibrated, exports without pickle |
| Thompson sampling | Exploration | Natural exploration that stays OPE-evaluable (Monte Carlo propensities) |
| IPS / SNIPS / DR | Offline evaluation | Standard estimators; they cross-check each other |
| DVC + MLflow | Data / pipeline versioning + run tracking | Reproducible datasets + experiment records |
| Prometheus + Grafana | Metrics and alerts | Pull-based, PromQL, unit-testable alert rules |
| Claude / Ollama (qwen3.5:9b) | Creative review | Structured outputs; a free local model for development and evals |
| Docker + Azure Container Apps | Deployment | Revisions, jobs, secrets and ingress without operating Kubernetes |

---

## 13. Cross-question drill

Cover the answer, then check yourself.

**Architecture**
1. *Why not call RabbitMQ directly from the API?* Because publishing after the database commit can
   lose events in a crash, and publishing before it can send phantom events. The outbox makes the
   decision and its event atomic, and keeps the broker off the serving path.
2. *Why is Redis derived-only?* So it can be lost without losing truth. Billing reconciles from
   Postgres, and counters rebuild. That allows aggressive timeouts (50 ms) and failing closed.
3. *Why one transaction per decision instead of writing asynchronously?* The decision log *is* the
   evaluation data: without it, propensities and labels can't be trusted. Atomicity beats a few
   milliseconds. At 100× scale, partition and batch it rather than dropping atomicity.
4. *Is the API stateless?* On the real backend, yes. State is in Postgres and Redis; each replica
   holds only the catalog snapshot (refreshed every 5 s) and the model.

**Correctness and consistency**
5. *What if the relay crashes after publishing but before marking?* The row is published again,
   and consumers deduplicate by event or message ID. Delivery is at-least-once; processing is
   effectively once.
6. *Can an advertiser be billed twice?* No. Each charge has an idempotency key, and the ledger is
   append-only, with credits for reversals.
7. *Can an advertiser be billed above budget?* No. The serve-time reservation needs one worst-case
   event of headroom; anything beyond the budget is logged as overdelivery and absorbed.
8. *What does the user see if Redis is down?* No ads. Readiness 503 takes the replica out of rotation
   if every replica is affected, and alerts fire within 2 minutes.
9. *How do you stop a client inventing clicks?* Events need the signed impression token bound to
   that user, unexpired. Anything else is logged as rejected and never billed.
10. *Can a user see an ad more often than the cap?* By one, if two requests for the same user race:
    the cap check and the impression record are separate Redis steps. The budget can't be overspent
    that way, because its check-and-reserve is one Lua script. The fix is to do the same for
    frequency.

**Scaling**
11. *Where's the first bottleneck?* The per-request Postgres work: the uncached profile read plus
    the 4-row write transaction. Then Redis round trips. CPU and the model aren't the problem.
12. *How do you get to 10×?* Profile cache, pipelined Redis, General Purpose Postgres, autoscaled
    API replicas, outbox pruning.
13. *When would you move to Kafka?* When replay, long retention and several independent consumers
    matter. The outbox keeps that change inside the relay.
14. *How would you go multi-region?* Pin users to a region and split budgets into per-region
    quotas with pacing. A single global budget counter is the hard part.

**Reliability**
15. *What's your blast radius for a bad model?* Gates block it offline, shadow catches it on live
    traffic, the circuit breaker stops it in serving, rollback restores the previous version, and
    the kill switch is the last resort.
16. *Why does the rate limiter fail open when everything else fails closed?* It protects capacity,
    not users. Safety state already fails closed, so blocking legitimate users during a Redis blip
    adds risk without adding safety. It's counted and alerted.
17. *How do you know an alert works?* `promtool` unit tests feed synthetic series, a test checks
    every metric used by alerts exists, and counters start at 0 so `increase()` sees the first event
    (a bug I found and fixed).

**ML and evaluation**
18. *Why log the full probability distribution?* IPS, SNIPS and DR need the probability of the logged
    action under the logging policy, and diagnostics need the support of the whole distribution.
19. *Why exclude degraded decisions from OPE?* Their served action isn't what the policy sampled, so
    their propensities would be wrong.
20. *Why wait 8 days for labels?* A 7-day click window plus 1 day of grace. Training earlier would
    teach the model that converting users didn't convert.
21. *Why don't you trust offline NE alone?* It stayed flat while ranking and revenue moved 15% in
    simulation. Promotion needs an A/B test.

**Security and privacy**
22. *How do you delete a user's data from immutable training snapshots?* Crypto-shredding: data is
    pseudonymised with a per-user key, and deleting the key makes those rows unlinkable.
23. *What do advertisers learn about users?* Nothing user-level: opaque click IDs, aggregate reports
    with minimum segment sizes (50 users / 10 conversions, plus complements), and identical
    responses for unknown and cross-tenant clicks.
24. *How do you stop prompt injection in ad review?* Normalised scanning, delimited and escaped
    text, structured outputs, no tools, deterministic grounding, a dual check and abstention. The
    model's verdict can only make things stricter.

**Cost**
25. *What does it cost to run?* About $220 a month on Azure at target scale, mostly always-on
    container reservations. Reservations are billed, not usage: the API used 60 MiB of 2 GiB, so I
    right-sized it.

---

## 14. Glossary

| Term | Meaning |
|---|---|
| **Propensity** | The probability the logging policy gave the action it took |
| **OPE** | Off-policy evaluation: estimating a new policy's value from old logs |
| **IPS / SNIPS / DR** | Inverse propensity scoring (unbiased, high variance) / self-normalised IPS (lower variance) / doubly robust (adds a reward model; robust if either part is right) |
| **ESS** | Effective sample size, (Σw)²/Σw²: how many "useful" rows the weights leave. Below 100 = inconclusive here. |
| **Importance weight** | π_target(a) / π_logging(a). Its mean should be about 1. |
| **NE** | Normalized entropy: log loss ÷ the entropy of the base rate. Below 1 beats guessing the average. |
| **ECE** | Expected calibration error: how far predicted probabilities are from observed rates |
| **Isotonic calibration** | A monotone mapping from raw scores to calibrated probabilities |
| **CUPED** | Using pre-period behaviour to reduce variance in A/B metrics |
| **MDE** | Minimum detectable effect: the smallest lift a test can reliably detect |
| **SRM** | Sample-ratio mismatch: arms not split as designed, which signals a bug |
| **Thompson sampling** | Sample from each option's posterior, then pick the best sample: exploration in proportion to uncertainty |
| **FQI** | Fitted Q iteration: offline RL that learns action values from logged data |
| **Holdout** | A permanent group kept out of a treatment (here: 2% uniform, 2% no ads) |
| **Fail closed / fail open** | On uncertainty, deny (no ad) / allow (rate limiter) |
| **Transactional outbox** | Messages written in the business transaction, then relayed |
| **DLQ** | Dead-letter queue: messages that failed permanently, kept for humans |
| **GCRA** | Generic cell rate algorithm: a smooth rate limiter |
| **Crypto-shredding** | Deleting data by destroying the key that links or decrypts it |
| **Shadow** | A model scoring live traffic without serving |
| **Circuit breaker** | Skip a failing component for a cooldown, instead of retrying it constantly |
