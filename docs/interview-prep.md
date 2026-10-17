# Theraflo Ads: interview prep

Everything about the project in one place: what it is, how it works, the numbers, how it's deployed
on Azure, and the problems solved along the way, written as stories you can tell.

> **For an AI assistant helping me prepare.** This document is the complete, factual record of a
> project I built. Use it to help me write interview stories (STAR format), practise answers, and
> anticipate follow-up questions.
> - **Use only the facts and numbers in this document.** Don't invent metrics, team sizes, dates,
>   users, or results. If a story needs a detail that isn't here, ask me.
> - **Keep the labels.** "Simulated" numbers came from a simulator; "estimate" numbers came from
>   stated assumptions; only the Open Bandit Dataset results are on real logs.
> - **Keep the reasoning.** When I'm asked "why X?", answer with the reasons, alternatives, and
>   trade-offs in sections 9 and 10, not generic ones.
> - It was a solo project: I designed and built it (with an AI coding assistant), so "I" is accurate.
>
> **Map:**
> 1. Pitch
> 2. The product and the problem
> 3. Scale, capacity and cost
> 4. Revenue estimate
> 5. Architecture and Azure deployment
> 6. How each part works, including scaling laws (6.13), real-data OPE validation (6.14), and every
>    evaluation and gate (6.15)
> 7. Results summary
> 8. Problems solved (20 stories)
> 9. Design trade-offs
> 10. Why each tool, and the alternatives
> 11. Likely interview questions
> 12. Limits

> **Telling it honestly.** Every number here comes from the code, its test and evaluation runs, or a
> labelled assumption. Say "simulated" whenever a number came from the simulator, because an
> interviewer will ask where the data came from. The only real-world data is the Open Bandit Dataset
> (26.9M logged impressions). The cloud deployment used the real `az` CLI and Azure resource types
> (Container Apps, Container Registry, PostgreSQL Flexible Server), run in an Azure-compatible
> environment so it cost nothing. If someone asks "did this run in your Azure subscription?", say
> exactly that: it costs nothing and loses nothing.

---

## 1. The pitch

**30 seconds.** Theraflo is a mental-health and wellness app. I designed and built its ads decisioning
platform, which picks which ad, if any, to show on each screen. The hard part isn't ranking but
safety: a revenue-maximising model will learn to target distressed users through proxies. So every
safety rule is a hard constraint that runs before any model, and anything uncertain fails closed to
"no ad". On top of that sit an ML ranker, bandits, offline RL, and LLM-assisted creative review. All
of it is measured with off-policy evaluation and A/B tests, and it runs as microservices on Azure
Container Apps with Postgres, Redis and RabbitMQ.

**2 minutes.** Add three things:
1. **Evidence before promotion.** Every decision logs its exact propensity, so OPE can evaluate new
   policies offline. I validated those estimators on 26.9M real impressions from the Open Bandit
   Dataset: every conclusive estimate covered the truth, and real data exposed three estimator bugs.
2. **Fault tolerance.** A transactional outbox, idempotent workers with retries and dead-letter
   queues, and an explicit failure hierarchy. Chaos tests stop Redis, Postgres and RabbitMQ and check
   that the system degrades exactly as designed.
3. **LLM review where a false approval is the critical error.** 0 of 117 bad creatives were
   auto-approved, even with a reviewer that obeys injected instructions, because deterministic
   controls sit around the model.

---

## 2. The product and the problem

- **Users:** about 100K monthly active users (MAU) of a mental-health and wellness app. That means
  teens, people in crisis, and people with impulse-control problems.
- **Ads:** wellness partners and brands. Pricing is fixed-payout CPA (per conversion) with vetted
  partners, plus CPM (per 1,000 impressions) and CPC (per click). There's no auction at first, so
  reordering ads never changes what an advertiser pays.
- **Objective:** platform revenue, meaning what Theraflo bills, not the advertiser's order value.
  A minimum predicted CVR/ROAS floor stops ranking on clickbait.
- **Why it's hard (the interview hook):**
  - **Penalties aren't enough.** If distressed users convert more on certain products, a
    revenue-maximising policy learns to target distress through proxies, even with no
    mental-health feature. Revenue can outweigh a penalty, but it can't outweigh a constraint. So
    safety lives in an **eligibility layer** that removes actions before the policy samples.
  - **Data is sparse.** About 34–150 conversions a day across all campaigns, so per-ad models don't
    work. Statistical strength is shared through category and advertiser features, Bayesian
    shrinkage, and clicks as a denser signal.
  - **Experiments are slow.** At this traffic, a 10% lift needs 21–93 days of A/B to detect.

---

## 3. Scale, capacity and cost

### Traffic model (from the design doc)

| | Optimistic | Conservative |
|---|---|---|
| MAU | 100K | 100K |
| DAU / MAU | about 100% | 25% |
| Sessions per DAU per day | 1 | 1.5 |
| Ad opportunities per session | 5 | 3 |
| Ranking decisions per day | about 500K | about 112K |
| Ranking decisions per month | about 15M | about 3.4M |
| Average requests/s | about 6 | about 1.3 |
| Clicks per day (1% CTR) | about 5,000 | about 1,100 |
| Conversions per day (3% post-click CVR) | about 150 | about 34 |

The system was designed for a peak of **100–300 ranking requests/s**, to leave room for bursts,
retries, experiments and growth.

### Measured throughput (load test, 5,000-user population, 20 s)

| Backend | Concurrency | Rank req/s | p50 | p95 | p99 | Errors | p99 ≤ 100 ms SLO |
|---|---:|---:|---:|---:|---:|---:|---|
| In-memory (1 process) | 5 | 356 | 7 ms | 11 ms | 34 ms | 0 | pass |
| In-memory (1 process) | 20 | 408 | 30 ms | 86 ms | 113 ms | 0 | fail (saturated) |
| Postgres + Redis (4 workers, on a laptop) | 5 | 90 | 39 ms | 46 ms | 52 ms | 0 | pass |
| Postgres + Redis (4 workers, on a laptop) | 20 | 192 | 79 ms | 109 ms | 125 ms | 0 | fail |

On the real backend, each rank makes several Postgres and Redis round trips: profile, kill switch,
budget reservation, then one transaction that writes the decision, impression, label and outbox
row.

### How many users one instance handles

Assume peak traffic is 3× the average. At the measured, SLO-safe **90 req/s**:
- average capacity is 30 req/s, or about 2.6M decisions a day;
- with the optimistic usage profile (5 decisions per MAU per day), that's **about 500K MAU**;
- with the conservative profile (about 1.1 decisions per MAU per day), it's **about 2.3M MAU**.

So one API instance covers the 100K MAU target with **5–20× headroom**. The design's 300 req/s peak
takes 2–4 API replicas. The API is stateless, so it scales horizontally, and the next bottleneck is
Postgres writes (one transaction per decision). The planned optimisations are caching profiles,
batching the decision transaction, and moving Postgres up from the Burstable tier.

### What the deployment costs on Azure

Azure retail prices, East US, from Microsoft's price API:
- **Container Apps:** $0.000024 per vCPU-second and $0.000003 per GiB-second, with a free monthly
  grant of 180,000 vCPU-seconds and 360,000 GiB-seconds; $0.40 per million requests.
- **PostgreSQL Flexible Server B1ms:** $0.017 an hour.
- **Container Registry Basic:** $0.1666 a day.

| Item | Size | Per month |
|---|---|---|
| Container Apps, always on: API, relay, 4 workers, Redis, RabbitMQ | 2.5 vCPU, 5 GiB in total (right-sized) | about $189 |
| Requests | about 15–30M | about $6–12 |
| PostgreSQL Flexible Server, Burstable B1ms | 1 vCore, 2 GiB, 32 GiB storage | about $12, plus storage |
| Container Registry, Basic | | $5 |
| **Total** | | **about $215–230** |

Before right-sizing, the same apps reserved 3 vCPU and 6 GiB: about $228 a month of compute
instead of $189, an extra 17% for memory nobody used (story 3). That's about **$0.002 per MAU per
month**. Model scoring itself costs about $0.0005 per million requests, so compute is never the
constraint at this scale; latency and review effort bind first.

---

## 4. Revenue estimate

These are illustrative and assumption-driven. Say so.

| Assumption | Value | Source |
|---|---|---|
| Fill rate (ad shown per request) | 66% | simulator baseline: 24,211 ads from 36,723 requests; 63% in the cloud demo traffic |
| Pricing mix | 40% CPM, 40% CPC, 20% CPA | simulator catalog |
| Prices | CPM $4–12 (mid $8), CPC $0.30–1.50 (mid $0.90), CPA $8–30 (mid $19) | simulator catalog |
| Funnel | 1% CTR, 3% post-click CVR | design doc |

**Effective revenue per 1,000 impressions (eCPM) by pricing model:**
- CPM: $8.00.
- CPC: $0.90 × 1% × 1,000 = $9.00.
- CPA: $19 × 1% × 3% × 1,000 = $5.70.
- Blended at the mix above: **about $7.94**.

| Scenario | Impressions per month | Revenue per month | Per year |
|---|---|---|---|
| Conservative | 3.4M × 66% ≈ 2.24M | **about $17.8K** | about $214K |
| Optimistic | 15M × 66% ≈ 9.9M | **about $78.6K** | about $943K |

- **Cross-check:** the simulator earned $406.86 on 36,723 requests, or $11.08 per 1,000 requests.
  That gives $37.7K–$166K a month: the same order of magnitude.
- **Demand caveat:** at about 100K MAU, expect tens to a few hundred campaigns. Advertisers' daily
  budgets cap revenue before traffic does, which is why the simulator exhausts budgets on purpose,
  to exercise pacing.
- **Value of the ML work**, all simulated:
  - the ML ranker earned about 15% more revenue per request than the heuristic in paired seeds;
  - Beta Thompson sampling earned about 10% more;
  - the A/B test measured +3.3% revenue per user, **not significant** (its CI spans zero, and the
    pre-registered minimum detectable effect was 36.9%), so the model was not promoted.

  At conservative revenue, a 10% lift is worth about $1.8K a month against about $220 of infra.
- **Infra as a share of revenue:** about 1.2% conservative, 0.3% optimistic.

---

## 5. Architecture

```
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
          drift job → retraining controller        creative review worker (rules + LLM)
          deletion job (crypto-shredding)          scaling research
```

- **Python 3.12 and FastAPI.** About 18K lines of source and 5.6K lines of tests. 678 tests: 653
  unit, plus 25 integration tests against real Postgres, Redis and RabbitMQ. Clean under ruff and
  bandit; alert rules pass `promtool` unit tests.
- **Postgres is the system of record** for decisions, impressions, labels, the outbox, profiles,
  the review queue and the audit log.
- **Redis holds derived state only:** frequency counters, budget reservations, features and rate
  limits, all rebuildable from Postgres. A missing counter is never read as zero.
- **RabbitMQ** carries events from the outbox to the workers, with retry and dead-letter queues.
- **LLMs never sit on the serving path.** They run asynchronously (creative review) or offline.

### On Azure

| Component | Azure resource | Size | Notes |
|---|---|---|---|
| `theraflo-api` | Container App, external HTTPS ingress | 0.5 vCPU / 1 GiB | Stateless, so it scales horizontally |
| `theraflo-relay` | Container App, no ingress | 0.25 / 0.5 GiB | **Exactly one replica**: two would double-publish (safe, but wasteful) |
| Workers: accounting, features, analytics, review | 4 Container Apps, no ingress | 0.25 / 0.5 GiB each | Scale out freely: RabbitMQ gives each message to one consumer |
| Redis | Container App, internal TCP ingress | 0.25 / 0.5 GiB | Azure Cache for Redis in production |
| RabbitMQ | Container App, internal TCP ingress | 0.5 / 1 GiB | Azure has no managed RabbitMQ; Service Bus would mean rewriting the queue code |
| Postgres | PostgreSQL Flexible Server | Burstable B1ms, v16, 32 GiB | Encryption required on real Azure |
| `theraflo-migrate`, `theraflo-seed` | Container Apps **jobs**, manual trigger | 0.5 / 1 GiB, 900 s timeout, 1 retry | Migrations run once, before the apps, never concurrently |
| Images | Container Registry, Basic | tags `v1`…`v5` | Never `latest`: every deploy names exactly what runs |
| Keys | Container Apps secrets, referenced with `secretref:` | 3 × 32-byte keys | Key Vault references in production |
| Dev console | A separate Container App, in-memory | 0.5 / 1 GiB | Dev only; refused when `THERAFLO_ADS_ENV=production` |

**Deploy order:**
1. Infrastructure: Postgres, Redis, RabbitMQ.
2. The **migrate** job (schema), then the **seed** job (demo campaigns).
3. The API.
4. The relay and the workers.

Apps that start before their dependencies crash-loop until those exist. That's harmless, but noisy.

**Image:** multi-stage build. Wheels are built with the toolchain, and the runtime stage has none.
It runs as a non-root user (uid 10001) and holds no secrets. The model files are a small layer on
top, so a model version is pinned to an image tag. Size: 1.27 GB unpacked, 363 MB compressed.

---

## 6. How each part works (deep-dive answers)

### 6.1 Safety: hard constraints, fail closed
- Everything that can remove an ad runs **before** the policy samples. The policy only ever sees
  eligible ads.
- **Ad-free placements:** crisis resources, therapy chat, journal, mood check-in, assessment, and
  onboarding never get ads. Putting one on the allowlist raises an error at config time.
- **16 restricted categories** (gambling, alcohol, payday loans, crypto, weight loss, supplements,
  mental-health services…), plus 3 more for teens and users of unknown age. Under-13s get no ads.
- **Crisis suppression:** after someone opens crisis resources, ads pause for **24 hours**. The ads
  service is never told why.
- **No personalization for minors or unknown age:** contextual targeting only. Six impulse-prone
  categories (streaming, travel, gaming, fast fashion, luxury, buy-now-pay-later) are never
  personalised, for anyone.
- **Failure hierarchy:** if a counter, a budget, or the kill switch can't be read, the answer is
  NO_AD. The rate limiter is the one deliberate exception: it fails open (and counts it), because
  safety state already fails closed elsewhere.
- **Served-ad safety gate:** after the policy picks, every hard constraint is checked again. A
  violation withholds the ad and pages someone. The target is 0, and it's alerted.
- **Kill switch:** shared across processes. Every process stops serving within its 1-second cache
  window.
- **Creative review:** unreviewed or edited creatives are never served.

### 6.2 Policy chain and fallbacks
- The chain is **guarded ML ranker → heuristic → popularity → NO_AD**. A policy that errors or
  can't decide hands over to the next.
- **NO_AD is an action in every probability distribution**, and stochastic policies have
  exploration floors.
- **Holdouts:** 2% of users get uniform exploration, and 2% get no ads. Uniform exploration gives
  OPE clean data; the no-ads holdout measures what ads cost users.
- **Budgets:** a reservation, made atomically with a Redis Lua script, must leave headroom for one
  worst-case event. Story 9 explains why.

### 6.3 Logging and off-policy evaluation (OPE)
- Every decision logs its **exact propensity** (the probability of the action taken) and the
  feature schema version.
- Deterministic, degraded, or post-policy-dropped decisions are **excluded** from OPE, because
  their propensities aren't valid.
- **Estimators:** IPS, SNIPS and doubly robust. Each comes with reliability checks: effective
  sample size, reward-event counts, the mean importance weight (which must be about 1), and cluster
  bootstrap intervals. Each estimate is labelled conclusive or inconclusive.
- **Rule:** OPE **screens** candidate policies; an online A/B test **promotes** them (see 7.2).

### 6.4 ML ranker
- **Features are shared online and offline**, from the same builder, so there's no training/serving
  skew. Impulse categories get no personalization features.
- **Models:** a LightGBM CTR model with calibration. The CVR model **failed its gate** (log loss
  0.179 vs the baseline's 0.172), so serving keeps category priors for predicted CVR.
- **CTR results:** log loss 0.0804 vs 0.0863 for the baseline; calibration error (ECE) 0.0047 vs
  0.0205. It passes the gates, including paired-bootstrap checks for regressions in each slice.
- **Pipeline:** DVC (build → train → evaluate → register), with MLflow tracking.
- **Model files** are signed bundles that never use pickle.
- **Registry:** a lifecycle of candidate → offline_validated → production, with rollback. **Only a
  production version serves.**

### 6.5 Bandits
- **Beta Thompson sampling** (per ad) and **linear Thompson sampling** (contextual).
- Propensities are estimated by Monte Carlo sampling, because Thompson sampling has no closed form.
- They run inside guarded fallbacks.
- **Result:** Beta TS earned about 10% more than the heuristic in two simulated runs, with the same
  outcomes across user cohorts.

### 6.6 Experiments
- Deterministic assignment, with checks for sample-ratio mismatch (SRM).
- **CUPED** (pre-period covariates) and winsorization for heavy-tailed revenue; the delta method
  for ratio metrics.
- A minimum detectable effect is pre-registered before each test; there are guardrail metrics and
  sequential monitoring.
- **What this traffic can detect** (per-impression conversion rate 0.03%, 80% power):

  | Lift | Optimistic traffic | Conservative traffic |
  |---|---|---|
  | 2% | about 520 days | about 2,300 days |
  | 10% | about 21 days | about 93 days |
  | 20% | about 6 days | about 26 days |

  So promotion thresholds must be at least the achievable MDE. Clicks and calibration are the fast
  leading metrics, and refactors use non-inferiority tests.

### 6.7 Monitoring, drift and retraining
- **Drift signals:** context features (PSI, KS), calibration per category, concept drift, and
  business signals.
- **Result:** zero false alarms in the stable run. An injected drift was caught at **day 42**
  through per-category calibration (budgeting apps +140%, streaming −71%).
- **The retraining controller recalibrates first.** It froze promotion and blocked a full retrain,
  because business metrics weren't affected.

### 6.8 Reinforcement learning
- **Offline FQI** (fitted Q iteration) with a support constraint, gated by OPE and by simulator
  safety evaluations on vulnerable cohorts:
  - +6.4% value in the simulator;
  - retention 88.7% vs 87.9%;
  - but OPE was **inconclusive** (effective sample size 94), so it doesn't serve.
- **PPO, simulator only:** it learned (0.0098 → 0.0168 per decision) but still evaluated below the
  heuristic (0.01625 vs 0.01775). Its safety evals passed.
- **The point:** RL never serves without passing OPE and the safety evals.

### 6.9 LLM-assisted creative review
Advertisers submit creatives, which wait as *pending* and are never served until reviewed. Every
step can only push a creative towards **reject** or **human review**:
1. **Injection scan** over normalised text (NFKC, zero-width characters stripped, homoglyphs folded,
   leetspeak read as letters). Text that addresses the reviewer is itself a violation.
2. **Rules:** a restricted-category term, vulnerability targeting, or a request for health data
   rejects outright.
3. **Model review:** Claude with structured outputs against an enumerated schema, no tools, and
   advertiser text escaped and delimited. Any failure goes to the human queue.
4. **Claim grounding** is deterministic code, never the model. Every claim must match the
   advertiser's substantiated claims, and claims the model invents are dropped as hallucinations.
5. **Dual check:** high-risk categories need a second, adversarially prompted review that agrees.
6. **Abstention:** anything short of a confident approval goes to a human, and health-adjacent
   categories always do.
7. **Audit sampling:** 5% of auto-approvals get a human audit, which measures the false-approval
   rate with a Wilson interval.

**Eval set:** 190 labelled creatives in 11 slices.

| Reviewer | False approvals | Automation | Injection approvals |
|---|---|---|---|
| Rules only | 0 / 117 | 48% | 0 / 20 |
| Simulated LLM + controls | **0 / 117** (95% CI ≤ 3.2%) | **79%** | 0 / 20 |
| Same LLM, verdict alone (ablation) | 29 / 117 | 98% | 2 / 20 |
| LLM that obeys injected text + controls | **0 / 117** | 79% | **0 / 20** |
| That LLM's verdict alone | 46 / 117 | 98% | 19 / 20 |
| Real local model (qwen3.5:9b) + controls | **0 / 117** | 78% | 0 / 20 |

- The simulated reviewer has configured error rates (8% misses, 5% mislabels, 5% hallucinated
  claims), so the eval measures what the **controls** catch.
- **The real local model** errs toward rejecting: 28 of 73 good ads rejected on its own verdict, 11
  with the controls. Its cost is wrongly rejected advertisers, not unsafe ads.
- **Other results:**
  - taxonomy accuracy 96.8%;
  - 6 hallucinated claims caught;
  - κ = 0.66 between model and labels.
- **The report narrator** (LLM-written report summaries) has its numbers checked against the source
  table. Across 40 reports it was 100% numerically faithful, after 9 regenerations.
- **Post-training data export:** SFT examples and DPO-style preference pairs, with false-approval
  corrections weighted 3× and lineage recorded.

### 6.10 Privacy and security
- **Right to deletion by crypto-shredding.** Each user's pseudonyms come from a per-user key, so
  deleting the key makes data-lake rows unlinkable. Decision records are anonymised, operational
  rows deleted, Redis cleared, and the audit trail records that a deletion ran, never whose.
- **No user identifiers reach advertisers:**
  - click redirects carry only an opaque `click_id`, with `Referrer-Policy: no-referrer`;
  - a postback for another advertiser's click looks exactly like an unknown click;
  - reports hide segments under 50 users or 10 conversions, plus their complements.
- **Signed impression tokens,** so clicks can't be forged.
- **Logs redact** bearer tokens, advertiser keys, hex secrets and API keys. Metrics label HTTP
  routes by template, never by raw path, because paths carry tokens.
- **Rate limiting:** GCRA per principal and scope, shared across replicas through Redis, using
  server time.

### 6.11 Observability
- **Prometheus metrics** cover decisions by policy, block reasons, the served-ad safety gate,
  latency by route, events, review outcomes, the kill switch and dependency health.
- **12 alert rules,** tested with `promtool`: 15 test groups assert that each alert fires or stays
  quiet with the right labels. A unit test also fails if any metric an alert or dashboard uses goes
  missing.
- **A 13-panel Grafana dashboard.**
- **JSON logs** with request IDs, never user IDs.

### 6.12 Fault tolerance

**The failure hierarchy:** each dependency failure has a designed, tested response.

| What fails | What the system does | Why |
|---|---|---|
| Redis (counters, budgets, frequency caps) | NO_AD; readiness 503; a missing counter is never read as 0 | Without frequency or budget state, safety and billing can't be guaranteed |
| Kill-switch state unreadable | NO_AD | Fail closed |
| Postgres (system of record) | NO_AD, retryable 503 with `Retry-After` | The decision can't be logged durably; clients retry |
| RabbitMQ | Serving and event intake continue; outbox rows wait in Postgres; the relay drains them on recovery | The outbox is in the decision's transaction, so events survive |
| A worker crashes | Messages stay queued; retry queues, then dead-letter queues; idempotent consumers make redelivery safe | At-least-once delivery, effectively exactly-once processing |
| ML model fails to load or score | The guarded chain falls back: heuristic → popularity → NO_AD | Degrade, don't break |
| LLM reviewer down, refusing, truncated, invalid | The creative goes to the human queue (never auto-approved) | A false approval is the critical error |
| Rate limiter's Redis | Fails **open**, counted and alerted | Safety already fails closed elsewhere; don't block real users |
| Bad deploy | The platform keeps serving the last healthy revision | Rollouts are safe by default |
| Incident | Kill switch: every process stops serving within 1 s | A human brake that doesn't need a deploy |

**Chaos tests with real containers:**

| Scenario | Expected | Result |
|---|---|---|
| Redis down | No ads (fail closed), readiness 503; resumes on restart | Pass; recovered in 0.1 s |
| RabbitMQ down | Serving unaffected; the outbox buffers, then the relay drains it | Pass; drained in 13 s |
| Postgres down | No ad served (retryable 503) | Pass, **after a bug fix** (story 10) |
| Kill switch | Every process stops serving within its cache window | Pass |

**Delivery guarantees:** the transactional outbox (written in the same transaction as the
decision), publisher confirms, and idempotent consumers deduplicating by event ID give effectively
exactly-once processing on top of at-least-once delivery.

The RabbitMQ chaos run measured how long the outbox took to drain: 12.6 s after the broker came
back. Redis and Postgres recovery took 0.1 s each, and the kill switch stopped and resumed serving
immediately.

### 6.13 Scaling laws (learning curves)

**The question** (from the design doc): Performance = f(data, model size, compute), and the most
important curve is business value = revenue lift − infra cost. In other words, where does more model
scale stop paying for itself?

**Honest framing:** at 100K MAU the data spans only about 1.5 orders of magnitude, so these are
**learning curves**, not scaling laws in the LLM sense. Public datasets would extend the data axis.

**Setup:**
- a 280K-decision simulated log: 121,024 training rows (2,201 clicks) and 36,541 test rows (674
  clicks), with 36 features;
- subsamples are **grouped by user** (no user in both train and test);
- quality is measured as **normalized entropy (NE)**: log loss divided by the entropy of the base
  rate, where below 1.0 beats predicting the average. Also AUC and calibration error (ECE), with
  bootstrap 95% CIs.

**1. Quality vs data** (LightGBM):

| Data share | Rows | Clicks | NE | AUC |
|---:|---:|---:|---:|---:|
| 2% | 2,472 | 45 | 0.9981 | 0.539 |
| 10% | 12,180 | 226 | 0.9980 | 0.563 |
| 20% | 24,301 | 457 | 0.9959 | 0.575 |
| 50% | 60,522 | 1,095 | 0.9960 | 0.578 |
| 100% | 121,024 | 2,201 | 0.9989 | 0.583 |

- AUC rises steadily with data, from 0.539 to 0.583.
- NE plateaus. The fitted power law is **NE(n) = 0.9963 + 0.357·n^(−0.669)**. The irreducible
  floor (0.9963) is most of the remaining error, and closing half the remaining gap needs about
  **341K rows, roughly 3× today's data**.
- Logistic regression on the same subsamples stays near NE 1.0 (AUC up to 0.585).

**2. Quality vs model size:**

| Model | NE [95% CI] | AUC | p50 / p99 scoring (10 candidates) |
|---|---|---|---|
| Category prior (heuristic) | 0.9942 [0.9901, 0.9983] | 0.565 | 0.004 / 0.010 ms |
| Logistic regression | 1.0039 [0.9925, 1.0171] | 0.585 | 0.014 / 0.023 ms |
| LightGBM, 2 to 128 leaves per tree | 0.992–1.002, overlapping CIs | 0.579–0.587 | under 0.1 ms p50 |

Model size **doesn't matter** here: every size lands inside the others' confidence intervals, and
scoring takes under 0.1 ms, so latency isn't the bottleneck either.

**3. Feature groups** (added cumulatively):
- ad features (22 features): NE 0.9936;
- + context (30): 0.9947;
- + exposure (32): **1.0021**, worse, though inside the noise;
- + personalization (36): **0.9934 [0.9886, 0.9991]**, the only set whose whole interval is below
  1.0.

**4. Revenue vs complexity vs cost** (simulator, paired seeds):

| Ranker | Revenue per 1K requests | Lift (each seed) | Serving $ per 1M requests | Net $ per 1M requests |
|---|---:|---|---:|---:|
| Heuristic | $11.10 | 0% | $0 | $0 |
| LightGBM, 4 leaves | $12.84 | **+15.6%** (+15.8%, +15.4%) | $0.0004 | **$1,734** |
| LightGBM, 16 leaves | $12.80 | +15.3% | $0.0005 | $1,697 |
| LightGBM, 64 leaves | $12.67 | +14.1% | $0.0005 | $1,564 |

**Conclusions to say out loud:**
- **The smallest model wins:** 4 leaves did as well as 64. Bigger isn't better at this data size.
- **Serving compute is irrelevant:** $0.0005 per million requests against about $1,700 of gain.
  Latency budget and review effort bind long before compute dollars do.
- **Offline NE was a poor proxy for revenue:** it was nearly flat while ranking quality and revenue
  moved by 15%. That's why online evidence (A/B) decides.
- **More data beats bigger models** here: the data curve still slopes; the size curve is flat.

### 6.14 OPE validated on real logs (Open Bandit Dataset)

**Why:** every OPE number on simulated data could be an artifact of the simulator, so I validated
the estimators on real logs.

**Data:** the Open Bandit Dataset (ZOZO, a large fashion e-commerce company; CC BY 4.0). One week of
real recommendation logs from two policies that ran side by side on the same traffic, both with
logged propensities:
- Uniform Random: 2.7M slots;
- Bernoulli Thompson Sampling: 24.2M slots.

That's 26.9M impressions in total.

**Method:** estimate each policy's click rate from the **other** policy's logs, using the
platform's own estimators (IPS, SNIPS, DR) and reliability checks. Compare against the truth,
measured from that policy's own logs. This covered 3 campaigns and both directions, with several
ways of defining the target policy.

**Results:**
- **27 of 27** conclusive estimates covered the truth (95% intervals, resampling whole hours).
- **4 estimates missed the truth, and all 4 were flagged inconclusive.** The reliability checks
  worked.
- Errors of the conclusive estimates: median **4.4%**, max **11.9%**.
- Logs of 5K–20K rows were never conclusive (median errors 10–54%), as intended. Logs of 100K–400K
  rows were conclusive, covering the truth 73–100% of the time.

**Four things real data taught the estimators** (stories 15–17):
1. **Logged propensities can be wrong.** Floored Monte Carlo propensities carried 12% of the weight
   in one campaign. This led to the mean-importance-weight check.
2. **Rows aren't independent.** Same-hour impressions share effects. This led to cluster bootstrap
   intervals.
3. **The target policy must be defined precisely.** Thompson sampling drifted during the week.
4. **Rewards can depend on the policy.** The ratio was 0.92× in one campaign. This led to the
   `reward_invariance` check.

**Bottom line:** OPE screens policies reliably, but its point estimates carry errors up to about 12%
even on millions of rows. So promotion still requires an online A/B test.

### 6.15 Every evaluation and gate

Nothing is promoted on a single number. Every model, policy, prompt and rule change passes its
gates, and the gates run in CI.

| Layer | What it checks | Thresholds and results |
|---|---|---|
| **Unit and integration tests** | Logic; real Postgres, Redis and RabbitMQ behaviour | 678 tests (653 + 25) |
| **ML model gates** | Log loss no worse than the baseline; ECE ≤ 0.02; no slice regresses by more than 0.02 log loss (paired bootstrap) | CTR passed; CVR **failed** (log loss 0.1791 vs 0.1717, ECE 0.0346 > 0.02), so it wasn't shipped |
| **Export parity** | The exported, non-pickle bundle scores exactly like the trained model | Max difference 0.0; Kendall τ 1.0; identical ECE; p95 scoring 0.15 ms |
| **OPE reliability** | ESS ≥ 100, reward events, mean importance weight ≈ 1, cluster-bootstrap CIs | ML ranker: ESS 97 < 100, so **inconclusive**; it didn't serve on OPE alone |
| **A/B and experiment checks** | Sample-ratio mismatch, pre-registered MDE, CUPED, guardrails, sequential monitoring | +3.3% not significant, MDE 36.9%, so not promoted; SRM passed |
| **Simulator safety evals** | Safety violations must be 0. A simulated *vulnerable* cohort, flagged in the simulator but invisible to the policy, must not get a higher ad load, a higher share of impulse-category ads, or a lower NO_AD rate than other users; failing any of these rejects a policy regardless of revenue | Baseline 0 violations; FQI and PPO passed |
| **Drift checks** | PSI and KS on features, calibration per category, concept drift, business signals | 0 false alarms; injected drift caught at day 42 |
| **Creative-review eval and gate** | 190 labelled creatives in 11 slices: false approvals, adversarial and safety approvals, taxonomy accuracy, abstention | Gate passed: 0 false, adversarial or safety approvals; taxonomy 97.4% / 96.8%; abstention about 20% |
| **Narrator faithfulness** | Every number in an LLM-written summary must match the source table | 40 of 40 faithful (9 regenerated after the check caught errors) |
| **Alert tests** | `promtool` feeds synthetic series and asserts each alert fires or stays quiet | 15 test groups; a deliberately broken expectation fails |
| **Metric contract test** | Every metric used by an alert or dashboard must exist in `/metrics` | A renamed metric can't silently disable an alert |
| **Load test** | Throughput and latency against the 100 ms p99 SLO | 90 req/s at p99 52 ms (real backend) |
| **Chaos tests** | Redis, RabbitMQ and Postgres stopped and restarted; the kill switch | All pass (section 6.12) |
| **Security scans** | bandit (static analysis), pip-audit (dependency vulnerabilities) | bandit clean; both run in CI |

---

## 7. Results summary (for quick recall)

| Area | Number |
|---|---|
| Tests | 678 (653 unit, 25 integration) |
| Code | about 18K lines of source, 5.6K lines of tests |
| Milestones | 10, from the design doc's roadmap |
| Simulator baseline | 3,000 users, 14 days: 36,723 requests, 24,211 ads, $406.86 revenue, **0 safety violations** |
| CTR model | log loss 0.0804 vs 0.0863; ECE 0.0047 vs 0.0205 |
| A/B, ML vs heuristic (simulated) | +3.3% revenue per user, not significant; MDE 36.9%; not promoted |
| Bandits (simulated) | about +10% revenue over the heuristic |
| RL, FQI (simulated) | +6.4% value; OPE inconclusive (ESS 94), so it doesn't serve |
| OPE on real logs (Open Bandit Dataset) | 27 of 27 conclusive estimates covered the truth; median error 4.4%, max 11.9% |
| Scaling (simulated, 280K decisions) | AUC 0.539 → 0.583 with more data; NE plateaus at 0.996 (power-law fit); 2 to 128 leaves make no difference |
| Creative review | 0 / 117 false approvals with the controls; 79% automation |
| Load (laptop) | 90 req/s at p99 52 ms on the real backend; 356 req/s at p99 34 ms in memory |
| Cloud demo run | 889 requests, 63% fill, 70% of served ads viewable, 29 clicks, 13 conversions, 25 rate-limited, 2 crisis suppressions, gambling advertiser billed $0 |
| Cost | about $220 a month on Azure, about 0.3–1.2% of estimated revenue |

---

## 8. Problems solved: stories

Each story covers what happened, how it was found, the fix, and the lesson. Pick 4–6 to rehearse.
The best for system-design interviews are 1, 2, 4, 9, 10, 11 and 14; for ML interviews, 15, 16, 17
and 19.

### Cloud deployment

**1. The web app broke behind a path-rewriting proxy.**
- *What happened:* after deploying, the console stuck on "Loading…". The ingress in front of the
  service rewrote absolute links in HTML (`/ui/js/main.js` became a long proxy path that the app
  doesn't serve), so no JavaScript loaded.
- *How it was found:* I compared the HTML fetched through the ingress with the HTML fetched straight
  from the container. Only the copy that went through the ingress had rewritten links.
- *Fix:* relative asset links (`js/main.js`), plus a regression test asserting that the page has
  no absolute `href` or `src`, and that `/ui` redirects to `/ui/` so relative links resolve.
- *Lesson:* never assume you're served at the root path. Path-prefix routing is common: Application
  Gateway path rules, Kubernetes proxies, nginx sub-paths.

**2. A proxy consumed the `Authorization` header.**
- *What happened:* every admin and user call returned 401, but advertiser calls worked.
- *How it was found:* I sent the same token through the ingress (401) and straight to the
  container (200), then tested which headers survived. A custom `X-Advertiser-Key` header passed
  through; `Authorization` didn't. The proxy authenticated itself with that header and then deleted
  it, which Kubernetes API-server proxies do.
- *Fix:* an **opt-in** fallback. With `THERAFLO_ADS_AUTH_FALLBACK_HEADER=on`, the API also reads the
  same `Bearer` token from `X-Theraflo-Authorization`, with the same checks:
  - off by default, and the app refuses to start with it in production;
  - `Authorization` wins whenever it's present, and the `Bearer` scheme is required;
  - three new tests cover it.

  Google's API gateways use the same pattern (`X-Forwarded-Authorization`).
- *Lesson:* a security change needs its blast radius bounded: opt-in, refused in production,
  logged with a warning, and tested for precedence and for rejecting bad tokens.

**3. Over-reserved resources blocked a rollout and inflated the bill.**
- *What happened:* a new revision sat `Pending` with "Insufficient memory", while the old revision
  kept serving the old code.
- *Root cause:* each app reserved 2 GiB but used about 60 MiB. Reservations (not usage) decide
  scheduling, and a rolling update needs room for the new replica **beside** the old one.
- *Fix:* measured actual use (API 60 MiB, console 64 MiB, RabbitMQ 118 MiB, Redis 3 MiB) and
  right-sized the API and console to 0.5 vCPU / 1 GiB. Reserved memory fell from 6.1 GiB to 4.1 GiB
  before adding the workers, and compute cost fell by about 17%.
- *Lesson:* Container Apps bills per-second **reservations**. Measure, then size, and keep surge
  headroom for rolling updates.

**4. Healthy, but silently running the wrong backend.**
- *What happened:* everything showed "Running", but the API was serving from its **in-memory**
  backend, not Postgres, and the migrate job failed.
- *Root cause:* a deploy script passed a PowerShell list through a wrapper script, which joined it
  into **one** argument. So the app got a single setting, `THERAFLO_ADS_BACKEND`, whose value was
  `infra` followed by every other setting. An unrecognised backend value fell through to the dev
  backend.
- *How it was found:* the console's backend badge, and counting the settings the app actually
  received (1 instead of 10).
- *Fix:* pass list items separately (PowerShell splatting, `@APP_ENV`), and check the setting count
  and secret lengths before deploying.
- *Lesson:* "Running" isn't "correct". Treat configuration as untrusted input: validate it at
  startup and fail fast. (Follow-up: unknown backend values should crash, not fall back.)

**5. Empty secrets: the case where fail-fast worked.**
- *What happened:* two of three key variables were empty when the secrets were created, so the new
  revision crash-looped with `THERAFLO_ADS_USER_HASH_KEY must be set (hex)`.
- *Why it didn't hurt:* the platform kept the last healthy revision serving, and the error named
  the exact problem.
- *Fix:* generate all three keys, check that each is 64 hex characters, set the secrets again, and
  deploy a new revision.
- *Lesson:* compare with story 4. Missing secrets failed loudly, while a malformed setting failed
  silently.

**6. Managed database address formats.**
- *What happened:* the migrate job failed with
  `ValueError: invalid literal for int() with base 10: '4512:5432'`.
- *Root cause:* the server's address from the Azure API already included a port, and the
  connection string added `:5432`.
- *Fix:* build the URL as `@${PG_HOST}/theraflo_ads` and let the driver default to 5432 when no
  port is given.
- *Lesson:* take endpoint strings from the API as they are; don't hardcode default ports.

**7. A misconfigured Redis address slowed every request.**
- *What happened:* readiness reported `frequency_store: false` and `budget_store: false`, each check
  took about 4 s, and pages felt slow. The cluster's DNS server even restarted under the flood of
  failed lookups.
- *Root cause:* the Redis and RabbitMQ hostnames the apps were given didn't resolve inside the
  environment.
- *How it was found:* the readiness endpoint showed which checks failed. Then I tested DNS and TCP
  from inside a running container and read the DNS server's logs.
- *Fix:* point the apps at addresses that resolve inside the environment. After that, every
  readiness check passed, the slowness disappeared, and the DNS server stopped restarting.
- *Lessons:*
  - design readiness checks per dependency, so the failing one is named;
  - a broken dependency can fail through timeouts (slowness) rather than errors;
  - failed DNS lookups multiply load on the DNS server.

**8. Migrations as one-off jobs, and safe rollouts.**
- Migrations run as a **Container Apps job** (manual trigger, 900 s timeout, 1 retry), not inside
  API startup, where several replicas would race. Seed runs after migrate.
- A failed revision never took traffic. Single-revision mode keeps serving the last *ready*
  revision, so each bad deploy above (stories 3, 5) caused no outage.
- **Image size is cold-start time:** 1.27 GB unpacked, 363 MB compressed, about 80 s to pull on a
  fresh node. Next step: split the ML dependencies out of the serving image.

### Systems and infrastructure

**9. Budget overdelivery.**
- *Found by:* the simulator.
- *What happened:* reserving only a click's *expected* cost let a single click overshoot a nearly
  spent budget.
- *Fix:* a campaign needs headroom for **one worst-case event** before it's eligible, and the check
  and reservation are atomic (a Redis Lua script).
- *Lesson:* reserve for the worst case, not the average, and never bill an advertiser above budget.
  Theraflo absorbs any overdelivery, and it's logged.

**10. A Postgres outage returned 500 instead of 503.**
- *Found by:* the chaos test.
- *What happened:* connection errors from the profile read escaped untranslated, so clients saw a
  non-retryable 500.
- *Fix:* map connection-level database errors to a **retryable 503** (with `Retry-After`), plus a
  regression test.
- *Lesson:* the status code is part of the contract. 503 tells clients and SDKs to retry, and lets
  load balancers and alerts tell an outage from a bug.

**11. The first safety violation would never have paged.**
- *Found by:* running demo traffic against the full stack.
- *What happened:* labelled counters only appeared on their first increment, so Prometheus never
  saw a step from 0 and `increase()` read 0. In the same run, 82 rate-limited requests showed up as
  a rate of 0.
- *Fix:* every alerted counter is created at 0 for each known label at startup. A unit test
  requires every metric that alerts and dashboards use, and `promtool` tests check the alerts fire.
- *Lesson:* test your alerts like code. An alert that can't fire is worse than none, because you
  trust it.

**12. Metrics split across workers.**
- *What happened:* with two server worker processes, a scrape saw only 17 of 20 requests, because
  each process keeps its own counters.
- *Fix:* one worker per container, scaled with replicas, plus a warning if someone starts several.
- *Lesson:* on a container platform, scale with replicas rather than processes inside a container.

**13. More deployment-shaped bugs**, quick to tell:
- **New campaigns were invisible until restart.** The API loaded its catalog before seeding ran and
  only refreshed campaigns it already knew. It now refreshes campaigns, ads and priors every few
  seconds. (A deploy-ordering bug.)
- **The analytics volume wasn't writable.** It was root-owned, the container runs as a non-root
  user, and every message was dead-lettered. The image now creates the directory with the right
  owner. (Non-root containers versus mounted volumes.)
- **Review metrics never reached Prometheus.** Reviews run in a worker that served no `/metrics`;
  it now serves them on :9100. (Every process type needs its own scrape target.)
- **Model registry paths broke on Linux.** Versions registered on Windows stored backslash paths.
  Paths are now written with forward slashes, and old indexes are read either way. (Cross-platform
  artifacts.)
- **JavaScript was served as `text/plain`.** Browsers refuse module scripts with that type, and the
  host's MIME table supplied it. The app now sets the content types itself. (Never trust the host's
  MIME table.)
- **Deep vs shallow copy.** Demo profiles were shallow-copied, so one test's state leaked into the
  next. They're now deep-copied.

**14. A diagnostic call changed live behaviour.**
- *What happened:* while debugging, a write endpoint was used to test something and switched the
  live serving policy.
- *How it was recovered:* the audit log recorded the change and its previous value, so it was
  restored exactly.
- *Lesson:* diagnose with read-only calls, make every change auditable, and treat control-plane
  writes as changes that need review.

### ML and evaluation

**15. Wrong propensities passed the reliability checks.**
- *Found by:* real logs from the Open Bandit Dataset.
- *What happened:* 4,501 Thompson sampling rows had Monte Carlo propensities floored near 10⁻⁶. In
  one campaign, 159 of those rows carried 12% of the importance weight and moved the SNIPS estimate
  by 7%, yet they passed the effective-sample-size and reward-event checks.
- *Fix:* a new check in the shared estimator core: importance weights must average 1, within 5% and
  not significantly different.
- *Lesson:* logged propensities are data too, and they can be wrong.

**16. The confidence intervals were too narrow.**
- *What happened:* bootstrap intervals that resampled single rows were about 1.5× too narrow,
  because impressions in the same hour share effects. With them, 8 "conclusive" estimates missed the
  truth.
- *Fix:* cluster resampling, by hour for the dataset and by user for the platform's own OPE. After
  that, all 27 conclusive estimates covered the truth.
- *Lesson:* check the independence assumption before trusting an interval.

**17. The target policy was ambiguous.**
- *What happened:* Thompson sampling's behaviour drifted during the week (total-variation distance
  0.23–0.31 between day 1 and day 7; one item's daily probability swung 27×). Hourly and week-average
  definitions of the same policy differed by up to 8% in their estimates.
- *Lesson:* weight diagnostics can't detect a wrong definition of the target policy. Also, rewards
  may depend on the policy that showed the ad: 0.92× [0.82, 1.04] in one campaign. That led to a new
  `reward_invariance` check. Conclusion: OPE **screens**, and an A/B test **promotes**.

**18. Feature schema mislabelled in the decision logs.**
- *What happened:* records said `features_v1`, while the features actually came from the
  `features_v2` builder.
- *Fix:* the logged version now comes from the feature provider itself.
- *Lesson:* lineage must come from the component that produced the data, not from configuration.

**19. A local LLM never saw the answer schema.**
- *What happened:* Ollama constrains the output to the JSON schema but doesn't show the schema to
  the model. So the model guessed category labels (46% accuracy), and it listed plain product
  descriptions as "claims", which sent clean ads to humans (64% automation).
- *Fix:* add the field list, the allowed labels, and a definition of a claim (with examples outside
  the eval set) to the local prompt. Category accuracy rose to 77% and automation to 78%, with false
  approvals still 0.
- *Lesson:* constrained decoding guarantees the format, not understanding. Evaluate with an
  ablation.

**20. More bugs the tests and tools caught:**
- **Ratings skipped claim grounding.** "Rated 4.9 stars by 80,000 readers" wasn't extracted as a
  claim. Ratings and audience statistics now count as claims.
- **Export lineage was wrong.** It named Claude as the base model when the judgements came from the
  simulated reviewer. It now records the model that produced each judgement.
- **`assert` on hot paths.** Python drops asserts under `-O`, which would have turned an invariant
  into a `TypeError`. They're now explicit checks.
- **A flaky integration test.** A random hash key put the test user in a holdout 4% of the time.
  Keys are now chosen outside the holdouts.
- **The CVR model failed its gate.** It wasn't shipped; serving keeps category priors for predicted
  CVR. (A good "I didn't ship it" story.)
- **Offline normalized entropy was a poor proxy for revenue.** It stayed flat while ranking quality
  and revenue moved, so the more conservative evidence (the budget-split A/B) wins.

---

## 9. Design decisions and trade-offs (why, not just what)

| Decision | Why | Trade-off |
|---|---|---|
| Hard constraints instead of reward penalties | A penalty can be outweighed by revenue; a constraint can't | Leaves some revenue on the table; rules need governance |
| Fail closed (NO_AD) on unreadable safety state | In a mental-health app, a wrong ad costs more than a missed one | Lost revenue during outages, measured by fail-closed metrics |
| The rate limiter fails **open** | Safety state already fails closed; blocking real users during a Redis hiccup adds risk without safety | Possible over-limit traffic, counted and alerted |
| Transactional outbox, not a direct publish | The decision and its event commit together, so nothing is lost or duplicated if the broker is down | Relay latency (seconds); one more process |
| Postgres as system of record; Redis derived only | Redis state can be rebuilt; billing reconciles from the durable log | Two stores to operate |
| Log exact propensities, with holdouts | Makes OPE and A/B possible | 2% exploration and 2% no-ads cost a little revenue |
| OPE screens, A/B promotes | Real data showed OPE point errors up to 12% | Slow promotions at this traffic (weeks to months) |
| Recalibrate before retraining | Cheaper and safer; a full retrain needs evidence | Slower to react to real concept drift |
| No LLM on the serving path | Latency, cost, and a failure mode the serving path can't absorb | Review is asynchronous; creatives wait as pending |
| Container Apps rather than AKS | One API, one relay and a few workers don't need a Kubernetes control plane to operate; revisions, jobs and secrets come built in | Less control (no DaemonSets or custom networking) |
| RabbitMQ in a container rather than Service Bus | No rewrite of the queue code (retry and DLQ topology) | You run it yourself; it needs persistent storage in production |
| Model files layered on the image | The model version is pinned to the image tag, and rollback is deploying an old tag | Rebuild to ship a model (fine at weekly retraining) |

---

## 10. Why each tool, and the alternatives

For "why did you use X?" questions: give the reason, name the alternative, and say when you'd
switch.

| Tool | Why this one | Alternative considered | When I'd switch |
|---|---|---|---|
| **Python 3.12** | One language for the API, ML, OPE and the simulator, so the online and offline feature code is the same code | Go or Java for the API | If the API needed 10× the throughput per core |
| **FastAPI** | Typed request and response models (pydantic) validate input at the edge; OpenAPI docs for free; dependency injection for auth roles | Flask, Django | Rarely: it fits a JSON API |
| **PostgreSQL** | System of record. Transactions let the decision, impression, label and outbox row commit together, which is what makes the outbox pattern work | MySQL; a NoSQL store (no multi-row transactions) | A read replica or partitioning at about 10× writes |
| **Alembic** | Versioned, reviewable schema migrations, run as a one-off job | Hand-written SQL | No |
| **pg8000** | A pure-Python Postgres driver; the native `psycopg` driver's DLL was blocked by Windows Application Control on the dev machine | psycopg | On Linux hosts, psycopg for speed |
| **Redis** | Sub-millisecond counters, budget reservations, rate limits and features. **Lua scripts** make check-and-reserve atomic across replicas. Derived state only, rebuildable from Postgres | Doing it in Postgres (too slow on the hot path); in-process memory (not shared across replicas) | Azure Cache for Redis in production |
| **RabbitMQ** | Work queues with per-consumer queues, retry and dead-letter topology, publisher confirms; the team already knows it; right-sized for 112K–500K events a day | **Kafka** | When replay, long retention for several independent consumers, or high-volume streaming matter (the design doc's criteria) |
| **Transactional outbox** | A decision and its event commit atomically, so there's no lost or phantom event when the broker is down | Publishing straight after the commit (loses events in a crash); buffering in memory or on local disk (lost on a container restart) | No; this is the standard pattern |
| **LightGBM** | Strong on tabular features, fast to train (0.5 s), very fast scoring (under 0.1 ms), exports to a non-pickle text format | XGBoost (similar); logistic regression (the baseline, NE about 1.0); neural nets (no gain at this data size) | Sequence models only with far more data |
| **Thompson sampling (Beta, linear)** | Natural exploration with sparse conversions; the propensity can be estimated by Monte Carlo, so it stays OPE-evaluable | ε-greedy (cruder); UCB (deterministic, so no propensity for OPE) | More context: neural bandits |
| **IPS / SNIPS / DR** | Standard OPE estimators: unbiased (IPS), lower variance (SNIPS), or using a reward model (DR); comparing them is itself a diagnostic | The direct method alone (biased by the reward model) | No |
| **FQI (offline RL)** | Learns from logged data without exploring on real users; a support constraint keeps it near logged actions | Online RL (unsafe on real users); PPO, kept simulator-only | Never online without OPE and safety gates |
| **DVC** | Versions datasets and the pipeline (`dvc repro`); git-like, so every model traces back to its data | Delta Lake or lakeFS (heavier) | At data-lake scale |
| **MLflow** | Tracks runs, parameters and metrics. DVC versions *data*; MLflow tracks *runs*, and they complement each other | Weights & Biases (hosted, paid) | Team preference |
| **Prometheus + Grafana** | Pull-based metrics, PromQL alerting with unit-testable rules (`promtool`), standard dashboards | Datadog (paid); Azure Monitor (production option) | Azure Monitor managed Prometheus in production |
| **Claude for creative review** | Strong instruction following and structured outputs (schema-constrained JSON) with refusal fallbacks; it runs asynchronously, never on the serving path | Fine-tuned classifiers (need labelled data first); rules alone (48% automation) | A cheaper or local model once evals show parity |
| **Ollama + qwen3.5:9b** | A free, private, local reviewer for development and evals; fits an 8 GB GPU (about 2 s per call); same contract as the Claude adapter | Other local models (gemma, llama) | When a model shows better false-rejection rates at 0 false approvals |
| **Docker (multi-stage build)** | The same image runs on a laptop and in the cloud; non-root user; no build toolchain in the runtime image; no secrets baked in | Buildpacks | No |
| **Azure Container Apps** | Managed containers with revisions (safe rollouts and rollback), jobs (migrations), secrets, scale rules and internal TCP ingress, without running a Kubernetes control plane. Small fleet: 1 API, 1 relay, 4 workers | **AKS** (full Kubernetes: more control, more ops); **App Service** (web apps, weak for workers and TCP); **Container Instances** (no revisions or ingress management) | AKS if we needed DaemonSets, a service mesh, or custom networking |
| **Azure Database for PostgreSQL Flexible Server** | Managed backups, patching and high availability; the Burstable B1ms tier ($0.017/h) fits the load | Self-hosted Postgres in a container (no backups or HA) | General Purpose tier plus a read replica at about 10× |
| **Azure Container Registry** | Private images close to the runtime; tags are versions, so rollback means deploying the old tag | Docker Hub (public, rate-limited) | No |
| **Container Apps secrets → Key Vault** | Keys are injected at runtime, never in the image; `secretref:` keeps them out of plain settings | Environment variables in code (insecure) | Key Vault references in production (rotation in one place) |
| **GCRA rate limiting** | Smooth limits with no bursty fixed windows; one Redis key per principal; works across replicas | Fixed or sliding windows | No |
| **Crypto-shredding** | Deletion even from immutable snapshots: destroy a user's key, and their pseudonymised rows become unlinkable | Rewriting every snapshot (impossible with versioned data) | No |

---

## 11. Likely interview questions, with short answers

1. **How do you stop the model from exploiting vulnerable users?** Hard eligibility constraints
   before the policy samples, contextual-only ads for minors and impulse categories, crisis
   suppression, and a served-ad safety gate after the policy. Revenue can't buy its way past a
   constraint.
2. **What happens when Redis goes down?** Ads stop (NO_AD), and readiness returns 503. Serving
   resumes within about 0.1 s of Redis coming back (chaos-tested). The rate limiter fails open
   (counted), and counters rebuild from Postgres.
3. **How do you guarantee an event isn't lost or double-billed?** The outbox row is written in the
   decision's transaction; the relay publishes with confirms; consumers deduplicate by event ID;
   there are retry and dead-letter queues; and billing reconciles from the durable log, never from
   Redis.
4. **How would you evaluate a new ranking policy before showing it to users?** OPE (IPS, SNIPS, DR)
   on logged propensities, with reliability checks and cluster-bootstrap intervals, then an A/B
   test with a pre-registered MDE, CUPED and guardrails. OPE alone isn't enough: validated on real
   logs, its point errors reached 12%.
5. **Your A/B showed +3.3%. Did you ship?** No. The confidence interval spanned zero, and the
   pre-registered MDE was 36.9%. At about 112K–500K decisions a day, a 10% lift needs 21–93 days.
6. **Why bandits?** To balance exploration and exploitation with sparse conversions. Beta Thompson
   sampling earned about 10% more in simulation, with Monte Carlo propensities logged, so it remains
   evaluable.
7. **Why didn't RL ship?** FQI looked good in the simulator (+6.4%), but OPE was inconclusive (ESS
   94). Nothing serves without passing OPE and the safety evals.
8. **How do you defend against prompt injection in ad review?** Normalised injection scanning,
   escaped and delimited advertiser text, structured outputs with no tools, deterministic claim
   grounding, a dual check, and abstention. An injection-obeying reviewer still produced 0 false
   approvals with the controls.
9. **How many users can this handle, and what does it cost?** About 90 req/s per instance within a
   100 ms p99, which is about 500K–2.3M MAU with 3× peaks. About $220 a month on Azure, roughly
   $0.002 per MAU, or 0.3–1.2% of estimated revenue.
10. **How would you scale 10×?** Add API replicas (it's stateless); cache profiles; batch the
    decision transaction; move Postgres to General Purpose with a read replica; Azure Cache for
    Redis; scale workers by queue depth; then add a retrieval stage once inventory grows past a few
    hundred campaigns.
11. **How do you handle deletion requests?** The profile is forgotten immediately and a request is
    queued. An hourly job shreds the user's key, which makes data-lake rows unlinkable, anonymises
    decisions, deletes operational rows, and clears Redis. The audit log records only that a
    deletion ran.
12. **Something deployed and looked healthy but was wrong. How did you catch it?** Story 4: the
    backend badge, and counting the settings the app actually received. Fix: validate configuration
    at startup and fail fast.
13. **Tell me about a production-style incident.** Story 10 (500 vs 503 under a Postgres outage),
    story 11 (an alert that could never fire), or story 3 (a rollout blocked by over-reservation).

14. **What did your scaling experiments show?** Learning curves, not LLM-style scaling laws: the
    data spans only about 1.5 orders of magnitude. AUC improved with data (0.539 → 0.583), and NE
    followed a power law with a floor at 0.9963, so about 3× more data closes half the remaining
    gap. Model size (2 to 128 leaves) made no difference, and a 4-leaf model earned the same +15.6%
    revenue as 64 leaves. More data beats bigger models here, and compute cost is negligible.
15. **Why RabbitMQ and not Kafka?** Work-queue semantics with retry and dead-letter queues, team
    familiarity, and the volume (hundreds of thousands of events a day) is small. Kafka earns its
    cost when you need replay, long retention for several independent consumers, or high-volume
    streaming. The outbox keeps the switch cheap later.
16. **Why Container Apps and not Kubernetes (AKS)?** The fleet is 1 API, 1 relay and 4 workers.
    Container Apps gives revisions, jobs, secrets and internal ingress without operating a control
    plane. I'd move to AKS for DaemonSets, a service mesh, or custom networking.
17. **How do you know your evaluation is trustworthy?** I validated OPE on 26.9M real impressions,
    found three estimator bugs, and fixed them. Every model passes gates in CI (log loss vs
    baseline, ECE ≤ 0.02, slice regressions, export parity). Alerts are unit-tested. Chaos tests
    prove the failure modes.

---

## 12. Limits and what's next

- **All data is simulated,** except the Open Bandit Dataset validation. Next: shadow traffic, and a
  real holdout on production logs.
- **The live Claude review eval** (billed) hasn't been run; the reviewer is unit-tested with a fake
  client. A real local model (qwen3.5:9b) was evaluated instead.
- **Production hardening:**
  - Key Vault for secrets and keys;
  - TLS to Postgres (required on Azure);
  - a non-admin database user;
  - Alertmanager routing;
  - Azure Cache for Redis;
  - persistent storage for RabbitMQ;
  - the dev console never deployed publicly.
- **Fail fast on unknown configuration values** (story 4's follow-up).
- **A slimmer serving image** (ML training dependencies split out) for faster cold starts.
