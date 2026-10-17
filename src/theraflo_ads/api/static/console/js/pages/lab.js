// Ranker lab: one request through every stage (hard constraints → ranker scores → each
// policy's distribution), with every candidate policy side by side. Nothing is served,
// spent, or counted. Switch the serving policy, inspect the bandits, promote a model.

import { allUsers, as, ids } from "../api.js";
import {
  badge, bars, busy, callout, card, confirmDialog, disclosure, empty, errorToast, field, fmt, h, kv, meter, mount, select, table, toast,
} from "../ui.js";
import { BLOCK_REASONS, PLACEMENT_NAMES, REJECTIONS, servedBy } from "../shared.js";

const POLICY_NAMES = {
  heuristic: "Heuristic (ε-greedy)",
  ranker: "ML ranker (ε-greedy)",
  beta_ts: "Beta Thompson sampling",
  lin_ts: "Linear Thompson sampling",
  popularity: "Popularity (fallback)",
  uniform: "Uniform (holdout)",
};

export default {
  id: "lab",
  title: "Ranker lab",
  icon: "⚗",
  async render(root, { scope }) {
    const users = allUsers();
    const placements = [...(ids.boot?.placements?.ads || ["home_feed", "content_library"]), ...(ids.boot?.placements?.ad_free || [])];
    const form = {
      user: h("select", {}, h("option", { value: "" }, "synthetic new user (no history)"), users.map((u) => h("option", { value: u.user_id }, u.user_id))),
      placement: select(placements.map((p) => [p, `${PLACEMENT_NAMES[p] || p}${ids.boot?.placements?.ad_free?.includes(p) ? " (ad-free)" : ""}`]), "home_feed"),
      consent: select([["", "(profile)"], ["personalized", "personalized"], ["contextual_only", "contextual only"]], ""),
      age: select([["", "(profile)"], ["adult", "adult"], ["teen", "teen"], ["under_13", "under 13"], ["unknown", "unknown"]], ""),
      device: select([["", "(none)"], ["mobile", "mobile"], ["web", "web"]], "mobile"),
      hour: select([["", "now"], ...Array.from({ length: 24 }, (_, i) => [String(i), `${String(i).padStart(2, "0")}:00 UTC`])], ""),
      suppressed: select([["", "(profile)"], ["true", "yes"], ["false", "no"]], ""),
      hidden: h("div", { class: "chips" }),
      auto: h("input", { type: "checkbox", checked: true }),
    };
    const hidden = new Set();
    const resultEl = h("div", { class: "stack" });
    const servingEl = h("div");
    const banditEl = h("div");
    const modelEl = h("div");
    const runBtn = h("button", { class: "btn", type: "button" }, "Run preview");

    const renderHiddenChips = async () => {
      try {
        const { data } = await as.admin("GET", "/v1/dev/catalog", { quiet: true });
        const ads = data.campaigns.flatMap((c) => c.ads);
        mount(form.hidden, ads.map((a) => {
          const box = h("input", { type: "checkbox", checked: hidden.has(a.ad_id) });
          box.addEventListener("change", () => {
            if (box.checked) hidden.add(a.ad_id);
            else hidden.delete(a.ad_id);
            if (form.auto.checked) run();
          });
          return h("label", { class: "check small" }, box, a.ad_id);
        }));
      } catch {
        /* optional */
      }
    };

    mount(
      root,
      h("div", { class: "page-head" }, h("div", { class: "grow" },
        h("h1", {}, "Ranker lab"),
        h("p", { class: "lede" },
          "One ad request through every stage of the decision, scored by every candidate policy at once. A preview reads counters but never serves, spends budget, or counts an impression. Change the request, compare the rankers and rerankers, and switch what serves."))),
      h("div", { class: "grid cols-2" },
        card("What-if request", h("div", { class: "stack" },
          h("div", { class: "form-grid" },
            field("user", form.user, "a real profile, with its exposures"),
            field("screen", form.placement),
            field("consent", form.consent),
            field("age band", form.age),
            field("device", form.device),
            field("time of day", form.hour, "daypart is a feature"),
            field("suppressed", form.suppressed)),
          field("hidden by the user", form.hidden),
          h("div", { class: "row" }, runBtn, h("label", { class: "check small" }, form.auto, "re-run on every change"))),
        ),
        servingEl,
      ),
      h("div", { class: "section" }, resultEl),
      h("div", { class: "grid cols-2 section" }, banditEl, modelEl),
    );

    const body = () => {
      const b = { placement: form.placement.value, policies: Object.keys(POLICY_NAMES) };
      if (form.user.value) b.user_id = form.user.value;
      if (form.consent.value) b.consent = form.consent.value;
      if (form.age.value) b.age_band = form.age.value;
      if (form.device.value) b.device = form.device.value;
      if (form.hour.value !== "") b.hour = Number(form.hour.value);
      if (form.suppressed.value) b.suppressed = form.suppressed.value === "true";
      if (hidden.size) b.hidden_ads = [...hidden];
      return b;
    };

    const run = async () => {
      try {
        const { data } = await as.admin("POST", "/v1/dev/rank-preview", { body: body() });
        renderResult(data);
        renderServing(data.serving);
      } catch (err) {
        mount(resultEl, callout(`Preview failed: ${err.message}`, "danger"));
      }
    };
    runBtn.addEventListener("click", () => busy(runBtn, run()));
    for (const el of [form.user, form.placement, form.consent, form.age, form.device, form.hour, form.suppressed]) {
      el.addEventListener("change", () => form.auto.checked && run());
    }

    // ----- serving -----

    const serve = async (policy) => {
      try {
        await as.admin("PUT", "/v1/dev/serving-policy", { body: { policy } });
        toast(`Now serving with ${POLICY_NAMES[policy]} (then the heuristic and popularity as fallbacks).`);
        await run();
      } catch (err) {
        errorToast(err);
      }
    };

    const renderServing = (serving) => {
      mount(
        servingEl,
        card("What serves live traffic", h("div", { class: "stack" },
          h("div", { class: "row" }, badge(POLICY_NAMES[serving.policy] || serving.policy, "solid")),
          h("div", { class: "small muted" }, "The fallback chain: each level serves only if every level above it fails (an error, an invalid distribution, an open circuit breaker, or a blown latency budget)."),
          h("ol", { class: "small", style: { margin: 0, paddingLeft: "18px" } },
            serving.chain.map((level, i) => h("li", {}, servedBy(level.served_by), " ", h("span", { class: "mono tiny" }, level.version), i === 0 ? h("span", { class: "tiny muted" }, "  ← first choice") : null)),
            h("li", { class: "muted" }, "NO_AD if every level fails")),
          h("div", { class: "row tight" },
            serving.servable.map((p) => h("button", {
              class: `btn small ${p === serving.policy ? "" : "secondary"}`,
              type: "button",
              disabled: p === serving.policy,
              on: { click: (e) => busy(e.currentTarget, serve(p)) },
            }, POLICY_NAMES[p] || p))),
          h("p", { class: "tiny faint" }, "The uniform policy only serves the 2% exploration holdout. RL rerankers (FQI, PPO) are offline by design: train and evaluate them in Jobs; they never serve without passing OPE and safety gates."))),
      );
    };

    // ----- results -----

    const renderResult = (r) => {
      const parts = [];
      const u = r.user;
      parts.push(h("div", { class: "row" },
        badge(PLACEMENT_NAMES[r.placement] || r.placement, "accent"),
        badge(u.synthetic ? "synthetic user" : u.user_id),
        badge(`consent used: ${fmt.label(r.effective_consent)}`, r.effective_consent === "personalized" ? "accent" : ""),
        badge(`age: ${fmt.label(u.age_band)}`),
        badge(`daypart: ${r.daypart}`),
        u.suppressed ? badge("suppressed", "danger") : null,
        !u.synthetic && u.holdout !== "none" ? badge(`live traffic puts this user in the ${fmt.label(u.holdout)} holdout`, "warn") : null,
      ));
      if (r.blocked) {
        parts.push(callout(h("div", {}, h("strong", {}, `Blocked before any policy runs: ${fmt.label(r.blocked)}. `), BLOCK_REASONS[r.blocked] || ""), "danger"));
      }
      parts.push(stageEligibility(r));
      if (!r.blocked) {
        parts.push(stageScores(r));
        parts.push(stagePolicies(r));
        parts.push(h("div", { class: "section" }, disclosure(`Feature vectors (the ads-safe inputs; non-zero values, ${Object.keys(r.features).length} ads)`, featureTable(r.features))));
      }
      mount(resultEl, parts);
    };

    const stageHead = (n, title, sub) => h("h2", {}, h("span", { class: "step" }, String(n)), title, sub ? h("span", { class: "muted" }, sub) : null);

    const stageEligibility = (r) => {
      const eligible = r.candidates.filter((c) => c.eligible).length;
      return h("div", { class: "section" },
        stageHead(1, "Eligibility", `hard constraints, before any policy: ${eligible} of ${r.candidates.length} ads eligible`),
        card(null, table(
          [
            { label: "ad", render: (c) => h("span", { class: "mono" }, c.ad_id) },
            { label: "advertiser", key: "advertiser_id" },
            { label: "category", render: (c) => fmt.label(c.category) },
            { label: "pricing", render: (c) => `${c.pricing_model.toUpperCase()} ${fmt.money(c.price)}` },
            { label: "budget left today", render: (c) => (c.budget_remaining === null ? "–" : h("div", { class: "row tight nowrap" }, meter(c.budget_remaining, c.daily_budget, c.budget_remaining / c.daily_budget < 0.1 ? "danger" : ""), fmt.money(c.budget_remaining))) },
            { label: "seen 24h", num: true, render: (c) => c.exposures_24h ?? "–" },
            { label: "review", render: (c) => badge(c.review_status, c.review_status === "approved" ? "ok" : c.review_status === "rejected" ? "danger" : "warn") },
            {
              label: "verdict",
              render: (c) => (c.eligible
                ? badge("eligible", "ok")
                : c.rejection
                  ? badge(REJECTIONS[c.rejection] || fmt.label(c.rejection), "danger")
                  : badge(r.blocked ? "request blocked" : "not eligible", "")),
            },
          ],
          r.candidates,
          { rowClass: (c) => (c.eligible ? "" : "dim") },
        )),
      );
    };

    const stageScores = (r) => {
      const ml = r.scores.some((s) => s.ml);
      return h("div", { class: "section" },
        stageHead(2, "Ranker scores", "category priors vs the calibrated ML model; expected revenue per 1,000 impressions (eCPM)"),
        r.reference?.ml_error ? callout(`ML ranker unavailable: ${r.reference.ml_error}`, "warn") : null,
        card(null, table(
          [
            { label: "ad", render: (s) => h("span", { class: "mono" }, s.ad_id) },
            { label: "pCTR prior", num: true, render: (s) => fmt.pct(s.prior?.p_ctr, 2) },
            { label: "pCTR ML", num: true, render: (s) => (s.ml ? fmt.pct(s.ml.p_ctr, 2) : "–") },
            { label: "pCVR prior", num: true, render: (s) => fmt.pct(s.prior?.p_cvr, 1) },
            { label: "pCVR ML", num: true, render: (s) => (s.ml ? fmt.pct(s.ml.p_cvr, 1) : "–") },
            { label: "eCPM prior", num: true, render: (s) => fmt.money(s.prior?.ecpm) },
            { label: "eCPM ML", num: true, render: (s) => (s.ml ? fmt.money(s.ml.ecpm) : "–") },
            {
              label: "Δ",
              num: true,
              render: (s) => {
                if (!s.ml || !s.prior) return "–";
                const d = s.ml.ecpm - s.prior.ecpm;
                return h("span", { class: d > 0.0005 ? "" : d < -0.0005 ? "" : "faint" }, `${d >= 0 ? "+" : ""}${d.toFixed(2)}`);
              },
            },
          ],
          r.scores,
        )),
        h("p", { class: "tiny faint" }, ml
          ? "CPM ads earn their price per impression whatever the click rate, so only CPC and CPA ads move with pCTR. pCVR stays on the priors when no CVR model is in production (conversions are sparse at this scale)."
          : "No ML model loaded: train one with `theraflo-ads ml all`."),
      );
    };

    const stagePolicies = (r) => {
      const available = r.policies.filter((p) => p.available);
      const actions = [...new Set(available.flatMap((p) => Object.keys(p.probabilities)))].sort((a, b) => (a === "NO_AD") - (b === "NO_AD") || a.localeCompare(b));
      const matrix = h("table", { class: "data heat" },
        h("thead", {}, h("tr", {}, h("th", {}, "action"), available.map((p) => h("th", { class: "policy" }, POLICY_NAMES[p.name] || p.name, p.serving ? " ●" : "")))),
        h("tbody", {}, actions.map((a) => h("tr", {},
          h("td", { class: "mono" }, a),
          available.map((p) => {
            const v = p.probabilities[a] ?? 0;
            const cell = h("td", { class: `cell ${p.top_action === a ? "top" : ""}`, title: `${POLICY_NAMES[p.name]}: P(${a}) = ${v.toFixed(4)}` }, fmt.prob(v));
            cell.style.background = `rgba(var(--heat), ${Math.min(v, 1) * 0.55})`;
            return cell;
          })))),
      );
      const cards = r.policies.map((p) => policyCard(p));
      return h("div", { class: "section" },
        stageHead(3, "Policies (rerankers)", "each policy's full distribution over the eligible ads and NO_AD; the service samples from exactly what it logs"),
        card("Side by side", h("div", { class: "stack" },
          h("div", { class: "table-wrap" }, matrix),
          h("p", { class: "tiny faint" }, `● serves live traffic. Bold: each policy's most likely action. Expected revenue uses one yardstick for every policy: ${r.reference?.source || "category priors"}.`))),
        h("div", { class: "grid auto-wide", style: { marginTop: "14px" } }, cards),
      );
    };

    const policyCard = (p) => {
      if (!p.available) {
        return card(POLICY_NAMES[p.name] || p.name, h("div", { class: "stack" }, h("p", { class: "desc small muted" }, p.description), callout(p.error, "warn")), { cls: "policy-card" });
      }
      const items = Object.entries(p.probabilities).sort((a, b) => b[1] - a[1]).map(([a, v]) => ({
        label: a,
        value: v,
        display: fmt.prob(v),
        highlight: a === p.top_action,
        title: p.predictions[a] ? Object.entries(p.predictions[a]).map(([k, x]) => `${k}=${x.toFixed(4)}`).join("  ") : "",
      }));
      return h("section", { class: `card policy-card ${p.serving ? "serving" : ""}` },
        h("div", { class: "card-head" },
          h("h3", {}, POLICY_NAMES[p.name] || p.name),
          p.serving ? badge("serving", "solid") : null,
          servedBy(p.served_by)),
        h("div", { class: "card-body" },
          h("p", { class: "desc" }, p.description, h("span", { class: "mono tiny faint" }, `  ${p.version}`)),
          h("div", { class: "mini-stats" },
            h("div", { class: "stat" }, h("span", { class: "label" }, "eRPM"), h("span", { class: "value" }, fmt.money(p.expected_revenue * 1000)), h("span", { class: "sub" }, "expected $ / 1k requests")),
            h("div", { class: "stat" }, h("span", { class: "label" }, "P(NO_AD)"), h("span", { class: "value" }, fmt.pct(p.p_no_ad))),
            h("div", { class: "stat" }, h("span", { class: "label" }, "differs from live"), h("span", { class: "value" }, p.tv_from_serving === undefined ? "–" : fmt.pct(p.tv_from_serving, 0)), h("span", { class: "sub" }, "TV distance")),
            h("div", { class: "stat" }, h("span", { class: "label" }, "latency"), h("span", { class: "value" }, fmt.ms(p.latency_ms)), h("span", { class: "sub" }, p.stochastic ? "stochastic" : "deterministic"))),
          bars(items, { max: 1 }),
          p.servable && !p.serving
            ? h("div", { class: "row", style: { marginTop: "12px" } }, h("button", { class: "btn secondary small", type: "button", on: { click: (e) => busy(e.currentTarget, serve(p.name)) } }, "Serve live traffic with this"))
            : null));
    };

    const featureTable = (features) => {
      const ads = Object.keys(features);
      const names = [...new Set(ads.flatMap((a) => Object.keys(features[a])))].sort();
      if (!names.length) return empty("No features.");
      return h("div", { class: "table-wrap" }, h("table", { class: "data" },
        h("thead", {}, h("tr", {}, h("th", {}, "feature"), ads.map((a) => h("th", { class: "num mono" }, a)))),
        h("tbody", {}, names.map((n) => h("tr", {}, h("td", { class: "mono" }, n), ads.map((a) => h("td", { class: "num" }, features[a][n] === undefined ? "·" : fmt.num(features[a][n], 4))))))));
    };

    // ----- bandits -----

    const renderBandit = async () => {
      try {
        const { data } = await as.admin("GET", "/v1/dev/bandit", { quiet: true });
        const restricted = new Set(ids.boot?.taxonomy?.restricted || []);
        // Arms that can serve someone: approved, and not in a category restricted for everyone.
        const arms = data.beta_ts.arms.filter((a) => a.review_status === "approved" && !restricted.has(a.category));
        const top = Math.max(...arms.map((a) => a.ci95[1]), 0.05);
        const ci = (a) => {
          const box = h("div", { class: "ci", title: `posterior mean ${fmt.pct(a.posterior_mean, 2)}, 95% CI ${fmt.pct(a.ci95[0], 2)}–${fmt.pct(a.ci95[1], 2)}, prior ${fmt.pct(a.prior_ctr, 2)}` });
          const range = h("div", { class: "range" });
          range.style.left = `${(a.ci95[0] / top) * 100}%`;
          range.style.width = `${((a.ci95[1] - a.ci95[0]) / top) * 100}%`;
          const mean = h("div", { class: "mean" });
          mean.style.left = `${(a.posterior_mean / top) * 100}%`;
          const prior = h("div", { class: "prior" });
          prior.style.left = `${(a.prior_ctr / top) * 100}%`;
          box.append(range, prior, mean);
          return box;
        };
        const lin = data.lin_ts;
        mount(
          banditEl,
          card("Bandit state", h("div", { class: "stack" },
            h("h3", {}, "Beta Thompson sampling: click-rate posteriors"),
            table(
              [
                { label: "ad", render: (a) => h("span", { class: "mono" }, a.ad_id) },
                { label: "impr.", num: true, render: (a) => fmt.num(a.impressions) },
                { label: "clicks", num: true, render: (a) => fmt.num(a.clicks) },
                { label: "posterior CTR", num: true, render: (a) => fmt.pct(a.posterior_mean, 2) },
                { label: "95% interval (dotted: prior)", render: ci },
              ],
              arms,
            ),
            h("p", { class: "tiny faint" }, `Warm start: each arm starts at its category prior with a pseudo-count of ${data.beta_ts.prior_strength}, so it moves quickly once real clicks arrive. Updated from every served impression and click, whichever policy served it.`),
            h("h3", {}, "Linear Thompson sampling: posterior weights"),
            h("div", { class: "row tight" }, badge(`${fmt.num(lin.updates)} updates`, "accent"), badge(`${lin.dimensions} features`), lin.learns_online ? badge("learns online", "ok") : badge("warm start only on this backend", "warn")),
            bars(lin.weights.map((w) => ({ label: w.feature, value: Math.abs(w.mean), display: `${w.mean >= 0 ? "+" : "−"}${Math.abs(w.mean).toFixed(4)}`, title: `sd ${w.sd.toFixed(4)}` }))),
            h("p", { class: "tiny faint" }, "Bayesian ridge regression on the ads-safe feature vector, warm-started to predict the category prior (prior_ctr weight 1)."))),
        );
      } catch (err) {
        mount(banditEl, card("Bandit state", callout(err.message, "danger")));
      }
    };

    // ----- models -----

    const registryAction = async (action, name, version) => {
      const promote = action === "promote";
      const ok = await confirmDialog(
        promote ? `Promote ${name}/${version} to production?` : `Roll back ${name}?`,
        promote
          ? "Walks the version through shadow and canary into production in models/registry/registry.json (the same steps as `theraflo-ads ml promote`), and the ML ranker becomes servable."
          : "Points production at the previous version (or the known-safe one); the current one is retired. Serving falls back to the heuristic if no production model remains.",
        { confirmText: promote ? "Promote" : "Roll back", danger: !promote },
      );
      if (!ok) return;
      try {
        await as.admin("POST", `/v1/dev/models/${action}`, { body: promote ? { name, version } : { name } });
        toast(promote ? `${name}/${version} is in production.` : `${name} rolled back.`);
        await Promise.all([renderModels(), run()]);
      } catch (err) {
        errorToast(err);
      }
    };

    const renderModels = async () => {
      try {
        const { data } = await as.admin("GET", "/v1/dev/models", { quiet: true });
        const loaded = data.loaded;
        const validation = loaded?.metrics?.validation || {};
        const reg = data.registry;
        mount(
          modelEl,
          card("Models", h("div", { class: "stack" },
            loaded
              ? [
                  h("div", { class: "row tight" }, badge(loaded.source, loaded.production ? "ok" : "warn"), badge(loaded.algorithm || "?"), badge(loaded.feature_schema_version || "?")),
                  kv([
                    ["CTR model", h("span", { class: "mono" }, loaded.ctr_version)],
                    ["CVR model", loaded.cvr_version ? h("span", { class: "mono" }, loaded.cvr_version) : "category priors"],
                    ["validation AUC", fmt.num(validation.auc, 3)],
                    ["calibration (ECE)", validation.ece !== undefined ? validation.ece.toExponential(1) : "–"],
                    ["log loss", fmt.num(validation.log_loss, 4)],
                    ["rows / positives", validation.n ? `${fmt.num(validation.n)} / ${fmt.num(validation.positives)}` : "–"],
                  ]),
                  loaded.missing_features.length
                    ? callout(h("span", {}, h("strong", {}, `${loaded.missing_features.length} of ${loaded.model_features} model features are not in this catalog's feature vector `), `(they score as 0): ${loaded.missing_features.join(", ")}. The model was trained on the simulator's catalog; on this demo catalog it ranks mostly on price, priors, and history.`), "warn")
                    : callout("Every model feature is present in the serving vector.", "ok"),
                  loaded.importance.length ? [h("h3", {}, "Feature importance (gain)"), bars(loaded.importance.map((i) => ({ label: i.feature, value: i.share, display: fmt.pct(i.share, 0) })))] : null,
                ]
              : callout(`No model found in ${data.model_dir}. Train one: \`theraflo-ads ml all\`.`, "warn"),
            data.problems.length ? callout(data.problems.join("; "), "danger") : null,
            Object.keys(reg).length
              ? [
                  h("h3", {}, "Registry"),
                  ...Object.entries(reg).map(([name, model]) => h("div", { class: "stack" },
                    h("div", { class: "row tight" }, h("strong", {}, name), model.production ? badge(`production: ${model.production}`, "ok") : badge("no production version", "warn"),
                      model.production && name !== "creative_reviewer" ? h("button", { class: "btn ghost small", type: "button", on: { click: () => registryAction("rollback", name) } }, "Roll back") : null),
                    table(
                      [
                        { label: "version", render: (v) => h("span", { class: "mono" }, v.version) },
                        { label: "stage", render: (v) => badge(fmt.label(v.stage), v.stage === "production" ? "ok" : v.stage === "retired" ? "" : "info") },
                        { label: "last change", render: (v) => (v.history.length ? `${fmt.label(v.history.at(-1).to)}: ${v.history.at(-1).reason}` : "–") },
                        {
                          label: "",
                          render: (v) => (name !== "creative_reviewer" && !["production", "retired"].includes(v.stage)
                            ? h("button", { class: "btn small secondary", type: "button", on: { click: () => registryAction("promote", name, v.version) } }, "Promote to production")
                            : ""),
                        },
                      ],
                      model.versions,
                    )))]
              : null)),
        );
      } catch (err) {
        mount(modelEl, card("Models", callout(err.message, "danger")));
      }
    };

    await renderHiddenChips();
    await run();
    renderBandit();
    renderModels();
    scope.every(5000, renderBandit, { immediate: false });
  },
};
