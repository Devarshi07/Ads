// Shared pieces: names and explanations, the decision viewer, creative copy for ad cards.

import { as } from "./api.js";
import { badge, bars, disclosure, fmt, h, json, kv, modal, table, errorToast } from "./ui.js";

export const PLACEMENT_NAMES = {
  home_feed: "Home feed",
  content_library: "Content library",
  journal: "Journal",
  mood_checkin: "Mood check-in",
  crisis_resources: "Crisis resources",
  therapy_chat: "Therapy chat",
  assessment: "Assessment",
  onboarding: "Onboarding",
  crisis: "Crisis",
};

export const BLOCK_REASONS = {
  kill_switch: "The global kill switch is on: no ads anywhere until an admin turns it off.",
  placement_not_allowed: "This screen is ad-free by policy (journal, mood check-in, crisis, therapy…). No policy ever runs here.",
  under_13: "Users under 13 never see ads.",
  suppression_window: "The user opened crisis resources recently, so ads are paused for them (the ads service never learns why).",
  ad_load_cap: "The user has seen their daily maximum of ads (frequency cap).",
  frequency_unavailable: "Frequency counters could not be read, so the request fails closed: no ad.",
  budget_unavailable: "Budget counters could not be read, so the request fails closed: no ad.",
  no_eligible_ads: "Every ad was filtered out by a hard constraint (caps, budgets, targeting, review, hidden ads).",
};

export const REJECTIONS = {
  not_approved: "not approved (pending or rejected review)",
  restricted_category: "restricted category for this user",
  campaign_inactive: "campaign paused",
  placement_targeting: "campaign does not target this screen",
  hidden_by_user: "hidden by the user",
  frequency_cap: "frequency cap (per ad, advertiser, or 10-minute gap)",
  budget: "daily budget exhausted",
};

export const SERVED_BY_KIND = {
  heuristic: "info",
  ranker: "violet",
  bandit: "accent",
  rl: "violet",
  popularity: "warn",
  no_ad: "",
  uniform_holdout: "warn",
  no_ads_holdout: "warn",
  house_ad: "",
};

export function servedBy(value) {
  return badge(fmt.label(value), SERVED_BY_KIND[value] ?? "");
}

export function reviewStatus(value) {
  const kind = { approved: "ok", rejected: "danger", pending: "warn" }[value] || "";
  return badge(value, kind);
}

export function outcomeBadge(value) {
  const kind = { auto_approved: "ok", auto_rejected: "danger", human_review: "warn" }[value] || "";
  return badge(fmt.label(value), kind);
}

// Copy for the demo ads (they predate creative review, so they have no creative record).
export const DEMO_COPY = {
  ad_sleepwell: { headline: "Sleep deeper tonight", body: "Weighted blankets and sleep masks, from SleepWell." },
  ad_trailco: { headline: "Boots for every trail", body: "Waterproof hiking boots from TrailCo, made to last." },
  ad_bookly: { headline: "Your next favorite book", body: "Cozy reads delivered to your door. Bookly." },
  ad_fitco: { headline: "Find a class you'll love", body: "Yoga, spin, and strength classes near you, from FitCo." },
  ad_luckyspin: { headline: "LuckySpin", body: "(restricted category: never served)" },
};

const COPY_KEY = "theraflo.console.creatives";
const copyCache = new Map();

export function rememberCreative(adId, copy) {
  copyCache.set(adId, copy);
  try {
    const stored = JSON.parse(localStorage.getItem(COPY_KEY) || "{}");
    stored[adId] = copy;
    localStorage.setItem(COPY_KEY, JSON.stringify(stored));
  } catch {
    /* storage unavailable */
  }
}

/** Headline and body for an ad card: demo copy, then this browser's submissions, then the
 *  creative record (admin view of the review queue). */
export async function creativeCopy(adId) {
  if (DEMO_COPY[adId]) return DEMO_COPY[adId];
  if (copyCache.has(adId)) return copyCache.get(adId);
  try {
    const stored = JSON.parse(localStorage.getItem(COPY_KEY) || "{}");
    if (stored[adId]) return stored[adId];
  } catch {
    /* ignore */
  }
  try {
    const { data } = await as.admin("GET", `/v1/admin/review-queue/${encodeURIComponent(adId)}`);
    const copy = { headline: data.creative.headline, body: data.creative.body };
    copyCache.set(adId, copy);
    return copy;
  } catch {
    return { headline: adId, body: "" };
  }
}

// ----- decisions -----

export function propensityBars(actions) {
  return bars(
    actions.map((a) => ({
      label: a.action === "NO_AD" ? "NO_AD (show nothing)" : `${a.action}${a.category ? ` · ${fmt.label(a.category)}` : ""}`,
      value: a.probability,
      display: fmt.prob(a.probability),
      highlight: a.chosen,
      title: a.chosen ? "sampled and served" : "",
    })),
    { max: 1 },
  );
}

export function decisionDetails(detail, { withLabel = true } = {}) {
  const d = detail.decision;
  const chosen = d.chosen_actions?.[0]?.ad_id || "NO_AD";
  const parts = [
    h(
      "div",
      { class: "row" },
      servedBy(d.served_by),
      d.policy_version ? badge(d.policy_version) : null,
      d.blocked_reason ? badge(`blocked: ${fmt.label(d.blocked_reason)}`, "danger") : null,
      d.degraded ? badge("degraded", "warn") : null,
      d.ope_eligible ? badge("OPE-eligible", "ok", "stochastic, not degraded, served what the policy sampled") : badge("not OPE-eligible"),
      d.holdout !== "none" ? badge(`holdout: ${fmt.label(d.holdout)}`, "warn") : null,
    ),
    d.blocked_reason ? h("p", { class: "small muted" }, BLOCK_REASONS[d.blocked_reason] || "") : null,
    kv([
      ["request", h("span", { class: "mono" }, d.request_id)],
      ["placement", PLACEMENT_NAMES[d.placement] || d.placement],
      ["consent used", fmt.label(d.consent_scope)],
      ["chosen", h("span", { class: "mono" }, chosen)],
      ["propensity", fmt.prob(d.action_probabilities?.[chosen])],
      ["degraded", d.degraded_reasons?.length ? d.degraded_reasons.join(", ") : "no"],
      ["safety policy", d.safety_policy_version],
      ["features", d.feature_schema_version],
      ["time", fmt.datetime(d.timestamp)],
    ]),
  ];
  if (detail.actions?.length > 1 || (detail.actions?.length === 1 && !d.blocked_reason)) {
    parts.push(h("h3", { class: "section" }, "Action probabilities (logged propensities)"), propensityBars(detail.actions));
  }
  if (d.predicted && Object.keys(d.predicted).length) {
    parts.push(
      h("h3", { class: "section" }, "Predictions for the chosen ad"),
      kv(Object.entries(d.predicted).map(([k, v]) => [fmt.label(k), fmt.prob(v)])),
    );
  }
  if (d.shadow && Object.keys(d.shadow).length) {
    parts.push(h("h3", { class: "section" }, "Shadow policies (scored, never served)"), json(d.shadow));
  }
  if (withLabel && detail.label) parts.push(h("h3", { class: "section" }, "Attribution label"), labelView(detail.label));
  if (d.features && Object.keys(d.features).length) {
    const nonzero = Object.entries(d.features).filter(([, v]) => v);
    parts.push(
      h("div", { class: "section" }, disclosure(`Feature vector of the chosen ad (${nonzero.length} non-zero)`,
        table([{ key: "k", label: "feature" }, { key: "v", label: "value", num: true }], nonzero.map(([k, v]) => ({ k, v: fmt.num(v, 4) }))))),
    );
  }
  parts.push(h("div", { class: "section" }, disclosure("Raw decision log record", json(d))));
  return h("div", { class: "stack" }, parts);
}

export function labelView(label) {
  const kind = { pending: "", partially_observed: "info", finalized: "ok", reversed: "danger" }[label.state] || "";
  return h(
    "div",
    { class: "stack" },
    h("div", { class: "row" }, badge(fmt.label(label.state), kind), label.viewable ? badge("viewable", "info") : null, label.click_id ? badge("clicked", "accent") : null, label.conversions ? badge(`${label.conversions} conversion(s)`, "ok") : null),
    kv([
      ["impression", h("span", { class: "mono" }, label.impression_id)],
      ["ad", `${label.ad_id} (${label.pricing_model} ${fmt.money(label.price)})`],
      ["click id", label.click_id ? h("span", { class: "mono" }, label.click_id) : "–"],
      ["platform revenue", fmt.money(label.platform_revenue, 4)],
      ["order value", label.conversion_value ? fmt.money(label.conversion_value) : "–"],
      ["served", fmt.time(label.served_at)],
    ]),
  );
}

export async function openDecision(requestId) {
  try {
    const { data } = await as.admin("GET", `/v1/dev/decisions/${encodeURIComponent(requestId)}`);
    modal(`Decision ${requestId.slice(0, 12)}…`, decisionDetails(data));
  } catch (err) {
    errorToast(err, "decision: ");
  }
}

export function recentDecisionsTable(decisions, { onRow } = {}) {
  return table(
    [
      { label: "time", render: (d) => h("span", { class: "nowrap" }, fmt.time(d.timestamp)) },
      { label: "screen", render: (d) => PLACEMENT_NAMES[d.placement] || d.placement },
      { label: "served by", render: (d) => servedBy(d.served_by) },
      { label: "ad", render: (d) => (d.chosen_action === "NO_AD" ? h("span", { class: "faint" }, "NO_AD") : h("span", { class: "mono" }, d.chosen_action)) },
      { label: "p", num: true, render: (d) => fmt.prob(d.propensity) },
      { label: "eligible", num: true, key: "eligible" },
      { label: "consent", render: (d) => fmt.label(d.consent_scope) },
      {
        label: "notes",
        render: (d) =>
          h(
            "span",
            { class: "chips" },
            d.blocked_reason ? badge(fmt.label(d.blocked_reason), "danger") : null,
            d.degraded_reasons.length ? badge("degraded", "warn", d.degraded_reasons.join(", ")) : null,
            d.holdout !== "none" ? badge(fmt.label(d.holdout), "warn") : null,
          ),
      },
    ],
    decisions,
    { onRow: onRow || ((d) => openDecision(d.request_id)), emptyText: "No decisions yet: open a screen in the App simulator or start demo traffic." },
  );
}
