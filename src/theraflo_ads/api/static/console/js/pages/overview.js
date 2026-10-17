// Overview: is the platform healthy and safe, what is serving, and what traffic is doing.

import { as, ids } from "../api.js";
import {
  ask, badge, bars, busy, callout, card, dot, errorToast, field, fmt, h, kv, mount, sparkline, stat, table, toast,
} from "../ui.js";
import { PLACEMENT_NAMES, recentDecisionsTable, servedBy } from "../shared.js";

const HISTORY = 40;

export default {
  id: "overview",
  title: "Overview",
  icon: "◎",
  async render(root, { scope, refreshHeader }) {
    const history = []; // [{t, rank, served, blocked}]
    const statusEl = h("div", { class: "grid cols-4" }, h("p", { class: "muted" }, "Loading…"));
    const liveEl = h("div");
    const trafficEl = h("div");
    const countsEl = h("div", { class: "grid auto-wide" });
    const decisionsEl = h("div");
    const catalogEl = h("div");

    mount(
      root,
      h(
        "div",
        { class: "page-head" },
        h("div", { class: "grow" },
          h("h1", {}, "Overview"),
          h("p", { class: "lede" },
            "Health, safety, and what is serving right now. Counts are this API process since it started; Prometheus and Grafana keep the history. Start demo traffic below to see everything move.")),
      ),
      statusEl,
      h("div", { class: "grid cols-2 section" }, liveEl, trafficEl),
      h("div", { class: "section" }, h("h2", {}, "Counters", h("span", { class: "muted" }, "totals since the API started")), countsEl),
      h("div", { class: "section" }, h("h2", {}, "Recent decisions", h("span", { class: "muted" }, "click a row for the full decision log record")), decisionsEl),
      h("div", { class: "section" }, h("h2", {}, "Inventory", h("span", { class: "muted" }, "campaigns, budgets, and ads in the live catalog")), catalogEl),
    );

    const killSwitch = async (enable) => {
      const reason = await ask(enable ? "Turn the kill switch ON" : "Turn the kill switch off",
        "Reason (kept in the audit log)", { placeholder: enable ? "e.g. incident 42: unsafe creative live" : "e.g. incident resolved", confirmText: enable ? "Stop all ads" : "Resume ads", danger: enable, minLength: 3 });
      if (!reason) return;
      try {
        await as.admin("PUT", "/v1/admin/kill-switch", { body: { enabled: enable, reason } });
        toast(enable ? "Kill switch ON: no ads are served anywhere." : "Kill switch off: ads resume.", enable ? "warn" : "");
        await refresh();
        refreshHeader();
      } catch (err) {
        errorToast(err);
      }
    };

    const renderStatus = (o) => {
      const totals = o.metrics.totals;
      const violations = Object.values(totals.safety_violations || {}).reduce((a, b) => a + b, 0);
      const health = Object.entries(o.health);
      mount(
        statusEl,
        card("Readiness", h("div", { class: "stack" },
          health.map(([name, ok]) => h("div", { class: "row tight" }, dot(ok ? "ok" : "danger"), h("span", {}, fmt.label(name)), ok ? null : badge("down", "danger"))),
          h("div", { class: "small muted" }, `${o.backend} backend · ${o.reviewer ? `reviewer: ${o.reviewer}` : ""}`))),
        card("Kill switch", h("div", { class: "stack" },
          h("div", { class: "row" }, o.kill_switch ? badge("ON: every request gets NO_AD", "danger") : badge("off", "ok")),
          h("p", { class: "small muted" }, "The incident brake: blocks every ad before any policy runs, on every replica."),
          h("button", { class: `btn ${o.kill_switch ? "secondary" : "danger secondary"} small`, type: "button", on: { click: () => killSwitch(!o.kill_switch) } },
            o.kill_switch ? "Turn off" : "Turn on…"))),
        card("Serving policy", h("div", { class: "stack" },
          o.serving
            ? [
                h("div", { class: "row tight" }, badge(o.serving.policy, "accent")),
                h("ol", { class: "small", style: { margin: 0, paddingLeft: "18px" } },
                  o.serving.chain.map((level) => h("li", {}, servedBy(level.served_by), " ", h("span", { class: "mono tiny" }, level.version)))),
                h("a", { href: "#/lab", class: "small" }, "Compare and switch in the Ranker lab →"),
              ]
            : h("span", { class: "muted" }, "unavailable"))),
        card("Safety gate", h("div", { class: "stack" },
          h("div", { class: "row" }, violations ? badge(`${violations} violation(s)`, "danger") : badge("0 violations", "ok")),
          h("p", { class: "small muted" }, "Every served ad is re-checked against the hard constraints before it leaves the API. Must stay 0; a violation withholds the ad and pages."),
          h("div", { class: "small" }, `Review queue: ${o.review_queue.review ?? 0} to review, ${o.review_queue.audit ?? 0} audits`),
          h("a", { href: "#/review", class: "small" }, "Open review & safety →"))),
      );
    };

    const renderLive = (o) => {
      const now = Date.now();
      const totals = o.metrics.totals;
      const rank = Object.values(totals.decisions || {}).reduce((a, b) => a + b, 0);
      history.push({ t: now, rank });
      if (history.length > HISTORY) history.shift();
      const rates = [];
      for (let i = 1; i < history.length; i += 1) {
        const dt = (history[i].t - history[i - 1].t) / 1000;
        rates.push(dt > 0 ? Math.max(history[i].rank - history[i - 1].rank, 0) / dt : 0);
      }
      const lat = o.metrics.rank_latency_ms;
      mount(
        liveEl,
        card("Live", h("div", { class: "stack" },
          h("div", { class: "stats" },
            stat("decisions / s", fmt.num(rates.at(-1) ?? 0, 1), `last ${HISTORY * 3}s below`),
            stat("decisions", fmt.num(rank)),
            stat("fill rate", o.fill.window ? fmt.pct(o.fill.served / o.fill.window) : "–", `ads served in the last ${o.fill.window} decisions`),
            stat("rank p50", fmt.ms(lat.p50), `p95 ${fmt.ms(lat.p95)} · p99 ${fmt.ms(lat.p99)}`)),
          sparkline(rates.length ? rates : [0, 0]),
          h("p", { class: "tiny faint" }, "Fill rate is low by design: ad-free screens, caps, and the 4% holdouts all serve NO_AD."))),
      );
    };

    // The traffic form is built once: the 3-second refresh must not wipe what you type.
    const rate = h("input", { type: "number", min: 0.5, max: 40, step: 0.5, value: 8 });
    const minutes = h("input", { type: "number", min: 0, max: 240, step: 1, value: 10 });
    const population = h("input", { type: "number", min: 1, max: 2000, step: 1, value: 300 });
    const floods = h("input", { type: "checkbox", checked: true });
    const start = h("button", { class: "btn", type: "button" }, "Start traffic");
    const stop = h("button", { class: "btn danger secondary hidden", type: "button" }, "Stop");
    const trafficForm = h("div", { class: "form-grid" },
      field("screens / second", rate), field("minutes (0 = until stopped)", minutes), field("users", population),
      h("label", { class: "check" }, floods, "rate-limit burst every minute"));
    const trafficState = h("div", { class: "stack" });
    start.addEventListener("click", () =>
      busy(start, as.admin("POST", "/v1/dev/traffic", {
        body: { rate: Number(rate.value), minutes: Number(minutes.value), population: Number(population.value), floods: floods.checked },
      }).then(() => { toast("Demo traffic started."); refresh(); }).catch(errorToast)));
    stop.addEventListener("click", () =>
      busy(stop, as.admin("DELETE", "/v1/dev/traffic").then(() => { toast("Demo traffic stopped."); refresh(); }).catch(errorToast)));
    mount(
      trafficEl,
      card("Demo traffic", h("div", { class: "stack" },
        h("p", { class: "small muted" }, "Plays a population of test users (mixed consent and age) and the dev advertisers against this API: feed and ad-free screens, viewable events, clicks, conversions, hidden ads, crisis suppressions, creatives, and a burst a minute that the rate limiter cuts off."),
        trafficForm,
        h("div", { class: "row" }, start, stop),
        trafficState)),
    );

    const renderTraffic = (t) => {
      const s = t.stats || {};
      trafficForm.classList.toggle("hidden", t.running);
      start.classList.toggle("hidden", t.running);
      stop.classList.toggle("hidden", !t.running);
      const errors = (s.transport_error || 0) + Object.entries(s).filter(([k]) => /^(rank|creative)_\d/.test(k)).reduce((a, [, v]) => a + v, 0);
      mount(
        trafficState,
        t.running
          ? callout(h("span", {}, h("strong", {}, "Running. "), `${t.rate}/s from ${t.population} test users${t.ends_at ? `, until ${fmt.time(t.ends_at)}` : ", until stopped"}.`), "ok")
          : null,
        Object.keys(s).length
          ? kv([
              ["screens", fmt.num(s.rank || 0)], ["ads served", fmt.num(s.ad_served || 0)], ["no ad", fmt.num(s.no_ad || 0)],
              ["viewable", fmt.num(s.ad_viewable || 0)], ["clicks", fmt.num(s.click || 0)], ["conversions", fmt.num(s.conversion || 0)],
              ["hidden", fmt.num(s.ad_hidden || 0)], ["creatives", fmt.num(s.creative || 0)], ["suppressions", fmt.num(s.suppression || 0)],
              ["rate-limited (429)", fmt.num(s.rate_limited || 0)], ["errors", fmt.num(errors)],
            ])
          : null,
      );
    };

    const counterCard = (title, data, { danger = false, help } = {}) => {
      const items = Object.entries(data || {}).sort((a, b) => b[1] - a[1]).map(([label, value]) => ({ label: fmt.label(label || "(none)"), value, danger: danger && value > 0 }));
      return card(title, h("div", { class: "stack" }, help ? h("p", { class: "tiny faint" }, help) : null, bars(items)));
    };

    const renderCounts = (o) => {
      const t = o.metrics.totals;
      mount(
        countsEl,
        counterCard("Decisions by policy", t.decisions, { help: "who answered each rank request" }),
        counterCard("Blocked before the policy", t.blocked, { help: "placement_not_allowed = ad-free screens working" }),
        counterCard("Safety violations", t.safety_violations, { danger: true, help: "expected 0 for every kind" }),
        counterCard("Client events", t.client_events, { help: "viewable / hidden / reported, by ingestion status" }),
        counterCard("Rate limited", t.rate_limited, { help: "requests refused with 429, by scope" }),
        counterCard("Creative review outcomes", t.review_outcomes),
        counterCard("Degraded decisions", t.degraded, { help: "fallbacks and fail-closed answers" }),
        counterCard("HTTP status", t.http_requests),
      );
    };

    const refresh = async () => {
      const { data } = await as.admin("GET", "/v1/dev/overview", { quiet: true });
      renderStatus(data);
      renderLive(data);
      renderTraffic(data.traffic);
      renderCounts(data);
    };
    const refreshDecisions = async () => {
      const { data } = await as.admin("GET", "/v1/dev/decisions", { query: { limit: 15 }, quiet: true });
      mount(decisionsEl, card(null, recentDecisionsTable(data.decisions)));
    };
    const refreshCatalog = async () => {
      const { data } = await as.admin("GET", "/v1/dev/catalog", { quiet: true });
      mount(
        catalogEl,
        card(null, table(
          [
            { label: "campaign", render: (c) => h("span", { class: "mono" }, c.campaign_id) },
            { label: "advertiser", key: "advertiser_id" },
            { label: "pricing", render: (c) => `${c.pricing_model.toUpperCase()} ${fmt.money(c.price)}` },
            { label: "billed today", render: (c) => h("div", { class: "row tight" }, h("span", { class: "nowrap" }, `${fmt.money(c.billed_today)} / ${fmt.money(c.daily_budget, 0)}`)) },
            { label: "screens", render: (c) => (c.placements ? c.placements.map((p) => PLACEMENT_NAMES[p] || p).join(", ") : "all ad screens") },
            { label: "priority", num: true, key: "priority" },
            { label: "ads", render: (c) => h("span", { class: "chips" }, c.ads.map((a) => badge(`${a.ad_id} · ${a.review_status}`, a.review_status === "approved" ? "ok" : a.review_status === "rejected" ? "danger" : "warn", fmt.label(a.category)))) },
          ],
          data.campaigns,
        )),
      );
    };

    scope.every(3000, refresh);
    scope.every(3000, refreshDecisions);
    scope.every(15000, refreshCatalog);
    if (!ids.boot) toast("identities not loaded", "error");
  },
};
