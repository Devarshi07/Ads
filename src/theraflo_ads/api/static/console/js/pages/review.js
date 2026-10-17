// Review & safety: the human review queue and audits, the machine review record behind each
// item, substantiated claims, the kill switch, and the audit log.

import { as, ids } from "../api.js";
import {
  ask, badge, busy, callout, card, disclosure, empty, errorToast, field, fmt, h, json, kv, mount, select, stat, table, toast,
} from "../ui.js";
import { outcomeBadge } from "../shared.js";

export default {
  id: "review",
  title: "Review & safety",
  icon: "⛨",
  async render(root, { scope, refreshHeader }) {
    let kind = "review";
    let selected = null;
    const queueEl = h("div");
    const detailEl = h("div");
    const auditSummaryEl = h("div");
    const killEl = h("div");
    const claimsEl = h("div");
    const logEl = h("div");
    const tabs = h("div", { class: "segmented" });

    mount(
      root,
      h("div", { class: "page-head" }, h("div", { class: "grow" },
        h("h1", {}, "Review & safety"),
        h("p", { class: "lede" }, "What the machine review could not decide goes to a person; a sample of what it approved is audited by one. Creative text is advertiser-supplied and shown as untrusted text."))),
      h("div", { class: "grid cols-4" }, auditSummaryEl, killEl, h("div", { class: "span-2", style: { gridColumn: "span 2" } }, claimsEl)),
      h("div", { class: "section", style: { display: "grid", gridTemplateColumns: "minmax(0, 360px) minmax(0, 1fr)", gap: "14px", alignItems: "start" }, id: "review-split" },
        card("Queue", h("div", { class: "stack" }, tabs, queueEl)),
        detailEl),
      h("div", { class: "section" }, h("h2", {}, "Audit log", h("span", { class: "muted" }, "who changed what, newest first")), logEl),
    );
    if (matchMedia("(max-width: 900px)").matches) document.getElementById("review-split").style.gridTemplateColumns = "minmax(0, 1fr)";

    const renderTabs = () => {
      mount(tabs, ["review", "audit"].map((k) => h("button", {
        type: "button",
        class: k === kind ? "active" : "",
        on: { click: () => { kind = k; selected = null; renderTabs(); loadQueue(); mount(detailEl, empty("Pick an item.")); } },
      }, k === "review" ? "To review (not live)" : "Audits (already live)")));
    };

    const loadQueue = async () => {
      const { data } = await as.admin("GET", "/v1/admin/review-queue", { query: { kind }, quiet: true });
      mount(queueEl, data.items.length
        ? h("ul", { class: "list" }, data.items.map((item) => h("li", {
            class: `clickable ${selected === item.ad_id ? "selected" : ""}`,
            on: { click: () => { selected = item.ad_id; loadQueue(); openItem(item); } },
          },
          h("div", { class: "row between" }, h("strong", { class: "mono small" }, item.ad_id), item.priority ? badge("priority", "warn") : null),
          h("div", { class: "small muted" }, `${item.advertiser_id} · ${fmt.ago(item.enqueued_at)}`),
          h("ul", { class: "tiny muted", style: { margin: "4px 0 0", paddingLeft: "16px" } }, item.reasons.slice(0, 3).map((r) => h("li", {}, r))))))
        : empty(kind === "review" ? "Nothing waiting. Submit a creative in the Advertiser portal (try a health-adjacent category or the urgency preset)." : "No audits open. A sample of auto-approved creatives lands here."));
    };

    const openItem = async (item) => {
      try {
        const { data } = await as.admin("GET", `/v1/admin/review-queue/${encodeURIComponent(item.ad_id)}`);
        renderDetail(item, data.creative, data.review);
      } catch (err) {
        mount(detailEl, callout(err.message, "danger"));
      }
    };

    const renderDetail = (item, creative, review) => {
      const allowed = ids.boot?.taxonomy?.allowed || [];
      const minor = ids.boot?.taxonomy?.minor_restricted || [];
      const restricted = ids.boot?.taxonomy?.restricted || [];
      const category = h("select", {},
        h("optgroup", { label: "allowed" }, [...allowed, ...minor].map((c) => h("option", { value: c, selected: c === (review?.category || creative.declared_category) }, fmt.label(c)))),
        h("optgroup", { label: "restricted (cannot be approved)" }, restricted.map((c) => h("option", { value: c, selected: c === review?.category }, fmt.label(c)))));
      const violations = h("div", { class: "chips" }, (ids.boot?.violations || []).map((v) => h("label", { class: "check small" }, h("input", { type: "checkbox", value: v, checked: review?.violations?.includes(v) }), fmt.label(v))));
      const notes = h("textarea", { rows: 2, maxlength: 1000, placeholder: "notes (optional, kept with the decision)" });
      const decide = async (verdict) => {
        const chosen = [...violations.querySelectorAll("input:checked")].map((i) => i.value);
        try {
          await as.admin("POST", `/v1/admin/review-queue/${encodeURIComponent(item.ad_id)}`, {
            body: { kind: item.kind, verdict, category: category.value, violations: verdict === "reject" ? chosen : [], notes: notes.value },
          });
          toast(`${item.ad_id}: ${verdict}d.${verdict === "approve" && item.kind === "review" ? " It can serve now." : ""}`);
          selected = null;
          mount(detailEl, empty("Decided. Pick another item."));
          await Promise.all([loadQueue(), loadAuditSummary(), loadLog()]);
        } catch (err) {
          errorToast(err);
        }
      };
      const approve = h("button", { class: "btn", type: "button" }, item.kind === "audit" ? "Machine was right" : "Approve");
      approve.addEventListener("click", () => busy(approve, decide("approve")));
      const reject = h("button", { class: "btn danger", type: "button" }, item.kind === "audit" ? "Overturn: take it down" : "Reject");
      reject.addEventListener("click", () => busy(reject, decide("reject")));
      mount(
        detailEl,
        h("div", { class: "stack" },
          card(`Creative ${creative.ad_id}`, h("div", { class: "stack" },
            h("div", { class: "row tight" }, badge(creative.advertiser_id), badge(creative.campaign_id), badge(`declared: ${fmt.label(creative.declared_category)}`, "info")),
            h("div", { class: "tiny faint" }, "Advertiser-supplied text (untrusted: the reviewer treats it as data, never as instructions)"),
            kv([
              ["headline", h("div", { class: "untrusted" }, creative.headline)],
              ["body", creative.body ? h("div", { class: "untrusted" }, creative.body) : "–"],
              ["landing URL", h("span", { class: "mono small" }, creative.landing_url)],
              ["landing text", creative.landing_text ? h("div", { class: "untrusted" }, creative.landing_text) : "–"],
              ["submitted", fmt.datetime(creative.submitted_at)],
            ]))),
          review ? machineReview(review) : callout("No machine review yet (the review worker has not run).", "warn"),
          card(item.kind === "audit" ? "Audit decision" : "Human decision", h("div", { class: "stack" },
            item.kind === "audit"
              ? h("p", { class: "small muted" }, "This ad is live (auto-approved). Overturning it takes it down immediately and counts as a false approval in the measured rate.")
              : h("p", { class: "small muted" }, "Not live until a person approves it."),
            field("category", category),
            field("violations (for a rejection)", violations),
            notes,
            h("div", { class: "row" }, approve, reject))),
        ),
      );
    };

    const machineReview = (r) => {
      const claimRows = r.claims || [];
      return card("Machine review", h("div", { class: "stack" },
        h("div", { class: "row tight" }, outcomeBadge(r.outcome), badge(`category: ${fmt.label(r.category)}`), r.audited ? badge("audit sample", "info") : null, r.error ? badge("reviewer error", "danger") : null),
        kv([
          ["reasons", r.reasons.length ? r.reasons.join("; ") : "–"],
          ["violations", r.violations.length ? h("span", { class: "chips" }, r.violations.map((v) => badge(fmt.label(v), "danger"))) : "none"],
          ["rule signals", r.rule_signals.length ? r.rule_signals.join(", ") : "none"],
          ["injection signals", r.injection_signals.length ? h("span", { class: "chips" }, r.injection_signals.map((s) => badge(s, "danger"))) : "none"],
          ["model", r.model || "none (rules only)"],
          ["prompt", r.prompt_version || "–"],
          ["versions", `${r.pipeline_version} · ${r.taxonomy_version} · ${r.rules_version}`],
          ["error", r.error || "–"],
        ]),
        claimRows.length
          ? [h("h3", {}, "Claims and their grounding"), table(
              [
                { label: "claim", render: (c) => h("span", { class: "small" }, c.text) },
                { label: "status", render: (c) => badge(fmt.label(c.status), { grounded: "ok", partial: "warn", ungrounded: "danger", not_in_creative: "danger" }[c.status] || "") },
                { label: "health", render: (c) => (c.health ? badge("health", "warn") : "–") },
                { label: "matched claim", render: (c) => c.claim_id || "–" },
                { label: "coverage", num: true, render: (c) => fmt.pct(c.coverage, 0) },
              ],
              claimRows,
            )]
          : h("p", { class: "small muted" }, "No factual claims extracted."),
        r.judgements?.length ? disclosure(`Reviewer judgements (${r.judgements.length})`, json(r.judgements)) : null,
        disclosure("Full review record", json(r)),
      ));
    };

    const loadAuditSummary = async () => {
      const { data } = await as.admin("GET", "/v1/admin/review-audit", { quiet: true });
      mount(auditSummaryEl, card("Measured false approvals", h("div", { class: "stack" },
        stat("false-approval rate", data.false_approval_rate === null ? "–" : fmt.pct(data.false_approval_rate),
          `95% CI ${fmt.pct(data.ci95[0])}–${fmt.pct(data.ci95[1])} over ${data.audited} audits`),
        h("div", { class: "small muted" }, `${data.overturned} overturned · ${data.open_audits} audits open`))));
    };

    const loadKill = async () => {
      const { data } = await as.admin("GET", "/v1/admin/kill-switch", { quiet: true });
      const toggle = h("button", { class: `btn small ${data.kill_switch ? "secondary" : "danger secondary"}`, type: "button" }, data.kill_switch ? "Turn off" : "Turn on…");
      toggle.addEventListener("click", async () => {
        const reason = await ask(data.kill_switch ? "Resume ads" : "Stop all ads", "Reason (audit log)", { minLength: 3, danger: !data.kill_switch, confirmText: data.kill_switch ? "Resume" : "Stop all ads" });
        if (!reason) return;
        try {
          await as.admin("PUT", "/v1/admin/kill-switch", { body: { enabled: !data.kill_switch, reason } });
          await Promise.all([loadKill(), loadLog()]);
          refreshHeader();
        } catch (err) {
          errorToast(err);
        }
      });
      mount(killEl, card("Kill switch", h("div", { class: "stack" },
        data.kill_switch ? badge("ON: no ads anywhere", "danger") : badge("off", "ok"),
        h("div", { class: "tiny faint" }, data.safety_policy_version),
        toggle)));
    };

    const loadClaims = async (advertiser) => {
      const advertisers = (ids.boot?.advertisers || []).map((a) => a.advertiser_id);
      advertiser = advertiser || advertisers[0];
      const picker = select(advertisers, advertiser);
      picker.addEventListener("change", () => loadClaims(picker.value));
      const { data } = await as.admin("GET", "/v1/admin/claims", { query: { advertiser_id: advertiser }, quiet: true });
      const text = h("input", { type: "text", placeholder: "the claim, as it may appear in ads", maxlength: 1000, style: { width: "100%" } });
      const evidence = h("input", { type: "text", placeholder: "evidence on file, e.g. study-2025.pdf", maxlength: 500, style: { width: "100%" } });
      const add = h("button", { class: "btn secondary small", type: "button" }, "Add substantiated claim");
      add.addEventListener("click", () => busy(add, as.admin("POST", "/v1/admin/claims", {
        body: { claim_id: `clm_${advertiser.replace(/^adv_/, "")}_${Date.now().toString(36)}`, advertiser_id: advertiser, text: text.value.trim(), evidence: evidence.value.trim() },
      }).then(() => { toast("Claim added: creatives stating it are now grounded."); loadClaims(advertiser); }).catch(errorToast)));
      mount(claimsEl, card("Substantiated claims", h("div", { class: "stack" },
        h("div", { class: "row" }, field("advertiser", picker)),
        data.claims.length
          ? h("ul", { class: "list small" }, data.claims.map((c) => h("li", {}, h("div", {}, c.text), h("div", { class: "tiny faint" }, `${c.claim_id} · evidence: ${c.evidence}`))))
          : h("p", { class: "small muted" }, "No claims on file: any statistic or health claim in this advertiser's ads is ungrounded."),
        h("div", { class: "form-grid" }, h("div", { class: "span-2" }, text), evidence),
        add), { sub: "what ads may state as fact, checked by a policy owner" }));
    };

    const loadLog = async () => {
      const { data } = await as.admin("GET", "/v1/admin/audit", { query: { limit: 60 }, quiet: true });
      mount(logEl, card(null, table(
        [
          { label: "time", render: (e) => h("span", { class: "nowrap" }, fmt.time(e.at)) },
          { label: "event", render: (e) => badge(fmt.label(e.event), /kill|reject|false|overturn|deletion/.test(e.event) ? "warn" : "") },
          { label: "actor", key: "actor" },
          {
            label: "details",
            render: (e) => h("span", { class: "small mono" }, Object.entries(e).filter(([k]) => !["event", "actor", "at"].includes(k)).map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join("  ")),
          },
        ],
        data.entries,
        { emptyText: "Nothing audited yet." },
      )));
    };

    renderTabs();
    mount(detailEl, empty("Pick an item from the queue."));
    await Promise.all([loadQueue(), loadAuditSummary(), loadKill(), loadClaims(), loadLog()]);
    scope.every(4000, async () => {
      await Promise.all([loadQueue(), loadAuditSummary(), loadLog()]);
    }, { immediate: false });
  },
};
