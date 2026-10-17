// App simulator: be a Theraflo user. Open screens, see ads (or why not), send the events
// the app SDK would send, click, convert, hide, ask "why this ad?", change consent, open
// crisis resources, delete your data, and watch the decision behind each screen.

import { allUsers, as, forgetTestUser, ids, saveTestUser } from "../api.js";
import {
  badge, busy, callout, card, confirmDialog, errorToast, field, fmt, h, kv, meter, modal, mount, select, toast, uid,
} from "../ui.js";
import { BLOCK_REASONS, PLACEMENT_NAMES, creativeCopy, decisionDetails, labelView } from "../shared.js";

const SCREENS = [
  ["home_feed", true],
  ["content_library", true],
  ["journal", false],
  ["mood_checkin", false],
  ["crisis_resources", false],
  ["therapy_chat", false],
];
const LAST_USER_KEY = "theraflo.console.appUser";

export default {
  id: "app",
  title: "App simulator",
  icon: "▣",
  async render(root, { scope }) {
    const state = {
      userId: null,
      device: "mobile",
      screen: "home_feed",
      session: uid("s"),
      last: null, // {rank, detail, ad}
      events: [],
      autoViewable: true,
      crisisSuppresses: true,
    };
    try {
      state.userId = localStorage.getItem(LAST_USER_KEY);
    } catch {
      /* ignore */
    }
    if (!allUsers().some((u) => u.user_id === state.userId)) state.userId = allUsers()[0]?.user_id;

    const usersEl = h("div", { class: "stack" });
    const profileEl = h("div");
    const phoneEl = h("div");
    const hoodEl = h("div", { class: "stack" });

    mount(
      root,
      h("div", { class: "page-head" }, h("div", { class: "grow" },
        h("h1", {}, "App simulator"),
        h("p", { class: "lede" }, "Use the app as any user. Every action calls the real API exactly as the Theraflo app and backend would; the right-hand panel shows the decision behind each screen. Open the API-calls drawer (⇄, top right) to see each request."))),
      h("div", { class: "app-layout" },
        h("div", { class: "stack" }, usersEl, profileEl),
        phoneEl,
        h("div", { class: "hood" }, hoodEl)),
    );

    // ----- users -----

    const renderUsers = () => {
      const users = allUsers();
      const picker = select(users.map((u) => [u.user_id, `${u.user_id}${u.kind === "test" ? ` (test, ${fmt.label(u.age_band || "")})` : ""}`]), state.userId);
      picker.addEventListener("change", () => switchUser(picker.value));
      const consent = select([["personalized", "personalized ads"], ["contextual_only", "contextual only"]], "personalized");
      const age = select([["adult", "adult"], ["teen", "teen (13–17)"], ["under_13", "under 13"], ["unknown", "unknown age"]], "adult");
      const create = h("button", { class: "btn secondary small", type: "button" }, "Create test user");
      create.addEventListener("click", () =>
        busy(create, as.admin("POST", "/v1/dev/users", { body: { consent: consent.value, age_band: age.value } }).then(({ data }) => {
          saveTestUser({ user_id: data.user_id, token: data.token, token_kind: data.token_kind, consent: consent.value, age_band: age.value });
          toast(`Created ${data.user_id}.`);
          switchUser(data.user_id);
        }).catch(errorToast)));
      const current = users.find((u) => u.user_id === state.userId);
      mount(
        usersEl,
        card("Signed in as", h("div", { class: "stack" },
          field("user", picker),
          current?.kind === "test"
            ? h("button", { class: "btn ghost small", type: "button", on: { click: () => { forgetTestUser(state.userId); switchUser(allUsers()[0]?.user_id); } } }, "Forget this test user")
            : null,
          h("details", { class: "disclosure" }, h("summary", {}, "New test user"), h("div", { class: "stack" },
            h("div", { class: "form-grid" }, field("consent", consent), field("age band", age)),
            create,
            h("p", { class: "tiny faint" }, ids.boot?.test_user_tokens === "session"
              ? "Session tokens signed with THERAFLO_ADS_SESSION_KEY: they survive API restarts."
              : "Tokens last until the API restarts (memory backend). Fresh users are not at their frequency caps yet."))),
        )),
      );
    };

    const switchUser = (userId) => {
      state.userId = userId;
      state.last = null;
      state.events = [];
      state.session = uid("s");
      try {
        localStorage.setItem(LAST_USER_KEY, userId);
      } catch {
        /* ignore */
      }
      renderUsers();
      openScreen(state.screen);
    };

    // ----- profile -----

    const loadProfile = async () => {
      if (!state.userId) return;
      try {
        const { data: p } = await as.admin("GET", `/v1/dev/users/${encodeURIComponent(state.userId)}`, { quiet: true });
        renderProfile(p);
      } catch (err) {
        mount(profileEl, callout(`Profile unavailable: ${err.message}`, "danger"));
      }
    };

    const renderProfile = (p) => {
      const personalized = p.consent === "personalized";
      const consentBtn = h("button", { class: "btn secondary small", type: "button" }, personalized ? "Turn personalized ads off" : "Turn personalized ads on");
      consentBtn.addEventListener("click", () =>
        busy(consentBtn, as.user(state.userId, "PUT", "/v1/users/me/consent", { body: { consent: personalized ? "contextual_only" : "personalized" } })
          .then(() => { toast("Consent updated: it applies from the next request."); loadProfile(); })
          .catch(errorToast)));
      const crisisBtn = h("button", { class: "btn secondary small", type: "button" }, "Simulate crisis resources (pause ads 24h)");
      crisisBtn.addEventListener("click", () => busy(crisisBtn, suppress()));
      const liftBtn = h("button", { class: "btn secondary small", type: "button" }, "Lift suppression");
      liftBtn.addEventListener("click", () =>
        busy(liftBtn, as.admin("DELETE", `/v1/dev/users/${encodeURIComponent(state.userId)}/suppression`).then(() => { toast("Suppression lifted (dev only)."); loadProfile(); }).catch(errorToast)));
      const deleteBtn = h("button", { class: "btn danger secondary small", type: "button" }, "Delete my ads data…");
      deleteBtn.addEventListener("click", async () => {
        const ok = await confirmDialog("Delete this user's ads data?", "Right to deletion: the ads profile goes now; the deletion job shreds the user's key and erases counters, interaction history, and pseudonyms in snapshots.", { confirmText: "Delete", danger: true });
        if (!ok) return;
        try {
          await as.user(state.userId, "DELETE", "/v1/users/me");
          toast(ids.boot?.backend === "infra" ? "Deletion queued: run “Run queued deletions” in Jobs (a cron job in production)." : "Deletion scheduled and run inline (memory backend).");
          loadProfile();
        } catch (err) {
          errorToast(err);
        }
      });
      const exposures = p.exposures_24h;
      mount(
        profileEl,
        card("Profile", h("div", { class: "stack" },
          h("div", { class: "row tight" },
            badge(fmt.label(p.consent), personalized ? "accent" : ""),
            p.effective_consent !== p.consent ? badge(`used as ${fmt.label(p.effective_consent)}`, "warn", "minors and unknown ages are always contextual-only") : null,
            badge(fmt.label(p.age_band)),
            p.suppressed ? badge("ads paused (suppression)", "danger") : null,
            p.holdout !== "none" ? badge(`holdout: ${fmt.label(p.holdout)}`, "warn", "2% see no ads, 2% get uniform-random ads, for unbiased measurement") : null),
          kv([
            ["user hash", h("span", { class: "mono tiny" }, `${p.user_hash.slice(0, 16)}…`)],
            ["region", p.region || "–"],
            ["suppressed until", p.suppressed ? fmt.datetime(p.suppressed_until) : "–"],
            ["hidden ads", p.hidden_ads.length ? p.hidden_ads.join(", ") : "none"],
            ["ads in last 24h", exposures ? h("div", { class: "row tight" }, `${exposures.total} / ${p.caps.per_user_24h}`, meter(exposures.total, p.caps.per_user_24h, exposures.total >= p.caps.per_user_24h ? "danger" : "")) : "unavailable"],
            ["per ad", exposures && Object.keys(exposures.by_ad).length ? Object.entries(exposures.by_ad).map(([a, n]) => `${a}: ${n}/${p.caps.per_ad_24h}`).join(", ") : "–"],
          ]),
          h("p", { class: "tiny faint" }, `Caps: ${p.caps.per_user_24h} ads a day per user, ${p.caps.per_ad_24h} per ad, ${p.caps.per_advertiser_24h} per advertiser, ${p.caps.min_gap_same_ad_s / 60} minutes between repeats of one ad.`),
          h("div", { class: "row tight" }, consentBtn),
          h("div", { class: "row tight" }, p.suppressed ? liftBtn : crisisBtn),
          h("div", { class: "row tight" }, deleteBtn),
        )),
      );
    };

    const suppress = async () => {
      try {
        // The Theraflo backend calls this when the user opens crisis resources; the reason is never sent.
        await as.service("POST", "/v1/internal/suppressions", { body: { user_id: state.userId, hours: 24 } });
        toast("Ads paused for 24h for this user (the ads service is never told why).");
        loadProfile();
      } catch (err) {
        errorToast(err);
      }
    };

    // ----- phone -----

    const renderPhone = async () => {
      const tabs = h("div", { class: "phone-tabs", role: "tablist" },
        SCREENS.map(([placement, ads]) => h("button", {
          class: `phone-tab ${placement === state.screen ? "active" : ""} ${ads ? "" : "adfree"}`,
          type: "button",
          role: "tab",
          "aria-selected": String(placement === state.screen),
          on: { click: () => openScreen(placement) },
        }, PLACEMENT_NAMES[placement])));
      const screen = h("div", { class: "phone-screen" }, await screenContent());
      const device = select([["mobile", "mobile"], ["web", "web"]], state.device);
      device.addEventListener("change", () => { state.device = device.value; openScreen(state.screen); });
      const viewable = h("input", { type: "checkbox", checked: state.autoViewable });
      viewable.addEventListener("change", () => (state.autoViewable = viewable.checked));
      const crisis = h("input", { type: "checkbox", checked: state.crisisSuppresses });
      crisis.addEventListener("change", () => (state.crisisSuppresses = crisis.checked));
      mount(
        phoneEl,
        h("div", { class: "stack" },
          h("div", { class: "phone" },
            h("div", { class: "phone-notch" }),
            h("div", { class: "phone-top" }, h("span", { class: "greet" }, "Theraflo"), h("span", { class: "tiny muted" }, state.userId)),
            tabs,
            screen),
          h("div", { class: "row small" },
            field("device", device),
            h("button", { class: "btn secondary small", type: "button", on: { click: () => openScreen(state.screen) } }, "Reload screen"),
            h("button", { class: "btn ghost small", type: "button", on: { click: () => { state.session = uid("s"); toast("New session."); } } }, "New session")),
          h("label", { class: "check small" }, viewable, "send AD_VIEWABLE when an ad renders (as the SDK does)"),
          h("label", { class: "check small" }, crisis, "opening Crisis resources pauses ads (as the backend does)")),
      );
      // Scroll the tab strip only (scrollIntoView would also scroll the page).
      const active = tabs.querySelector(".phone-tab.active");
      if (active) tabs.scrollLeft = active.offsetLeft - (tabs.clientWidth - active.clientWidth) / 2;
    };

    const screenContent = async () => {
      const placement = state.screen;
      const skel = (title, widths = ["w80", "w60"]) => h("div", { class: "content-block" }, h("strong", { class: "small" }, title), widths.map((w) => h("div", { class: `skel ${w}` })));
      const slot = await adSlot();
      switch (placement) {
        case "home_feed":
          return [skel("Good evening. How are you feeling today?"), slot, skel("A 5-minute breathing exercise", ["w80", "w80", "w60"]), skel("Your streak: 4 days")];
        case "content_library":
          return [skel("Sleep: wind-down routines"), skel("Understanding stress"), slot, skel("Mindful walking")];
        case "journal":
          return [skel("Journal · today", ["w80", "w80", "w80", "w60"]), slot];
        case "mood_checkin":
          return [skel("How do you feel right now?  😞 😐 🙂 😄"), slot];
        case "crisis_resources":
          return [
            h("div", { class: "crisis-panel" },
              h("strong", {}, "You are not alone."),
              h("p", { class: "small", style: { marginTop: "6px" } }, "In the US, call or text 988 (Suicide & Crisis Lifeline), any time. If you are in immediate danger, call 911."),
              h("p", { class: "tiny muted" }, "Simulated screen. Opening it suppresses ads for 24 hours; the ads service only receives “suppress this user”, never the reason.")),
            slot,
          ];
        default:
          return [skel("Therapist · today", ["w60"]), skel("You", ["w80"]), skel("Therapist", ["w80", "w60"]), slot];
      }
    };

    const adSlot = async () => {
      const last = state.last;
      if (!last) return h("div", { class: "no-ad" }, "Loading…");
      if (last.error) return h("div", { class: "no-ad" }, `Request failed: ${last.error}`);
      const ad = last.rank.ads[0];
      const d = last.detail?.decision;
      if (!ad) {
        const reason = d?.blocked_reason;
        return h("div", { class: "no-ad" },
          h("div", {}, h("strong", {}, "No ad here. "), `served_by: ${last.rank.served_by}`),
          reason ? h("div", { class: "tiny", style: { marginTop: "4px" } }, BLOCK_REASONS[reason] || reason) : null,
          !reason && d?.chosen_actions?.length === 0 && d?.policy_action === "NO_AD" ? h("div", { class: "tiny" }, "The policy sampled NO_AD (it is always an action).") : null,
          d?.holdout === "no_ads" ? h("div", { class: "tiny" }, "This user is in the 2% no-ads holdout.") : null);
      }
      const copy = await creativeCopy(ad.ad_id);
      const advertiser = last.detail?.actions?.find((a) => a.action === ad.ad_id)?.advertiser_id;
      return h("div", { class: "ad-card" },
        h("div", { class: "sponsored" }, h("span", {}, "Sponsored"), h("span", { class: "mono" }, ad.ad_id)),
        h("div", { class: "headline" }, copy.headline),
        copy.body ? h("div", { class: "body" }, copy.body) : null,
        advertiser ? h("div", { class: "domain" }, `${advertiser.replace(/^adv_/, "")}.example`) : null,
        h("div", { class: "actions" },
          h("button", { class: "btn small", type: "button", on: { click: (e) => busy(e.currentTarget, clickAd(ad)) } }, "Open ad"),
          h("button", { class: "btn secondary small", type: "button", on: { click: (e) => busy(e.currentTarget, sendEvent(ad, "AD_HIDDEN").then(loadProfile)) } }, "Hide"),
          h("button", { class: "btn secondary small", type: "button", on: { click: (e) => busy(e.currentTarget, sendEvent(ad, "AD_REPORTED")) } }, "Report"),
          h("button", { class: "btn ghost small", type: "button", on: { click: (e) => busy(e.currentTarget, why(ad)) } }, "Why this ad?")));
    };

    const openScreen = async (placement) => {
      state.screen = placement;
      state.last = null;
      if (!state.userId) return;
      try {
        if (placement === "crisis_resources" && state.crisisSuppresses) await suppress();
        const { data: rank } = await as.user(state.userId, "POST", "/v1/ads/rank", {
          body: { session_id: state.session, placement, context: { device: state.device } },
        });
        let detail = null;
        try {
          detail = (await as.admin("GET", `/v1/dev/decisions/${rank.request_id}`)).data;
        } catch {
          /* the decision panel shows what it can */
        }
        state.last = { rank, detail };
        if (rank.ads[0] && state.autoViewable) await sendEvent(rank.ads[0], "AD_VIEWABLE", { quietToast: true });
      } catch (err) {
        state.last = { error: err.status === 401 ? "this user's token is no longer valid (API restarted?): create a new test user" : err.message };
      }
      await renderPhone();
      renderHood();
      loadProfile();
    };

    // ----- ad actions -----

    const sendEvent = async (ad, kind, { quietToast = false } = {}) => {
      const event = {
        event_id: uid("evt_"),
        event_type: kind,
        impression_id: ad.impression_id,
        ad_id: ad.ad_id,
        impression_token: ad.impression_token,
        client_timestamp: new Date().toISOString(),
      };
      let status;
      try {
        const res = await as.user(state.userId, "POST", "/v1/events", { body: event });
        status = res.data.status;
      } catch (err) {
        status = err.data?.status || `error ${err.status}`;
      }
      state.events.unshift({ at: new Date(), kind, status });
      if (!quietToast) toast(`${kind}: ${status}`);
      await refreshLabel();
      renderHood();
    };

    const clickAd = async (ad) => {
      // The real click URL: the API records the click and redirects to the landing page with
      // a click_id. The redirect is not followed here (the landing domains are examples).
      await as.public("GET", ad.click_url, { redirect: "manual" });
      state.events.unshift({ at: new Date(), kind: "CLICK", status: "redirected to landing page" });
      await refreshLabel();
      renderHood();
      const label = state.last?.detail?.label;
      toast(label?.click_id ? "Click recorded; the advertiser can now report a conversion for it." : "Click sent.");
    };

    const why = async (ad) => {
      try {
        const { data } = await as.user(state.userId, "GET", `/v1/ads/${ad.impression_id}/why`);
        modal("Why am I seeing this ad?", h("div", { class: "stack" },
          h("ul", {}, data.factors.map((f) => h("li", {}, f))),
          h("div", { class: "small muted" }, "Controls: ", data.controls.join(" · ")),
          h("p", { class: "tiny faint" }, `Templated from the logged decision (${data.template_version}). Suppression windows, age, consent history, and wellness activity have no template, so they can never appear.`)), { narrow: true });
      } catch (err) {
        errorToast(err);
      }
    };

    const refreshLabel = async () => {
      const impression = state.last?.rank?.ads?.[0]?.impression_id;
      if (!impression || !state.last.detail) return;
      try {
        state.last.detail.label = (await as.admin("GET", `/v1/dev/impressions/${impression}`, { quiet: true })).data;
      } catch {
        /* keep the previous label */
      }
    };

    const conversion = async (value) => {
      const label = state.last?.detail?.label;
      const advertiser = state.last?.detail?.actions?.find((a) => a.chosen)?.advertiser_id;
      if (!label?.click_id || !advertiser) return;
      try {
        const { data } = await as.advertiser(advertiser, "POST", "/v1/postbacks/conversion", {
          body: { click_id: label.click_id, order_id: uid("ord_"), value, occurred_at: new Date().toISOString() },
        });
        state.events.unshift({ at: new Date(), kind: "CONVERSION (advertiser postback)", status: data.status });
        toast(`Conversion postback: ${data.status}`);
      } catch (err) {
        state.events.unshift({ at: new Date(), kind: "CONVERSION (advertiser postback)", status: `error ${err.status}` });
        errorToast(err);
      }
      await refreshLabel();
      renderHood();
    };

    // ----- under the hood -----

    const renderHood = () => {
      const last = state.last;
      const parts = [];
      if (!last || last.error) {
        parts.push(card("Under the hood", h("p", { class: "muted" }, last?.error || "Open a screen to see its decision.")));
      } else if (last.detail) {
        const label = last.detail.label;
        const valueInput = h("input", { type: "number", min: 1, max: 10000, step: 0.01, value: 49.0, style: { width: "100px" } });
        parts.push(card("Under the hood", decisionDetails(last.detail, { withLabel: false }), { sub: PLACEMENT_NAMES[state.screen] }));
        if (label) {
          parts.push(card("Attribution", h("div", { class: "stack" },
            labelView(label),
            label.click_id
              ? h("div", { class: "row" }, field("order value", valueInput),
                  h("button", { class: "btn small", type: "button", on: { click: (e) => busy(e.currentTarget, conversion(Number(valueInput.value))) } }, "Send conversion postback (as the advertiser)"))
              : h("p", { class: "tiny faint" }, "Click the ad to get a click_id; the advertiser reports conversions against it server-to-server."))));
        }
      } else {
        parts.push(card("Under the hood", kv([["request", last.rank.request_id], ["served by", last.rank.served_by]])));
      }
      parts.push(card("Events sent", state.events.length
        ? h("ul", { class: "list small" }, state.events.slice(0, 12).map((e) => h("li", {}, h("span", { class: "faint" }, `${e.at.toLocaleTimeString()}  `), h("strong", {}, e.kind), ` → ${e.status}`)))
        : h("p", { class: "muted small" }, "AD_VIEWABLE, AD_HIDDEN, AD_REPORTED, clicks, and conversions for this screen appear here.")));
      mount(hoodEl, parts);
    };

    renderUsers();
    await loadProfile();
    await openScreen(state.screen);
    scope.every(10000, loadProfile, { immediate: false });
  },
};
