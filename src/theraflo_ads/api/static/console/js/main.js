// Console shell: navigation, header status, theme, and the API-calls drawer.

import { as, calls, ids, loadIdentities, onCall } from "./api.js";
import { Scope, badge, callout, dot, h, json, mount } from "./ui.js";
import overview from "./pages/overview.js";
import appSim from "./pages/app.js";
import lab from "./pages/lab.js";
import advertiser from "./pages/advertiser.js";
import review from "./pages/review.js";
import jobs from "./pages/jobs.js";

const PAGES = [
  { group: "Monitor", items: [overview] },
  { group: "Test", items: [appSim, lab, advertiser, review] },
  { group: "Offline", items: [jobs] },
];
const ALL = PAGES.flatMap((g) => g.items);

let scope = null;

function currentPage() {
  const id = (location.hash.replace(/^#\/?/, "").split("?")[0] || "overview").trim();
  return ALL.find((p) => p.id === id) || overview;
}

function renderNav() {
  const nav = document.getElementById("sidenav");
  const active = currentPage();
  mount(
    nav,
    PAGES.map((group) =>
      h(
        "div",
        { class: "nav-group" },
        h("div", { class: "nav-label" }, group.group),
        group.items.map((page) =>
          h(
            "a",
            { class: `nav-link ${page === active ? "active" : ""}`, href: `#/${page.id}`, "aria-current": page === active ? "page" : null },
            h("span", { class: "nav-icon", "aria-hidden": "true" }, page.icon),
            page.title,
          ),
        ),
      ),
    ),
    h(
      "div",
      { class: "nav-foot" },
      h("div", { class: "faint tiny" }, "Elsewhere"),
      h("a", { href: "/docs", target: "_blank", rel: "noopener" }, "Swagger UI ↗"),
      ids.boot?.backend === "infra"
        ? [
            h("a", { href: "http://localhost:3000", target: "_blank", rel: "noopener" }, "Grafana ↗"),
            h("a", { href: "http://localhost:9090/alerts", target: "_blank", rel: "noopener" }, "Prometheus alerts ↗"),
            h("a", { href: "http://localhost:15672", target: "_blank", rel: "noopener" }, "RabbitMQ ↗"),
          ]
        : h("span", { class: "faint tiny" }, "Grafana and Prometheus run with the Docker stack."),
    ),
  );
}

async function render() {
  scope?.dispose();
  scope = new Scope();
  const page = currentPage();
  renderNav();
  document.title = `${page.title} · Theraflo Ads Console`;
  const main = document.getElementById("main");
  const root = h("div", { class: "page" });
  mount(main, root);
  try {
    await page.render(root, { scope, refreshHeader });
  } catch (err) {
    console.error(err);
    root.append(callout(`This page failed to load: ${err.message}`, "danger"));
  }
}

// ----- header status -----

async function refreshHeader() {
  const el = document.getElementById("topbar-status");
  const items = [badge(`${ids.boot?.backend || "?"} backend`, ids.boot?.backend === "infra" ? "info" : "accent", ids.boot?.backend === "infra" ? "Postgres, Redis, RabbitMQ" : "in-memory stores (resets on restart)")];
  try {
    const ready = await fetch("/readyz", { cache: "no-store" });
    const body = await ready.json();
    const down = Object.entries(body.checks || {}).filter(([, ok]) => !ok).map(([k]) => k);
    items.push(
      h("span", { class: "badge", title: down.length ? `down: ${down.join(", ")}` : "every dependency is reachable" },
        dot(body.ready ? "ok" : "danger"), body.ready ? "ready" : `not ready: ${down.join(", ")}`),
    );
  } catch {
    items.push(h("span", { class: "badge danger" }, dot("danger"), "API unreachable"));
  }
  try {
    const { data } = await as.admin("GET", "/v1/admin/kill-switch", { quiet: true });
    items.push(data.kill_switch ? badge("KILL SWITCH ON: no ads", "danger") : badge("kill switch off", "ok"));
  } catch {
    /* shown by readiness */
  }
  items.push(h("span", { class: "faint tiny" }, `v${ids.boot?.version || "?"}`));
  mount(el, items);
}

// ----- API calls drawer -----

function callRow(entry) {
  const status = entry.status;
  const kind = status === "network" || status >= 500 ? "danger" : status >= 400 ? "warn" : "ok";
  return h(
    "li",
    {},
    h(
      "details",
      {},
      h(
        "summary",
        {},
        h("span", { class: "faint" }, entry.at.toLocaleTimeString()),
        h("span", { class: "mono" }, entry.method),
        badge(String(status), kind),
        h("span", { class: "path", title: entry.path }, `${entry.path}  ·  as ${entry.role}`),
        h("span", { class: "right faint" }, entry.ms !== null ? `${entry.ms.toFixed(0)} ms` : ""),
      ),
      entry.request !== undefined ? [h("div", { class: "tiny faint", style: { margin: "0 14px" } }, "request body"), json(entry.request)] : null,
      h("div", { class: "tiny faint", style: { margin: "0 14px" } }, "response"),
      json(entry.response),
    ),
  );
}

function setupCalls() {
  const drawer = document.getElementById("calls");
  const list = document.getElementById("calls-list");
  const toggle = () => {
    drawer.hidden = !drawer.hidden;
    if (!drawer.hidden) mount(list, calls.map(callRow));
  };
  document.getElementById("calls-toggle").addEventListener("click", toggle);
  document.getElementById("calls-close").addEventListener("click", () => (drawer.hidden = true));
  document.getElementById("calls-clear").addEventListener("click", () => {
    calls.length = 0;
    list.replaceChildren();
  });
  onCall((entry) => {
    if (drawer.hidden || entry.role === "console") return;
    list.prepend(callRow(entry));
    while (list.children.length > 80) list.lastChild.remove();
  });
}

// ----- theme -----

function setupTheme() {
  const key = "theraflo.console.theme";
  try {
    const saved = localStorage.getItem(key);
    if (saved) document.documentElement.dataset.theme = saved;
  } catch {
    /* ignore */
  }
  document.getElementById("theme-toggle").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem(key, next);
    } catch {
      /* ignore */
    }
  });
}

async function boot() {
  setupTheme();
  setupCalls();
  try {
    await loadIdentities();
  } catch (err) {
    mount(
      document.getElementById("main"),
      h("div", { class: "page" }, callout(h("div", {}, h("strong", {}, "The console cannot start. "),
        `GET /v1/dev/identities failed (${err.message}). The console needs dev credentials: it is not mounted when THERAFLO_ADS_ENV=production.`), "danger")),
    );
    return;
  }
  window.addEventListener("hashchange", render);
  await render();
  refreshHeader();
  setInterval(refreshHeader, 5000);
}

boot();

