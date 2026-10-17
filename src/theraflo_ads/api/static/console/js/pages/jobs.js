// Jobs & reports: the offline side. Run allowlisted `theraflo-ads` commands (simulator,
// review evals, experiments, offline RL, drift) and read every report under reports/.

import { as } from "../api.js";
import {
  badge, busy, callout, card, empty, errorToast, field, fmt, h, json, markdown, mount, select, table, toast,
} from "../ui.js";

const STATUS_KIND = { running: "info", succeeded: "ok", failed: "danger", cancelled: "warn", timed_out: "danger" };

export default {
  id: "jobs",
  title: "Jobs & reports",
  icon: "⚙",
  async render(root, { scope }) {
    const catalogEl = h("div", { class: "grid auto-wide" });
    const jobsEl = h("div");
    const outputEl = h("div");
    const reportsEl = h("div");
    const viewerEl = h("div");
    let watching = null;
    let reportFilter = "";

    mount(
      root,
      h("div", { class: "page-head" }, h("div", { class: "grow" },
        h("h1", {}, "Jobs & reports"),
        h("p", { class: "lede" }, "Offline work runs as allowlisted CLI commands with bounded parameters, one at a time, in this API's process environment. Nothing billed can start from here. Outputs land in reports/console/."))),
      h("div", { class: "section" }, h("h2", {}, "Run a job"), catalogEl),
      h("div", { class: "grid cols-2 section" },
        card("Runs", jobsEl, { sub: "latest first" }),
        outputEl),
      h("div", { class: "section" }, h("h2", {}, "Reports", h("span", { class: "muted" }, "every JSON and Markdown file under reports/")),
        h("div", { class: "grid", style: { gridTemplateColumns: "minmax(0, 360px) minmax(0, 1fr)", alignItems: "start" } }, reportsEl, viewerEl)),
    );

    const paramControl = (name, p) => {
      if (p.kind === "bool") return h("input", { type: "checkbox", checked: Boolean(p.default), dataset: { param: name } });
      if (p.kind === "choice") return select(p.choices, p.default, { dataset: { param: name } });
      return h("input", {
        type: "number", value: p.default, min: p.low ?? undefined, max: p.high ?? undefined,
        step: p.kind === "int" ? 1 : "any", dataset: { param: name },
      });
    };

    const renderCatalog = (catalog) => {
      mount(catalogEl, catalog.map((spec) => {
        const controls = Object.entries(spec.params).map(([name, p]) => {
          const control = paramControl(name, p);
          const hint = [p.low !== null && p.high !== null && p.kind !== "bool" && p.kind !== "choice" ? `${p.low}–${p.high}` : "", p.help].filter(Boolean).join(" · ");
          return p.kind === "bool" ? h("label", { class: "check small" }, control, fmt.label(name), p.help ? h("span", { class: "faint tiny" }, ` (${p.help})`) : null) : field(fmt.label(name), control, hint || undefined);
        });
        const runBtn = h("button", { class: "btn small", type: "button", disabled: Boolean(spec.unavailable) }, "Run");
        const container = h("div", { class: "form-grid" }, controls);
        runBtn.addEventListener("click", () => busy(runBtn, (async () => {
          const params = {};
          for (const el of container.querySelectorAll("[data-param]")) {
            const p = spec.params[el.dataset.param];
            params[el.dataset.param] = p.kind === "bool" ? el.checked : p.kind === "choice" ? el.value : Number(el.value);
          }
          try {
            const { data } = await as.admin("POST", "/v1/dev/jobs", { body: { name: spec.name, params } });
            toast(`Started: ${data.command}`);
            watching = data.job_id;
            await refreshJobs();
          } catch (err) {
            errorToast(err);
          }
        })()));
        return card(spec.title, h("div", { class: "stack" },
          h("p", { class: "small muted" }, spec.description),
          controls.length ? container : null,
          spec.unavailable ? callout(`Unavailable: ${spec.unavailable}`, "warn") : null,
          spec.outputs.length ? h("div", { class: "tiny faint" }, `writes ${spec.outputs.join(", ")}`) : null,
          h("div", { class: "row" }, runBtn)));
      }));
    };

    const refreshJobs = async () => {
      const { data } = await as.admin("GET", "/v1/dev/jobs", { quiet: true });
      mount(jobsEl, table(
        [
          { label: "job", render: (j) => h("strong", {}, j.name) },
          { label: "status", render: (j) => badge(fmt.label(j.status), STATUS_KIND[j.status] || "") },
          { label: "started", render: (j) => fmt.time(j.started_at) },
          { label: "duration", num: true, render: (j) => `${j.duration_s.toFixed(1)} s` },
          { label: "exit", num: true, render: (j) => (j.exit_code ?? "–") },
        ],
        data.jobs,
        { onRow: (j) => { watching = j.job_id; refreshOutput(); }, rowClass: (j) => (j.job_id === watching ? "selected" : ""), emptyText: "No runs yet." },
      ));
      if (!watching && data.jobs.length) watching = data.jobs[0].job_id;
      await refreshOutput();
      return data;
    };

    let lastShown = "";
    const refreshOutput = async () => {
      if (!watching) {
        mount(outputEl, card("Output", empty("Run a job to see its output here.")));
        return;
      }
      const { data: job } = await as.admin("GET", `/v1/dev/jobs/${watching}`, { quiet: true });
      const pre = h("pre", { class: "log" }, job.output || "(no output yet)");
      const cancel = h("button", { class: "btn danger secondary small", type: "button" }, "Cancel");
      cancel.addEventListener("click", () => busy(cancel, as.admin("DELETE", `/v1/dev/jobs/${job.job_id}`).then(refreshJobs).catch(errorToast)));
      const shown = `${job.job_id}:${job.status}:${job.output.length}`;
      if (shown === lastShown) return; // nothing new: keep the scroll position
      lastShown = shown;
      mount(outputEl, card(`Output: ${job.name}`, h("div", { class: "stack" },
        h("div", { class: "row tight" }, badge(fmt.label(job.status), STATUS_KIND[job.status] || ""), h("span", { class: "mono tiny" }, job.command), job.status === "running" ? cancel : null),
        pre)));
      pre.scrollTop = pre.scrollHeight;
    };

    // Built once: the periodic refresh must not reset the filter while you type.
    let reports = { root: "", reports: [] };
    const filter = h("input", { type: "search", placeholder: "filter", value: reportFilter });
    const list = h("div");
    const filesSub = h("span", { class: "muted small" });
    filter.addEventListener("input", () => { reportFilter = filter.value; draw(); });
    mount(reportsEl, h("section", { class: "card" },
      h("div", { class: "card-head" }, h("h3", {}, "Files"), filesSub),
      h("div", { class: "card-body" }, h("div", { class: "stack" }, filter, list))));
    function draw() {
        const rows = reports.reports.filter((r) => r.path.toLowerCase().includes(reportFilter.toLowerCase()));
        mount(list, table(
          [
            { label: "file", render: (r) => h("span", { class: "mono small" }, r.path) },
            { label: "size", num: true, render: (r) => fmt.bytes(r.bytes) },
            { label: "modified", render: (r) => fmt.ago(r.modified) },
          ],
          rows,
          { onRow: (r) => openReport(r.path), emptyText: "No reports match." },
        ));
    }
    const refreshReports = async () => {
      const { data } = await as.admin("GET", "/v1/dev/reports", { quiet: true });
      reports = data;
      filesSub.textContent = data.root;
      draw();
    };

    const openReport = async (path) => {
      try {
        const { data } = await as.admin("GET", `/v1/dev/reports/${path.split("/").map(encodeURIComponent).join("/")}`);
        let body;
        if (data.kind === "json") {
          try {
            body = json(JSON.parse(data.content));
          } catch {
            body = h("pre", { class: "code" }, data.content);
          }
        } else if (data.kind === "md") {
          body = markdown(data.content);
        } else {
          body = h("pre", { class: "code" }, data.content);
        }
        mount(viewerEl, card(path, body, { actions: h("button", { class: "btn ghost small", type: "button", on: { click: () => mount(viewerEl, "") } }, "Close") }));
      } catch (err) {
        errorToast(err);
      }
    };

    const first = await refreshJobs().catch((err) => {
      mount(jobsEl, callout(err.message, "danger"));
      return null;
    });
    if (first) renderCatalog(first.catalog);
    await refreshReports();
    mount(viewerEl, card("Viewer", empty("Pick a report. Try research/scaling.md, research/obd_ope.md, or review_compare.json.")));
    scope.every(1500, refreshJobs, { immediate: false });
    scope.every(10000, refreshReports, { immediate: false });
  },
};
