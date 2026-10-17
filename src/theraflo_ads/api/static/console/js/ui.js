// DOM helpers. Server data is only ever inserted as text (textContent), never as HTML:
// creatives are advertiser-supplied and untrusted, and reports render through `markdown`,
// which builds elements itself.

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "text") el.textContent = value;
    else if (key === "on") for (const [event, fn] of Object.entries(value)) el.addEventListener(event, fn);
    else if (key === "style") Object.assign(el.style, value);
    else if (key === "dataset") Object.assign(el.dataset, value);
    else if (key in el && typeof value !== "string") el[key] = value;
    else el.setAttribute(key, value === true ? "" : String(value));
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

export function mount(el, ...children) {
  el.replaceChildren();
  append(el, children);
  return el;
}

export function svg(tag, attrs = {}, ...children) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  for (const c of children) if (c) el.append(c);
  return el;
}

// ----- formatting -----

export const fmt = {
  num(v, digits = 0) {
    if (v === null || v === undefined || Number.isNaN(v)) return "–";
    return Number(v).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
  },
  pct(v, digits = 1) {
    if (v === null || v === undefined || Number.isNaN(v)) return "–";
    return `${(v * 100).toFixed(digits)}%`;
  },
  prob(v) {
    if (v === null || v === undefined) return "–";
    if (v === 0) return "0";
    if (v < 0.001) return v.toExponential(1);
    return v.toFixed(3);
  },
  money(v, digits = 2) {
    if (v === null || v === undefined || Number.isNaN(v)) return "–";
    return `$${Number(v).toFixed(digits)}`;
  },
  ms(v) {
    if (v === null || v === undefined) return "–";
    return v < 10 ? `${v.toFixed(2)} ms` : `${v.toFixed(0)} ms`;
  },
  time(iso) {
    if (!iso) return "–";
    const d = new Date(iso);
    return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  },
  datetime(iso) {
    if (!iso) return "–";
    return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
  },
  ago(iso) {
    if (!iso) return "–";
    const s = Math.round((Date.now() - new Date(iso).getTime()) / 1000);
    if (s < 5) return "just now";
    if (s < 60) return `${s}s ago`;
    if (s < 3600) return `${Math.round(s / 60)}m ago`;
    if (s < 86400) return `${Math.round(s / 3600)}h ago`;
    return `${Math.round(s / 86400)}d ago`;
  },
  bytes(n) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / 1024 / 1024).toFixed(1)} MB`;
  },
  label(s) {
    return String(s ?? "").replaceAll("_", " ");
  },
};

// ----- small components -----

export function badge(text, kind = "", title) {
  return h("span", { class: `badge ${kind}`, title }, text);
}

export function dot(kind) {
  return h("span", { class: `dot ${kind}` });
}

export function card(title, body, { actions, foot, cls = "", sub } = {}) {
  return h(
    "section",
    { class: `card ${cls}` },
    title !== null
      ? h("div", { class: "card-head" }, h("h3", {}, title, sub ? h("span", { class: "muted small" }, ` ${sub}`) : null), actions || null)
      : null,
    h("div", { class: "card-body" }, body),
    foot ? h("div", { class: "card-foot" }, foot) : null,
  );
}

export function stat(label, value, sub, { small = false } = {}) {
  return h(
    "div",
    { class: "stat" },
    h("span", { class: "label" }, label),
    h("span", { class: `value ${small ? "small" : ""}` }, value),
    sub ? h("span", { class: "sub" }, sub) : null,
  );
}

export function kv(pairs) {
  const dl = h("dl", { class: "kv" });
  for (const [k, v] of pairs) {
    if (v === undefined) continue;
    dl.append(h("dt", {}, k), h("dd", {}, v === null ? "–" : v));
  }
  return dl;
}

export function callout(content, kind = "") {
  return h("div", { class: `callout ${kind}` }, content);
}

export function empty(text) {
  return h("div", { class: "empty" }, text);
}

/** columns: [{key, label, render?(row), num?, cls?}] */
export function table(columns, rows, { onRow, rowClass, emptyText = "Nothing yet." } = {}) {
  if (!rows.length) return empty(emptyText);
  const thead = h("thead", {}, h("tr", {}, columns.map((c) => h("th", { class: c.num ? "num" : "" }, c.label))));
  const tbody = h("tbody");
  for (const row of rows) {
    const tr = h("tr", { class: [onRow ? "clickable" : "", rowClass ? rowClass(row) : ""].join(" ") });
    if (onRow) tr.addEventListener("click", () => onRow(row, tr));
    for (const c of columns) {
      const value = c.render ? c.render(row) : row[c.key];
      tr.append(h("td", { class: [c.num ? "num" : "", c.cls || ""].join(" ") }, value ?? "–"));
    }
    tbody.append(tr);
  }
  return h("div", { class: "table-wrap" }, h("table", { class: "data" }, thead, tbody));
}

/** items: [{label, value, display?, highlight?, danger?, title?}] */
export function bars(items, { max, emptyText = "No data yet." } = {}) {
  if (!items.length) return empty(emptyText);
  const top = max ?? Math.max(...items.map((i) => i.value), 0);
  return h(
    "div",
    { class: "bars" },
    items.map((item) =>
      h(
        "div",
        { class: `bar-row ${item.highlight ? "highlight" : ""} ${item.danger ? "danger" : ""}`, title: item.title || "" },
        h("span", { class: "bar-label", title: item.label }, item.label),
        h("span", { class: "bar-track" }, h("span", { class: "bar-fill", style: { width: `${top > 0 ? Math.max((item.value / top) * 100, item.value > 0 ? 1.5 : 0) : 0}%`, display: "block" } })),
        h("span", { class: "bar-value" }, item.display ?? fmt.num(item.value)),
      ),
    ),
  );
}

export function meter(value, max, kind = "") {
  const share = max > 0 ? Math.min(Math.max(value / max, 0), 1) : 0;
  return h("div", { class: `meter ${kind}`, title: `${fmt.num(value, 2)} / ${fmt.num(max, 2)}` }, h("div", { style: { width: `${share * 100}%` } }));
}

export function sparkline(values, { height = 44, width = 300 } = {}) {
  const el = svg("svg", { class: "spark", viewBox: `0 0 ${width} ${height}`, preserveAspectRatio: "none" });
  if (values.length < 2) return el;
  const top = Math.max(...values, 1e-9);
  const step = width / (values.length - 1);
  const points = values.map((v, i) => [i * step, height - 3 - (v / top) * (height - 8)]);
  const line = points.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  el.append(svg("path", { class: "area", d: `${line} L${width},${height} L0,${height} Z` }));
  el.append(svg("path", { class: "line", d: line }));
  return el;
}

export function segmented(options, value, onChange) {
  const wrap = h("div", { class: "segmented", role: "group" });
  for (const opt of options) {
    const [val, label] = Array.isArray(opt) ? opt : [opt, opt];
    wrap.append(
      h("button", {
        type: "button",
        class: val === value ? "active" : "",
        "aria-pressed": String(val === value),
        on: {
          click: () => {
            for (const b of wrap.children) {
              b.classList.remove("active");
              b.setAttribute("aria-pressed", "false");
            }
            const btn = [...wrap.children].find((b) => b.textContent === label);
            btn?.classList.add("active");
            btn?.setAttribute("aria-pressed", "true");
            onChange(val);
          },
        },
      }, label),
    );
  }
  return wrap;
}

export function select(options, value, attrs = {}) {
  const el = h("select", attrs);
  for (const opt of options) {
    const [val, label] = Array.isArray(opt) ? opt : [opt, opt];
    el.append(h("option", { value: val, selected: val === value }, label));
  }
  if (value !== undefined) el.value = value;
  return el;
}

export function field(label, control, hint) {
  return h("label", { class: "field" }, label, control, hint ? h("span", { class: "hint" }, hint) : null);
}

export function json(value) {
  const pre = h("pre", { class: "code" });
  const text = JSON.stringify(value, null, 2) ?? "null";
  // Token-colored, still text nodes only.
  const re = /("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false|null)\b|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g;
  let last = 0;
  for (const m of text.matchAll(re)) {
    if (m.index > last) pre.append(text.slice(last, m.index));
    if (m[1]) pre.append(h("span", { class: m[2] ? "json-key" : "json-str" }, m[1]), m[2] || "");
    else if (m[3]) pre.append(h("span", { class: "json-lit" }, m[3]));
    else pre.append(h("span", { class: "json-num" }, m[4]));
    last = m.index + m[0].length;
  }
  pre.append(text.slice(last));
  return pre;
}

export function disclosure(summary, content, open = false) {
  return h("details", { class: "disclosure", open }, h("summary", {}, summary), h("div", {}, content));
}

// ----- markdown (a safe subset, built as elements) -----

function inline(text) {
  const out = [];
  const re = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\[[^\]]+\]\((https?:\/\/[^)\s]+)\))|(\*[^*\s][^*]*\*)/g;
  let last = 0;
  for (const m of text.matchAll(re)) {
    if (m.index > last) out.push(text.slice(last, m.index));
    if (m[1]) out.push(h("code", {}, m[1].slice(1, -1)));
    else if (m[2]) out.push(h("strong", {}, m[2].slice(2, -2)));
    else if (m[3]) out.push(h("a", { href: m[4], target: "_blank", rel: "noopener noreferrer" }, m[3].slice(1, m[3].indexOf("]"))));
    else if (m[5]) out.push(h("em", {}, m[5].slice(1, -1)));
    last = m.index + m[0].length;
  }
  out.push(text.slice(last));
  return out;
}

export function markdown(source) {
  const root = h("div", { class: "markdown" });
  const lines = source.replaceAll("\r\n", "\n").split("\n");
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (line.startsWith("```")) {
      const body = [];
      i += 1;
      while (i < lines.length && !lines[i].startsWith("```")) body.push(lines[i++]);
      i += 1;
      root.append(h("pre", {}, body.join("\n")));
      continue;
    }
    const heading = /^(#{1,4})\s+(.*)$/.exec(line);
    if (heading) {
      root.append(h(`h${Math.min(heading[1].length, 3)}`, {}, inline(heading[2])));
      i += 1;
      continue;
    }
    if (line.trim().startsWith("|")) {
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) rows.push(lines[i++]);
      const cells = (r) => r.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const tableEl = h("table");
      const header = cells(rows[0]);
      tableEl.append(h("thead", {}, h("tr", {}, header.map((c) => h("th", {}, inline(c))))));
      const body = h("tbody");
      for (const r of rows.slice(1)) {
        if (/^\s*\|?\s*:?-{2,}/.test(r)) continue;
        body.append(h("tr", {}, cells(r).map((c) => h("td", {}, inline(c)))));
      }
      tableEl.append(body);
      root.append(h("div", { class: "table-wrap" }, tableEl));
      continue;
    }
    if (/^\s*[-*]\s+/.test(line)) {
      const ul = h("ul");
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) ul.append(h("li", {}, inline(lines[i++].replace(/^\s*[-*]\s+/, ""))));
      root.append(ul);
      continue;
    }
    if (/^\s*\d+\.\s+/.test(line)) {
      const ol = h("ol");
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) ol.append(h("li", {}, inline(lines[i++].replace(/^\s*\d+\.\s+/, ""))));
      root.append(ol);
      continue;
    }
    if (!line.trim()) {
      i += 1;
      continue;
    }
    const para = [];
    while (i < lines.length && lines[i].trim() && !/^(#{1,4}\s|```|\s*\||\s*[-*]\s|\s*\d+\.\s)/.test(lines[i])) para.push(lines[i++]);
    if (!para.length) {
      // A line that looks like the start of a block but did not parse as one.
      para.push(lines[i++]);
    }
    root.append(h("p", {}, inline(para.join(" "))));
  }
  return root;
}

// ----- toasts and dialogs -----

export function toast(message, kind = "") {
  const root = document.getElementById("toasts");
  const el = h("div", { class: `toast ${kind}` }, message);
  root.append(el);
  setTimeout(() => el.remove(), kind === "error" ? 7000 : 3800);
}

export function errorToast(err, prefix = "") {
  const status = err?.status ? `${err.status}: ` : "";
  toast(`${prefix}${status}${err?.message || err}`, "error");
}

export function modal(title, body, { actions = [], narrow = false, onClose } = {}) {
  const root = document.getElementById("modal-root");
  const close = () => {
    root.replaceChildren();
    document.removeEventListener("keydown", onKey);
    onClose?.();
  };
  const onKey = (e) => {
    if (e.key === "Escape") close();
  };
  document.addEventListener("keydown", onKey);
  const dialog = h(
    "div",
    { class: `modal ${narrow ? "narrow" : ""}`, role: "dialog", "aria-modal": "true", "aria-label": title },
    h("div", { class: "modal-head" }, h("h3", {}, title), h("button", { class: "btn ghost small", type: "button", on: { click: close } }, "Close")),
    h("div", { class: "modal-body" }, body),
    actions.length ? h("div", { class: "modal-foot" }, actions.map((a) => (typeof a === "function" ? a(close) : a))) : null,
  );
  const backdrop = h("div", { class: "modal-backdrop", on: { click: (e) => e.target === backdrop && close() } }, dialog);
  root.replaceChildren(backdrop);
  dialog.querySelector("input, textarea, select, .btn:not(.ghost)")?.focus();
  return close;
}

export function ask(title, label, { placeholder = "", initial = "", confirmText = "OK", danger = false, minLength = 0 } = {}) {
  return new Promise((resolve) => {
    let done = false;
    const input = h("input", { type: "text", value: initial, placeholder, style: { width: "100%" } });
    const finish = (value, close) => {
      done = true;
      close();
      resolve(value);
    };
    const submit = (close) => {
      if (input.value.trim().length < minLength) {
        input.focus();
        return;
      }
      finish(input.value.trim(), close);
    };
    const close = modal(title, h("div", { class: "stack" }, field(label, input)), {
      narrow: true,
      onClose: () => !done && resolve(null),
      actions: [
        (c) => h("button", { class: "btn secondary", type: "button", on: { click: () => finish(null, c) } }, "Cancel"),
        (c) => h("button", { class: `btn ${danger ? "danger" : ""}`, type: "button", on: { click: () => submit(c) } }, confirmText),
      ],
    });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") submit(close);
    });
  });
}

export function confirmDialog(title, message, { confirmText = "Confirm", danger = false } = {}) {
  return new Promise((resolve) => {
    let done = false;
    modal(title, h("p", {}, message), {
      narrow: true,
      onClose: () => !done && resolve(false),
      actions: [
        (close) => h("button", { class: "btn secondary", type: "button", on: { click: () => { done = true; close(); resolve(false); } } }, "Cancel"),
        (close) => h("button", { class: `btn ${danger ? "danger" : ""}`, type: "button", on: { click: () => { done = true; close(); resolve(true); } } }, confirmText),
      ],
    });
  });
}

// ----- page lifecycle -----

/** Timers and listeners that a page registers; all cleared when the page changes. */
export class Scope {
  constructor() {
    this.timers = [];
    this.cleanups = [];
    this.alive = true;
  }
  every(ms, fn, { immediate = true } = {}) {
    let busy = false;
    const tick = async () => {
      if (!this.alive || busy || document.hidden) return;
      busy = true;
      try {
        await fn();
      } catch (err) {
        console.warn("poll failed", err);
      } finally {
        busy = false;
      }
    };
    if (immediate) tick();
    this.timers.push(setInterval(tick, ms));
  }
  later(ms, fn) {
    this.timers.push(setTimeout(() => this.alive && fn(), ms));
  }
  onCleanup(fn) {
    this.cleanups.push(fn);
  }
  dispose() {
    this.alive = false;
    for (const t of this.timers) {
      clearInterval(t);
      clearTimeout(t);
    }
    for (const fn of this.cleanups) fn();
  }
}

export function busy(button, promise) {
  const label = button.textContent;
  button.disabled = true;
  button.textContent = "Working…";
  return Promise.resolve(promise).finally(() => {
    button.disabled = false;
    button.textContent = label;
  });
}

export function uid(prefix = "") {
  const bytes = new Uint8Array(5);
  crypto.getRandomValues(bytes);
  return prefix + [...bytes].map((b) => b.toString(16).padStart(2, "0")).join("");
}
