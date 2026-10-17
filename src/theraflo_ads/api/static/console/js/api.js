// API client for the console. Every call goes through `request`, which records it for the
// "API calls" drawer (method, path, role, status, timing, bodies), never the credentials.

export class ApiError extends Error {
  constructor(status, detail, data) {
    super(detail || `HTTP ${status}`);
    this.status = status;
    this.data = data;
  }
}

const listeners = new Set();
export const calls = [];
const MAX_CALLS = 80;

export function onCall(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function record(entry) {
  if (entry.quiet) return; // polling: kept out of the drawer
  calls.unshift(entry);
  if (calls.length > MAX_CALLS) calls.length = MAX_CALLS;
  for (const listener of listeners) listener(entry);
}

function detailOf(data) {
  if (!data) return null;
  if (typeof data === "string") return data.slice(0, 300);
  if (typeof data.detail === "string") return data.detail;
  if (Array.isArray(data.detail)) {
    return data.detail.map((d) => `${(d.loc || []).slice(1).join(".")}: ${d.msg}`).join("; ");
  }
  if (data.status) return String(data.status);
  return null;
}

export async function request(method, path, { body, query, role = "public", auth = {}, redirect, quiet = false } = {}) {
  const headers = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (auth.bearer) {
    headers.Authorization = `Bearer ${auth.bearer}`;
    // The same token again, for proxies that consume Authorization (the Kubernetes API
    // server's, in LocalStack's Container Apps). The API reads it only when started with
    // THERAFLO_ADS_AUTH_FALLBACK_HEADER on.
    headers["X-Theraflo-Authorization"] = headers.Authorization;
  }
  if (auth.advertiserKey) headers["X-Advertiser-Key"] = auth.advertiserKey;
  let url = path;
  if (query) {
    const params = new URLSearchParams();
    for (const [k, v] of Object.entries(query)) if (v !== undefined && v !== null && v !== "") params.set(k, v);
    const qs = params.toString();
    if (qs) url += `?${qs}`;
  }
  const started = performance.now();
  const entry = { method, path: url, role, quiet, at: new Date(), request: body, status: null, ms: null, response: null };
  let response;
  try {
    response = await fetch(url, {
      method,
      headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      redirect: redirect || "follow",
      cache: "no-store",
    });
  } catch (err) {
    entry.status = "network";
    entry.ms = performance.now() - started;
    entry.response = String(err);
    record(entry);
    throw new ApiError(0, `network error: ${err.message}`);
  }
  entry.ms = performance.now() - started;
  if (response.type === "opaqueredirect") {
    entry.status = 302;
    entry.response = "(redirect to the landing page, not followed)";
    record(entry);
    return { status: 302, data: null, redirected: true };
  }
  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  entry.status = response.status;
  entry.response = data;
  record(entry);
  if (!response.ok) throw new ApiError(response.status, detailOf(data) || response.statusText, data);
  return { status: response.status, data, headers: response.headers };
}

// ----- identities -----

export const ids = {
  boot: null, // GET /v1/dev/identities
  get admin() { return { bearer: this.boot?.admin_token }; },
  get service() { return { bearer: this.boot?.service_token }; },
};

export async function loadIdentities() {
  const { data } = await request("GET", "/v1/dev/identities", { role: "console" });
  ids.boot = data;
  return data;
}

const TEST_USERS_KEY = "theraflo.console.testUsers";

export function testUsers() {
  try {
    return JSON.parse(localStorage.getItem(TEST_USERS_KEY) || "[]");
  } catch {
    return [];
  }
}

export function saveTestUser(user) {
  const users = testUsers().filter((u) => u.user_id !== user.user_id);
  users.unshift(user);
  try {
    localStorage.setItem(TEST_USERS_KEY, JSON.stringify(users.slice(0, 40)));
  } catch {
    /* storage unavailable: the user lasts for this page only */
  }
}

export function forgetTestUser(userId) {
  try {
    localStorage.setItem(TEST_USERS_KEY, JSON.stringify(testUsers().filter((u) => u.user_id !== userId)));
  } catch {
    /* ignore */
  }
}

/** Every user the console can act as: built-in dev users, then test users made here. */
export function allUsers() {
  const builtIn = (ids.boot?.users || []).map((u) => ({ ...u, kind: "dev" }));
  const made = testUsers().map((u) => ({ ...u, kind: "test" }));
  return [...builtIn, ...made];
}

export function userToken(userId) {
  return allUsers().find((u) => u.user_id === userId)?.token;
}

export function advertiserKey(advertiserId) {
  return (ids.boot?.advertisers || []).find((a) => a.advertiser_id === advertiserId)?.key;
}

// ----- role helpers -----

export const as = {
  admin: (method, path, opts = {}) => request(method, path, { ...opts, role: "admin", auth: ids.admin }),
  service: (method, path, opts = {}) => request(method, path, { ...opts, role: "service", auth: ids.service }),
  user: (userId, method, path, opts = {}) =>
    request(method, path, { ...opts, role: `user:${userId}`, auth: { bearer: userToken(userId) } }),
  advertiser: (advertiserId, method, path, opts = {}) =>
    request(method, path, { ...opts, role: `adv:${advertiserId}`, auth: { advertiserKey: advertiserKey(advertiserId) } }),
  public: (method, path, opts = {}) => request(method, path, { ...opts, role: "public" }),
};
