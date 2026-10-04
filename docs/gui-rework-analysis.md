# Cascade GUI — analysis before the rework (2026-10-04)

Scope: `src/web/dashboard.html`, `src/web/settings.html`, `src/web/mod.rs` and the
backend they call (`src/handlers.rs`, `src/router.rs`, `src/registry.rs`,
`src/state.rs`, `src/config.rs`, `src/cascade_features.rs`), plus the data sources
the UI *does not* reach today (gpu-provisioner, stack registry, `cascade.db`).

Goal of the rework: make the gateway UI the **single place** where provisioned
GPUs and external API providers are managed, because that is where those nodes
actually land in the routing registry. Reference point: Bifrost LLM gateway
(provider management · key management · routing rules · health · observability in
one UI).

---

## 1. What the GUI is today

Two pages, both hand-written, both shipped as `include_str!` blobs.

### 1.1 `dashboard.html` — `/web/` (and `/`, same document)

| Region | Data source | `fetch()` | Notes |
|---|---|---|---|
| KPI cards (Total Requests, Uptime, Providers, Fallbacks, Session Cache, Multimodal) | `handlers::dashboard_api` → prometheus counters + `state` | `GET api/dashboard` | See defect **D1** |
| Bar + doughnut chart | same payload (`requests_by_backend`) | (same call) | Chart.js **from a CDN** — defect **D2** |
| "Inference Nodes" grid | `handlers::admin_upstreams` (gateway registry only) | `GET api/v1/admin/upstreams` | See **D3** |
| admin key field | `localStorage['cascade_admin_key']` | — | admin key injected by Caddy, so the field is only for the raw LAN port — **D4** |
| activate/deactivate toggle | `handlers::{activate_node,deactivate_node}` | `POST api/v1/inference-nodes/{id}/{activate,deactivate}` | |
| remove node | `handlers::delete_upstream` | `DELETE api/upstreams/{role}?id=` | |
| register node form | `handlers::put_upstream` | `PUT api/upstreams/{role}` | |
| "Probe now" | `handlers::probe_upstreams` | `POST api/v1/admin/upstreams/probe` | |
| "Refresh Models" | `handlers::refresh_models_upstreams` | `POST api/v1/admin/upstreams/refresh-models` | uses the implicit global `event` — **D5** |
| Image generation quick test | `media::image_generation_handler` | `POST v1/images/generations` | **404 in production** — **D6** |
| "Configured Providers" table | `handlers::list_providers` (runtime providers) | `GET api/providers` | read-only |
| "Local Models" table | `handlers::list_models` | `GET models` (= `/web/models`) | |
| nav → `metrics` | `cascade_features::metrics_handler` | raw Prometheus text in a new tab | |

### 1.2 `settings.html` — `/web/settings`

| Region | Data source | `fetch()` |
|---|---|---|
| Live Status (active main/aux model) | `handlers::active_models` → probes each node's `/v1/models` on **every call** | `GET api/active-models` |
| routing thresholds / default route / hybrid / marker mode | `handlers::{get_settings,update_settings}` | `GET`+`PUT api/settings` |
| chat upstream overrides | `update_settings` → registers a `manual-{role}` registry node | `PUT api/settings` |
| audio endpoints (TTS/STT) | runtime settings | `PUT api/settings` |
| provider list + add provider | `handlers::{list_providers,add_provider,delete_provider}` | `GET/POST api/providers`, `DELETE api/providers/{id}` |
| link to `docs/routing-settings.md` | `handlers::docs_page` | `GET /web/docs/<name>` |

### 1.3 Backend route inventory (`src/router.rs`) the GUI could use but does not

* `GET /extraction/v1/backends` — extraction backend pool (weights, health,
  masked keys). **Never shown.**
* `GET /health` — `extraction_backends`, reachability, uptime. **Never shown.**
* `GET /metrics` — Prometheus. **Only as a raw text tab.**
* `registry.strategy()` (`round_robin` / `least_latency`) is *displayed* in the
  dashboard (`id="node-strategy"`) and **cannot be changed from anywhere** —
  **D7**.
* No endpoint at all for: provider update, provider enable/disable, provider key
  rotation, provider test without re-adding, gpu-provisioner anything.

### 1.4 Data sources not reachable from the gateway at all today

| Source | Endpoint | Used by the GUI? |
|---|---|---|
| GPU provisioner | `GET /api/v1/health`, `/instances`, `/inference-nodes`, `/budget`, `/costs`, `/providers`, `/settings`, `/flavors`, `/model-catalog`; `POST /inference-nodes`, `/inference-nodes/pollinations`, `/inference-nodes/{id}/{activate,deactivate,terminate}`, `/cloud-toggle`, `/reconcile`, `PUT /budget` | **no** |
| Stack LLM registry | `config/llm-registry.yaml` (read by `scripts/llm-registry.sh` / `stack-health-api`) | **no** (only reachable through the runtime `ProviderConfig` list) |
| Persisted state | `data/cascade-llm/cascade.db` (settings + providers, `db.rs`) | indirectly |

Verified live (2026-10-04, container `netai-cascade-llm: hybrid`, image
`netai/cascade-llm:hybrid`):

```
GET :3006/api/v1/admin/upstreams  -> {"nodes": [], "active_roles": {...all false},
                                     "defaults": {"main": "http://netai-vllm-main:18000/v1/chat/completions", ...}}
GET :8091/api/v1/inference-nodes  -> 6 nodes, all "terminated"
GET :8091/api/v1/budget           -> weekly_cap 10.0, spend 0.2909, remaining 9.7091
GET :8091/api/v1/costs            -> daily[] + provider_monthly{"vastai":0.2909}, 180 records
GET :8091/api/v1/health           -> billed_usd 4.4483, cloud_enabled true, max_instances 3
```

So the deployed GPU-provisioner surface (6 inference-node records, 180 cost
records, a budget, a cloud toggle, a backoff map) is **entirely invisible** in the
gateway UI. The user must switch to a second dashboard at `/gpu-provisioner/`.

---

## 2. Concrete defects and gaps (with evidence)

### D1 — Total Requests and both charts silently undercount (server-side)

`handlers::dashboard_api` sums a hardcoded list:

```rust
let known_backends = ["small", "large", "large_multimodal", "session_affinity",
                      "stt_proxy", "ocr", "auxiliary", "inference"];
```

`cascade_features::MetricsRegistry::init` pre-initialises a **different** set:

```rust
for backend in &["small", "large", "large_multimodal", "large_text", "fallback",
                 "ocr", "auxiliary", "inference", "main", "image", "rag_worker",
                 "extraction"] { requests_total.with_label_values(&[backend]); }
```

Consequences, both directions:
* `main`, `image`, `rag_worker`, `extraction`, `large_text` and `fallback` are
  counted by the router but **never summed** into `requests_total`, so the KPI is
  too low and those backends never appear in the charts.
* `session_affinity` and `stt_proxy` are read but never initialised, so they are
  always 0 — two permanently dead legend entries.

Fix: derive the label set from the registry instead of hardcoding it twice.

### D2 — Chart.js is loaded from a CDN (violates gotcha 13)

`dashboard.html:10`
```html
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
```
The stack is offline-by-requirement ("Vendor all frontend assets locally (no
CDN)"). On an air-gapped install the two canvases stay blank with **no error
message** — `initCharts()` throws on the undefined `Chart` global and the two
`chart-box` cards render as empty white/dark rectangles.

Fix: no third-party chart library at all — the two charts are hand-rolled inline
SVG (~60 lines), which is smaller, themeable and cannot 404.

### D3 — "Inference Nodes" shows only what already reached the registry

The grid is built from `GET api/v1/admin/upstreams`, i.e. the gateway's
**in-memory** registry. A GPU that is `provisioning`, `booting`, `stopped`,
`terminating` or `terminate_failed` — every state that actually costs money —
does not exist yet in that list, and the registry forgets everything on a gateway
restart (it is only re-seeded by the provisioner's `reconcile_once`). The
dashboard therefore looks empty ("No dynamic nodes registered") while the
provisioner holds six node records and a live budget. There is no cost roll-up,
no terminate action and no budget anywhere in the gateway UI.

### D4 — Admin-key UX implies the key is required in production

`dashboard.html` opens with a password field for `x-cascade-admin-key`. Caddy
injects it server-side (`header_up X-Cascade-Admin-Key {$CASCADE_ADMIN_KEY}` in
all three `/cascade/*` blocks) and `auth-validator` 403s non-admins first, so in
the real deployment the field is dead weight and a user who types a wrong key
makes **every** write fail with a message about the key they were never supposed
to need. The UI has no notion of "admin session" vs "anonymous read-only".

### D5 — `refreshModels()` reads the implicit global `event`

```js
async function refreshModels() {
  const btn = event.target;   // global `event`, works only via inline onclick
```
It also `alert()`s results. Breaks under CSP, in strict mode, and whenever the
handler is called from anywhere but an inline `onclick`; the reference to the
live `/web/settings` page's own toast/UI conventions is inconsistent with the
dashboard's `alert()`s.

### D6 — Image quick-test 404s in production (same class as the `/models` bug)

`dashboard.html:538` → `fetch('v1/images/generations')`. Caddy serves the UI via

```
handle_path /cascade/* { rewrite * /web{path} ; reverse_proxy netai-cascade-llm:3000 }
```

so a relative `v1/images/generations` resolves to `/web/v1/images/generations`.
There is no such route — `/v1/images/generations` is only mounted unprefixed
(`src/router.rs:66`) and reached through Caddy's separate `/cascade/v1/*` block.
The deployed image test therefore answers **HTTP 404** and renders
`HTTP 404: {"error":...}` under the form, which reads as "image generation is
broken" rather than "the UI built the wrong URL". Exactly the defect class of the
already-fixed `models` → `/v1/models` fetch (`dbe2119`/`609445e`).

The same relative-path fragility applies to every other `fetch('api/...')` in both
files: they only work because both `/` and `/web/` happen to end in a slash.

### D7 — Routing strategy is displayed but not settable

`registry.set_strategy()` exists and `RoutingStrategy::parse` handles
`round_robin|weighted|least_latency|latency`, but no HTTP route reaches it. The
dashboard shows `strategy: round-robin` as static text.

### D8 — Providers: no edit, no enable/disable, no key management, no test

`ProviderConfig` carries `enabled` and `api_key`, but:
* the only mutating routes are `POST /api/providers` (create) and
  `DELETE /api/providers/{id}`;
* `redact_provider` blanks `api_key` on **every** read, so the UI cannot even
  show whether a key is set — the settings page renders the provider name, type,
  URL, models and a Remove button, nothing else;
* there is no way to disable a provider without deleting it (and deleting it
  drops the registry node created at add time — `DELETE /api/providers/{id}`
  never removes that node, so the pool keeps a node whose provider is gone: **D8b**);
* `add_provider` runs a live chat probe, but there is no "test" for an existing
  provider.

### D9 — `/health`'s `providers` count is always 0

`handlers::health_check` reports `"providers": state.config.providers.len()`, and
`config.providers` is initialised to `Vec::new()` and never written
(`config.rs from_env_base`). The real list lives in `runtime.providers`. Any
consumer trusting `/health.providers` reads zero forever.

### D10 — Chat upstream overrides revert after Save

`update_settings` registers `manual-main` / `manual-auxiliary` registry nodes
when the submitted URL differs from the env default — but `get_settings` returns
`state.config.to_settings()`, i.e. the **env** values. After `saveSettings()` the
page reloads the form with the old URL, so the typed value disappears and the
node that was actually created is not shown anywhere. Clearing the field sends
`null` and removes nothing, so an override can be created but never undone
except by deleting the node by its generated id.

### D11 — Two hand-written pages, two copies of the chrome and the fetch layer

`nav`, `header`, `.card`, `.table`, `.btn` and the fetch/error handling are
duplicated across 566 + 410 lines; the two nav bars have drifted (the dashboard
has a `#nodes` anchor, the settings page does not), the settings page's "active"
cog links to `href=""` (a reload, not a control), and there is no shared
loading/error/empty vocabulary — some regions say "Loading…", some render an
empty tbody, one renders "Checking…" forever.

### D12 — Unescaped API data interpolated into `innerHTML` (both files)

`settings.html` builds provider rows, `active-models` cards and the toast with
template literals injected via `innerHTML`; `dashboard.html` does the same for
the model description and the `x-cascade-route` doc text. The stack hardened its
own UIs against exactly this (`tests/test_ui_xss_roles.py::test_no_innerhtml_with_api_data`),
and provider names / node labels / model ids are operator- and upstream-supplied
strings. `esc()` exists in the dashboard but is not applied to every
interpolation (e.g. `m.id`, `p.name` in `settings.html`).

### D13 — Polling storms with no cancellation, no back-off and no staleness marker

`setInterval(refresh, 5000)` + `setInterval(refreshNodes, 10000)` +
`setInterval(updateImageStatus, 10000)`. Three independent timers, each issuing
1–2 requests, none abortable, none aware that the previous run is still in
flight. After a gateway restart every card silently keeps its last value or `—`;
there is no "last updated" indicator anywhere, so a frozen dashboard is
indistinguishable from a quiet one.

### D14 — Observability is a raw Prometheus text dump

`nav → metrics` opens `/web/metrics` in a new tab: 20+ series of
`cascade_requests_total{selected_backend="…"}`, no aggregation, no error rates,
no latency, and no fallback-reason breakdown even though the reason label is
collected. The dashboard shows a *cumulative* fallback count with no rate.

### D15 — `/v1/models` mislabels and duplicates

Live `GET :3006/web/models` today:

```
netai-hybrid        (owned_by netai-stack, gateway identity)
bge-m3              (model_type "Main")
bge-reranker-v2-m3  (model_type "Main")
netai-aux           (model_type "Main")
netai-hybrid        (model_type "Main", OVMS)   <-- duplicate id
netai-hybrid-think  (model_type "Main")
```

* `netai-hybrid` appears **twice** (gateway identity + upstream discovery);
* the two embedding models are tagged `"Main"`, i.e. as if they were chat models
  of the main role. That is a consequence of `fetch_models_from` labelling
  everything found on the main origin — under OVMS the shim serves the chat and
  embedding graphs from one port, so the label is structurally undecidable from
  the origin alone. The honest fix is to stop claiming a capability and to report
  *where* a model was discovered (`discovered_via`), not to guess.

### D16 — The model list in the GUI is filtered by a hardcoded owner allow-list

```js
const localModels = models.data.filter(m => m.owned_by === 'netai-stack' ||
  m.owned_by === 'llamacpp' || m.owned_by === 'OVMS' || m.owned_by === 'vllm');
```
Any new engine tag (openarc, ggml, an upstream's own `owned_by`) silently drops
rows, and the "empty" state reads as "no local models configured". Same class of
bug as D15.

### D17 — Stale hardcoded model names in the description text

```js
m.id === 'cascade-hybrid-v1' ? 'Unified routing identity…'
m.id === 'docling-serve'     ? 'Document/OCR service for ingestion…'
```
`cascade-hybrid-v1` is the *pre-OVMS* router name (gotcha 40/68: the advertised
id must be a working served name, now `netai-hybrid`). The branch is dead and the
comment lies to whoever reads the table.

### D18 — `active-models` probes the network twice per page load

`role_active_status` calls `fetch_reported_model` → `GET <origin>/v1/models` for
main and again for auxiliary, on every request. The settings page triggers it on
load with no loading state, and the two probes are sequential.

---

## 3. What must survive the rework

* **Hybrid routing** (`CASCADE_HYBRID_AUTO`) and the `AUX_MODEL_RETARGET` map —
  `x-cascade-route` must still answer `auxiliary-server` for a short prompt and
  `inference-server` for a complex one (`tests/test_cascade_routes.py::
  TestCascadeChatRouting`).
* **`/v1/models` discovery** must keep advertising a *working* id (gotcha 40) and
  must not lose entries.
* **The admin-key flow**: Caddy strips any client-supplied `X-Cascade-Admin-Key`
  and injects the real one after `forward_auth`; the raw loopback port :3006 still
  requires the key. The UI must therefore treat "admin" as ambient (never ask for
  the key on the Caddy path) and only ask when a call actually comes back 403.
* **`x-cascade-route` hints** (`inference-server`, `auxiliary-server`,
  `auto-small`, `auto-large`, `session-affinity`, `ocr`) — gotcha 25.
* **Direct-binding exceptions** (gotcha 32) must stay visible in the UI, because
  "cascade offers every route" is not the same as "everything uses it".

---

## 4. Plan

### 4.1 Backend (Rust)

1. `src/provisioner.rs` — new. Allowlisted, admin-guarded **server-side**
   passthrough to `http://netai-gpu-provisioner:8091` (config:
   `GPU_PROVISIONER_URL`, `GPU_PROVISIONER_API_KEY` falling back to
   `KNOWLEDGE_API_KEY`). Route: `GET|POST|DELETE /web/api/provisioner/{allowlisted-path}`.
   Only the exact paths the UI needs are accepted; anything else is 404. The
   browser therefore never sees the provisioner API key, and the admin key stays
   the single credential (Caddy → cascade).
2. Provider management completes `add`/`delete`: `PUT /web/api/providers/{id}`
   (name, base_url, models, use_as, enabled, api_key rotation/clear),
   `POST /web/api/providers/{id}/test`, and `api_key` handling that re-registers
   or withdraws the registry node — closing **D8/D8b**.
3. `list_providers` reports `has_key` + `key_masked` (last 4, same convention as
   `mask_bearer`/`list_extraction_backends`).
4. `PUT /web/api/routing-strategy` — closes **D7**.
5. `GET /web/api/overview` — one authoritative header payload (health,
   metrics, routing, registry summary, provider counts, provisioner reachability)
   so the dashboard needs one call and cannot show a half-initialised header.
6. `list_models` — dedupe by id and add `discovered_via` instead of pretending to
   know a capability — **D15**.
7. `dashboard_api` — derive the backend label set from the registry/active roles
   — **D1**.
8. `health_check` — report the real provider count — **D9**.
9. `get_settings` — return the *runtime* chat URLs so overrides do not revert
   — **D10**.

### 4.2 Frontend

* **One page** (`src/web/app.html`) replacing `dashboard.html` +
  `settings.html`; `/web/settings` serves the same document (hash selects the
  section) so bookmarks keep working.
* A **derived API base** from `location.pathname` — the structural fix for D6 and
  the whole "relative fetch breaks under `/web/`" class.
* Left rail navigation with sections: **Overview · Backends · GPU fleet ·
  Providers · Models · Settings · Diagnostics**.
* A shared render vocabulary: `skeleton` → `data` → `empty` → `error`, with the
  error carrying the real HTTP status and body; one toast; no `alert()`.
* **No `innerHTML` with API data** — DOM construction only (`textContent`),
  matching `tests/test_ui_xss_roles.py`.
* **No CDN** — the theme CSS/JS and the logo come from the stack's own
  `/theme/*`; charts are inline SVG.
* One polling loop with a visible "last updated" stamp and an abort of the
  in-flight request — **D13**.

### 4.3 Stack-side

Only the deploy line is needed; no compose/Caddy change is required because the
provisioner is reached from inside the gateway and `/cascade/*` already forwards
`/api/v1/*` and `/web/*` to the gateway.

---

## 5. Known risks / not in scope

* Renting a GPU from the gateway UI is **not** added: the flavor/model-catalog
  selection flow (region rules, VRAM filters, `min_vram_gb` estimation) belongs to
  the provisioner's own dashboard. The gateway UI gets the *lifecycle* controls
  (activate/deactivate/terminate/reconcile/budget/cloud-toggle) and the hosted-API
  registration (Pollinations), plus a deep link into the provisioner UI.
* `admin_key` remains mandatory for all mutations; the UI cannot relax that.
* The registry is still in-memory: a gateway restart empties it until the
  provisioner's `reconcile_once` repopulates it. That is pre-existing and is now
  *visible* (the GPU-fleet section shows which provisioner records are active in
  the provisioner but absent from the registry).

---

## 6. Defects found *while* reworking — found by rendering and clicking, not by reading

Analysis of existing code finds the bugs you can see. Rendering the new console
against the live stack found five more, all of the same family: **a UI that
guesses a path, or a source of truth that differs from the server's, fails
silently and looks correct.**

### D19 — a relative fetch resolves one level too deep under Caddy

The new console was written with the correct fix for D6 (derive the API prefix
instead of assuming one). It was still 404ing in production, because the first
derivation was wrong:

```
page at /cascade/  →  PREFIX "/cascade"  →  "/cascade" + "/web/api/overview"
                    →  Caddy rewrites to "/web/web/api/overview"  →  404
```

Caddy serves the console with `handle_path /cascade/* { rewrite * /web{path} }`,
so the literal `/web` segment of a *gateway* path is **not** part of the public
URL. The working rule is one function over gateway paths:

```js
gw = (path) => PREFIX === "" ? path : PREFIX + path.replace(/^\/web/, "")
```

`/` → `/web/api/overview` · `/web/` → `/web/api/overview` ·
`/cascade/` → `/cascade/api/overview` · `/cascade/v1/chat/completions` → stays.

Verified against all five mounts a user can land on, through Caddy with a real
admin session. **This is the same defect class as D6** — which is why the fix is
structural (one function) and not another patched URL.

### D20 — two console reads could not reach their data *at all* through the proxy

`/health` and `/extraction/v1/backends` are root-level gateway paths. Caddy's
`/cascade/*` rewrite puts a request for them under `/web`, where no such route
exists, so both 404 in the browser while working on the raw port. The console
could not display extraction backends at all. Fixed with namespaced aliases
(`/web/api/health`, `/web/api/extraction-backends`) that the console uses; the
root paths are unchanged for every other consumer.

### D21 — the Remove button called a route that did not exist

The node table's delete used `DELETE /web/api/v1/upstreams/{role}` — the natural
sibling of the `/web/api/v1/inference-nodes/{id}/…` routes used for
activate/deactivate. That route was never registered. The click 404'd, the node
stayed in the pool, and the table kept rendering it: a UI that guesses a path
gets a **silent no-op**. Found by clicking the button, not by reading the code.
The admin namespace is now complete and pinned by a test that exercises every
mutation on both mounts.

### D22 — `active-models` reported an embedding model as the main chat model

```rust
.and_then(|a| a.first())   // data[0] of the upstream's /v1/models
```

One OVMS process serves the chat *and* the embedding graphs from one port, and
its list leads with `bge-m3`. The settings page therefore displayed
`main → bge-m3`, which is simply false. List order carries no meaning on a
multi-graph node; membership does. `fetch_reported_model` now takes the id the
role is configured (or the registered node declares) to serve and returns it when
the upstream lists it, falling back to the first entry only when it does not.

### D23 — a disabled provider came back into the pool on the next restart

`update_provider` correctly withdrew the registry node, and the setting persisted
— but `main.rs`'s boot-time restore registered every persisted provider
**unconditionally**. So "Disable" applied until the next deploy, then silently
stopped applying. That is the worst possible shape for a disable control: the UI
and the DB both agree, and the pool disagrees. The boot path now calls the same
`sync_provider_node` helper as create/edit, so the relationship has exactly one
implementation.

### D24 — `model` is required, and its absence reported as "Invalid JSON"

The chat handler's deserialization error branch returned a flat
`"Invalid JSON"` for *every* failure. `ChatCompletionRequest::model` is not
optional, so the single most common client mistake — omitting `model` — produced
an error naming the wrong thing with no way to diagnose. The console's own
routing self-test hit it. The message now carries the serde error, and the
self-test defaults the model field to the gateway identity.

### D25 — a column answered from a cache nobody filled

The Providers view's "where does this sit in routing" column read
`cache.upstreams`, which only the Backends view populates. A user opening
Providers first therefore saw **"no registry node"** on a provider that was in the
pool — a false negative on exactly the question the screen exists to answer. The
registry is now fetched on demand, and every provider mutation invalidates both
views so one panel cannot say "disabled" while the other still shows it in the
pool.

### D26 — horizontal overflow clipped every panel

Grid items default to `min-width: auto`, so one long endpoint URL in the node
table pushed the whole page into a horizontal scroll and cut off the right-hand
column of *all* panels. Caught by measuring `scrollWidth` against
`clientWidth` at three widths rather than by looking at it. Fixed with
`min-width: 0` on the shell and panel grids; the tables scroll inside `.tw`.

---

## 7. Verification performed

* `cargo build --release` and `cargo test --release` — **100 tests, 0 failures**
  (17 new in `tests/console_api.rs`, 2 new unit tests in `handlers.rs`), each
  named after the defect it pins.
* Deployed **only** through the stack wrapper:
  `scripts/compose-up.sh up -d --no-build --force-recreate llm-gateway`
  (gotcha 22 — never raw `docker compose`).
* All five console mounts exercised through Caddy with a real admin session:
  `/cascade/`, `/cascade#<section>`, `/cascade/settings`, `/cascade/metrics`,
  `/cascade/docs/*`, plus every API path the UI calls. **Zero "Could not load"
  states, zero console errors, zero horizontal overflow** at 800/1440/1920 px.
* Interactive paths driven in a real browser: node register → inline weight edit
  → strategy change → deactivate → remove → empty state; provider
  create → test → disable → enable → remove, including the registry node
  appearing and disappearing; provisioner node-log modal; reconcile.
* **Hybrid routing unchanged**: a short prompt returns
  `x-cascade-route: auxiliary-server` with `model: netai-aux` (the retarget), a
  long Eurocode prompt returns `x-cascade-route: inference-server` with
  `model: netai-hybrid`.
* `/web/api/models` returns the five OVMS models plus the gateway identity, with
  the duplicate `netai-hybrid` entry gone.

## 8. What remains

* **Renting a GPU stays in the provisioner UI.** Flavor selection, region rules
  and VRAM estimation are that product's surface; the gateway links to it.
* **The GPU-rules panel is read-only here** by design — it is editable in the
  provisioner UI, and two writable copies of the same rules would drift.
* **The registry is still in-memory.** A gateway restart empties it until
  `reconcile_once` repopulates it. Now *visible* (the fleet view shows provisioner
  records that are absent from the registry) rather than fixed; persisting the
  pool is a routing-layer change, not a UI one.
* **`x-cascade-route` self-test latency** is the gateway's own overhead, not the
  model's — the number shown is wall clock for a full round trip.
* Provider **key rotation** is supported, but there is no key *history*: the old
  value is overwritten and never recoverable by design.

