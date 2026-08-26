# Comprehensive Refactoring & Feature Specification: Dynamic Control API, Multi-Modal Routing & Admin GUI Extensions for Cascade LLM

We need to upgrade the `cascade-llm` infrastructure. This update expands routing for specialized workloads (OCR, Image Generation, LightRAG), introduces dedicated administrative control REST APIs, and extends the user interface to manage dynamic upstreams and cloud GPU nodes (e.g., Vast.ai, RunPod, SkyPilot).

In parallel you need to work on /home/edgar/NetAI-Stack-SE project, where cascade is a central piece of. Parts of this refactoring are located there. (GPU provisioner, docker compose endpoints, config file endpoints) some configuration variables might need to be relocated to a cascade_config.yaml or .env file where it makes sense. netAI-Stack-SE will only have cascade llm as one central endpoint for everything from this time on. Part of this session is also to analyze if this makes sense in every place or not. you can decide. at the end of this round you need to deliver a working solution and NetAI-Stack-SE test suite needs to run through.
---

### 1. Dynamic Upstream & Provisioner Control APIs (Backend Layer)

Implement a dynamic, thread-safe `UpstreamRegistry` (e.g., `Arc<RwLock<Registry>>`) inside `cascade-llm` to support real-time node management without restarting the gateway.

#### Required Control Endpoints & Auth
All administrative endpoints must be protected by header validation (`x-cascade-admin-key: <ADMIN_KEY>`).

- **`PUT /web/api/upstreams/{role}`**
  - **Purpose:** Registers or updates an active backend endpoint for a designated role (`main`, `auxiliary`, `ocr`, `image`, `rag_worker`).
  - **Payload:**
    ```json
    {
      "endpoint_url": "[http://10.0.0.45:8080/v1/chat/completions](http://10.0.0.45:8080/v1/chat/completions)",
      "bearer_token": "optional-node-auth-token",
      "weight": 1,
      "max_context_length": 131072
    }
    ```
- **`DELETE /web/api/upstreams/{role}`**
  - **Purpose:** Instantly deregisters a backend for a given role (e.g., when an instance is terminated or idling).

- **`POST /api/v1/inference-nodes/{id}/activate` & `/deactivate`**
  - **Purpose:** Explicitly toggles an inference node in/out of the active routing pool after health checks or weight loading completes.

- **`GET /api/v1/admin/upstreams`**
  - **Purpose:** Returns current pool status, node health, active roles, VRAM metrics, and latency stats for all managed nodes.

---

### 2. Multi-Modal Payload Inspection & Routing Rules

Update the core proxy pipeline (`POST /v1/chat/completions`, `POST /v1/images/generations`, `POST /v1/rag/extract`) to forward traffic dynamically based on the active node in the `UpstreamRegistry`:

- **Route A (Image Generation):** Route text generation prompts to the active `image` endpoint (e.g., SDXL / Flux service).
- **Route B (Vision / OCR):** Detect base64 image data or document extraction flags in chat messages and route to the active `ocr` vision model.
- **Route C (Context / Reasoning):** Forward standard context or agent scratchpad tasks to `auxiliary` or `main` inference nodes depending on context depth.
- **Route D (Heavy RAG Extraction):** Direct LightRAG graph extraction jobs to backends assigned to the `rag_worker` role (including temporary cloud GPU instances).

---

### 3. GUI & Admin Control Interface Extension

Extend the frontend (LibreChat extension / React Admin Panel / Gradio Management View) to expose an interactive control center for administrators and users:

#### A. Real-Time Node Status Dashboard
- **Active Node Grid:** Display all connected backends (`main`, `auxiliary`, `ocr`, `image`, `rag_worker`), showing their endpoint IP, assigned model name, health status, and active role.
- **VRAM & Hardware Monitoring:** Show real-time VRAM usage and context window consumption for local GPUs (e.g., Intel Arc Pro Battlemage) and remote instances.
- **Manual Node Overrides:** Toggle switches allowing admins to manually activate, deactivate, or hot-swap backends for specific roles.

#### B. Dynamic Cloud GPU Provisioning Controls 
- **On-Demand Triggers:** Interface controls to trigger, monitor, or terminate cloud GPU instances (Vast.ai, RunPod, SkyPilot) directly from the UI when launching heavy LightRAG extraction jobs.
- **Cost & Runtime Badges:** Show active runtime and estimated hourly costs for provisioned cloud instances.

#### C. Routing & Multi-Modal Transparency
- **Routing Indicator:** Display subtle visual metadata on chat responses showing which active model/backend processed the request (e.g., "Processed by Ornith-9B" or "OCR via LFM2.5-VL").
- **Task Preview Panel:** Provide status views for asynchronous background jobs such as document extraction pipelines or image generation queues.

---

### 4. Health Checks, Fallbacks & Lifecycle Integration

- **Automated Health Checks:** Periodically probe all registered upstreams (`GET /health`). Automatically demote or deactivate failing nodes.
- **Graceful Fallbacks:** If a specialized node (`ocr` or `rag_worker`) is offline or undergoing provisioning, notify the UI or fall back to the `main` node when capable.
- **Zero-Downtime Hot Swapping:** Ensure backend switches (e.g., shifting main models or starting cloud workers) occur seamlessly without dropping active user chat sessions.

---

### Deliverables Required

1. **Rust Gateway Refactoring (`cascade-llm`):**
   - Implemented REST endpoints for dynamic upstream management and administrative monitoring.
   - Dynamic routing engine with context/payload inspection.
2. **GUI / Admin Frontend Updates:**
   - React / Web UI components for node status, manual role toggles, hardware telemetry, and cloud instance management.
3. **Configuration & Docs:**
   - Updated `docker-compose.yml` exposing administrative keys and frontend routes.
   - API & UI documentation for registering backends and managing cloud compute nodes.
4. **NetAI-Stack-SE update:**
   - /home/edgar/NetAI-Stack-SE/ needs to be updated too. the router is a central component and all endpoints in the project and docker-compose etc. will route through cascade llm from now on. 

---

### Development Rules

- **clean code:** don't over complicate and write readable code.
- **documentation:** all the features and structural diagrams need to end up in Agents.md, but keep it as compact as possible. more detailed documentation goes to docs/. delivered code change rounds need to be documented by every subagent too.
- **performance and scalability:** keep in mind that this router could possibly manage hundreds of requests / minute, so it has to be multithreaded and need to have enough scalability to work for huge enterprises or cloud installations.
- **subagents:** use subagents as often aspossible for sub tasks, research etc. if explore agents fail use general agents. 
- **branch:** feature/cascade_central is the current branch on both cascade_llm and NetAI-Stack-SE. this should be the home for this big refactoring and it needs to be worked on in parallel, so commits have to happen in both repositories
