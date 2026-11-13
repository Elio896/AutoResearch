# EvoResearch

First milestone: a local CSV classification executor. No API keys or LLMs needed.

## Setup (PowerShell, only after dependency-install approval)

```powershell
cd C:\workspace\EvoResearch
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## CSV contract

One header row, one sample per row, numeric or categorical features, and an explicitly named target column. Target labels must be nonempty and contain at least two classes. Each class needs enough samples for stratified train/validation/test splits (60/20/20). Exclude IDs and any features derived from the outcome. Grouped or time-dependent data requires a different splitter and is not supported yet.

```powershell
.\.venv\Scripts\python.exe executor.py data\samples.csv --target label
```

Returns real validation macro-F1, accuracy, timing, dataset hash and split sizes. Supported models: logistic_regression, random_forest. Missing values and categorical encoding are fitted using training data only.

Use one `ResearchSession` object for all experiments in a research task. Choose the best experiment using validation results, then call `finalize(experiment_id)` once. It evaluates the held-out test set and closes the session. CLI `--finalize` is for a completed single-experiment task; do not repeatedly run it to tune against test results.

The standalone executor is in memory; use the persistent task workflow below for cross-process history and final-test isolation. No HTTP API, model export, or automatic coding-tool switching yet.

Verified on 2026-11-13: all 3 unittest checks passed in the project-local virtual environment. A real bundled Iris CSV smoke run (seed 42, 90/30/30 split) achieved validation macro-F1 0.9333 for both supported models and final test macro-F1 0.9333 for the selected logistic regression. Local results: artifacts/iris_smoke.json. This is an executor check, not an Agent benchmark.

## Development handoff

Both Codex and Cline must open this directory. Read AGENTS.md and HANDOFF.md before work; update HANDOFF.md after verified changes. DeepSeek in Cline is a separate paid API service and requires manual handoff from ChatGPT.

## Local Qwen research loop

The separate orchestrator uses the existing AgentWork service at `http://127.0.0.1:11435/api/chat` (Ollama-shaped JSON, not OpenAI chat completions). It does not launch or alter that service. Model default: `Qwen2.5-0.5B-Instruct:latest`. An alternative endpoint must implement the same protocol; changing model families may also need prompt adjustments.

```powershell
.\.venv\Scripts\python.exe orchestrator.py data\iris.csv --target target --max-experiments 3 --max-calls 4 --max-seconds 120 --output artifacts\research_001.json
```

Use a fresh output path for each new task. The baseline counts toward the experiment limit; every model request, including busy, invalid and duplicate attempts, counts toward the call limit. Output is capped at 160 tokens per request. Time limits stop new work and bound HTTP waits; they do not interrupt an already running fit or inference on the server. Server-reported token counts are telemetry, not billing guarantees, and may be incomplete for failed requests.

Only validated model/parameter JSON reaches the executor. Unknown actions, out-of-range parameters and duplicate experiments are rejected. One complete JSON code fence is tolerated; arbitrary surrounding text is rejected. No dataset rows are sent to the model; it receives validation history. No model-generated code is executed. Selection uses validation macro-F1 only. The orchestrator never calls `finalize` or evaluates the test set.

Repeated invalid responses, unavailability, or validation stagnation return `escalation_required` when cloud is disabled. Optional DeepSeek recovery is described below. New CLI research runs also create a durable task in artifacts/research.sqlite3 and return task_id. The research journal is flushed per event; use the task workflow below to inspect and manually continue experiments after a normal restart.

Live smoke check on 2026-11-13: Qwen generated a valid logistic-regression candidate with C=0.001. Validation macro-F1 fell from 0.9333 (baseline) to 0.7917, so the baseline was retained. One local call, zero cloud calls, no test evaluation; artifacts/qwen_loop_001.json contains the actual result. This establishes execution, not research-quality improvement.

## Optional DeepSeek recovery

Cloud is OFF by default. Cline's own provider configuration does not configure this project's API. To opt in, run the helper in an interactive PowerShell terminal:

```powershell
cd C:\workspace\EvoResearch
.\tools\run-cloud.ps1
```

It prompts for a hidden DeepSeek API key, passes it to the child process through its environment, restores the previous environment afterwards, and never writes the key to disk. It uses the bundled Iris CSV by default; pass `-Csv data\your.csv -Target label` for your own data. No execution-policy changes are made; if script execution is blocked, use your organization's approved launch method.

Alternatively set `DEEPSEEK_API_KEY` privately in your terminal, then:

```powershell
.\.venv\Scripts\python.exe orchestrator.py data\iris.csv --target target --enable-cloud --cloud-model deepseek-flash --max-cloud-calls 1 --output artifacts\cloud_research_001.json
```

The expert receives only the recent validation/model history and a fixed reason, not CSV rows or held-out scores. Its recovery candidate passes exactly the same whitelist validator. One valid recovery experiment ends the run, retaining the best validation model even if the cloud suggestion is worse. Cloud does not get to trigger final test evaluation.

Limits: one cloud request by default (configurable 0..3), 256 output tokens per request, 12000-byte prompt limit. All attempted requests count, including failures. No transport retries and no HTTP redirects. These are request/output limits, NOT a guaranteed dollar cap. Provider usage is recorded when available; missing usage is marked incomplete. `cloud_cost` is null because provider billing/rates have not been integrated; it must not be interpreted as zero cost.

Official protocol reference: https://api-docs.deepseek.com/api/create-chat-completion/ . This adapter uses /chat/completions, JSON output and non-thinking mode. Model is configurable via --cloud-model or DEEPSEEK_MODEL. 2026-11-13: protocol, rejection, budget, and recovery paths tested against a temporary local HTTP server and real sklearn execution. Live paid DeepSeek authentication/inference has NOT been verified because the project environment has no API key.

Later verification: the user's saved cloud run proves one live DeepSeek response (238 prompt / 44 completion tokens). A local sklearn failure on numeric-looking labels was fixed with training-fitted label encoding. The saved cloud candidate was successfully replayed offline; no new paid call was made for that fix.

## Durable task history and one-time final evaluation

New orchestrator CLI runs save task metadata, each experiment, events and the final research report to SQLite. No additional package is needed. Existing older JSON results are not automatically imported. Keep the original CSV unchanged and retain the database.

```powershell
# List task IDs, including runs started by run-cloud.ps1:
.\.venv\Scripts\python.exe task_store.py list
# Inspect a task (replace TASK_ID with the actual ID):
.\.venv\Scripts\python.exe task_store.py show TASK_ID
# Continue an open task manually, within its original experiment budget:
.\.venv\Scripts\python.exe task_store.py run TASK_ID --model random_forest
# Only when research is finished: select best validation model and evaluate once:
.\.venv\Scripts\python.exe task_store.py finalize TASK_ID
```

You can also start a manual task with `task_store.py create data\iris.csv --target target --max-experiments 3`. The manual workflow makes no LLM calls. Finalize automatically selects the highest validation macro-F1 (earliest experiment wins ties), deterministically refits that fixed configuration on the training split, evaluates the held-out test once, and seals the task. It does not tune on or refit with the test set. Dataset hashes and numpy/pandas/sklearn versions are checked before reproduction.

SQLite transactions reserve attempts before training; failed attempts consume budget. A durable finalizing claim is committed before test access, blocking competing/restarted processes. If finalization fails or the process crashes after the claim, it remains sealed; no automatic retry. A crash during a training attempt leaves it marked running and blocks automatic continuation until explicit recovery is implemented.

Scope: experiment histories and experiment budgets survive restarts. Autonomous planner continuation and model-call budget recovery after a crash are not implemented; do not restart the orchestrator as if it were the same task. Creating another task/database or using executor.py directly is outside the task guard; this is per-task governance, not a tamper-proof dataset-wide test-access system.

## Development HTTP API (dependency installation pending approval)

api.py exposes health, task creation from CSV text, task list/detail, manual experiments and one-time finalization. It uses a separate artifacts/api/research.sqlite3 database by default, so CLI-created tasks in artifacts/research.sqlite3 are not listed there. Use create_app(data_directory) to choose another isolated API workspace. No cloud calls or automatic Agent run endpoint yet.

After explicit dependency-install approval:

```powershell
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements-api.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m uvicorn api:create_app --factory --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs for the interactive API. POST /tasks accepts JSON with csv_text, target and max_experiments (no server-side file paths). POST /tasks/{id}/experiments accepts model and parameters. POST /tasks/{id}/finalize is an explicit irreversible research-stage action: it consumes the final test evaluation and seals the task.

Development only: bind to loopback. No authentication, deployment hardening, background job queue or strict total HTTP request-size middleware has been implemented. CSV payload limit is validated after parsing JSON. Do not expose this service on a public network.

Current verification: Python syntax check passes. The 2 HTTP tests cannot run until FastAPI/httpx are installed; API runtime is NOT yet verified.

## Research workbench (2026-11-13)

The installed API dependencies have since been verified by yuyu (23 tests passed), superseding the earlier pending-install note. The workbench and local research job integration now pass the complete 26-test suite.

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:create_app --factory --host 127.0.0.1 --port 8001
```

Open http://127.0.0.1:8001/ . Port 8000 was already serving an older API; the development workbench uses 8001 to avoid interrupting it. The page supports UTF-8 CSV text upload (2 MB), a bundled Iris example, target selection, experiment/local-call/time budgets, autonomous LOCAL Qwen research, polling of real experiments/events, previous task selection, and an explicit confirmation dialog before one-time final testing. No frontend packages, fonts or external CDN are required. API JSON upload remains parsed before size validation; do not expose this development service publicly.

POST /tasks/{id}/research queues a background job and returns 202. Its input contains max_calls (0..10) and max_seconds (5..600); the experiment budget is fixed when the task is created. This web branch uses the local Qwen endpoint only and never calls paid cloud APIs. When local quality is insufficient, it reports escalation_required; the CLI cloud recovery path remains separate.

Research claims are saved in SQLite, only one job is admitted per API database at a time, and manual experiments/finalization are locked while queued/running. Completed runs retain report, experiment and event history. An abrupt server crash leaves the job claim active and fails closed; automatic recovery/cancellation, cross-database scheduling and autonomous resume are not implemented. Use a single API server/worker with this prototype. Database history is not a distributed job queue.

Verification: all 26 Python tests pass, node --check static/app.js passes, and real HTTP + existing local Qwen completed two experiments with one model request, best validation macro-F1 0.9333, zero cloud calls, no test evaluation. Finalization while running returned 409. Evidence: artifacts/workbench_http_verification.json. Background job tests use real sklearn training and injected deterministic proposals; the HTTP smoke run used actual Qwen.

Browser verification script: node tests/verify_workbench.mjs (uses installed Edge, no npm install). It is intended to verify demo/start/render/reload/mobile layouts and save screenshots, but the headless Edge renderer crashed in the current execution environment, so visual/browser-interaction acceptance is still pending. No successful screenshots or browser E2E pass are claimed. Existing Starlette/httpx deprecation warning remains; it does not fail the suite.
