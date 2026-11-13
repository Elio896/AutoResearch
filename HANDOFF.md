# HANDOFF.md — 开发交接

## 项目位置

- 规范工作目录：`C:\workspace\EvoResearch`（VS Code 已打开此目录；后续所有开发与交接记录均在此维护）。

## 当前进度

- 阶段：**项目初始化**。
- 已完成：工具配置与交接文件初始化（AGENTS.md / HANDOFF.md / .clinerules/01-handoff.md）。
- **尚未验证项目运行**：未安装依赖、未编写业务代码、未执行任何实验。

## 已验证内容

- **无**。（当前没有任何可验证的运行结果。）

## 待解决问题

1. 项目运行尚未验证：Python、FastAPI、scikit-learn 等依赖是否就绪未知。
2. 尚未实现「不依赖 LLM 的分类实验执行器」。
3. 尚未确定 CSV 数据格式规范与数据存放位置。
4. 本地轻量模型与云端大模型的接入方式尚未设计（不影响第一步）。
5. 密钥管理方案尚未落地（当前阶段不涉及密钥）。

## 下一步

1. 实现「**不依赖 LLM 的分类实验执行器**」：
   - 读取 CSV 表格数据；
   - 划分训练集 / 测试集；
   - 使用 scikit-learn 训练分类器；
   - 输出真实评估指标（测试集仅最终评估一次）。
2. 完成后真实运行并记录输出，更新本文件「已验证内容」。
3. 暂不引入任何 LLM 调用。

## 约束提醒

- 密钥不得写入代码、本文件或日志。
- 测试集不得用于迭代调参。
- 实验必须真实执行，禁止伪造结果。

## Codex update 2026-11-13

Canonical workspace: C:\workspace\EvoResearch.
Implemented executor.py, tests/test_executor.py, requirements.txt, README.md, .gitignore.
Executor: CSV classification, fixed stratified 60/20/20 splits; train-only preprocessing; logistic regression/random forest; validation macro-F1 and accuracy; one final test evaluation per in-memory session. No LLM calls.
Verified: python -m py_compile executor.py tests/test_executor.py passed; python executor.py --help passed.
Full test command: python -m unittest discover -s tests -v. Result: 1 passed, 2 errors (test_missing_target_rejected: pandas missing; test_real_training_and_reserved_test_set: sklearn missing). No real training results obtained. Do not describe this milestone as runtime verified.
Dependencies were NOT installed. AGENTS.md requires explicit confirmation first.
Next: obtain approval to create project-local .venv and install requirements.txt; run full tests, fix failures, record actual results. FastAPI and model-provider integration remain deferred.
Important: session state is not persisted; restarting bypasses the in-memory final-test lock. Add persistence before serving multiple research requests.

## Verified checkpoint 2026-11-13

User explicitly approved project-local .venv installation. Created .venv and installed binary wheels from requirements.txt successfully. System Python was not modified.
Installed: pandas 3.1.1, scikit-learn 1.9.1, numpy 2.5.4, scipy 1.18.1.
Full suite: .\.venv\Scripts\python.exe -m unittest discover -s tests -v -> 3 tests passed.
Real smoke run: bundled sklearn iris -> data/iris.csv; seed 42; train/validation/test = 90/30/30. Logistic regression and random forest (50 trees, max_depth 5) both validation macro-F1 = 0.9333333333333332. Tie selected exp_001 (logistic regression). One final test evaluation: macro-F1 = 0.9333333333333332, accuracy = 0.9333333333333333.
Actual results: artifacts/iris_smoke.json. These are executor smoke checks, not evidence of Agent performance or generalization across datasets.
The earlier missing-dependency blocker is resolved. Next: persistent task state and experiment history, then FastAPI integration; keep final-test isolation across process restarts. Do not call LLMs or claim automatic ChatGPT-to-DeepSeek switching is implemented.
Cline must open C:\workspace\EvoResearch and use .venv\Scripts\python.exe.

## 进度同步通道 2026-11-13（鱼鱼）

目的是让 Cline / Codex / 鱼鱼三方实时看到彼此在动什么。

**新增**
- `.agent-log/events.jsonl` — 唯一进度源，一行一个 JSON，只追加。规则写进 AGENTS.md「三方进度共享」。
- `tools/agents-watch.py` — 合并时间线：进度源 + Cline 会话日志 + Codex rollout + 项目文件改动 + 进程快照 + 每方最后活跃时间。用法 `python tools\agents-watch.py`（`--follow` 每 3 秒刷新）。
- `~/Documents/Cline/Hooks/*.ps1` — 9 个 Cline 钩子（TaskStart / TaskResume / TaskCancel / TaskComplete / TaskError / PreToolUse / PostToolUse / UserPromptSubmit / SessionShutdown），自动把 Cline 的每个动作写进进度源。

**已验证（真实执行）**
- 钩子契约从 Cline 4.1.26 bundle 中读出：事件名大小写不敏感，支持扩展名 `.ps1` 等；Windows 下钩子统一由 `powershell -ExecutionPolicy Bypass -File` 执行，超时 30s；stdin 收 JSON，stdout 必须为空或合法 JSON，退出码必须为 0。
- 用 5 条合成 payload 直接喂 `PreToolUse.ps1`：5 次均 exit=0，进度源正确落 5 行，中文字段无乱码（第一版有乱码，改为按 UTF-8 读原始字节后修正）。
- `tools/agents-watch.py` 实跑成功：正确显示 Cline 会话（deepseek-v4-pro / status）、Codex 运行日志时间、项目文件改动时间线。

**注意**
- Cline 当前工作目录曾是 `c:\Users\junzi\.vscode` 而不是本项目，需要在 VS Code 里打开 `C:\workspace\EvoResearch` 才能正确用 `.venv`。
- 钩子「被 Cline 真正触发」这一步，要等 Cline 下一次跑任务才能确认；目前验证到的是钩子本身可执行 + 契约匹配。真实首条记录会出现在进度源里。



## Local Qwen orchestrator checkpoint 2026-11-13

Implemented orchestrator.py (separate from unchanged executor.py), tests/test_orchestrator.py, and README instructions. No dependencies added, models downloaded, or AgentWork service settings changed.
Default endpoint: http://127.0.0.1:11435/api/chat; model Qwen2.5-0.5B-Instruct:latest. Strict JSON whitelist, numeric type/range checks, model-specific parameters, duplicate rejection, bounded model calls/experiments, retry on HTTP 429/503. Deterministic baseline and validation-only best-model selection. No finalize call.
Full suite: .venv\Scripts\python.exe -m unittest discover -s tests -v -> 10 tests passed. py_compile passed. Includes actual temporary HTTP-server test of 429 and Ollama protocol.
Live verification: .venv\Scripts\python.exe orchestrator.py data\iris.csv --target target --max-experiments 2 --max-calls 3 --max-seconds 90 --output artifacts\qwen_loop_001.json. Completed in 2.98s (loop only); 1 local request; prompt/completion tokens 190/32 as reported by server. Baseline validation F1=0.9333; Qwen candidate C=0.001 F1=0.7917; exp_001 retained. No test evaluation or cloud calls. Not evidence of research improvement.
Usage: choose a NEW --output path. Per-event journal written alongside result for inspection; no resumable session implemented. Time limit stops new work but does not hard-cancel fits or remote inference. In-memory executor lock still has no cross-process enforcement.
Cloud escalation currently returns escalation_required only. Next: wire budgeted cloud diagnostic provider; persistent task state/final-test lock; API/UI later. Keep executor free of LLM calls.


## Optional cloud recovery checkpoint 2026-11-13

Added cloud_provider.py and opt-in DeepSeek recovery to orchestrator.py. DeepSeek /chat/completions, JSON output, non-thinking, default deepseek-flash; checked official API docs. No new packages or models. executor.py unchanged.
Default: cloud disabled. Enable with --enable-cloud and DEEPSEEK_API_KEY; --max-cloud-calls defaults to 1 (range 0..3). tools/run-cloud.ps1 securely prompts for the key in an interactive terminal and launches one bounded task without storing the key. Script parse verified, interactive credential flow not executed.
Cloud receives bounded validation/model history only; no CSV rows, no test scores. A cloud candidate passes the same validator; one valid recovery experiment ends the run and best validation model is retained. Request failures count and do not auto-retry; redirects blocked; exception text sanitized. HTTP error handles explicitly closed.
Verification: .venv\Scripts\python.exe -W error::ResourceWarning -m unittest discover -s tests -v -> 14 tests passed, no warnings. Python compile and PowerShell script parse passed. Protocol/HTTP failures/redirect rejection/recovery tested against a temporary local server; actual sklearn training used. CLI smoke artifacts/cloud_disabled_check_001.json returned 0 cloud calls and no test evaluation.
No project DEEPSEEK_API_KEY found. Live paid authentication/inference NOT VERIFIED. Do not extract Cline credentials or claim paid cloud was tested. User must enter key locally via helper.
Cloud limits: per-request output max 256 tokens; prompt max 12000 bytes; all calls counted. These are not a guaranteed money cap. Reported token telemetry can be incomplete; cloud_cost=null means unknown, not zero.
Next: live cloud verification once key is locally configured; persistent research-task state and cross-process final-test lock; then API/UI.


## Cloud training failure fixed 2026-11-13

User completed live DeepSeek call: artifacts/cloud_a731770b17cd4f18a571ca1d53968139.json records 1 cloud call, reported prompt/completion tokens 238/44, and an accepted random_forest proposal (100 trees, depth 10, min leaf 1, balanced). Original training failed; authentication and response are now evidenced, superseding prior no-live-call status.
Root cause reproduced offline: installed sklearn class-weight path fails on numeric-looking string labels ('0','1','2'), while integer IDs and nonnumeric names work. executor.py had converted every label to string. Added training-fitted LabelEncoder and stable integer IDs; no feature preprocessing or test-scoring changes. Added regression covering numeric-looking and named labels.
Full suite: .venv\Scripts\python.exe -W error::ResourceWarning -m unittest discover -s tests -v -> 15 tests passed. Replayed all saved candidates without network: artifacts/cloud_candidate_replay_fixed.json. Cloud RF now validation macro-F1=0.9333, tie with original baseline, which remains selected. No final test evaluation, no new paid API request. Original failed run preserved.
Next: persistent task state and cross-process final-test lock; then API/UI. A new paid end-to-end run is optional, not required to establish this local fix.


## Persistent tasks checkpoint 2026-11-13

Added task_store.py: SQLite task metadata, experiments, events, report, runtime/data fingerprints; bounded experiment attempts reserved before fit; atomic finalizing claim before test access. Final test picks best validation model, refits only the fixed train split, evaluates once and seals the task. Failure/crash after claim remains sealed.
Orchestrator CLI now persists NEW tasks automatically and returns task_id (default artifacts/research.sqlite3). StoredResearchSession bridges executor calls; providers and executor remain unchanged. Existing old JSON not imported. Manual resume: task_store.py run TASK_ID; list/show/finalize commands available. Automated planner crash resume and persistent model-call budget recovery remain unimplemented. Interrupted running fit blocks further automatic actions; explicit recovery not implemented.
Full suite: .venv\Scripts\python.exe -W error::ResourceWarning -m unittest discover -s tests -v -> 21 tests passed. py_compile and CLI help passed. Real CLI run then separate-process show/run/finalize/repeat-finalize check passed: artifacts/persistence_verification.json, dedicated artifacts/persistence_smoke.sqlite3. Final macro-F1=0.9333; second finalize blocked. Zero model API requests.
Per-task guard only: new databases/tasks or direct executor access can bypass it. No dataset-wide tamper-proof claim. Next: API for frontend; confirm before installing any new dependencies.


## API prepared, installation pending 2026-11-13

Prepared api.py create_app factory: health, task list/create/detail, manual experiment, one-time finalize. JSON CSV text upload, no arbitrary server paths, local-development scope. No Agent/background/cloud run endpoint. API uses separate artifacts/api/research.sqlite3 by default; CLI tasks are not automatically shared. Added tests/test_api.py and requirements-api.txt (FastAPI, uvicorn, httpx plus existing requirements).
Syntax check passed. Full suite has 21 passing tests and 2 API test errors due to missing fastapi; API runtime NOT VERIFIED. No packages installed. Await explicit permission required by AGENTS.md before project-local installation, then run full tests, fix any failures and verify actual HTTP server. No paid model calls performed. README contains startup commands and limitations.

## API runtime verified 2026-11-13 (yuyu)

Installed requirements-api.txt into the project `.venv` only (system Python untouched): fastapi 0.162.5, uvicorn 0.54.0, httpx 0.28.1, starlette 1.10.0, pydantic 2.14.3, plus transitive deps. Nothing installed outside `C:\workspace\EvoResearch\.venv`.

Full suite: `.venv\Scripts\python.exe -W error::ResourceWarning -m unittest discover -s tests -v` → **23 tests passed, 0 failures** (previously 21 pass + 2 errors from missing fastapi).

Server started exactly as documented: `.venv\Scripts\python.exe -m uvicorn api:create_app --factory --host 127.0.0.1 --port 8000`. `/health` returned `{"status":"ok","cloud_enabled":false}`.

End-to-end verification against the live server (real HTTP, real sklearn training, no LLM, no paid calls), script kept at `tests/verify_api_e2e.mjs`:

1. `GET /health` → 200 `{"status":"ok","cloud_enabled":false}`
2. `POST /tasks` (data/iris.csv as csv_text, target=`target`, max_experiments=3) → 201, `task_id=c9106da37e4448f484418d69a9c78874`
3. `POST /tasks/{id}/experiments` logistic_regression → 200, `exp_001`, validation macro-F1 **0.9333**, split 90/30/30, duration 0.0256 s, params `{max_iter:1000}`
4. `POST /tasks/{id}/experiments` random_forest(n_estimators=50, max_depth=5) → 200, `exp_002`, validation macro-F1 **0.9333**, duration 0.0650 s
5. `GET /tasks/{id}` → 200, returns state=`open`, dataset_hash, versions, and both experiments in history
6. `POST /tasks/{id}/finalize` → 200, `test_f1` **0.9333**, `test_accuracy` **0.9333` (single final evaluation)
7. `POST /tasks/{id}/finalize` again → **409** `{"detail":"Task is not open; final test cannot be repeated."}` — one-time lock holds over HTTP
8. `GET /tasks` → 200, 1 task

Constraints upheld: executor.py unchanged and LLM-free; no API key written anywhere; no paid model call made (`cloud_enabled:false`); server bound to 127.0.0.1 only.

Known gaps unchanged: no Agent/background/cloud run endpoint in the API; API database (`artifacts/api/research.sqlite3`) is separate from CLI tasks; no authentication (local-dev scope only).



## Workbench + autonomous local research 2026-11-13 (codex)

Implemented static/index.html, style.css, app.js: CSV UTF-8 upload / Iris example, target/budgets, start local research, polling of real history/events, prior tasks, validation winner, confirmation before final test. No frontend dependencies. UI only local model, no browser key entry or paid calls.
Added research_jobs.py and research_jobs SQLite table; /tasks/{id}/research returns 202 and runs ResearchOrchestrator in a background thread. /demo serves sklearn Iris; / serves UI; /static serves assets. Durable per-database single-job admission, duplicate prevention and task manual-action/finalize locks. StoredResearchSession uses a private research token for job-owned experiments; no token exposed in task JSON. States persist; abrupt crash remains fail-closed, no recovery/cancel endpoint. Single server worker only.
Connection interruption left app.js missing; restored it after reconnect. Final full suite: 26/26 passed; node --check static/app.js passed. Existing Starlette/httpx deprecation warning remains.
Old 8000 API was already running and lacked new endpoints. Preserved it; NEW server runs at http://127.0.0.1:8001/ (uvicorn api:create_app --factory --host 127.0.0.1 --port 8001), process shown by uvicorn as 45708, exec session 15609. Do not kill another agent's old server.
Real HTTP + actual local Qwen verified, evidence artifacts/workbench_http_verification.json: task 273935f2aed64961b2a1792a070f833a, completed, 2 experiments, 1 local call, best validation F1=.9333, 0 cloud calls, test not evaluated, finalization while running blocked (409), 6 events.
Browser automation script tests/verify_workbench.mjs added (Edge via CDP, no npm packages); renderer crashed / websocket closed in this execution environment. Browser visual/interaction/mobile/reload acceptance NOT VERIFIED. No screenshots were successfully created. Codex preview opener also failed. User can visit 8001 directly; yuyu may verify browser manually.
Known gaps: web local-only (cloud recovery CLI separate); no automatic crash resume/cancel, no authentication/public deployment support. Next: browser acceptance and visual iteration; then bounded cloud opt-in in UI or experience memory/benchmark, without fabricating results.


## Integrity guard complete 2026-11-13 (codex)

Added integrity.py and known_leaks.json; exact bundled t1 hash + generator evidence confirms record_id leakage (sort target, assign IDs, shuffle). Generic candidate detection uses training rows only: ID names, ordered unique integers, target matches, strong binary association, unique numeric feature that class-separates when sorted. Renamed t1 column still triggers candidate review. Candidate is not proof; explicit keep with provenance reason allowed, confirmed leak must exclude. No silent dropping; cleared linter is not proof of no leakage.
ResearchSession computes risk report, blocks .run while unresolved, applies explicit feature_decisions to all models and includes exclusions/reasons in results. TaskStore migrations add decisions and risk report, /review endpoint saves decisions atomically before any attempted experiment; frozen afterwards. _session guards manual/autonomous/final actions. Legacy histories unchanged, but show() attaches risk warning and UI masks suspicious best/final conclusion while preserving old table scores. Research start on manual-history tasks remains disabled with a clear explanation.
UI: risk panel with per-column action+reason, save decisions, start reviewed fresh task; existing form says new task. Frontend checks OpenAPI for /tasks/{task_id}/review before allowing mutations, so old backend cannot be mistaken for guarded service. JS syntax passed.
Full suite after reconnect: 34/34 passed; existing Starlette/httpx deprecation warning unchanged. Real HTTP verification tools/verify_integrity_http.py completed before disconnect, evidence artifacts/integrity_http_verification.json: task e8a785624d6546f798bfc49964bbf483; unreviewed manual+Agent409, 0 attempts, explicit record_id exclusion -> logistic validation F1 .9440154440, post-training review409; no test evaluation and 0 model API calls. Do NOT compare this holdout score to 5-fold CV threshold.
No benchmark thresholds, generation scripts, source CSV, sentinel/unit handling, or dependencies changed. Older history remains a legacy result, not re-labeled as a new clean experiment.
8001/8002 old servers were unreachable after reconnect; guarded server is being restarted on 8002. URL http://127.0.0.1:8002/ . Launch: .venv\Scripts\python.exe -m uvicorn api:create_app --factory --host 127.0.0.1 --port 8002. Verify availability if another machine disconnect occurs. Browser visual acceptance remains unverified (earlier Edge renderer crash). Next: user/browser review; then benchmark-cleaning design and optional Jev integration, separately.


## Optional Jev escalation gate 2026-11-13 (codex)

Found jev_client.py recently contributed by another agent; preserved it unchanged. New jev_router.py is a separate narrow adapter used by orchestrator. Verified official docs: https://docs.typesafe.ai/api/system-one.md and introduction/quickstart.md . POST https://api.typesafe.ai/v1/systemone, one Noul worth_cloud_diagnosis question, response answers.<id>.noul + model + usage input/output tokens.
Jev OFF by default; CLI --enable-jev requires explicit --enable-cloud and positive cloud-call budget. TYPESAFE_API_KEY/JEV_API_KEY are read only from process environment. No key configured in this runtime; no paid Jev request performed. tools/run-jev.ps1 prompts hidden Jev + DeepSeek keys, restores previous process env, defaults to t3_clean and caps each cloud provider at one request. Script syntax checked only; interactive flow not executed.
Rule eligibility still checks remaining experiments, cloud budget and time BEFORE any Jev request. Jev only gates cloud diagnosis after local failure/stagnation; threshold default .75 (engineering setting, not empirically calibrated). A decline stops with jev_declined; malformed/failed Jev or zero Jev budget falls back to existing rules within cloud limits. All failed attempts count; no hidden retry/redirect. Test set, feature decisions and hard budgets cannot be overridden. State contains only validation model/parameter history, trigger, call count and remaining budgets.
Reports: jev_calls, cloud_calls (DeepSeek), total_remote_calls, escalation_gate (source, probability/threshold/model/reported usage); jev_cost/cloud_cost null means unknown, not free. No claimed budget/quality improvement yet.
Full suite .venv\Scripts\python.exe -W error::ResourceWarning -m unittest discover -s tests -v: 40/40 passed. Temporary loopback protocol + HTTP failure tests, reject/allow/fallback and hard-budget precedence use actual sklearn training; holdout not accessed by gate. py_compile, CLI --help and PowerShell AST parse passed. Existing Starlette/httpx deprecation warning unchanged.
Web remains local-only; paid gates are CLI only. Earlier EvoResearch_GitHub_20261113.zip predates this adapter and is NOT updated. Next: user can privately configure keys and verify a real request, then compare rule vs Jev gating under matched budget before claiming savings. Existing 34-test integrity guard unchanged.
