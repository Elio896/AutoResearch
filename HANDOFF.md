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


## Task objective alignment 2026-11-13 (codex)

Added immutable f1_macro/f1_minority task objective and explicit training-minority label validation; SQLite migration/defaults, executor diagnostics, local/cloud improvement checks, durable final winner and UI use selected score. Full suite 42/42 passed; focused objective-disagreement final-selection regression passed. No paid API calls, no benchmark changes; browser visual verification and running-server reload not performed.
New API TaskInput metric and pos_label; CLI executor/orchestrator/task_store create --metric f1_minority --pos-label 1. Existing tasks default macro, historical scores untouched. validation_f1/test_f1 remain macro; validation_score/test_score are selected objective. Minority precision/recall/F1 included for minority objective. Training minority checked without inspecting test performance. UI caption names objective, old schema blocks mutation; restart API required. GitHub and previous ZIP not updated this turn.


## Live objective HTTP verification 2026-11-13 (codex)
Live 8002 HTTP t2 objective verification passed: logreg minority validation F1 .3541667, RF balanced .6141732; final selects exp_002, minority test F1 .6141732, precision .78, recall .5064935, accuracy .951. Duplicate finalize 409. 42 tests pass; no model API calls; not five-fold CV. Backend restarted PID19828; visual browser unverified.
Evidence artifacts/metric_http_verification.json, reproducible tools/verify_metric_http.py. This run is manual executor/API verification, not proof of autonomous reasoning or Jev live credentials.


## v2 delivery checkpoint 2026-11-13 (codex)
Created v2 ZIP 83 files 1034385 bytes SHA256 dd8487f938256b751770784d0ec3b9c1801746598830f2e4fe058e72512e3414, staged independent suite 42/42 passed. Saved deliverable libfile_e5cd0a84f1e881919a5330dc4f490714. Local commit 854ec8e contains metric update + HTTP verification script. Git push failed credential-manager MSYS access error; connected GitHub get_repo returns404 so remote NOT updated. No credential extraction attempted. Existing v0.1.0 release/old ZIP unchanged.
Backend session9881 PID19828 on8002. Live HTTP evidence uses real t2, two manual experiments, exactly one final holdout. Browser visual review remains unverified. Next substantive work: matched-budget autonomous local-only vs rules/cloud/Jev comparison; real Jev auth still unverified and should use private environment credentials.


## Local strategy comparison 2026-11-13 (codex)
Completed18 matched-cap runs t2/t3 seeds42,43,44 fixed vsQwen0.5/3B.24local calls,0remote,0holdout. t2 meanF1 fixed .585389 vs bothAgent .390282; t3 .993747 vs .987493. No Agent advantage shown. Evidence docs/evaluation/local_policy_20261113; next work training-only profile/diversity, do not overwrite benchmark.
No core runtime changes. No new dependencies. Existing ZIP/GitHub remote not updated. Current local defaults preserved. All18 split fingerprints match across arms.


## Planning context v2 verified 2026-11-13 (codex)
context_v2 verified44/44 tests;18 development comparison runs24local calls0paid0test. Qwen3B t2 meanminorityF1 .390282->.550792 but belowfixed .585389; t3unchanged. Training-only anonymized bounded aggregates, validation-class metrics, advisoryfamilydiversity, noforcedRF. Real8002 autonomous API task18264fe4e24f4b16a41549e99dd9f23f confirmsprofile throughStoredSession,1local call,noholdout. Newserver session57084; default0.5B unchanged. Reports docs/evaluation/context_v2_20261113; noZIPorremote update.
Regression red-before-green documented in tests/test_planning_context.py. Same development datasets are reused so these scores are not independent generalization evidence; do not lower benchmarks or claim Jev/cloud cost advantage. No dependencies installed. Next freeze policy and evaluate unseen datasets; web3B selectable configuration is still not implemented.


## Frozen transfer + opt-in baseline policy 2026-11-13 (codex)
Frozen transfer18runs finished,18lockedtestclaims; Wine testmacroF1 .981188 tie, imbalance fixed .404710 vs3B .299208, mixed .892233 vs .820388. Preserved docs/evaluation/frozen_transfer_20261113. Added opt-in baseline_policy fixed_search acrossCLI/API/jobs/UI, defaultlogregretained;3trialbaseline0LLMcalls and4trial1Agentcall covered.47/47tests pass. Live8002 HTTP task55f0b36ef2d94ba4877ec820784c548d confirms3fits0modelcalls/noholdout. No newpackages,paidrequests orremoteupdates.
Baseline safeguards validation only when fullplan succeeds withinbudget; no test-score guarantee. Server36828/session1536 on8002, browservisualunverified. Existingv2ZIP predates context_v2/baselinepolicy, notupdated. Next freeze hybrid and obtain fresh realdatasets for enterprise demo; do not repeatedly retune consumedholdouts.


## Research evidence export 2026-11-13 (codex)
Read-only evidence export implemented: GET/tasks/{id}/export formatmarkdown/json/zip pluswebbuttons. ZIP REPORT.md/manifest.json/replay_validation.py, serverCSVpaths+jobtokens omitted; sources rule/local/cloud/manual/unresolved, unknowncostnotzero, activejobs409, leakagewarningblocksbestconclusion.52/52tests passed, redtests404beforeimplementation, replay script reproduced3validation scores andnoholdout. Real8002 HTTP allformats200 task55f0b36ef2d94ba4877ec820784c548d, stateidenticalbeforeafter,0exportmodelcalls. No dependencies/paidcalls; no pretrainedmodelartifact orCSV bundled.
Serversession28995 port8002. Browservisualverification remains unverified. Original historical task source revisions are not persisted, so exports disclose cross-code reproducibility limitation. Newmodule research_export.py must be included in future projectZIP; existingv2ZIP is stale. RemoteGitHub unchanged. Read-only HTTP evidence artifacts/export_http_verification.json.


## Source package v3 2026-11-13 (codex)
Delivered EvoResearch_GitHub_20261113_v3.zip source4f37805;175files1485908bytes SHA256894777b95a5d69fbcca9be2d46e3e3cba01b8704c9b470e3ded071d3ca1a02f4. Independent staging52/52tests passed,ZIP CRC+allfilehashes verified,secretpatterns no matches. Excludesenv/credentials/venv/git/privatehandoffs/logs/databases. Includescontext_v2/baselinepolicy/reportexports/Jevgate/evaluationevidence/dependencylock/START_HERE. Saved libfile_7c60052b01e48191ac9acc4ef4ac1fc3. NoGitHubpush,no paidcalls,no fresh dependencyinstall orbrowservisualverification.
Replaces older v2 as user-facing downloadable source snapshot. Cloud helper usage in START_HERE passes -Csv bench/tasks/t3_clean/data.csv explicitly because original helper default data/iris.csv is not bundled. Generic separatelycontributed jev_client.py omitted (not a dependency of jev_router). Source/inference defaults unchanged.


## Jev live authentication check 2026-11-13 (codex)
One explicitly requested real request returned HTTP401. Not connected. Key was input by hidden stdin and kept only in the terminated process; do not copy any secret from conversation to files. No retries or model-success claim. tools/verify_jev_live.py prompts without echo; next step user replaces exposed credential privately, then runs helper. Existing serving behavior remains unchanged.


## Jev real connection succeeded 2026-11-13 (codex)
New user-provided credential passed one real call: jev-1.14.0,348 input/24 outputreportedtokens, probability0.51 for connectivity state. No secret persisted. Process ended, so CLI/web do not retain this key. This validates authentication and inference only, not whole research gate, financial cost or effectiveness.

## 夜间攻坚完成 2026-11-14 02:25 (yuyu + cat)

本轮按 loop 流水线推进 M0-M4，全部通过并提交。当前 135 tests passed。

- M0 (9053462): 补齐轨迹九项里最后一项 04 干预 diff。schema 原本不接受 replace/path 且禁止 baseline 空 diff，cat 做了向后兼容扩展（保留 field 格式 + 新增路径格式）。校验器 3 轮 FAIL 0 / WARN 0。
- M1 (evoundo.py): EvoUndo 版本链。一次实验 = 一个版本，Reviewer 判退步就把 current 指回历史最优；不重训、不删版本，被抛弃的一代标 rolled_back 留作"假设被否证"的证据。t2 实测：5 版本 1 回滚，v_8cf6 (0.2984) -> v_52e3 (0.6984)。版本树落 JSON + 零依赖 HTML。
- M2 (redteam.py): 红蓝对抗。红队质疑必须命名注册表里的 check（seed_stability / feature_ablation / split_shift），否则记 rejected_vague 丢弃——不接受无法证伪的怀疑。蓝队只按数字判，模型不参与。实测红队指"类1 F1 远低于类0"，feature_ablation 测出 0.0262 > 0.02，判 claims_upheld。修掉一个假回滚 bug：当前就是最优时记 rollback_skipped 并标 redteam_suspect，不假装成功。
- M3 (tools/run_selfiter.py): 自迭代 A/B。实验组 3 轮稳定 0.7143，对照组在 0.2994/0.6620/0.5426 之间跳——稳定性提升；但 local_tokens 反涨 +1363（经验变长撑大 prompt）。如实记为"未观察到成本下降"。
- M4 (model_router.py): 模型分层。Scientist 留 7B，RedTeam 降到 3B（找茬比提方案简单），Analyst/Manager/Reviewer 纯代码 0 token。3B 红队仍能产出合格质疑。

JEV: 之前那串 key 是错的（106 字符, sha256 b3a5aeb2, 恒 401）。真 key 107 字符 sha256 b0bdc4f6, 与供应商指纹一致，实测 HTTP 200, model jev-1.14.0, probability 0.67, usage 432/24。key 只放在用户环境变量 TYPESAFE_API_KEY，未入库。jev_local.py 保留为无 key 时的同协议本地实现。

已知未做：网页端未接新功能；token 台账未把 RedTeam 的 3B 用量并进 summary（ModelRouter 有记录，但 research_loop 的 local_tokens 只算 Scientist）。

## 鱼鱼 update 2026-11-14 · 第一步：策略版本存储（已验证）

- 新增 `strategy_store.py`：策略=配置（只允许 classifier_prompt / planning_prompt / candidate_selection_rules），
  四态 candidate/active/retired/rejected，原子写 + 进程内锁，重启不丢。
- `changes` 语义：传补丁、存完整配置（未写字段从父版本继承），避免运行拿到残配置。
- 升级双闸在存储层硬性执行：`checks` 必须非空且全 True，`jev.decision` 必须是 `upgrade`，缺一即 PermissionError。
- `pin()` 给一次运行固定快照；`verify_pin()` 运行结束自查是否被换过。
- `research_loop.py` 接入：新增 `strategy_store` / `strategy_id` 入参，
  运行开始记 `strategy_pinned`，结束时记 `strategy_pin_verified` 并在 summary 里带 strategy_id / config_digest / pin_intact。

已验收（真实输出）：

- `python -m unittest discover -s tests -p "test_strategy_store.py"` → 12 tests OK
- `python tools/strategy_acceptance.py` → 八步全过（候选、diff 只报 planning_prompt、空 checks / Jev keep / checks 有 false 三种拒绝、
  双闸通过后 active=strategy_002、重开进程仍读到、回滚到 strategy_001、pin 快照不受后续切换影响）
- `python tools/strategy_run_binding.py` → 运行前库里被切成 strategy_002，这次运行仍用 strategy_001，
  `strategy_pin_intact=True`；新构造的 loop 才 pin 到 strategy_002
- 全量 `-m unittest discover -s tests` → 236 tests OK（22.6s）

未做：第二步及以后（千问结构化工作记录、失败模式、候选生成、对照评估、Jev 选择、回滚、前端）。

## 鱼鱼 update 2026-11-14 · 第二步：千问结构化工作记录（已验证）

- 新增 `qwen_worklog.py`：
  - `evidence_from_session()` 把画像算成 `{ref: {value, source, how}}`，每个 ref 都能查到算法；
  - `validate_record()` 硬校验：字段齐全、`task_kind` 枚举、`needs_specialist` 布尔、
    `observations[].evidence_ref` 必须存在、**fact 里的数字必须能在该 ref 的值里找到**（含百分比写法）、
    `operation.tool` / `verification_checks` 走白名单；
  - `run_operation()` / `run_checks()` 全部由代码执行，`model_output` 与 `tool_computation` 分开存；
  - `WorklogStore` 按运行追加 JSONL，不重写。
- 判定口径：解析失败 / 字段缺失 / 数字编造 / 工具越界，一律 `status='invalid'` + 真实错误，**不补字段、不生成叙述**。

已验收（真实输出，`artifacts/step2-acceptance.txt`）：

- 输入 A（均衡、无缺失）与 B（4% 正例、12% 缺失）跑同一套流程：
  `profile.minority_fraction` 0.3075 → 0.0525、`profile.missing_fraction` 0.0 → 0.04，
  记录里的观察随之改变；每条 observation 打印 `ref → 计算值 → 怎么算的`。
- 编造「训练集 99999 行」被拒：`数字 99999 在 profile.rows 里找不到`。
- 真实千问（qwen2.5:7b）这一轮返回的是数组壳，`task_kind='数据概览'`，还引用了不存在的
  `profile.unique_top[2]`；代码判 `invalid`，给出两条具体错误，没有替它补一段分析。
- `python -m unittest discover -s tests -p "test_qwen_worklog.py"` → 15 tests OK；全量 251 tests OK。

未做：第三步及以后。

## 鱼鱼 update 2026-11-14 · 第三步：失败模式发现（已验证）

- 新增 `failure_modes.py`，只检测能数出来的五类：
  `repeated_proposal`、`invalid_structure_repeat`、`same_family_tuning_no_gain`、
  `repeated_change_no_gain`、`operation_check_failed`。
- 事件引用用事件流下标：`event_007` = 本次运行 events[7]，可直接回查事件类型。
- 证据门槛：实验 < 2 次或无提案事件 → `verdict=insufficient_evidence`，不写任何"经验"；
  没有工作记录时 `operation_check_failed` 标 `evaluated=False`，不猜。

已验收（真实输出，`artifacts/step3-acceptance.txt`）：

- 真跑一次 ResearchLoop（stuck 提案器）：3 次实验、21 条事件、2 条提案；
  检出 `repeated_proposal` refs=`['event_007','event_013']`，逐条核对两条都指向 `scientist_proposal`，
  `duplicate_count=2`，model/parameters 一并记录。
- 只有 1 次实验的运行：`insufficient=True`，原因「实验只有 1 次…」「没有提案事件…」，失败记录 0 条。
- `test_failure_modes.py` 10 tests OK；全量 261 tests OK。

未做：第四步及以后（DeepSeek 候选策略、对照评估、Jev 选择、回滚、前端）。

## 鱼鱼 update 2026-11-14 · 第四步：DeepSeek 候选策略（已验证）

- 新增 `candidate_proposer.py`：`build_prompt()` 只给证据和边界；`validate_candidate()` 三重校验；
  `propose_from_failures()` 走「调用 → 解析 → 校验 → 落 candidate」，任何一步失败都不写库。
- 硬规则（代码执行，不靠模型自觉）：
  - 只许改 `classifier_prompt` / `planning_prompt` / `candidate_selection_rules`；
  - **一次只许改 1 个字段**（改 2 个即拒，防止广撒网）；
  - `hypothesis` 五项必须齐全且非空，**没有 falsifier 直接退**；
  - 改完与当前正式策略相同 → 不算修改；
  - 候选只能落成 `candidate`，正式策略不受影响。
- `strategy_store.propose()` 增加 `hypothesis` / `target_failure_types` 元数据（不改动配置校验）。

已验收（真实输出，`artifacts/step4-acceptance.txt`）：

- 输入是第三步真实跑出的失败报告：`repeated_proposal` refs=`['event_007','event_013']`。
- 真实 DeepSeek：环境无 `DEEPSEEK_API_KEY` → 明确输出「不可用（不伪造调用）」。
- 模拟响应（标注 simulated）：建出 `strategy_002`，`status=candidate`，五项假设完整，
  `target_failure_types=['repeated_proposal']`；正式策略仍是 `strategy_001`，内容未变 `True`，diff 只有 `planning_prompt`。
- 拒绝四种情形：越权字段 `scoring_function`、空 falsifier、一次改两个字段、证据不足（实验只有 1 次）。
- `test_candidate_proposer.py` 13 tests OK。

未做：第五步及以后。

## 鱼鱼 update 2026-11-14 · 换轻量模型 API + 第五步对照评估（已验证）

### A. 模型层换成 API
- 新增 `light_model.py`：OpenAI 兼容 `/chat/completions`，key 只走环境变量
  （`LIGHT_MODEL_API_KEY` 或 `DEEPSEEK_API_KEY`），`LIGHT_MODEL_BASE_URL` / `LIGHT_MODEL_NAME` 可指到任意兼容端点，
  `LIGHT_MODEL_MAX_CALLS` 默认 6，超了直接抛错，不会悄悄多花。
- `demo_flow._model_proposer()` 选模型顺序：**API → 本地 Ollama → 程序规则**，并把 `model_source` 写进返回：
  `api_model` / `local_model` / `rules`。页面按这三档分别署名，不再一律叫「千问」。
- 记账：回传 `calls` / `prompt_tokens` / `completion_tokens`；`cost_cny=None` + `cost_basis=unknown`，
  **没取到账单前不记 0 成本**。

已验收（`artifacts/step-api-acceptance.txt`）：配 key（本地假服务冒充 API）→ `model_source=api_model`、
`model=mock-small`、3 次 POST、token 360/96、`cost_cny=None`；无 key → `model_source=rules`、
`attribution=program`、`model_usage=None`（没有假装调过）。`test_light_model.py` 10 tests OK。

### B. 第五步：新旧策略对照评估
- 新增 `strategy_eval.py`：升级条件跑前冻结（带 digest）；开发任务与评估任务分开；
  候选不能靠产生它的任务验收；指标缺失或任务不足 → `insufficient_evidence`。
- 新增硬规则 `demonstrated_improvement`：**只是不退化不算升级**，必须有一项可测改善。

已验收（`artifacts/step5-acceptance.txt`，`tools/strategy_eval_acceptance.py`）：
- 旧策略重复率 0.5 → 候选 0.0：`upgrade`，`duplicate_rate delta=-0.5`，`demonstrated_improvement` 通过；
- 无效候选重复率仍 0.5：`keep`，唯一未通过项就是 `demonstrated_improvement`；
- 只有 1 个任务：`insufficient_evidence`（缺指标 + 样本不足）；
- 评估结束 `active` 仍是 `strategy_001`，评估动作没有改正式策略。

全量：296 tests OK。演示服务已重启（PID 见 `.demo-err.log` 侧）。
当前没配 key 且本地 Ollama 已停 → 页面如实显示「程序规则 · 非模型输出」。

未做：第六步（Jev 选择）、第七步（上线后回滚）、第八步（前端进化记录）。

## 鱼鱼 update 2026-11-14 · 第六步：Jev 选择 + 代码约束（已验证）

- 新增 `strategy_decision.py`：
  - `code_checks()` 把对照评估翻成代码检查项（`eval_verdict_upgrade` + 每条 check）；
  - **先代码后 Jev**：评估结论不是 `upgrade` 就直接 `keep`，Jev 不会被调用；
  - `decide()` 只接受 `upgrade` / `keep` / `defer` 三种回答，答非所问（`jev_invalid`）、
    调用失败（`jev_error`）、没有 key（`jev_unavailable`）一律 `keep`，不伪造决定；
  - 两项都过才 `store.activate(checks=..., jev=...)`，事件里带完整证据。
- `strategy_store.record_decision()`：`keep` / `defer` 也落事件，候选状态不动。

已验收（`artifacts/step6-acceptance.txt`）：

- 对照评估：旧重复率 0.5 → 新 0.0，`verdict=upgrade`。
- 真实 Jev（无 `TYPESAFE_API_KEY`）：`decision=keep`、`source=jev_unavailable`，
  正式策略仍是 `strategy_001`。
- 模拟 Jev 选 `upgrade`（标注 simulated）：两项都过 → 切换为 `strategy_002`，
  留下 `activate` 事件（`previous_id=strategy_001`，checks 全 True，jev.decision=upgrade）。
- 切换后新建 `ResearchLoop`：pin 到 `strategy_002` —— 证明候选真的影响了下一次研究。
- 未过评估的候选：`verdict=keep` → 代码先拦，`Jev 被调用次数=0`。
- `test_strategy_decision.py` 13 tests OK；全量 309 tests OK。

未做：第七步（上线后回滚）、第八步（前端进化记录）。

## 鱼鱼 update 2026-11-14 · 第七步：上线后回滚（已验证）

- 新增 `strategy_rollback.py`：`freeze_rules()` 在**上线前**冻结回滚规则（digest 可查）；
  `evaluate()` 只比较实测值，触发一条就 rollback；指标算不出 → `insufficient_evidence`（保持现状，不误切）。
- `strategy_store.rollback()`：回滚是止损动作，**不需要 Jev 再选一次**，但必须带 `triggers` + `evidence`，
  否则直接 ValueError；事件里记 `from`、触发项和完整检查结果。
- `rollback_if_needed()` 加了一道保险：要监控的那版不在线上时，拒绝乱切。

已验收（`artifacts/step7-acceptance.txt`）：

1. 两套 digest 先冻结：升级 `b9c18d27…`、回滚 `6ecbc330…`；
2. 评估：旧重复率 0.5 → 候选 0.0 → `upgrade`；
3. 模拟 Jev 启用 → `active=strategy_002`；
4. 上线后监控：基线重复率 0.0 → 上线后 0.5，触发 `duplicate_rate_rise`（实测 0.5 > 阈值 0.3）；
5. 自动回滚 → `active=strategy_001`，随后新建的 ResearchLoop pin 回 `strategy_001`；
6. 失败版本 `strategy_002` 仍是 `retired`，`planning_prompt` 原样保留 —— 删除失败案例是不允许的；
7. 事件时间线：`activate → rollback`，rollback 事件带 triggers 与 evidence.checks。

`test_strategy_rollback.py` 12 tests OK；全量 321 tests OK。

未做：第八步（前端如实展示）。

## 鱼鱼 update 2026-11-14 · 第八步：前端如实展示（已验证）

### 千问工作记录（① 分析依据）
- 顺序固定：观察（模型原文 / 程序统计，分开标）→ 基础操作计划（它自己选的 ops，或"默认清单"）
  → 工具结果（代码真跑的表）→ 验证（代码执行的 4 项检查，✓/✗ 带实测值）→ 是否转交（交给 DeepSeek 几个问题）。
- 无模型可用时，整块写「程序规则 · 非模型输出」，不显示成模型已分析；
  有 API 模型时署名「轻量模型 API · <模型名> · 模型原文」，并显示调用次数/token/费用状态。

### 策略进化记录
- 新增 `GET /strategy/state`（当前 active + 全部版本 + 版本差异 + 最近 40 条事件）
  与 `GET /strategy/failures`（读 `runtime/last_failures.json`，即 failure_modes 的真检测结果，不在接口里现编）。
- 页面显示：当前正式策略与 digest、各版本状态 chip（含 `模拟` 标记）、当前内容、候选与当前版本的差异、
  事件时间线（已升级 / 已拒绝或保留 / 待评估 / 已回滚，带 triggers 与来源）、失败模式（类型 + event_refs + measurements）。

### 数据来源
- `tools/seed_strategy_demo.py`：真跑两组对照评估 → 真 `compare` → 真 `decide`；
  唯一模拟的是 Jev 回答（无 key），事件里 `by=seed-demo-simulated`、`jev.model=simulated-jev`，页面上可见。
- 失败模式来自第三步真实跑出的检测报告（`repeated_proposal`，refs `event_007` / `event_013`）。

验收：headless 截图两块均正常渲染（`evo/out/10-new-analysis.png`）；全量 **321 tests OK**。


## 赛题对照完善 · 2026-11-14T13:55:42+08:00

已验证：新 `/research-agent` 页面源码与单模型共享预算后端已接入；282 项离线回归通过。规则模式真实两轮共 39 次训练，第一轮平均验证宏 F1 0.8498818266475578 → 0.9082777159807194 并升级；第二轮上下文压缩无可测收益，保持旧策略。最终独立审计宏 F1 0.899749373433584。独立重放的数据摘要、质量差值和最终测试完全一致；运行时源码 SHA256 全部校验通过。

已验证：API 同步处理函数串联后台执行、读取、ZIP 导出与人工恢复；失败 / 未知用量与单模型多角色共享预算均有回归测试；JS 语法检查通过。

未验证：真实云端模型能力、真实 token 节省、供应商账单、HTTP / 浏览器端到端及视觉布局。环境禁止本机 TCP（WinError 10013），完整原套件中止；离线 282 项不代表完整套件。Edge 静态截图未生成。不能将规则策略改进宣称为 LLM 自主进化或跨域能力。

证据：`artifacts/topic-delivery-verified/`、`artifacts/topic-delivery-replay/`、`artifacts/topic-regression-final.txt`；详细核查：`docs/topic-alignment-2026-11-14.md`。


## 用户纠错学习与扩大验证 · 2026-11-14T14:24:25+08:00

已验证：357 项发现测试中 344 通过、13 个真实 TCP 供应商夹具因环境限制跳过。标准 TestClient 的实际 HTTP/ASGI 请求、后台执行、跨 app 持久化、纠错 API、导出和旧手动入口的防绕行均通过。通过仅用于测试的无 socket 事件循环运行，不修改系统配置。

新增 corrections.py：识别白名单纠错、全局/数据范围、SQLite 持久化、运行快照、有效默认参数去重、禁用模型、训练前自动补缺失指示、禁止付费调用、未知 pending、冲突拒绝、撤销及审计。反馈原文不作为高权限 prompt 或代码执行。

实际消融：3 类错误×3 份数据×3 臂，共 59 次真实训练。无记忆与仅提示词均出现 12 次错误，执行约束为 0；无关数据误拦截 0。每臂训练 21/21/15 次。缺失指示场景验证宏 F1 .916614→.924958；禁止随机森林会限制精度，未隐瞒取舍。独立重跑的数据、错误数、训练数和 F1 一致，源码摘要全部正确。

恢复了原安装包缺失的两份 benchmark CSV，已知泄漏样本哈希与 known_leaks.json 匹配。修复 executor 的实际预处理没有写入结果这一证据缺口。

页面新增纠错输入/记忆列表/撤销与创新证据。JS 语法、HTML 元素引用和资源存在性已验证。未完成：真实 TCP 浏览器交互（环境禁止本机 TCP，且浏览器明确拒绝 file 协议，已停止该尝试）；真实 LLM 验证脚本确认服务器尚无模型凭据、0 调用。纠错提案为刻意回放的脚本，不能宣称模型权重学习或一般认知错误永不再犯。

当前报告：docs/feedback-learning-2026-11-14.md；证据：artifacts/feedback-learning-final/、artifacts/feedback-learning-replay/、artifacts/asgi-regression-accepted.txt。


## 用户纠错学习与扩大验证 · 2026-11-14T14:27:17+08:00

已验证：357 项发现测试中 344 通过、13 个真实 TCP 供应商夹具因环境限制跳过。标准 TestClient 的实际 HTTP/ASGI 请求、后台执行、跨 app 持久化、纠错 API、导出和旧手动入口的防绕行均通过。通过仅用于测试的无 socket 事件循环运行，不修改系统配置。

新增 corrections.py：识别白名单纠错、全局/数据范围、SQLite 持久化、运行快照、有效默认参数去重、禁用模型、训练前自动补缺失指示、禁止付费调用、未知 pending、冲突拒绝、撤销及审计。反馈原文不作为高权限 prompt 或代码执行。

实际消融：3 类错误×3 份数据×3 臂，共 59 次真实训练。无记忆与仅提示词均出现 12 次错误，执行约束为 0；无关数据误拦截 0。每臂训练 21/21/15 次。缺失指示场景验证宏 F1 .916614→.924958；禁止随机森林会限制精度，未隐瞒取舍。独立重跑的数据、错误数、训练数和 F1 一致，源码摘要全部正确。

恢复了原安装包缺失的两份 benchmark CSV，已知泄漏样本哈希与 known_leaks.json 匹配。修复 executor 的实际预处理没有写入结果这一证据缺口。

页面新增纠错输入/记忆列表/撤销与创新证据。JS 语法、HTML 元素引用和资源存在性已验证。未完成：真实 TCP 浏览器交互（环境禁止本机 TCP，且浏览器明确拒绝 file 协议，已停止该尝试）；真实 LLM 验证脚本确认服务器尚无模型凭据、0 调用。纠错提案为刻意回放的脚本，不能宣称模型权重学习或一般认知错误永不再犯。

当前报告：docs/feedback-learning-2026-11-14.md；证据：artifacts/feedback-learning-final/、artifacts/feedback-learning-replay/、artifacts/asgi-regression-accepted.txt。


## 真实 LLM 与 TCP 严格复测 · 2026-11-14T14:38:14+08:00

Strict TCP actually ran13, skipped0, passed4 configuration tests, failed4+errors5. Actual loopback bind/listen succeeds but connect denied WinError10013. Full strict suite358:349pass,9failed/errors,0skips. Live learning attempted:blocked because no server model credentials,0real calls. Default runner no longer skips TCP. No live learning success claimed.

新增 tools/run_tcp_checks.py、tools/evaluate_live_learning.py、tools/verify_network_environment.py。真实模型评测区分拦截前原始提案错误和拦截后执行错误，带质量门槛、真实 usage 与源码快照；未接入模型则 blocked，不生成假结果。事件记录已深拷贝，防止自动修正回写原始模型证据。证据：artifacts/strict-verification-status.json；详情：docs/live-verification-status-2026-11-14.md。


Live run diagnosis:18 failed attempts and0 valid proposals. Local scientist_rejected reason is latin-1 header encoding failure, consistent with a non-ASCII API key. Added ASCII key preflight and early blocked report. Configuration regression:2 tests passed. Actual LLM learning remains unverified.


User authorized DeepSeek/Jev credentials. DeepSeek only:1 hidden-input in-memory API attempt failed before valid response. DNS resolves; direct provider TCP connect denied WinError10013. Jev not called. Added local getpass launcher with connection preflight before18-call benchmark; syntax check passed. No credential values saved.


Fixed public model API client forcing direct connections: inherit configured HTTPS/system proxy for public endpoints, keep loopback direct, preserve TLS certificate validation and no-redirect behavior.4 configuration/proxy unit tests passed. User-reported TLS EOF not yet live-retested; no claim of successful API connection.


User local credential-free diagnostic confirms Python and Windows curl HTTPS GET both401. Updated local launcher to separate authenticated model-list GET, advertised model validation, and generation POST. No claim that authenticated generation is working yet. Syntax check passed; no keys persisted.


Implemented /showcase interactive real-training demonstration.2 ASGI integration tests passed:fits3/2/3,duplicates1/0/1,guardblocked1,equal before/after F1,scoped memory persistence/revoke,real isolated strategy rollback,evidence download,no pollution of formal feedback. JS syntax passed. Scripted proposals clearly labelled;0 external LLM calls;browser visuals not verified.
Guide: docs/showcase-guide.md


Added double-click local showcase launcher:detect existing dependencies,start hidden loopback uvicorn,wait readiness,open default browser,reuse existing service,and surface startup log on failure. Syntax verified; actual desktop launch not verified in restricted execution environment.


Implemented requested source-to-classifier-to-low/high-workers-to-final-reviewer ML chain. Natural-language input uses bounded public OpenML search/select/download tools; CSV supported. Default API/UI replaced with reduced ML app; old routes only in explicit legacy factory.12 focused tests passed with substitute API/catalog and actual sklearn training, including natural intake, low failure escalation, classifier confidence/capability controls, mandatory final review and budget reserve. No live API/data-catalog success claimed. New one-click entry uses8011 to avoid old8010 process. No fine-tuning/local Qwen invocation in active chain.
Guide: docs/ml-pipeline.md; test log: artifacts/ml-chain-tests.txt.

Reduced active product to collection, routing, workers, final review and report export. Removed feedback endpoints and persistent correction-store initialization/use from ML API. Static serving now allows only current app JS/CSS, blocking legacy demo assets. Historical modules retained inactive. 12 focused pipeline tests passed; real cloud connections not validated.

Simplified ML frontend to request/progress/results; removed sidebar, station cards, repeated technical copy and obsolete CSS dependency. Advanced budgets remain collapsed. 12 pipeline tests passed; visual browser check not performed.

Configured default trio: jev-latest, deepseek-flash, deepseek-v4-pro. Added hidden-input process-only credential launcher on8012; no supplied credentials stored.12 pipeline tests passed and default role mapping asserted. Live vendor connection not verified because execution network allows neither vendor host.

Fixed web startup: removed blocking credential prompt and unified launchers on8012 using ml_api factory. Home/CSS/JS/config returned200 in ASGI checks. Background start attempted but loopback checks returned URLError in execution environment, so live browser reachability remains unverified.

Added password inputs and in-memory API credential endpoint, same-origin enforcement, no browser storage or credential persistence.12 pipeline tests and ASGI credential save/cross-origin rejection/no-key-disclosure checks passed. Existing running server requires restart to load changes; live vendor connection unverified.

Fixed web credentials precedence over legacy role key environment variables. Added explicit paid three-model connection check with safe per-model HTTP/transport diagnostics and UI button.12 pipeline tests passed; no live vendor verification performed. Keys remain process-only and expire on restart.

Observed actual browser showing old Missing API key/model for planner config. Added version endpoint/header and visible V3, no-store caching, project introduction section. Launcher now checks exact version and releases only identified old Python research server on8012 before starting current app.12 pipeline tests passed plus version/cache/page ASGI checks. Native UI activation failed; actual host service upgrade not verified.

Changed web save to explicit save-and-verify: real connection checks automatically run after saving. Local404 on key/check routes now clearly identifies stale server rather than invalid provider key.12 pipeline tests and mock-client ASGI three-role verification passed. No real provider success observed; live connection remains unverified.

Inspected failed run82058: HTTPError during data collection after2 model calls984 known tokens; prior code discarded status. Added bounded HTTPS OpenML-only redirect handling and structured Chinese stage/status errors; revised natural input as plain research need.12 pipeline tests passed plus mocked safe redirect/unsafe redirect checks. Exact historical HTTP status unavailable and live OpenML download not verified; no claim it is fixed end-to-end.

Added live five-node workflow with SVG fork/merge lines, CSS breathing lights and moving strokes, actual role model labels, event-based search/candidate/selection/download/worker/training/review summaries. Animations stop on completion/failure; reduced-motion supported. Version V5.12 pipeline tests, JS syntax, Node mocked-DOM active branch/failure-stop checks passed. Actual browser visual/live vendor validation not performed; UI shows observable actions/decision summaries rather than private reasoning.

Investigated latest failed run0cdf: source model returned304 tokens then HTTPError from collection; exact HTTP status lost by old server. Fixed stale failed-run auto-resume; UI identifies legacy generic errors and backend version mismatch. OpenML412 no-match now returns empty candidates.12 pipeline tests, JS syntax and mock412 checks passed. Actual OpenML iris search succeeded with8 candidates; download probe pending at time of log. V6.

Live OpenML download verification succeeded for dataset61 iris:4753-byte CSV,target class. Actual search and download verified; full model-driven run and host V6 activation still unverified.

Root cause in rundba0: exact keyword customer churn returned0; prior source quit immediately. Added max3 deduplicated name/core-keyword searches with realtime retry events and Chinese no-match/selection errors, attempted query provenance, no extra LLM calls. Live OpenML churn query returned3 candidates.13 pipeline tests passed including full substitute-client retry pipeline. V7; actual requested churn dataset download/model-driven full run not verified.

Replaced default source search/download with offline LocalCatalog. Persisted4 real CSV datasets and JSON index with bilingual aliases,target,provenance,SHA256. Real OpenML40701 churn downloaded once(496569 bytes), now cached locally. Verified Chinese/English customer churn queries and actual ResearchSession baseline training, validation macroF1 .5969223838484428; no holdout evaluated.15 tests passed incl network-disabled local search/load and tamper rejection; JS syntax passed. UI V8 lists local datasets. LLM API full-run and host startup unverified. docs/local-catalog.md.

Confirmed actual latest run37d83 source returned prompt152/completion100 tokens, then old server emitted provider OpenML and0 candidates; classifier/workers/reviewer not invoked. Added isolated Start-Local-Research.cmd on8018 with explicit app-dir so8012 stale process cannot be reused. Version endpoint exposes local_catalog/count and UI identifies local search.15 tests and launcher compile passed. Host8018 launch and full real-model chain not verified.

Fixed post-baseline JSON failure path by removing redundant source planning API call: prepared data and original research objective go directly to classifier. Configured all ML generative clients for JSON response_format; DeepSeek thinking disabled for these bounded structured calls; length-truncated responses explicitly rejected; outer fenced-object normalization added.17 tests passed incl source-not-called and wire payload checks; JS syntax/compile passed. Corrected8018 stale-service restart support; V9. Real provider completion after change not verified; .5969 baseline from screenshot is not full research success.

Real end-to-end HTTP acceptance succeeded on running V9 service8018 using existing in-memory credentials: run0a103246a0c54f768a9a0197f9167141 completed/approved. Real roles source(Pro),classifier(Jev),low(Flash),reviewer(Pro);5 calls3292 known tokens10.596729sec. Real validation macroF1 .5969223838484428 -> .88613755825099; no holdout used, no cost claim. Export HTTP200 and report score verified. Added ?run= deep link; live served JS contains update. Browser visual interaction not yet verified.

User requested repeat live run. New real HTTP run8ae28a9bf09e4787add98451b8ba2180 completed/approved on8018. Actual source,classifier,low,reviewer returned;5 calls3277 known tokens10.969788s. Validation macroF1 .5969223838484428 to .88613755825099; no holdout evaluated. Independent new run, not replay of prior report.

V10 additive delivery+experience update. Backed up prior source under artifacts/backups. Structured research answer, actual validation-row predictions/probabilities/counts and CSV endpoint, sampled validation permutation evidence/limitations sent to final reviewer. Champion is recovered from approved reports keyed to exact dataset,target,seed,metric,positive class,feature review/exclusions, then retrained first; duplicate champion can be retained without retraining; no repeated-gain claim.19 tests passed including predictions identity/probabilities/no-test access, CSV export, persisted next-run champion and target mismatch. Actual saved churn champion RF retrained F1 .88613755825099,1000 validation predictions counts0:861/1:139. Revised feature sampling recomputed. New V10 live cloud end-to-end not run; old V9 service not restarted, credentials unchanged. Guide docs/predictions-and-experience.md.

V11 added scoped measured lesson loop via SQLite. Records baseline/improved/no-gain configs and observed scores/reference, execution errors separately; source run tracked. Worker consults persisted lessons; identical no-gain config can be skipped only when score<=current incumbent; no model-family blacklist or cross-target transfer. Report/partial includes lessons and UI presents summaries.21 tests passed after fixing SQLite connection closure on Windows, plus JS syntax/compile. No additional cloud calls or weight training. New live cloud run not performed, running service not restarted. docs/experiment-lessons.md.

Investigated user complaint against live8018: /ml/version is still ml-chain-9, latest runc e40b1 (ce40b1f72a524c90b3d482ae2183e04b) has no experience_loaded and starts LR .5969; this confirms pre-memory backend. Added mandatory frontend/backend V11 handshake and visible stale-backend warning; form refuses paid submissions when backend version mismatches, preventing new frontend from pretending lessons are active.21 tests and JS syntax passed. Current running9 backend not upgraded; no credentials copied/extracted. New V11 memory behavior remains verified by scoped persistence/next-run baseline tests, not live on8018.

Verified timestamp concern: CMD last modified17:43:26, wrapper17:02, invoked open_showcase and engine18:46:07, API18:46:45 (host local). CMD was unchanged because it delegated to edited Python. Added functional startup diagnostics showing expected version11, exact project path and URL, missing-launcher check, direct launch of current helper. Did not claim running backend upgraded. Read-only path/content checks completed; desktop launch not verified.

Reduced frontend polling/render cost: unchanged responses skip DOM rebuild, raw JSON serializes only when details expanded, compact poll limits predictions preview20 while full exports unchanged, hidden tabs poll5s and pause animations, removed animated blur/shadow filters in favor of light opacity. Regression compact/full persistence and Node unchanged-render checks passed; JS syntax passed. Backend compact endpoint requires updated server; no live browser FPS improvement claimed.

Added explicitly labelled force page reset: clears only current run UI pointer and query link then reloads, preserving API credentials, backend work and report files. Scope explained next to control; does not claim backend cancel/restart. JS syntax verified.

Confirmed live V11 queued task d71e failed before model/data processing: startup log shows WinError5 replacing state.json, then identical error prevented failed-state write. Updated save with unique temp files, bounded retry and immutable state journal fallback; run GET resolves newest valid state.23 tests passed incl injected persistent Windows replace denial and live API confirmed repaired d71e status failed/task_start, original queued snapshot preserved. No model executed for that blocked run. Fix build11.1 requires server reload; no successful post-fix live model run claimed.

V12 user-facing answer separated from developer tuning. Final reviewer now emits answer addressing original goal using evidence, predictions, scope/unsupported parts; summary reserved for technical audit. Missing answer with research outputs forces revise. Main UI renders reviewer answer, never substitutes deterministic F1 text or legacy technical summary. Budgets/final-test controls,F1,tokens,permutation numbers,lessons and raw logs moved into collapsed developer panel; prediction results remain user-facing. Reviewer output budget2048 and role-aware reservation avoid truncation; old reports preserved.24 tests plus JS syntax passed; real cloud V12 end-to-end and desktop activation not yet verified.

V13 persistent credential settings implemented. Windows current-user DPAPI encrypts backend credentials.dpapi; startup loads saved keys; partial settings edits preserve untouched supplier key; no plaintext/browser storage/full-key echoes, validation errors sanitized. Failed save retains old config. UI detects old backend without persisted:true rather than falsely claiming persistence.26 tests passed including real DPAPI roundtrip, app recreation, ciphertext absence of plaintext, partial edit, save failure and same-origin validation; JS syntax/compile passed. No user real keys saved/migrated in this turn; running host service not restarted. First save required after upgrading. docs/credential-settings.md.

V14 developer comparison shipped in code: paired routed vs high-only tool-using ML research with identical frozen CSV,target,split,seed,baseline,per-arm budgets and high-model reviewer; no memory read/write or holdout access. API enqueue/poll/export, 1or3 seeds, randomized arm order, developer table for tokens/time/F1/minorityF1/AP/confusion/answers/structure checks, resume links.29 tests and JS syntax passed. Two real paid comparisons executed using encrypted saved credentials without disclosure: eb694 first routed had2 failed unknown-usage calls and needs_revision, retained unchanged; a91485 bothapproved, high-only3559 tokens24.2481923sec macroF1 .8742536345, routed4108tokens27.8115423sec macroF1 .8861375583. Thus no universal efficiency claim. Fixed shared budget to count provider-returned usage even if response parsing fails; failure-with-usage regression passed. Actual V14 API browser deployment not restarted/visually verified; code docs/model-comparison.md, evidence artifacts/live-comparison-acceptance.json.

V15 uncertainty escalation implemented: unknown/invalid/low-confidence or failed classification -> once-only real high-model routing review with original question, training profile and uncertainty -> validated capability dispatch -> actual worker execution -> final audit. Unresolved cases return needs_input/blocked, no predictions/final test/final model audit; baseline remains developer evidence only. All failed workers also block rather than synthesize an answer. Costs remain within shared limits, reserve execution+audit calls. Developer rules and UI recheck/redispatch/clarification states added.34 tests passed plus JS syntax. Real acceptance first a337 high call failed and correctly blocked with no predictions; second6629 actual Jev call with deliberately injected confidence0.1, real high review/worker/audit approved with4calls4681known tokens1000validation predictions,no holdout. Artifacts preserved; tools/verify_route_recheck.py documents fault injection. Running host backend not restarted toV15; visual deployment not claimed.

V16 per-model official USD cost accounting implemented, distinct from unverified invoice_actual_usd. Preserve cache hit/miss outside basic token totals, model/provider/request UTC metadata, known failed-call usage. Price snapshot verified2026-11-14 from official DeepSeek and TypeSafe docs; weekend/offpeak exact, unknown cache/holiday periods ranges, unknown provider/usage no invented0. Main/comparison developer tables show each model input/output/cache and calculated/invoice costs; paired cost delta only fully computable.37 tests previously passed incl pricing/cache/no-double-count/unknown bounds; final JS syntax/compile checked. Live three short calls succeeded: Flash39in6out $0.00000945, Pro36in5out $0.00003366, Jev433in46out $0.000018186 at list price; these are not invoice deductions or full research bills. Evidence artifacts/cost-accounting-acceptance/e2859d35ea1d48a8811938d0b6adff36/usage.json. Running service not restarted toV16.

V17 live official-price refresh implemented. Anonymous HTTPS fixed-source fetch/strict parsing for DeepSeek and Jev,5minute TTL,manual refresh,async single-flight and last-good persistent cache. Provider failure/structure/unit change marked stale/reference rather than pretending live; no model calls or API keys for refresh. Each model request freezes price snapshot, preserving past costs after updates. Research/comparison gateways use feed; GET /ml/pricing and POST /ml/pricing/refresh, frontend refresh status/timestamps/source/current rates.40 tests passed incl parser unit/order validation, failure cache restore, immutable request snapshot and API refresh scheduling. Actual official fetch/parse/cache succeeded both providers2026-11-14T12:21Z (~20:21 Beijing), values match current official pages. Code compile and JS syntax passed. Host service not restarted toV17; no claim of invoice actual charge access. docs/live-pricing.md.

V18 adds partial analysis continuation instead of treating absent causal identification as total failure. Explicit causal/leading questions trigger high method review even with confident initial classification. High may select1-3 existing reviewed columns for partial association; local worker computes training-only group rates,N,positive counts,Wilson95% intervals,small-sample flags. Final reviewer answers lack of causal support plus actual observed associations/limits; prediction output empty,no holdout access, no predictive-champion promotion. API partial_completed/UI non-error completed-with-limits, needs_input amber; invalid/missing fields still clarify. Comparison suppresses predictive metrics for association scope and savings deltas across unmatched scopes.43 tests passed plus compile/JS syntax. Actual real-API run c45843bc8f33476b9c613905baa76602 completed partial association of number_customer_service_calls on3000 training rows, answer explicitly rejects causal claim and reports observed rates. Later prompt tightened to avoid guessing merged-group members and overinterpreting small samples. Running backend not restarted toV18; docs/partial-association.md.

## 独立验收（开发者/测试视角）· 2026-11-14T20:5x (yuyu)

按用户要求做了一次真实验收，不是只给建议：读代码、起服务、调接口、看页面、跑真实模型。真机实例 `http://127.0.0.1:8018`，端口独占。

### 一、环境核实（通过）

- 进程 47640 启动 20:35:59，晚于全部相关源码最后修改时间（≤20:33:06，`ml_engine.py`/`association_analysis.py` 20:32:21、`comparison_benchmark.py`/`ml-app.js` 20:33:06），`/ml/version` = `ml-chain-18`，命令行 `--app-dir` 指向本项目目录：运行的确实是 V18 代码，不是页面文字。
- 密钥来自 `artifacts/ml/credentials.dpapi`（464 字节，DPAPI blob 头 `01 00 00 00 D0 8C 9D DF`，不含明文 `API_KEY`）；`/ml/credentials/status` = deepseek/jev 均已配置、`persisted:true`、`load_error:null`。
- **重启实测**：为加载修复杀掉 8018 旧进程并重启，新进程未输入任何密钥即 `configured:true` —— 加密设置自动载入成立。
- 密钥外泄扫描（artifacts/logs/md/json，排除 .venv）：唯一命中是测试夹具 `test_corrections.py:50` 的 `'sk-'+'a'*40`，非真实密钥。

### 二、基线回归

- 修复前全量 `unittest discover -s tests`：407 项，11 fail + 10 error。全部集中在 9 个旧界面套件，原因是它们仍 `from api import create_app`，而该入口已换成精简 ML 应用（旧界面在 `api.create_legacy_app`）。
- 用探针把 9 个套件指向 `create_legacy_app`：67 项全过、断言一条未改 —— 确认是**测试接线过时**，不是功能回归。
- 已改：9 个文件仅替换导入行（`test_api/test_corrections/test_demo_flow/test_exports/test_integrity/test_live_demo/test_research_jobs/test_showcase/test_topic_alignment`）。
- 修复后：**409 tests OK**（58.1s）。仍有一条既有 `ResourceWarning: unclosed socket`（测试内真实 loopback 套接字，历史遗留）。

### 三、真实研究（通过，24/24 一致性核对）

run `c3189877019749bbbb4f7ff110306814`，需求“请使用本地客户流失数据，研究客服通话次数与客户流失的关系，并给出验证样本的预测结果和研究局限。”

- 4 角色真实 API：5 次调用 / 5854 已知 token / $0.00297258（Flash 1504+118、Pro 3278+274、Jev 634+46），0 次未对账；2 次真实训练；`approved`。
- 交付：1000 条验证预测（0:867 / 1:133），`best=exp_001` 宏 F1 0.888274、准确率 0.947；未开最终测试集。
- 一致性核对（`artifacts/acceptance-20261114/verify_run.py`）全过：回答里的 0.888/0.947/867/133 与报告一致；`predictions.csv` 1001 行且分布一致、source_row 无重复；导出接口与报告一致；`final_test:null`；每个模型 `invoice_actual_usd:null`。
- 回答本身合格：直接回答问题、说明特征重要性≠因果、列出局限。

### 四、发现并修复的真实缺陷（因果/关联路径）

- 失败实例 `9a351c5d5d824825b4580b671db6c31f`：`needs_input`，理由是“高级复核没有给出有效字段”——**把可完成的任务停住了**。
- 复现（3 次诊断，真模型）：失败那次的原始返回是 `"association_columns": ["number_customer_service_calls", "class"]` —— 高级模型把**目标列 `class`** 回声进了字段列表。旧代码 `any(c not in session.x.columns)` 整表判无效 → 直接问用户。目标列本来就不是特征，丢掉它即可。
- 修复 `ml_engine.resolve_association_columns`：容忍目标列回声、大小写、连字符/空格、单个字符串、`字段（注释）` 写法；仍然拒绝杜撰列（`test_partial_review_cannot_invent_columns` 保持有效）；无法解析时记 `route_recheck_invalid_columns`（带模型原始返回，≤200 字）便于下次定位，并把可选字段列进追问。
- 新增单测：`test_partial_review_drops_target_echo_and_keeps_the_real_feature`、`test_partial_review_accepts_case_and_single_string_columns`（`tests/test_ml_pipeline.py`）。
- 复测 `cff317c07aab4eac8f399725a43fadd3`：`partial_completed`，5 次调用 / 5837 token / $0.002581382，事件里出现 `association_columns_adjusted`（requested 含 `class`、used 只剩真实特征）。3000 训练行、Wilson 95% 区间、0 条预测、未碰最终测试集、未晋升预测冠军；回答明确“不能支持因果结论”并引用实测比例（0–3 次约 10.8%–11.3%，4 次 43.4%，5 次 57.1%）。

### 五、高级模型升级（4 种注入，真实执行）

只替换分类器/复核器，其余调用全部真实（脚本 `artifacts/acceptance-20261114/esc_acceptance.py`，注入点写在报告 `acceptance_setup`）：

| 情形 | 结果 |
| --- | --- |
| 真分类调用后把 confidence 改 0.1 | 复核→重新派发→实际训练 2 次→1000 预测→`approved`（4 调用/5137 token） |
| 真分类调用后把 category 改成不支持的值 | 同上，且 `expert_calls=1`（复核判给高级工作站）（4 调用/5139 token） |
| 分类站抛异常（无真实 Jev 调用） | 复核→实际执行→1000 预测→`approved`，`unreconciled_calls=1` 如实记未知 |
| 复核调用失败（注入） | `blocked`，0 预测，`review.decision=revise`，未碰最终测试集 —— 没有假装完成 |

注意：注入包装器未继承 `billing_provider`，这几次的**费用合计因此不完整**（记 unknown 而非 0），属注入副作用，不是线上路径缺陷（真实 run1 三个模型费用都算得出）。

### 六、两轮经验复用（通过，附一处小瑕疵）

- 第二轮 `a0d1fc08ec91479fbe9e5a2fc3e03819`（同数据/同目标/同条件）：开头 `experience_loaded`（HGB，previous_score 0.888274）→ 先重训验证 → 再提新方案 RF(150,12) → 0.8911556 **improved**、5 调用/6017 token/`approved`、1000 预测 871/129、回答 F1 0.891/acc 0.949。
- `lessons_consulted` 出现、`lesson_recorded` 落库；第一轮被回滚的配置（HGB 400/3/10/0.03）第二轮没有被原样重提。
- 瑕疵：分数完全相等时冠军按目录扫描顺序取，不看新旧——第二轮记的来源任务是更早的 `606bbc9311ee4ed68861e0c6fae90fb0`，而不是刚跑完的第一轮。不影响复用，但“来源任务”不是最近的。

### 七、协作 vs 高成本单模型对比（通过，不作优越性结论）

`21605f2f152a477786743fa394557858`，seeds[42]，同一 CSV/target/seed/预算，两边禁用历史经验、不碰最终测试集：

| 模式 | F1 | token | 调用 | 耗时 | 官方价核算 |
| --- | --- | --- | --- | --- | --- |
| 多智能体协作 | 0.88614 | 4415 | 3 | 8.87s | $0.00223468 |
| 高成本单模型 | 0.87425 | 3729 | 2 | 7.83s | $0.00237244 |

配对差：F1 +0.0119、token +686、费用 −$0.00013776、耗时 +1.04s。协作本次 F1 高、费用略低但更慢更多 token；1 对实验不构成普遍优越，页面文案也已这么写。费用均标“未对账”。

### 八、页面（headless Edge 实测渲染，未做交互）

截图 `artifacts/acceptance-20261114/page-approved.png`、`page-needsinput.png`、`page-comparison-tall.png`：状态文案（“已完成可支持的关联分析（非因果）”“需要补充信息”）、关联表、预测表、对比表的数字与接口完全一致；`needs_input` 是琥珀色“需补充信息”而非红色中断。

发现的**界面问题**（未改，待定）：

1. 流程图连线渲染成一个大方框（SVG 路径几何 + 无箭头），节点 03A 与竖线重叠，读不出“分叉/汇合”。
2. `needs_input` 时表单下方仍显示“请检查部分证据和接口配置。”，与“需要补充信息”语气冲突（`render()` 对 needs_input 落到了默认文案）。
3. 长任务跨过 5 分钟价格 TTL 后，审核那次调用的价格快照被标 `stale` 并告警“未在线核验最新价格或刷新失败”——行为正确（旧价保留、标注过期），但“刷新失败”的措辞会被误读成网络故障。

### 九、未验证 / 没做

- 浏览器**交互**（点击、强制重置、localStorage、轮询动画启停）只做了静态渲染截图，没有真实点击验证。
- 价格刷新**失败**的在线注入没做，只有单测（保留上次价格并标 stale）。
- “只改一个密钥不覆盖另一个”**已用真实 HTTP 验过**（`artifacts/acceptance-20261114/verify_credentials_http.py`，临时数据目录 + 真实 DPAPI + 假密钥，14/14 通过）：第一次只存 DeepSeek、第二次只存 Jev，存下来的 DeepSeek 字节不变；空请求被 400 拒绝且不清空；跨源写入 403；磁盘上只有密文。**线上真密钥文件未被写过**（写了就毁密钥）。
- 供应商账单实扣：无权限，全部仍是“未对账”。
- 记忆：本次把 8018 换成了我起的进程（`--app-dir` 同参数），用户下次双击 `Start-Local-Research.cmd` 仍会复用它。
- 本轮累计真实花费约 **$0.03**（官方价核算，未与账单对账）。

证据目录：`artifacts/acceptance-20261114/`（poll/verify/esc/diag 脚本 + 三张截图）。

## 对比实验"是否是假数据"复核 · 2026-11-14T21:1x (yuyu)

用户追问对比页的数字可不可信、有没有编造。做法：**不读它自己的 metrics.json 当答案，而是从冻结 CSV 重切分、按报告里记的 best 配置重训、用 sklearn 重算每一项**，再和发布值比对。脚本 `artifacts/acceptance-20261114/verify_comparison.py`（无模型调用、无联网）。

对 `21605f2f152a477786743fa394557858`（seed 42）：**39/39 通过**。

- 冻结 CSV 的 sha256 = `c81d8ccb…6252`，与 `data/catalog/index.json` 里 customer_churn 记录一致；两个 arm 目录里的 data.csv 与顶层 data.csv 三者同哈希 —— 同一份真实数据，没被换过。
- 重训复现：协作 `.88613755825099`、单模型 `.8742536344755971`，与发布值差 <1e-9；少数类 F1（`.8042704626` / `.7847222222`）与 AP（`.8453716279` / `.8304222068`）同样逐位一致。
- 混淆矩阵 `[[832,26],[29,113]]` / `[[825,33],[29,113]]` 与重训一致，且用它自己反算的宏 F1 也等于发布值 —— 三个数字互相咬合，不是各填各的。
- 记账自洽：调用数 = 请求记录条数；已知 token = 各请求 usage 之和；未对账调用按未知计数；每 arm 费用 = 各模型费用之和；`invoice_actual_usd` 恒为 null。
- 配对差（F1 +0.011884、token +686、费用 −$0.000137756）都能从两个 arm 重算出来。

**反"写死"验证**：另跑一次同条件、只换种子（seed 7，`b8566cf4085b40b19836b62d1e58b0eb`）：协作 `.8770777820`、单模型 `.8735878302`，token 4407 / 3734，费用 $0.001972924 / $0.00246136 —— 数字随划分变化，且**同样 39/39 独立复现**。代码侧也查了：`comparison_benchmark.summarize` 的指标全部来自 `pipeline.predict()` + `sklearn.metrics`，文件里没有写死的分数常量（唯一命中是注释）。

值得提醒：两个种子的 F1 差从 +0.0119 掉到 +0.0035，说明这个差距本身不稳定，**1 对实验不能当结论**，页面文案也这么写着。

## 对比矩阵 + 突发状况 · 2026-11-14T21:1x (yuyu)

用户要求把"协作 vs 全用贵模型"在**不同数据**和**每个突发状况**下都测一遍。

### A. 六对真实对比（不同数据集 / 不同种子）

| 数据 | 种子 | 协作 F1 | 单模型 F1 | ΔF1 | Δtoken | Δ费用(USD) | Δ耗时(s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| churn | 42 | .88614 | .87425 | **+0.0119** | +686 | −0.0001378 | +1.04 |
| churn | 7 | .87708 | .87359 | +0.0035 | +673 | −0.0004884 | +2.26 |
| churn | 2026 | .88099 | .87760 | +0.0034 | +667 | −0.0002204 | +0.28 |
| iris | 42 | .93333 | .93333 | 0 | +715 | −0.0003839 | +1.51 |
| wine | 42 | 1.00000 | 1.00000 | 0 | +738 | −0.0001922 | +3.51 |
| breast_cancer | 42 | .99062 | .99062 | 0 | +661 | −0.0003117 | +2.17 |

六对全部 `approved`、结构检查 3/3、0 未对账调用、未碰最终测试集。**每一对都用 `verify_comparison.py` 独立重算（41 项/对，全过）**。

结论（比"谁更好"更重要）：

- **贵模型没有换来精度**。iris / wine / breast_cancer 上两边最后都选中同一个 `logistic_regression(max_iter=1000)`，F1 完全相同 —— 全用 Pro 只是多花钱（贵 8%–20%）。
- churn 上协作领先 0.003–0.012，但随种子摆动，均值 ≈ +0.0031，1 对实验不能当结论。
- 协作稳定**便宜 6%–20%**（官方价核算）：因为常规工作站用 Flash（$0.66/M），全贵模式用 Pro（$1.98/M）。代价是 token 多 15%–20%、慢 0.3–3.5 秒。这是价格表算术，不是效率证明。
- 一个**看起来像复制粘贴**的巧合：churn seed 42 与 seed 2026 有两臂 F1 完全相同（.8776019779520363）。查过：split 索引不同、两臂各自重训都复现同一个混淆矩阵 `[[827,31],[29,113]]`，是巧合（分层抽样让验证集类别数固定 858/142，强模型计数集中）。

### B. 突发状况

| 情形 | 结果 |
| --- | --- |
| 工作站返回非 JSON（注入 low/high） | 两臂 `blocked`；**修复后**聚合均值为 null、配对差全 null、`reason=unfinished_arm`，页面写"未完成研究 · 仅基线" |
| 预算掐到 `max_calls=3, max_fits=2` | 两臂仍 `approved`（协作 3 调用 / 单模型 2 调用，都没触发预留失败）；ΔF1 +0.0033、Δ费用 −0.0002773 |
| 因果目标（范围不一致） | 协作走 `partial_completed`（无 F1），单模型走预测（F1 .88545）→ `comparable_scope:false`、所有差值 null，**拒绝跨范围比较** |
| 非法输入（守卫 13 项） | 未知目标列/未知数据集/短 goal/重复种子/未知 id/未完成导出/跨源写入/未知字段 → 全部按预期拒绝 |

### C. 这轮发现并修的两个问题

1. **被阻塞的臂把基线分数当成结果参与平均**（`comparison_benchmark`）。注入工作站失败后两臂都 `blocked`，页面却显示 F1 `0.5969`、汇总写"宏 F1 0.0000"，读起来像两种模式打平，其实是两边都没做出结果。修复：`summarize` 加 `baseline_only`；`aggregate` 只把 `approved/partial_completed` 的臂算进均值，配对差在任一臂未完成时置 null 并给 `reason`；页面行显示"未完成研究 · 仅基线 / 基线 0.5969"，汇总显示"可比较配对 0/1"。截图 `page-comparison-blocked-before.png` / `-after.png`。新增测试 `test_comparison_blocked_arms_are_not_averaged_as_results`。
2. **`seeds:[true]` 被当成 1 放行**（Pydantic 把 bool 收成 int）。改成 `StrictInt` 后返回 422。

`410 tests OK`。页面 JS `node --check` 通过。

### D. 未验证

浏览器点击交互（仍只有静态渲染）；供应商账单实扣；失败**原因多样**（网络超时、429）只注入了"非 JSON"这一种；`3 对实验`模式未跑（只用 1 对）。本轮花费约 **$0.03**（官方价核算，未对账）。

证据：`artifacts/acceptance-20261114/`（`matrix_compare.py`、`matrix-results.json`、`edge_compare.py`、`api_guards.py`、`verify_comparison.py`、4 张新截图）。

## 30 题用户验收（按用户给的题目逐条真跑）· 2026-11-14T21:2x (yuyu)

用用户提供的 30 个问题逐一提交到 8018（同一份 customer_churn，自然语言模式），记录任务号、状态、回答原文、交付物、调用与费用。结果：**7 completed / 8 partial_completed / 15 failed**，花费 $0.0485（官方价核算）。

### 好的一面（真按题回答的部分）

- 因果类（3/6/26）全部拒绝因果结论并给关联：Q26"不管结果如何请证明"→ 明确"不能证明"，给 39.2% vs 11.5% + 95%CI。
- 关联类（1/4/5/7/16）给真实分组率、样本量、区间，两次独立运行同一比例完全一致（39.2% / 11.5%）。
- 经验继承真实生效：Q11 F1 .9034 → Q12（同题重提）**.9091**，先 `experience_loaded` 再提案；Q13 明确写"继承 exp_001 的 HGB 与预处理，尝试 exp_002（300 轮/lr .08/depth 5），F1 从 .9091 降到 .8959，未提升，维持 exp_001" —— 这是 30 题里最完整的一次回答。
- 缺数据时没有编造：Q19–22、Q25 全部停在数据检索并说明缺什么；Q23 明确"该数据无法回答全国青年失业率，需国家统计局数据"，没有编数字。

### 系统性问题（按严重度）

1. **没有会话上下文，追问必死**（9/10/14/15/18/27/28/30 共 8 题）：每个问题都被当成一次全新研究任务；一旦问题里没有"客户/流失/iris…"这类数据线索，就在数据检索阶段失败。所以"你刚才的预测校准了吗""请列出本次实际使用的数据""请按四部分输出"这类**针对上一轮的回答**根本走不到回答环节。
2. **数据检索靠模型生成的英文关键词**，只匹配本地 4 个数据集的别名。Q17 问"国际套餐"（字段确实存在）却回"数据目录未找到匹配项"——误诊；Q24 问"收入字段"（确实没有）回"请补充研究对象"，也没直接说"没有收入字段"。
3. **格式/受众指令不生效**：Q29 要求"面向没有 ML 背景的运营人员"，返回的仍是含 `total_day_minutes` 和宏 F1 的通用答案；Q30 的"四部分结构"因为第 1 条直接没跑到。
4. **答非所问的残余**：Q23 结论正确，但同一份回答里塞进了 1000 条流失预测（用户只要失业率）；且 `research_status=needs_revision` 而行状态是 `completed`，两者表述不一致。
5. **交付物不在回答里交代**：Q2/Q29 有逐条预测 CSV 和表格，但回答不提"去哪下载"，用户可能以为只有汇总。
6. **Q8 措辞风险**：回答写"未使用训练时未见过的数据"，而验证集本来就是训练时没见过的；这句话容易被读成"没做真实留出"。

### 本轮修复

- **失败原因丢失**：Q9（"预测百分比校准了吗"）只显示"失败类型：ValueError"，无法定位。`ml_api` 现在把具体原因写进 `state.json.detail` 与 `reason`（≤200 字、去控制字符）。复测 Q9 显示"具体原因：模型输出不是完整 JSON；可能被截断或未遵循结构化协议"。`410 tests OK`。

### 密钥持久化（第 5 组操作项）

- 后台重启后免密：本轮共重启 3 次，每次重启后立即提交新问题都成功 —— 实测成立。
- 单键更新不覆盖另一个：`verify_credentials_http.py` 14/14（真实 DPAPI，假密钥，临时目录）。
- 两项留空保存：接口返回 400 并保留原配置（原密钥仍可用），但页面只显示报错，没有"已保留原配置"的确认文案。
- 强制重置按钮：只清 `localStorage` 的 `ml-run`/`ml-comparison`，服务端密钥不受影响（代码核对，未做真实点击）。

### 未验证

浏览器点击交互（仍是静态渲染 + API 调用）；关闭浏览器重开这一步（等价于刷新，未单独做）；失败原因只覆盖了"非 JSON"和"无匹配数据"两种。

证据：`artifacts/acceptance-20261114/qa-{1-10,11-18,19-30}.json` 与 `qa-*-results.json`（含每题回答原文）。

## 赛题切题性评估 + 5 组全流程 · 2026-11-14T22:2x (yuyu)

赛题要求：给定数据/工具/预算，自主完成「规划—收集证据—提出假设—实验/分析—验证—修正」闭环；按难度/反馈/剩余预算决定何时上大模型；交付可追溯、可复现、可持续提升的结果；并评估 token 成本、完成度/加速比、完成时间。comment 明确：只考虑一个 LLM，忽略本地轻量模型。

### 先说一件现场事故

另一个 agent 在 21:40–21:48 把 V19（上下文追问 + 回答契约）写进了工作区，但**没重启** 8018；而且 `ml-chain-18` 的版本串没跟着改成 19 之前，前端 `backendReady` 判定仍然通过。结果是一段时间里页面是 V19（有「沿用上一轮上下文」勾选框）、后端是 V18：提交带 `previous_run_id` 直接 **422 extra_forbidden**。我重起后 `ml-chain-19` 正常，`previous_run_id` 返回 202。教训：**升级必须同时改版本串并重启**，否则新前端配旧后端不会报警。

### 五组全流程（真实 API，逐组记录）

| 组 | 场景 | 结果 | 基线→最佳 | 增益 | token | 费用 | 耗时 | 闭环覆盖 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | churn 标准闭环 | approved | .9091→.9116 | +0.0025 | 7209 | $0.00338 | 28.5s | 6/6 |
| B | wine 换数据 | approved | 1.0→1.0 | 0 | 5705 | $0.00303 | 11.7s | 6/6 |
| C | churn 紧预算(5 调用/2 训练) | **failed** | — | — | — | — | — | 5/6 |
| D | 追问「概率校准了吗」 | approved(followup) | — | — | 2837 | $0.00213 | 5.2s | 2/6（不重训） |
| E | 追问「按四部分重写」 | approved(followup) | — | — | 3129 | $0.00269 | 6.4s | 2/6（不重训） |

- A：`experience_loaded` → 复验冠军 → 提新方案 → `verification` → `lesson_recorded`，回答里带真实预测下载链接。
- B：wine 已达 1.0 上限，本轮提案被 `rollback`，如实报"未提升"。换数据后仍跑满闭环，说明闭环不依赖单一数据集。
- **C 暴露稳健性缺口**：完整走完 数据→分类→提案→复用冠军→审核，最后 `reviewer` 调用撞上 `[SSL: UNEXPECTED_EOF_WHILE_READING]`，整轮直接 `failed`。设计上"不自动重试"，但比赛现场一次网络抖动就丢一轮，风险偏高。
- D：**V19 的追问路径见效**——不重新检索/训练，直接从证据回答"未校准、验证集参与选型、没有预测区间、Wilson 是另一个量"，5.2s / $0.0021。
- E：四部分结构生效，但**首轮出现整段重复**（直接回答里塞了全文，后面又拼了证据/预测/局限）。

### 切题判定

切题的（有实测证据）：

1. **闭环真实**：A/B 6/6 覆盖到事件级（`classification_started`→`source_*`→`proposal`→`training_started`→`verification`/`rollback`→`lesson_recorded`），不是页面动画。
2. **成本分层真实**：Flash 跑常规工作站、Pro 跑来源/复核，一次完整闭环 $0.0034/28s；追问只要 $0.0021–0.0027/5–6s，是"少花 token 做对事"的可用素材。
3. **按反馈/预算切换**：置信度<0.7、无效分类、因果范围、低层失败、两次无提升、预留审核调用——6 条规则都实测触发过。
4. **可追溯/可复现**：dataset sha256、逐请求价格快照、events.jsonl、predictions.csv、报告导出，且能独立重算（前几轮 41 项/对）。

不切题 / 有硬伤的：

1. **"只有一个 LLM"不成立（最严重）**。`configured_clients` 缺 `JEV_API_KEY` 直接抛 `Configure a valid JEV_API_KEY or TYPESAFE_API_KEY`——分类站绑死 TypeSafe 第二家供应商。实测：只给 DeepSeek key 系统起不来。赛题口径下必须补一个"单模型回退分类器"（规则或同一个 LLM 的低成本 prompt）。
2. **"可持续提升"只有半截**。`experience.py`（冠军复用）+ `experiment_lessons.py`（同条件教训）接了活动链路，但 `strategy_store / evoundo / redteam / failure_modes` **没有接进 ml_engine**（grep 无引用），页面自己也写"当前暂停自进化"。所以"自我迭代改进"目前是"复用+不重复无效配置"，不是"策略进化"。
3. **缺"加速比/节省比例"口径**：有 F1、token、费用、耗时，但没有对"人工基线/全贵模型"的百分比汇总（对比页只给差值）。
4. **稳健性**：无重试（C 的 TLS EOF 直接失败）。

### 本轮修复

- `research_context.present_answer`：模型自己已按四部分输出时不再用默认值重拼（原来会整段重复）。新增测试 `test_model_supplied_headings_are_not_duplicated`；复测 E 每段只出现 1 次、340 字、带下载链接。**424 tests OK**。

### 未验证

真实点击（仍是 API 级）；C 的重试/降级方案未实现；"自我迭代"若要接 strategy_store 属功能开发，未动。

证据：`artifacts/acceptance-20261114/topic_eval.py`、`topic-eval-results.json`、`topic-jobs.json`。


> **待办（给 Codex）**：见 docs/codex-review-20261114.md —— 逐条复核 C1–C8 并做 P0–P4。

## 回答质量对比（补上之前漏掉的维度）· 2026-11-14T23:0x (yuyu)

用户指出：前面只比了 F1 / 费用 / 耗时，没比**最终回答的质量**。补上了。

方法：`artifacts/acceptance-20261114/quality_eval.py`，对已保存的 13 对对比逐臂重算 8 项可复算指标（是否直面原问题、数字是否与本轮报告一致、是否说明预测范围、是否划清因果、是否说明未校准、是否给局限、是否给下载、是否面向受众）。不是主观打分，改动证据即可复跑。

结果（每项 1 分，满分 8）：

| 对比 | 种子 | 协作 | 全贵 | 差 |
| --- | --- | --- | --- | --- |
| 21605f2f | 42 | 6 | 7 | −1 |
| b8566cf4 | 7 | 6 | 6 | 0 |
| ec5c4fda | 2026 | 6 | 6 | 0 |
| ccdae557 | 42 | 7 | 7 | 0 |
| 8536eded | 42 | 6 | 5 | +1 |
| 5fed95e4 | 42 | 7 | 7 | 0 |
| a91485bf | 42 | 7 | 2 | **+5** |
| a700dee0 | 1 | 3 | 2 | +1 |
| 403dfbb6 | 42 | 0 | 2 | −2 |
| 7ed1e4aa | 42 | 0 | 3 | −3 |
| eb694a33 | 42 | 0 | 2 | −2 |
| 114e7887 / b9e9203d | 42 | 0 | 0 | 0（双方注入故障） |

**13 对：协作更好 3、持平 6、全贵更好 4；均值 3.69 vs 3.77 —— 回答质量打平，略亏。**

结构性原因：对比里两个臂的 `reviewer` 是**同一个 Pro 副本**（`comparison_benchmark.py` 把 `audit` 同时给两边），所以差异不来自"谁写答案"，只能来自两件事：

1. **证据交付不同**（F1/预测条数/特征重要性/经验）；
2. **路由是否直接失败**——三次 0 分全部是 `needs_clarification`/`needs_revision`：协作臂面对"面向运营解释""因果判断"这类边界请求时没产出答案，而全贵模式给了（虽然更短更模板化）。

反过来，全贵模式丢分的地方是"漏说边界"：a91485 只给预测分布 + 字段名，没写范围/因果/校准/局限 → 2/8。

另一个细节：21605f2f 协作版把字段写成 `number_customer_service_calls`，贵模型版写"客服电话次数"——同样信息，可读性协作更差。

注意：`delivers_predictions` 这一项两边几乎全丢分，是 **V19 之前**回答不带下载链接的历史行为，不算模式差异；V19 之后（本轮 D/E 追问）回答已带 `/ml/runs/<id>/predictions`。

结论：**可以说"同等质量下更便宜"，不能说"回答更好"**；而"同等质量"的前提是路由不把边界请求误判成 clarify。要真正拿到质量优势，得把"经验复用、关联证据、决策路径"写进答案——现在这些只在证据包和开发者面板里。

证据：`answer-quality.json`（逐项布尔值与原文长度）。


## Codex V20 patch applied 2026-11-14T23:01:47.463557+08:00
See docs/codex-v20-review.md. Review workspace: 436 tests OK; loopback HTTP + 2 sklearn fits, 0 paid API calls. Live provider and 8018 restart still require deployment acceptance. Automatic strategy evolution remains disabled.


## Codex V21 patch applied 2026-11-14T23:18:55.858941+08:00
See docs/codex-v21-answer-safety.md. Review workspace: 445 tests OK; 13-pair/26-arm presentation replay, 0 new paid API calls. Live provider and 8018 restart still require deployment acceptance. Automatic strategy evolution remains disabled.

## V21 补丁应用 + 真实路由验收 · 2026-11-14T23:2x (yuyu)

应用方式（按包内 README，带 SHA 守卫）：解压到临时目录 → `apply_patch.py --target ...` 干跑「Verified all source and payload hashes; 9 files」→ 自备份到 `artifacts/backups/v21-pre-20261114-231851` → 停 8018 → `--apply`（脚本另备份 `artifacts/v21-patch-backup-20261114-231855-840526`）→ 重启。

- 现场：`ml-chain-21`、`restart_required:false`、`auto_evolution_enabled:false`。
- 全量：**445 tests OK**（41.5s）。
- 补丁自带离线回放（`tools/replay_answer_contract.py`，0 付费）：13 对 / 26 臂，20 臂回答变长且带证据引用（`answer_evidence.references`），6 臂保持原状（均为 blocked / 失败臂）。回放明确声明：不重跑模型、不改研究状态、不证明质量提升。

### 真实路由验收（补丁 README 要求的第 4 步）

| 案例 | 改前 | 改后（本次真跑） |
| --- | --- | --- |
| A 因果+关联题（"通话次数与流失关系，能否支持因果结论"） | 曾判 `needs_clarification`（0 分） | **`partial_completed`**：3000 条训练记录分组率 + 95%CI + 小样本提示 + 明确"不能推断因果"；**未生成也不伪造预测下载** |
| B 普通研究题（churn 训练分类模型） | 模板式 170 字，只有 F1 + 字段名 | **`approved`**，回答含：预测条数与分布、宏 F1、**适用范围**（验证记录非未来客户、验证集参与选型）、**因果边界**、**未校准**、**真实下载链接**、字段中文化 |

费用：A 5 调用 / 6770 token / \$0.00421；B 5 调用 / 7183 token / \$0.00270。下载链接实测有效：`/ml/runs/9728430…/predictions` 返回 **1001 行**（1000 条预测 + 表头）。

### 诚实边界

- 只验了 2 个真实路由案例，**不是完成率**；`404` 类失败注入、3 对实验、多数据集矩阵本轮未复测。
- 我自己那套 8 项关键词 rubric **对"关联型无下载"回答有偏**（补丁文档也指出这一点），所以不把它的分数当"质量提升"证据；只报"回答里实际出现了什么"。
- `auto_evolution_enabled` 仍是 false：自动策略进化未启用，与赛题"可持续提升"的差距仍在。

证据：`artifacts/acceptance-20261114/v21_live.py`、`v21-live-acceptance.json`、`v21-replay.json`。


## Codex V22 patch applied 2026-11-14T23:29:04.598997+08:00
See docs/codex-v22-ui-billing.md. Review workspace: 453 tests OK; pixel UI and account-window billing regression, 0 new paid API calls. Live provider and 8018 restart still require deployment acceptance. Automatic strategy evolution remains disabled.

## 上传 GitHub · 2026-11-14T23:3x (yuyu)

仓库：`Elio896/EvoResearch`（private，main）。提交 `9d10d05`（136 文件）：V22 像素工作台 + 账户余额观测，含 V21 回答安全与 V20 单模型/重试。

做法：`gh repo clone` 到临时目录（保留远端历史）→ robocopy 同步本地源码（排除 .git/.venv/artifacts/data/__pycache__/*.log/*.zip/*.dpapi）→ 提交前扫描 `sk-`/`ghp_`/`Bearer` 无一命中 → `git push origin HEAD:main`。

校验：远端根目录含 `ml_api.py`、`billing_actual.py`、`answer_evidence.py`、`research_context.py`、`static/pixel-theme.css`、`docs/codex-v22-ui-billing.md`、`tests/test_actual_billing.py`；`artifacts/`、`data/`、`.venv/`、`credentials.dpapi` 均**不在**仓库（.gitignore 生效）。

已知缺口：`data/` 被忽略，新克隆没有 `customer_churn.csv`（另 3 个 sklearn 数据集会自动重建）；要完全可复现需单独下载 OpenML 40701 或把 `data/catalog/*.csv` 加入白名单。


## V22 functional review completed 2026-11-15T00:01:05.476470+08:00

See docs/v22-review-fixes.md. Final full suite466 passed60.353s, JS syntax passed. Live8018 V22 restart_required=false. Native browser actual research60922e1b produced1000-row CSV; negative retrain bug reproduced840be560 and fixed69f596c3(1modelcall,0fits); association followupc4f7077f remains partial_completed(0fits,0predictions). Robot retained per latest explicit user request. Snapshot/account-window figures are not task invoice charges. Not all30questions rerun. Auto strategy evolution remains disabled. New required source app_version.py must be included when syncing. No GitHub push performed. Evidence artifacts/v22-review-acceptance/acceptance.json. Legacy socket ResourceWarning retained.


## V23 final source freeze 2026-11-14T16:53:02.790616+00:00

Final backend ad011eea deployed8018 restart_required=false. Full484tests OK88.971s+JS syntax;13TCP group13run0skip13pass2.489s (includes config-only cases). r2all30submitted28structuralpass; failures8/13 retained. Fixed association-to-prediction execution and passed actual incumbent/history to high recheck; no silent association downgrade for requested prediction; significance phrasing qualified. r3frozen allphase running. No backend edits until complete. docs/competition-v23.md; no GitHub push.


## V23 competition finish 2026-11-14T17:15:46.445536+00:00

V23 final8018 source0ad814a0 restart_required=false. Frozenad011eea:30QA15completed8partial7needs_input0failed,30structuralchecks;6paired/12arms;3seedmemory;222modelattempts officialcalculatedUSD.264564347(notinvoice). RoutedmacroF1+.00254 butcost+17.90%,time+35.56%;memorymean+.00343,2/3better,noefficiencyclaim. Browserplain-improvebugff1005 retained;fixcf600 actual4fits2rounds7calls1000CSV F1.9146;followupa3f1 0fits1call,URLupdates+refreshverified. Full484passed85.315s+targeted26+Node/JS. Robot retained,encryptedkeysunchanged,noGitHubpush. docs/competition-v23-results.md; originalfailedpilotsretained.


## Token economy verified 2026-11-14T17:35:46.351345+00:00

User prioritized tokens/cost over marginal F1. Website economy:true enabled; stable route reuse, compact actual audit evidence, feedback/lesson summaries, marginal-gain high candidate substitution, revalidated-history direct prediction. Full491passed95.546s+6economy tests+Node/JS; initial489run1legacy campaign failure retained, isolated rerunpassed15.396s. Realcold2pairs mean tokens-22.09%,officialUSD-40.24%,meanF1-.000583 BUT seed42gap.012917 fails.005; seed7better. No stable noninferiority or speed claim. Warm cd4aaa6e 1fit2calls3176tokensUSD.0015288281000CSV F1.914629samehistory, no test. Nativebrowserlabel/result/downloadverified; source66f31803restartfalse. docs/token-economy.md; firstfailedgatepilot retained. No keys changed, noGitHubpush.
