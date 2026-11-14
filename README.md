# 原版千问 · 当前提交版

安装及演示请先阅读 [提交说明](docs/qwen-demo-submission.md)。保留原版千问3B，不使用已撤下的 v24 包。

> Website now defaults to token-saving mode. Real measured results and quality limits: [token-economy.md](docs/token-economy.md).

# V23 current entry

Start Start-Local-Research.cmd and open http://127.0.0.1:8018/. Current scope, bounded research rounds, evaluation and reproduction: [competition-v23.md](docs/competition-v23.md). Older sections below describe historical versions and are not the current acceptance status.

# EvoResearch

**当前主线：机器学习 API 工作站链路。** 自然语言/CSV → 来源模型（公开数据搜索、获取与规划）→ Jev 分类派发 → 低成本/高成本工作站 → 高性能模型审核整合。双击 `双击打开机器学习.cmd` 打开新入口。配置、边界与验收见 `docs/ml-pipeline.md`。下方历史演示和旧验证记录不代表当前默认应用的功能清单。

项目展示入口：`/showcase`。可按按钮实际演示“研究 → 用户纠错 → 新会话复验 → 撤销负对照”，以及独立策略副本的回滚。无需模型密钥。启动方式与验证边界见 `docs/showcase-guide.md`。

**当前严格复测状态**：13 项原跳过测试已全部执行，0 跳过，4 项配置校验通过、9 项协议/传输失败或报错；实际 TCP connect 被环境以 WinError 10013 拒绝。真实 LLM 评测因服务器无模型凭据而 blocked，不能宣称已验证学习效果。最新证据和运行方式见 `docs/live-verification-status-2026-11-14.md`。

`python tools/run_asgi_checks.py` 现在默认不跳过 TCP；`--offline` 仅为明确要求的离线检查。真实模型学习评测使用 `tools/evaluate_live_learning.py`，单独衡量原始模型提案和执行拦截结果。

最新验证与纠错学习：新增用户反馈 → 持久化约束 → 下次训练前检查，页面 `/research-agent` 可提交、查看和撤销纠错，并展示创新设计及真实消融结果。**9 个跨会话案例中，脚本错误提案造成的执行错误从 12 次降为 0，无关数据误拦截 0。** 完整证据、质量取舍和验证边界见 `docs/feedback-learning-2026-11-14.md`。

回归命令 `python tools/run_asgi_checks.py`：357 项中 344 项通过、13 项真实 TCP 供应商夹具受环境限制跳过。此为实际进程内 HTTP/ASGI 集成测试；真实浏览器及云端模型能力不由此推导。

面向轻量机器学习任务的**预算感知自主科研 Agent 系统**。当前赛题入口是 `/research-agent`：不要求本地模型，支持多个职责共用一个 LLM，以及真实训练的规则对照模式。

本轮对照赛题完善了实际策略绑定、独立任务评测、两轮策略迭代、共享 token / 调用预算、失败与未知用量记账、证据快照及回滚。完整核查和实测结果见 `docs/topic-alignment-2026-11-14.md`。

在已有依赖环境运行 `./start-research.ps1`，打开 `http://127.0.0.1:8002/research-agent`。无模型验证：`python tools/run_research_system.py --output artifacts/new-campaign`。规则结果不能视为 LLM 自主能力或云端节省效果。

## 可选工业分层模式（兼容原实现）

```
Jev Harness（0 token，纯规则）   →  这次失败值得上云吗？
        ↓ 放行
本地千问 3B（几十 token）        →  要调吗？调哪一档？
        ↓ 放行
云端 DeepSeek（贵，按 token 计费）→  真正解决本地说不出来的难题
```

类比看病：**分诊台 → 社区医生 → 专家门诊**。分诊台（Jev Harness）把"小感冒"（参数写错、死循环、微小改进）直接挡回去，这些根本不该占用专家。

## 五个角色

| 角色 | 实现 | Token |
|---|---|---|
| Analyst | pandas 统计画像（缺失、不平衡、小样本） | **0** |
| Manager | 预算算术 + 停/升级判据 | **0** |
| Scientist | 已配置的单个 LLM 或可选本地模型提方案（必带可证伪假设） | 按真实 usage |
| Experimenter | scikit-learn 真训练 | **0** |
| Reviewer | 规则判好坏 + 闸门 + 回滚标记 | **0** |
| RedTeam | 可选模型质疑，预算内执行代码检验 | 按真实 usage |

## EvoUndo：能被审计的回滚

每次实验是一个版本。Reviewer 判退步 → `current` 自动指回历史最优。**不重训、不删版本**——被抛弃的那代标 `rolled_back` 留着，它是"这条假设被否证"的证据。

```powershell
.\.venv\Scripts\python.exe tools\run_five_roles.py bench\tasks\t2_imbalanced\data.csv ^
    --target target --metric f1_minority --pos-label 1 ^
    --model qwen2.5:7b-instruct-q4_K_M --small-model qwen2.5:3b-instruct-q4_K_M ^
    --enable-cloud --enable-qwen-router --enable-jev-harness --max-cloud-calls 1
```

产物：
- `artifacts/evoundo/<run>/versions.json` + `versions.html`（零依赖版本树）
- `artifacts/trajectory/<run>/` 九项证据 + 哈希链，用 `docs/交付包/04-研究轨迹规范/validate_trace.py` 校验

## 研究轨迹的九项证据

按 `docs/交付包/04-研究轨迹规范/` 实现，**验收只看 E2 以上**：

身份血缘 / 输入快照 / **假设与可证伪预测** / 干预 diff / **执行凭证** / 结果与不确定性 / **闸门与回滚** / 预算台账 / 结论与下一步

实测：3 轮轨迹 `FAIL 0 / WARN 0`；同 seed 重跑 diff = 0（E2 达标，脚本 `tools/verify_determinism.py`）。

## 已验证的事实（不是设计稿）

| 项 | 证据 |
|---|---|
| 本地 7B | 100% GPU，4.8GB，64 tok/s，JSON 合规 40/40 |
| 本地 3B | 121 tok/s，红队用它照样提得出合格质疑 |
| JEV 官方 | `model: jev-1.14.0`，probability 0.67 |
| 云端 DeepSeek | 返回真 JSON；只在三层放行后被调用 |
| 测试 | 143 passed |

## 第一阶段（历史）

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

## Feature-integrity review (current guarded service: port 8002)

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:create_app --factory --host 127.0.0.1 --port 8002
```

Open http://127.0.0.1:8002/ . Old running processes on 8000/8001 do not hot-load the new Python guard. The updated frontend checks that the backend exposes /tasks/{task_id}/review and disables mutations when connected to an older backend. This capability check is for accidental version mismatch, not authentication. Do not use old API processes for research conclusions.

Task creation now returns risk_report. Candidate signals include ID-like names, near-unique ordered integers, exact matches to training labels, strong binary training association, and unique numeric features that perfectly separate classes when sorted on the TRAIN split. These are review candidates, not proofs of leakage. All statistical checks use training rows only; validation/test values and metrics do not determine exclusion.

known_leaks.json records the exact SHA-256 of the bundled t1 CSV and target/column/source evidence. bench/make_tasks.py explicitly sorts by target before assigning record_id and then shuffles; this exact fixture is confirmed leakage, not a generic ban on that column name. An altered/renamed copy can still trigger generic candidate review. Clearing the linter does not prove the absence of all possible leakage.

POST /tasks/{id}/review accepts:

```json
{"decisions":{"record_id":{"action":"exclude","reason":"Generated after sorting by the target, as documented in the generator."}}}
```

Every candidate needs an explicit exclude/keep decision and a reason. Confirmed leakage cannot be kept. No feature is silently dropped. The effective feature exclusions and review reasons are persisted per task and attached to experiment results, applying identically to every model. Decisions become immutable once any experiment has been attempted; create a new task to change the feature policy. Both manual experiments and autonomous research are blocked until review is cleared, without consuming experiment slots. Final test access also checks the guard.

The workbench shows the review panel, supports saving decisions and starting a reviewed fresh task. Existing tasks with experiments cannot start autonomous research; the page explains that a new task is required. Old experiment results are not overwritten: their current risk is labeled, and suspicious historical F1 is not presented as the current best reliable result. Old configuration cannot be retroactively changed.

CLI: task_store.py review TASK_ID --decisions JSON; executor.py and orchestrator.py accept --feature-decisions JSON. Source CSV files, benchmark thresholds, generation scripts, missing-sentinel/unit policies, and evaluation splits were not changed. The current holdout F1 is not comparable to the benchmark's 5-fold CV reference threshold.

Verification: 34 Python tests passed, including named and renamed leakage candidates, legitimate strong predictors that can be explicitly kept, immutable decisions, legacy-history labeling, train-only signals, and manual/autonomous API gates. Real HTTP evidence in artifacts/integrity_http_verification.json: t1 unreviewed endpoints returned 409, zero attempts consumed, reviewed record_id exclusion trained logistic regression with validation macro-F1 0.9440, later policy edits returned 409. No test evaluation and no model API calls. Browser visual acceptance remains pending from the earlier Edge renderer limitation.

## Optional Jev cloud-escalation gate

Jev routing is now implemented in jev_router.py and the CLI orchestrator. It is OFF by default. The separately contributed generic jev_client.py is preserved and is not used by this gate. Source protocol: https://docs.typesafe.ai/api/system-one.md and https://docs.typesafe.ai/introduction/quickstart.md .

The existing rules first decide whether escalation is eligible: local planner failure/stagnation, unused experiment slots, positive cloud-request budget, a configured expert and remaining time. Only then can a single Jev request evaluate whether a cloud diagnosis is likely to help. It sends a Noul question using POST https://api.typesafe.ai/v1/systemone and reads answers.worth_cloud_diagnosis.noul. Default threshold 0.75 is a configurable engineering choice, not a validated calibration or guaranteed economic optimum.

Jev receives only bounded validation/model history, trigger and remaining budgets. No CSV rows, test metrics, feature-review reasons or keys enter its state. It cannot authorize more experiments, final-test access, feature-policy changes or arbitrary tools. If it declines, the task stops with jev_declined. Errors/invalid responses or a zero Jev-call budget fall back to the original rule-based cloud decision, within the same cloud limits. There are no automatic transport retries or HTTP redirects.

Use an interactive PowerShell terminal:

```powershell
.\tools\run-jev.ps1
```

The helper securely prompts for a TypeSafe/Jev key and a DeepSeek key, uses them only in the process environment and restores prior values. Default CSV is bench/tasks/t3_clean/data.csv; defaults permit at most one Jev request and one DeepSeek request, and neither is guaranteed to be needed. To compare the rule baseline, use the existing cloud workflow without --enable-jev. The script does not modify execution policy or model service settings. The existing local Qwen service is still required for normal proposals.

CLI flags: --enable-cloud --enable-jev --max-cloud-calls 1 --max-jev-calls 1 --jev-threshold 0.75. Jev keys are read from TYPESAFE_API_KEY or JEV_API_KEY; DeepSeek uses DEEPSEEK_API_KEY. --jev-model can select a pinned model for reproducibility (default jev-latest); the returned model is recorded in the gate event. Jev requires an explicitly enabled cloud-recovery branch. The web workbench remains local-only and does not enable these paid routes.

Reports distinguish jev_calls, cloud_calls (DeepSeek), total_remote_calls and escalation_gate source (jev/rules/rule_fallback), probability and reported usage. jev_cost and cloud_cost remain null (unknown), not zero. Request/byte limits are not dollar caps. This introduces another paid provider; no cost or quality improvement is claimed before matched-budget comparison.

Verification: complete suite 40 tests passed; new tests cover the official HTTP shape via a temporary loopback server, sanitized HTTP failures, probability validation, deny/allow/fallback and hard-budget precedence with actual sklearn training. No new packages or paid requests. Live Jev authentication/inference has NOT been verified because no project Jev key is configured. tools/run-jev.ps1 is syntax-checked only; its interactive key flow has not been executed by Codex. Earlier GitHub ZIP is a prior snapshot and does not contain this new optional router.


## Task selection metric

New tasks default to `f1_macro`. For binary minority-class research (including t2), select `f1_minority` and explicitly specify the minority label, e.g. `--metric f1_minority --pos-label 1` in executor/orchestrator/task-store create, or `metric`/`pos_label` in POST /tasks. The class must be a minority in the training split; labels are not inferred from holdout performance. This immutable task setting is persisted in SQLite and used for local/cloud improvement checks, best-experiment selection, and final evaluation.

`validation_f1` / `test_f1` remain macro F1 for compatibility; `validation_score` / `test_score` carry the chosen objective. Minority tasks additionally report minority precision, recall and F1. UI tables display the chosen score and task caption names the objective. Existing tasks retain macro F1. Restart the API after upgrading; the UI refuses older backend schemas. Holdout 60/20/20 results are not interchangeable with the benchmark's five-fold CV reference. No benchmark threshold or dataset was changed.


## Planning context v2

The planner now receives bounded, anonymized training-only aggregate evidence and per-class validation precision/recall/F1. A prompt recommends trying an untried model family; no estimator is silently substituted for the model proposal. Reports label this policy `context_v2`. The executor remains LLM-free. Experiment/call/time caps are unchanged. See docs/evaluation/context_v2_20261113/REPORT.md for development comparisons against preserved v1 evidence. Qwen3B t2 mean minority F1 improved .3903 to .5508; fixed search remained better at .5854, and t3 did not improve. These are development-validation observations, not independent test results or proof of Jev benefits.


## Optional baseline-first policy

Choose `fixed_search` using the web research-strategy selector, `ResearchInput.baseline_policy` or CLI `--baseline-policy fixed_search`. Default remains `logreg`. The baseline plan is default logistic regression, balanced100-tree random forest, balanced logistic regression C=.1. Every fit consumes the same task experiment budget; time limits apply before new fits. With3 successful allowed fits and sufficient time, this reproduces the fixed-search candidate set with zero LLM calls. With4+ fits, remaining budget is available for validated Agent proposals. With1 or2 fits, only the corresponding prefix runs. Job settings and reports preserve the selected policy. This provides a validation candidate floor only when the baseline plan finishes; it does not guarantee test-set superiority or a new Agent gain.

Frozen transfer evidence is in docs/evaluation/frozen_transfer_20261113/REPORT.md. All validation selections were locked before one final test per arm/split. Wine tied; the2 synthetic tasks favored fixed search. Do not retune on these test scores and label the same splits unseen.


## Read-only evidence export

Use the workbench download buttons or GET /tasks/{id}/export?format=markdown|json|zip. ZIP includes REPORT.md, manifest.json and replay_validation.py. Export is blocked during active research/finalization, never fits models or evaluates test data, and omits server CSV paths and private job tokens. Unfinished tasks are explicitly labeled as having no successful final test. Historical leakage risk suppresses the best-validation conclusion while retaining observed records. Sources distinguish recorded rule/local/cloud proposals; missing provenance is labeled unresolved.

To replay successful validation configurations from an unzipped bundle, use the project environment: python replay_validation.py --project PATH_TO_SOURCE --csv ORIGINAL_CSV. It checks dataset SHA256 and dependency versions, applies recorded feature decisions, and never finalizes. Raw data and trained estimator files are not included. Historical tasks lack persisted source revision, so exact reproduction across source versions is not guaranteed. Unknown usage/cost is not represented as free.


## 原版千问可提交 Demo
保留原版千问3B、六组数据和主研究架构；最小修复及安装/演示说明见 [docs/qwen-demo-submission.md](docs/qwen-demo-submission.md)。首次安装运行 Setup.cmd；之后 Start-Local-Research.cmd。Ollama 需安装原版 qwen2.5:3b-instruct-q4_K_M。云端高级站可能收费，转交必须实际训练才计结果。
