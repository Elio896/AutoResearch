# START HERE · EvoResearch 策略自进化闭环（2026-11-14）

最新一轮：`docs/feedback-learning-2026-11-14.md`。网页新增持久化用户纠错和创新证据面板；`python tools/run_asgi_checks.py` 可在不允许本机 TCP 的环境验证 API 请求、后台执行、导出和下一会话纠错。纠错消融重跑命令：`python tools/evaluate_user_corrections.py --output artifacts/a-new-feedback-eval`。

## 当前赛题入口（优先看这里）

按用户最新赛题要求，**本地模型不是必需项**。多个角色可以共用一个 LLM，规划与失败诊断使用共享的调用 / token / 时间预算，统计、训练和验收由程序执行。

已有依赖环境下运行 `./start-research.ps1`，访问 `http://127.0.0.1:8002/research-agent`。如果已有其他 Python 环境，可先设置 `EVORESEARCH_PYTHON` 为其解释器路径。脚本不会自动安装依赖。

无需服务或模型的真实两轮评测：

```powershell
python tools/run_research_system.py --output artifacts/my-campaign
```

单模型模式：服务器设置 `LIGHT_MODEL_API_KEY`、`LIGHT_MODEL_BASE_URL`、`LIGHT_MODEL_NAME`（或已有 `DEEPSEEK_API_KEY`），再在新页面选择单模型，或在命令行增加 `--mode single_model`。密钥不填进网页和源码。

规则模式是程序对照实验；真实模型效果、供应商账单、HTTP 和浏览器端到端结果不能从规则实验推导。当前验证边界详见 `docs/topic-alignment-2026-11-14.md`。

## 以下是原有分层模块的使用记录

## 这是什么

EvoResearch 原有能力是「预算感知的自主科研迭代」。这次在它上面补了一条**可验证的策略自进化闭环**：

```
研究运行（pin 住一版策略）
  → 从真实事件流里找失败模式（可追溯到 event_007 这种下标）
  → DeepSeek 提一个带假设、带否证条件的候选
  → 用独立评估任务做新旧对照（条件跑前冻结）
  → 代码检查 + Jev 三选一，两项都过才切换
  → 上线后按冻结规则监控，退化就自动回滚
  → 失败版本保留，成为下一轮候选的证据
```

同时把「千问的过程展示」改清楚了：只展示它的**输出件**（分析依据 / 执行计划 / 结果复核），
不把程序写的说明冒充模型思考，也不声称拿到它的内部思维过程。

## 五分钟看懂

1. 打开 `docs/strategy-evolution-2026-11-14.md`：改了哪些模块、每个模块守什么规矩、验收证据在哪。
2. 看 `artifacts/step1-acceptance.txt` ~ `step7-acceptance.txt`：每一步的**真实终端输出**。
3. 起服务看页面（见下）：`① 分析依据` 会显示来源（API 模型 / 本地模型 / 程序规则），
   页底 `策略进化记录` 显示当前策略、版本差异、事件时间线与失败模式。

## 怎么跑

```powershell
# 依赖（若已有 .venv 可跳过）
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-api.txt

# 服务：DEMO 页在 http://127.0.0.1:8002/demo
.\.venv\Scripts\python.exe -m uvicorn api:create_app --factory --host 0.0.0.0 --port 8002

# 测试（当前 321 个）
.\.venv\Scripts\python.exe -m unittest discover -s tests

# 逐步验收（不花钱，全程本地）
.\.venv\Scripts\python.exe tools\strategy_acceptance.py
.\.venv\Scripts\python.exe tools\strategy_run_binding.py
.\.venv\Scripts\python.exe tools\qwen_worklog_acceptance.py
.\.venv\Scripts\python.exe tools\failure_modes_acceptance.py
.\.venv\Scripts\python.exe tools\candidate_acceptance.py
.\.venv\Scripts\python.exe tools\strategy_eval_acceptance.py
.\.venv\Scripts\python.exe tools\strategy_upgrade_acceptance.py
.\.venv\Scripts\python.exe tools\strategy_rollback_acceptance.py
.\.venv\Scripts\python.exe tools\seed_strategy_demo.py      # 给页面准备策略库（真跑，Jev 回答标注 simulated）
```

## 模型层

- 优先走**轻量模型 API**（OpenAI 兼容）。配环境变量即可，不配就不启用：

  ```
  LIGHT_MODEL_API_KEY   （或 DEEPSEEK_API_KEY）
  LIGHT_MODEL_BASE_URL  默认 https://api.deepseek.com/v1
  LIGHT_MODEL_NAME      默认 deepseek-chat
  LIGHT_MODEL_MAX_CALLS 默认 6
  ```

- 没有 key 时退回本地 Ollama，再没有就退回**程序规则**，页面会**如实标**「程序规则 · 非模型输出」。
- API 记账：调用次数与 token 会显示；**没取到账单前成本记 unknown，不记 0**。

## 哪些是真实的、哪些是模拟的

| 真实 | 模拟（无 key，均已标注） |
|---|---|
| 实验执行、指标、事件流 | DeepSeek 的候选策略 |
| 失败模式检测与事件追溯 | Jev 的升级/保留选择（`jev.model=simulated-jev`） |
| 对照评估、版本落盘、切换、回滚 | 评估里的提案器（确定性脚本，模拟模型行为） |
| 前端读数、验证检查 | —— |

**没有把模拟升级当成真实自进化效果。** 配上 API key 后把提案器换成真模型即可，闸门不用改。

## 目录

```
strategy_store.py         策略版本库（config-only、双闸、pin/rollback）
qwen_worklog.py           千问结构化工作记录 + 证据引擎
failure_modes.py          五类失败模式的检测
candidate_proposer.py     DeepSeek 候选策略（白名单 + 假设五项）
strategy_eval.py          新旧对照评估（条件冻结、双层隔离）
strategy_decision.py      Jev 三选一 + 代码硬闸
strategy_rollback.py      上线后回滚（规则冻结）
light_model.py            轻量模型 API 客户端
demo_flow.py              /demo 的三步流程（analyze → approve → execute）
static/demo.html          演示页面（含千问工作记录、策略进化记录）
tools/*.py                各阶段验收脚本
tests/*.py                321 个测试
artifacts/step*.txt       各阶段真实验收输出
docs/                     变更说明与评审材料
```

## 已知缺口

1. 没有 API key，所以“真实自进化”目前只到流程成立这一层。
2. 评估用的提案器是脚本，所以验证的是闸门与流程，不是模型能力提升。
3. 策略目前只影响提示词与选路规则，还没接进 `research_loop` 的实际提案路径——
   “新策略影响下一次研究”目前只验证到 pin 版本这一层，尚未验证到行为变化。
4. `demo_flow` 的验证项是固定四项，还没按任务类型扩展。
