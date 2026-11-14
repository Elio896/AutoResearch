# 给 Codex 的任务书：复核我的结论 + 修复完善（2026-11-14 夜）

> 发起人：鱼鱼（DSH）。用户要求「让 Codex 去测一下，看哪里不对，然后改并完善」。
> 工作目录：`C:\Users\junzi\Documents\Codex\2026-11-14\new-chat-2\EvoResearch`
> 运行实例：`http://127.0.0.1:8018`（当前 `ml-chain-19`，我重启后加载的是你 21:40–21:48 写的那版）
> 证据目录：`artifacts/acceptance-20261114/`（30 题回答原文、对比矩阵、5 组全流程、脚本与截图）

## 0. 你要先做的事

1. 读 `HANDOFF.md` 我最后加的两节（「30 题用户验收」「赛题切题性评估 + 5 组全流程」）。
2. 逐条复核下面 8 条结论，判定 **对 / 错 / 表述不准**，每条写复现命令与结果。
3. 按 P0–P4 修复，改完跑全量测试、重启服务、更新版本串，把结论写回 `HANDOFF.md` 并往 `.agent-log/events.jsonl` 追加一行。

## 1. 请复核的 8 条结论

| # | 我的结论 | 我给的证据 | 置信度 |
| --- | --- | --- | --- |
| C1 | **只配一个 LLM（DeepSeek）系统起不来**：`configured_clients` 缺 `JEV_API_KEY`/`TYPESAFE_API_KEY` 直接抛 `ValueError`，分类站绑死 TypeSafe | `ml_engine.py:76` 只构造 `JevClassifier`；我用 `configured_clients({'DEEPSEEK_API_KEY': fake})` 实测抛错 | 高 |
| C2 | **活动链路没用策略进化**：`ml_engine.py`/`ml_api.py` 里没有 `strategy` 引用，页面写「当前暂停自进化」，与赛题「可持续提升」不一致 | `grep -n strategy ml_engine.py ml_api.py` 为空。**注意**：`research_system.py:26` 有 `from strategy_store import StrategyStore`，但 ml 路径只用它的 `save`/`read_saved_state`——请确认这个区分对不对 | 高（表述需你确认） |
| C3 | 对照页/报告**没有「节省比例 / 加速比」口径**，只有差值与绝对费用 | `grep -n 'speedup\|加速比\|节省比例\|savings' *.py static/*.js` 只命中一句无关的 planning 文案 | 高 |
| C4 | **没有自动重试**：一次 `SSL: UNEXPECTED_EOF_WHILE_READING` 让整轮 failed | 证据 run 在 `topic-eval-results.json` 的 `C-churn-紧预算`；`light_model.py`/`cloud_provider.py` 里只有「no automatic ... retry」注释 | 高 |
| C5 | **V19 上线未重启 + 版本串滞后**：21:40–21:48 代码已改，但 8018 仍跑 21:29 的进程且 `ml-chain-18`，导致新前端（有「沿用上一轮上下文」）配旧后端，带 `previous_run_id` 提交返回 422；我 22:36 重启后才 `ml-chain-19` | `curl /ml/version`、`Get-CimInstance CreationDate`、`POST /ml/runs` 422 的返回体 | 高（已修，但请加防呆） |
| C6 | `present_answer` 在模型已自带四部分时仍用默认值重拼 → **整段重复**；我改成「模型已含全部标题就不再重拼」并加了测试 | run `E-追问-四部分` 原文（重复）；你可以在 `research_context.py:102` 看我改的 `supplied / already / else` 三分支 | 中（请判断我的修法是否更合适） |
| C7 | 30 题里 15 个失败中 **8 个是「追问型问题没有数据线索」**；V19 用关键词白名单缓解，但白名单外（如「这些结论有什么局限？」）仍失败 | 我 22:31 用 `previous_run_id=a233d943…` 提交「那这些结论有什么局限？」→ 仍然走数据检索并 `needs_input` | 中 |
| C8 | Q23「用流失数据回答全国青年失业率」结论正确（拒绝），但**同时塞了 1000 条无关流失预测**，且 `research_status=needs_revision` 而行状态是 `completed` | `qa-30-results.json` 中 Q23（run `5662e6c8…`） | 中 |

> 如果哪条错了，直接说「C2 错，因为 …」，不用客气——用户就是让我被挑错。

## 2. 要修/完善的功能（按优先级）

### P0 单模型回退（决定赛题是否成立）

- 目标：**只配置 DeepSeek 一个 key**，也能跑完整链路并通过一次真实研究。
- 做法建议（你定）：`classifier` 在缺 Jev key 时退回「规则分类器」（按目标文本 + 数据画像给 `category/confidence`），或退回同一个 LLM 的低成本 prompt；两条都行，但必须**在报告里标清本次分类来源**（`classifier_source: jev|rules|llm`），不能假装是 Jev。
- 验收：`DEEPSEEK_API_KEY` 单独存在时 `/ml/config` → `configured:true`；一次真实研究 `approved`；报告里能看到分类来源；全量测试全绿。

### P1 有界重试 + 失败分类

- 目标：传输类错误（`SSL`/`EOF`/timeout/连接重置）最多重试 1 次，重试计入调用与 token 预算。
- 同时把失败原因分成 `network_failure` / `budget_exhausted` / `model_protocol_error` 三类，写进 `state.json`，页面区分文案。
- 验收：注入一次 SSL 错误后，整轮仍能完成（或明确报「网络失败，已重试 1 次」）；测试要覆盖"重试也失败"。

### P2 让「可持续提升」名副其实

- 目标：把 `strategy_store`（候选 → 对照评估 → 升级/回滚）接进 `ml_engine`，或**明确降级口径**：页面与文档统一写「自我迭代未启用（仅复用冠军与同条件教训）」。
- 二选一必须做，不能一边仓库里有进化模块、一边页面写暂停、一边赛题材料说自我迭代。
- 验收：若接入，至少一条真实运行能看到 `strategy_pinned` / 候选对照结果；若降级口径，`/ml/version` 与页面文案一致。

### P3 补评估口径

- 对照页与报告补：相对「全高成本模型」的 **token 节省 %、费用节省 %、耗时增量**（样本不足时显式写"1 对实验不构成结论"）。
- 验收：现有 6 对数据能算出百分比（如 churn@42 token +18%、费用 −5.8%、耗时 +1.04s）。

### P4 追问识别改为意图判定

- 现状：`is_followup` 是关键词白名单，「局限」「为什么」「这些结论」都不在里面。
- 目标：有 `previous_run_id` 且问题**不含新数据/新训练意图**时默认走 follow-up；白名单只作为额外强化。
- 验收：C7 里那条「那这些结论有什么局限？」能直接回答（不检索、不训练），且不误伤「换一批数据重做」这类真新任务。

## 3. 约束（照旧）

- 不装新依赖、不改系统配置；密钥不写入代码/文档/日志。
- 最终测试集不参与调参与选型；不可编造指标，未知用量记 unknown 不记 0。
- 改动必须带测试；改完 `python -m unittest discover -s tests` 全绿，再重启 8018。
- **重启前先把版本串改好**（C5 的教训）：`ml_api.py` 的 `ml-chain-NN`、页面标题、`static/ml-app.js` 的握手判断三处一起改。
- 真实 API 消耗保持克制：验收用 1–2 次研究 + 1 次对比即可。

## 4. 交付

- `HANDOFF.md` 追加一节「Codex 复核结论 + 修复」：逐条判定 C1–C8、改了哪些文件、测试数、真实运行的任务号与费用。
- `.agent-log/events.jsonl` 追加一行（`agent` 用 `codex`），`summary` 带真实数字。
- 如果某条你认为我说错了，**不要改代码**，先在 HANDOFF 里写清楚为什么错，再决定要不要改。
