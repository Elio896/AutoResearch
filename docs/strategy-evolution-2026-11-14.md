# 策略自进化闭环 · 变更说明（2026-11-14）

八个阶段已实现并逐段验收。本文是「改了什么、凭什么算通过、哪些是模拟」的唯一说明。

## 一、新增模块

| 模块 | 作用 | 关键约束 |
|---|---|---|
| `strategy_store.py` | 策略=配置的版本库（candidate/active/retired/rejected），原子落盘、可 diff、可 pin | 只许改 3 个白名单字段；`activate` 要求 checks 全真 + jev.decision=upgrade |
| `qwen_worklog.py` | 千问结构化工作记录与证据引擎 | fact 里的数字必须能在其 `evidence_ref` 的值里找到；缺字段即 `invalid`，不补 |
| `failure_modes.py` | 从真实事件流找五类失败模式 | 每条带 `event_refs`（事件流下标）；证据不足 → `insufficient_evidence` |
| `candidate_proposer.py` | 云端模型提候选策略 | 一次只许改 1 个字段；hypothesis 五项齐全，**没有 falsifier 直接退** |
| `strategy_eval.py` | 新旧策略对照评估 | 条件跑前冻结；开发任务/评估任务分离；**不退化不等于改进**，必须有 `demonstrated_improvement` |
| `strategy_decision.py` | Jev 三选一 + 代码硬闸 | 评估没过不问 Jev；Jev 不可用/答非所问一律 keep，不伪造 |
| `strategy_rollback.py` | 上线后按预定规则回滚 | 规则上线前冻结；未知指标 → insufficient，不误切；必须带 triggers+evidence |
| `light_model.py` | 轻量模型 API 客户端（OpenAI 兼容） | 无 key 不可用；调用上限；成本未知记 unknown 不记 0；key 不外泄 |

前端：`static/demo.html` 增加「千问工作记录」（观察→计划→工具→验证→转交，逐块标来源）与
「策略进化记录」（当前策略、版本、差异、事件时间线、失败模式）。后端新增 `GET /strategy/state`、
`GET /strategy/failures`（只读，读真实报告，不现编）。

## 二、闭环怎么闭环

```
研究运行（pin 住策略版本）
  → failure_modes 从真实事件找失败（每条可追溯到 event_007 这种下标）
  → DeepSeek 提一个带假设的候选（candidate，正式策略不动）
  → strategy_eval 用独立评估任务对照（条件先冻结）
  → strategy_decision：代码检查 + Jev 三选一，两项都过才 activate
  → 上线后 strategy_rollback 按冻结规则监控，触发就切回旧版本
  → 失败版本保留（retired），成为下一轮候选的证据
```

## 三、验收证据（可复查）

| 阶段 | 命令 | 产物 | 关键结果 |
|---|---|---|---|
| 1 策略版本 | `python tools/strategy_acceptance.py`、`tools/strategy_run_binding.py` | `artifacts/step1-*.txt` | 12 tests；三种拒绝；重启后仍在；pin 后库里换版本也不影响本次运行 |
| 2 工作记录 | `python tools/qwen_worklog_acceptance.py` | `artifacts/step2-acceptance.txt` | 证据随输入变（0.3075→0.0525）；编造 99999 被拒；真实 qwen 输出判 invalid |
| 3 失败模式 | `python tools/failure_modes_acceptance.py` | `artifacts/step3-acceptance.txt` | 真跑 21 事件检出 `repeated_proposal` refs `event_007/013`；单实验运行判 insufficient |
| 4 候选策略 | `python tools/candidate_acceptance.py` | `artifacts/step4-acceptance.txt` | 越权字段/空 falsifier/改两个字段/证据不足 全部拒绝；active 未变 |
| 5 对照评估 | `python tools/strategy_eval_acceptance.py` | `artifacts/step5-acceptance.txt` | 好候选 upgrade（重复率 0.5→0.0）；无效候选 keep；样本不足 insufficient |
| 6 Jev 闸 | `python tools/strategy_upgrade_acceptance.py` | `artifacts/step6-acceptance.txt` | 真 Jev 无 key → keep；两关都过才切；切换后新运行 pin 到新版 |
| 7 回滚 | `python tools/strategy_rollback_acceptance.py` | `artifacts/step7-acceptance.txt` | 上线→退化→按规则回滚→后续运行用回旧版；失败版本保留 |
| 8 前端 | headless 截图 | `evo/out/10-new-analysis.png` | 两块均渲染；无模型时写「程序规则·非模型输出」 |
| 全部 | `python -m unittest discover -s tests` | `artifacts/suite-final.txt` | **321 tests OK** |

## 四、哪些是模拟，哪些是真的

- **真的**：实验执行、指标、事件流、失败检测、对照评估、版本落盘、切换/回滚、前端读数。
- **模拟**：DeepSeek 的候选（无 `DEEPSEEK_API_KEY`）、Jev 的选择（无 `TYPESAFE_API_KEY`）、
  对照评估里的提案器（确定性脚本，模拟模型行为）。
- 模拟的地方都在数据与界面上标了：事件里 `by=seed-demo-simulated`、`jev.model=simulated-jev`、
  版本上 `simulated` 标记、脚本输出里写明"仅验证流程"。
- **没有**把模拟升级当成真实自进化效果。真实自进化要等 key：配上 `LIGHT_MODEL_API_KEY`
  （或 `DEEPSEEK_API_KEY`）后，提案器换成真模型，同一套闸门不用改。

## 五、已知缺口

1. 没有 API key，模型层实际跑的是「程序规则」；`/demo` 页面第一块会如实这么写。
2. 对照评估的提案器是脚本，所以验证的是**闸门与流程**，不是模型能力提升。
3. 策略目前只影响提示词与选路规则，还没有接进 `research_loop` 的实际提案路径（下一步）。
4. `demo_flow` 的 `verification` 用的是固定四项检查，还没有按任务类型扩展。
