# V19 复核与 V20 修复（2026-11-14）

原工程：`C:\Users\junzi\Documents\Codex\2026-11-14\new-chat-2\EvoResearch`。
可写验证副本：`C:\workspace\EvoResearch-v19-review`。
原目录不在本轮可写范围；未部署、未重启原 8018，不能把下列测试当成线上验收。

## C1–C8 逐条核对

| 结论 | 判定与证据 | 处理 |
|---|---|---|
| C1 单 DeepSeek key 无法使用 | 成立：原 configured_clients 构造 JevClassifier(None) 抛 ValueError。但 HTTP 服务本身可以启动，是研究配置/提交被阻断。 | Jev 可选；无 Jev 使用显式 rules 分类，不计作 LLM 请求。ML_SINGLE_MODEL=1 让所有 LLM 角色共用同一客户端/模型。规则置信度是启发式路由分数，没有声称校准或 F1。 |
| C2 策略进化未接主链路 | 成立；research_system 的导入不能证明 MLResearch 使用它。经验复用、单任务调参不等于策略进化。 | ML 主链路接独立 ml-strategies.json，开局 pin，实际 Scientist 提示使用固定策略，报告携带 id/digest。自动候选生成、独立验收、激活及上线回滚仍未启用；UI 和接口明确 auto_evolution_enabled=false。禁止宣称自动自进化已经验证。 |
| C3 缺相对节省比例 | 成立。已有配对差值，不是百分比。 | 按配对计算 (高成本−协作)/高成本×100；速度比为高成本耗时/协作耗时。未知 usage/费用、零分母、未完成、不同范围不计算；负数为变差。前端及 JSON 显示。只代表比较协议内的工作站/审核阶段，排除共同来源采集；不是人工效率或全系统节省。 |
| C4 TLS 失败无重试 | 成立，C 原始文件明确为审核阶段 SSL EOF。 | TierBudget 至多重试一次瞬态网络/指定 HTTP 状态，重新通过共享调用/Token/时间门；失败未知 usage 保留预约，低层不能耗尽审核预留。401、协议错、证书验证错不重试。错误类别写入 state.json。 |
| C5 V19 文件配旧后台 | 设计缺陷成立；仅凭当前文件不能独立证明历史进程起止。 | V20 前后端版本同步，并比较加载时源码指纹与磁盘指纹，即使忘记改版本也会报告 restart_required 并阻止提交。仍需部署后重启。 |
| C6 四部分重复 | 修补方向正确，现有回归通过。原 topic 文件还保留修复前重复文本，不能把它当修复后结果。 | 保留 present_answer 修复并跑回归。 |
| C7 追问关键词漏判 | 成立，原 is_followup 不包含“这些结论有什么局限”。 | 显式 previous_run_id 且无新数据意图，默认解释型追问；换数据优先重开流程；要求重训仍沿用旧 CSV 并运行实验。 |
| C8 不匹配问题仍产预测 | 原问题有历史记录；V19 已在 run_fit 前调用 scope_gap，已覆盖 churn→失业、缺收入字段等明确情况。 | 保留并回归，未声称已覆盖所有跨领域不匹配。 |

## 实测数据核对与边界

直接读取原 artifacts/acceptance-20261114/topic-eval-results.json：
A：0.9090745788→0.9115826702，7209 Token，估算 $0.003378838，28.456995 秒。
B：1→1，5705 Token，估算 $0.003028714，11.732641 秒。
C：审核网络失败；D/E：不重训、1 次审核调用。
A/B 的原 stages 对“收集证据”标记为 false，不能脱离输入方式及事件范围笼统引用“6/6”。
追问和完整研究任务不同，不能用二者费用比证明同任务路由优化；费用为价格核算，不是已对账账单。

## 验证

- 新增回归先红后绿，覆盖单 key、同模型、规则来源、TLS 成功重试/二次失败/预算耗尽、不重试永久错误、策略实际进入提示、HTTP 接口配置、源码变更、解释追问和配对比例。
- 全量命令：`C:\workspace\EvoResearch\.venv\Scripts\python.exe -m unittest discover -s tests -q`。
- 最近完整结果 436 tests OK；进程结束另有一个旧测试套件的未关闭 socket ResourceWarning，未定位修复，不隐去。
- `node --check static/ml-app.js` 通过。
- tools/verify_v20_http.py：实际本机 HTTP 模型协议桩 + sklearn 两次真实训练；rules 分类、同一模型客户端、两次 HTTP 请求、验证 F1 0.9333333，未使用最终测试集，付费 API 调用 0。桩回复不是实模型智能表现。
- 没有重新跑真实供应商付费研究；没有新的节省百分比实验结论；没有启用或证明自动策略进化。

## 部署

补丁内 apply_patch.py 默认只检查。先在原工程停止现有研究并备份，再以该工程的 Python 运行 `apply_patch.py --target 原工程目录 --apply`。
脚本逐文件校验 V19 SHA256，检测并发改动则全量拒绝；自动备份后写入，保留 credentials、runtime、数据、原交接记录。
应用后以原启动方式重启 8018，确认 /ml/version 为 ml-chain-20、restart_required=false，强制刷新页面后再做一轮真实验收。
