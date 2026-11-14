# 各模型用量与费用

V16 在开发者面板增加按模型分账，比较组同步携带各模型用量和费用。接口记录实际模型、官方供应商标识、请求 UTC、输入/输出 Token，以及 DeepSeek 缓存命中/未命中计数。缓存分类不重复加进总 Token。解析失败但已返回 usage 的调用仍计入费用，未返回用量的失败不能按零费用处理。

价格核验日：2026-11-14。币种统一 USD，不自动用不明汇率换算人民币。DeepSeek off-peak 每百万 Token：Flash 缓存命中0.003、未命中0.15、输出0.6；Pro 缓存命中0.022、未命中0.66、输出1.98。高峰2倍。Jev输入0.042、输出免费。来源：
https://api-docs.deepseek.com/quick_start/pricing/
https://help.typesafe.ai/articles/5187202169-how-much-does-jev-cost
缓存字段： https://api-docs.deepseek.com/api/create-chat-completion/

周末或非高峰时段可按 off-peak 精确核算公开价。工作日高峰小时但没有核定中国节假日日历时，显示高低峰范围；缓存计数缺失时显示全命中至未命中范围。型号/供应商没有核验、请求用量缺失则不能声称精确费用，旧记录不补造缓存和时间信息。所有金额均与核验日的公开单价对应，不能保证未来价格不变。

官方价核算不等于账户实扣：赠送额度、账户币种、合同折扣和账单调整仍需供应商账单确认。当前API没有取得账单实扣，invoice_actual_usd 始终 null，页面明确显示未对账，不用估算冒充真实账单。对比仅在双方费用均可核算时计算美元差额，不根据总Token判断谁便宜。

实测：用加密保存的密钥对三个模型各发一次短请求，记录真实 usage 和缓存信息。Flash 输入39/输出6，核算$0.00000945；Pro输入36/输出5，$0.00003366；Jev输入433/输出46（免费），$0.000018186。此项是费用采集验证，不是完整研究任务账单。证据在 artifacts/cost-accounting-acceptance/e2859d35ea1d48a8811938d0b6adff36/usage.json。
