# 无法识别因果时继续可支持的分析

V18 区分因果证据不足与任务完全不可执行。明确因果/导致类问题即使分类置信度高，也会触发高级方法范围复核。复核可返回 partial，指定1至3个真实且经过特征审核的列。仅缺因果识别时不要求用户重复确认，继续执行本地关联工具；目标含糊或真正缺字段仍需补充信息，接口失败仍明确阻止执行。

关联工具使用训练行，给出相关列的分组目标类别比例、N、目标类别计数和Wilson95%比例区间。数值多于10个取值时用四分位组；类别/少量数值按原值分组，稀少组和较多类别合并。小于30行标记小样本。没有做因果识别、混杂调整或复杂抽样推断；区间不是因果效应区间，不推断合并组的成员。高级审核根据真实统计解释能回答和不能回答的部分，不编造效应方向或显著性。

该模式不生成分类预测表，不打开最终测试集，不把额外基线训练的F1当作关联研究成果，也不将部分结果存成预测最佳经验。状态 partial_completed，页面显示可支持的关联分析已完成；真正信息不足为 needs_input（黄色），技术故障为失败/阻塞。比较面板中关联任务不比较预测F1。

已用真实API对“现有数据能否支持增加客服通话导致客户流失，并完成可支持的关联分析”运行，成功进入 partial_completed，基于3000条训练记录完成 number_customer_service_calls 分组统计，审核返回无法确立因果并解释观察关联，未访问测试集。原记录在 artifacts/association-acceptance/c45843bc8f33476b9c613905baa76602。后续进一步明确小样本和合并组解释限制。

## 2026-11-14 独立验收补丁：字段名容忍与诊断

验收发现一次 `needs_input`（run `9a351c5d5d824825b4580b671db6c31f`）并非真的缺信息：高级复核返回 `association_columns: ["number_customer_service_calls", "class"]`，把**目标列**回声进了字段列表，旧校验 `any(c not in session.x.columns)` 于是整表判无效、直接问用户。

现在的解析 `ml_engine.resolve_association_columns`：目标列不在特征集内，回声会被丢掉而不是否掉整次复核；同时容忍大小写、空格/连字符、`"a, b"` 单字符串和 `字段（注释）` 写法。**仍然不接受杜撰列**——解析不出任何真实特征时才追问，且追问会列出可选字段；无法解析时记 `route_recheck_invalid_columns`（含模型原始返回，≤200 字）便于定位，能解析但被裁剪时记 `association_columns_adjusted`（requested / used）。

复测 run `cff317c07aab4eac8f399725a43fadd3`：`partial_completed`，3000 训练行，事件含 `association_columns_adjusted`。单测见 `tests/test_ml_pipeline.py`。

