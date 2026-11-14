# 本地数据搜索库

默认来源已切换为 LocalCatalog，不再在线查询 OpenML。数据位于 data/catalog，index.json 包含中英文别名、目标列、数据形状、SHA256 和原始来源。搜索后直接读取本地 CSV；未知需求不伪造数据。

已收录真实数据：客户流失（OpenML 40701 缓存）、鸢尾花、葡萄酒、乳腺癌（scikit-learn 自带）。客户流失数据不是合成示例。数据获取后已固定为本地快照。来源与适用范围应在具体研究时审查。

新增数据可通过 LocalCatalog.add(id,csv,target,aliases,provenance) 导入，必须明确目标列和来源。页面 /ml/catalog 展示可用目录。LLM 推理仍需 API，本地搜索和数据读取不需网络。
