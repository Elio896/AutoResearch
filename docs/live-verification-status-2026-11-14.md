# 真实 LLM 与 TCP 严格复测状态

这份状态更新优先于之前的“344 通过、13 跳过”说明。**本轮不跳过 TCP，不把规则回放当真实 LLM 学习。**

## 13 项测试实际执行结果

命令：`python tools/run_tcp_checks.py`。

| 项目 | 实际结果 |
|---|---:|
| 发现测试 | 13 |
| 实际运行 | 13 |
| 跳过 | **0** |
| 通过 | 4 |
| 断言失败 | 4 |
| 报错 | 5 |
| 整组通过 | **否** |

通过的 4 项是构造和环境配置校验，不证明 TCP 已连通。9 项真实协议/传输用例未通过，不能改标为通过。已额外加强两个协议错误测试：必须确认本地 HTTP fixture 收到请求，避免把连接失败误当作正确检测到了 malformed response。

真实 socket 探测：本机可以 bind/listen，但 connect 被拒绝，`PermissionError / WinError 10013`。没有修改防火墙、关闭沙盒、更换通道绕过限制，或替换为内存传输来假装验证了 TCP。

证据：`artifacts/tcp-strict-report.json`、`artifacts/tcp-strict-run.txt`、`artifacts/network-environment.json`。

`tools/run_asgi_checks.py` 默认现已运行全部测试，包括真实 TCP。仅显式添加 `--offline` 才跳过 TCP 夹具；离线通过不能再算完整验收。

## 真实 LLM 学习效果

实际执行：`python tools/evaluate_live_learning.py --output artifacts/live-learning-attempt-02`。

结果：**blocked，未配置服务器模型凭据，实际模型调用 0，学习效果未验证，passed=false。** 当前进程没有可用的 `LIGHT_MODEL_API_KEY` 或 `DEEPSEEK_API_KEY`，项目和原项目的 `.env` 路径也不存在。

没有使用假客户端、脚本提案或规则结果补成模型响应。之前的“12 次执行错误降到 0”仍然只是执行约束的实测结果，不能当真实 LLM 已学会。

真实模型评测入口已完成，冻结三个种子、三个实验臂，共最多 18 次真实调用：

1. 无记忆；
2. 仅提示词纠错；
3. 跨会话持久记忆与执行约束。

分别记录 **模型原始提案错误率（拦截前）** 和 **真正执行的错误率（拦截后）**、有效响应数、验证 F1、真实 usage 与耗时。如果只是执行器拦住错误，不会判为 LLM 学习；基线本来没犯错、输出不完整、质量证据不足时，也不会宣称提升。这是在上下文与检索记忆层做适应性测试，不是训练模型权重。

为避免混淆两种效果，还修复了事件证据的可变引用问题：记录时深拷贝，后续自动修正提案不能回写此前的“模型原始提案”。已增加回归测试。

证据：`artifacts/live-learning-attempt-02/report.json`、`protocol.json`、`source/`。

## 完成验证所需条件

需要一个允许正常本机 TCP 和已选模型 HTTPS 接口访问的执行环境，以及通过服务器环境变量配置好的模型凭据。不能在当前受限会话里通过改测试、关闭保护或绕行网络来制造通过结果。

配置已选模型的 `LIGHT_MODEL_API_KEY`（或 `DEEPSEEK_API_KEY`）、`LIGHT_MODEL_BASE_URL` 和 `LIGHT_MODEL_NAME` 后，在具备正常网络能力的环境执行：

```powershell
python tools/run_tcp_checks.py
python tools/run_asgi_checks.py
python tools/evaluate_live_learning.py --output artifacts/live-model-new-run --max-calls 18 --max-tokens 150000 --max-seconds 240
```

输出目录必须使用新名称，避免覆盖原始证据。真实模型费用只记供应商 usage；没有账单时不填零费用。

全套严格回归也已执行：358 项中 349 通过、4 断言失败、5 报错、0 跳过。失败组与这 13 项中的 9 项一致。日志 `artifacts/full-strict-regression.txt`。当前结论仍是未完成真实网络与真实 LLM 验收，不能改成已通过。
