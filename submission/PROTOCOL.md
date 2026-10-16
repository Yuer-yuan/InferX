# 评测口径与待确认差异

依据仅为根目录两份主办方 MD 与 eval.zip 内原版 README/runner。本目录不修改评测规则，也不自行发明分数公式。正式执行协议须主办方确认。

| 项目 | Qwen 基准 MD | eval.zip 的实际入口 | 当前准备方式 |
|---|---|---|---|
| 硬件 | 单张 48GB RTX 4090 | API 客户端，不核验服务端 GPU | 使用 UCloud 实测的单张 49,140 MiB 4090，另存硬件证据 |
| 并发与样本 | 1/2/4/8/3，不同样本数；C2 追加 100 | 1/2/4/8，各 100 | 先按 ZIP 默认整理；C3/额外复测是否必需待确认 |
| 输入 8192 | 模型非 special token ID | 默认 builtin=tiktoken/o200k_base；可显式模型 tokenizer | 建议严格模式并固定匹配 tokenizer revision；最终由主办方确认 |
| 数据与缓存 | 每档新种子，无公共前缀 | 同一确定数据/顺序，每档不同 cache_salt | 保持原版；验证服务确实支持缓存隔离，不静默忽略 |
| 预热、超时 | max(4,C)、180 秒 | 每档 8、600 秒 | 保持 ZIP 原值；记录正式协议 |
| 生成 | temperature=0，ignore_eos，512 | 同左，核验服务端输出数 | 输出不够 512 视为异常，不能伪造 usage |
| 精度 | IFBench 前 100；max_tokens=32768 | EvalScope 1.13.0、batch=2，只显式覆盖 max_tokens | 使用原版配置；完整保留展开后的 defaults |
| 上下文 | 示例服务 32768 | ≤32768 时跳过精度并报配置异常 | 需实际窗口大于 32768，并容纳最长提示词+输出预算 |
| 达标 | TTFT P95 10s、TPOT P95 40ms | “完整/异常”仅检查完整性，不判 SLO | 单独列两项 SLO 的联合通过情况 |
| 精度参考值 | 同名列分别 80%、76%，另有 MTP 循环 | 四项原生 IFBench score | 确认比较口径；参考数不作为我们的成绩 |

TPOT 为**每条请求** `(E2E−TTFT)/(输出 token 数−1)`，再对请求取百分位。不是每个 SSE 间隔的 P95，不是引擎 token 间隔，也不是 `1/总吞吐`。报告沿用 runner 的实际百分位实现和统计，不重新挑选成功子集掩盖异常。

性能的四档依次闭环运行，预热不计正式指标，网络计入耗时。官方客户端为 AIPerf 0.13.0，2 worker，关闭其 GPU telemetry；服务端另以 1 秒间隔采样 `nvidia-smi`。1 秒采样最大值不能宣称连续显存峰值。

IFBench 不继承性能阶段的温度、ignore_eos、512 输出长度、并发或随机种子。即使显式项只有 max_tokens，服务端的默认思考/模板/采样仍须记录。报告四项 `prompt_level_strict`、`inst_level_strict`、`prompt_level_loose`、`inst_level_loose`；完整性检查以 100 个样本为准，指令级计数可能超过 100。

原生输出必须保留：report.md、summary.json、configuration.json、environment.json、model_info.json；performance/ 下的数据、元数据和 c1/c2/c4/c8 全部原生导出；accuracy/ 下传入/展开配置、native 预测、review 和 report。GPU 采样、服务日志与环境锁另存为配套证据。

退出码：0=完整或 dry-run 成功，1=评测异常，2=参数/环境/启动错误，130=中断。必须同时核对实际请求、生成长度、错误和 SLO；进程退出 0 本身不是比赛达标证明。
