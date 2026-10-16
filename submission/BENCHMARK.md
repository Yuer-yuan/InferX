# Benchmark 报告与证据索引

## 单卡原版完整评测结果（本轮节点 ucloud1）

2026-10-16完成一轮；原版runner退出0，四档各100正式+8预热、seed42、精确模型tokenizer8192/512，同机回环。第二台只做部署与功能核验，不重复评测或合并成绩。该结果按eval.zip执行，主办方参考MD与ZIP差异仍见[协议说明](PROTOCOL.md)。

固定：单张49,140MiB RTX4090、450W；NInfer v2混合Q4/Q5、MTP4/lm-head-draft、活动上限8、prefill2048、int8 KV、64K上下文、KV容量131072、greedy/no-thinking、presence/frequency penalty0。[完整版本清单](model-manifest.json)、[机器盘点](../results/ninfer-performance/inventory/)、[配置](../configs/ninfer/qwen38-27b-baseline.template.json)。主仓运行快照8e974a9；后续只追加材料与离线分析，原版runner、引擎/API补丁和二进制未变。

| 并发 | 成功/100 | 实际ISL/OSL | TTFT P95(s) | TPOT P95(ms) | E2E P95(s) | 输出tok/s | 两项延迟SLO |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | 100/100 | 8192/512 | 3.45 | 9.01 | 8.05 | 73.59 | 是 |
| 2 | 100/100 | 8192/512 | 3.56 | 22.51 | 15.05 | 92.09 | 是 |
| 4 | 100/100 | 8192/512 | 7.00 | 45.93 | 27.42 | 94.61 | 否 |
| 8 | 100/100 | 8192/512 | 13.86 | 90.66 | 53.28 | 103.31 | 否 |

最高已测正式吞吐103.31 tok/s（C8）；同时满足TTFT P95≤10s、TPOT P95≤40ms的最高吞吐92.09 tok/s（C2）。这两项延迟通过不等于质量或提交验收通过。P95先计算每条TPOT再取原版插值百分位；排除预热，不丢弃失败样本，实际零请求错误。只测一轮，无跨轮置信区间；PILOT中的8请求数字另列。

| IFBench前100 / batch2 | prompt strict | instruction strict | prompt loose | instruction loose |
|---|---:|---:|---:|---:|
| 同一冻结服务 | 55.00% | 58.56% | 55.00% | 59.46% |
| 原生有效分母 | 100 | 111 | 100 | 111 |

传入generation_config只有max_tokens32768；展开temperature/top_p/seed为null，服务实际greedy/temperature0、top_p约0.8、seed42、零presence/frequency penalty。保留EvalScope默认retries5等项，无额外原生chat请求；100生成/100评分，零执行错误。输出4–2362 tokens，全部stop_token，没有触及32768输出上限。质量未达到参考表列出的80%/76%数值，但表格两列同名、比较口径待确认；没有同条件MTP0/高精度对照，不能归因于量化或MTP。

证据入口：

- [原版完整报告](../results/ninfer-performance/official-r1/runs/official-r1/report.md)、[原始结果/快照清单](../results/ninfer-performance/official-r1/)、[独立重算与覆盖状态](../results/ninfer-performance/official-r1/analysis.json)。
- [性能与瓶颈报告](../docs/reports/2026-10-16-ninfer-performance.md)：8种配置19档PILOT、原生计时/源码分析及正式等待数据。C8中prefill调度单元计时之和约占测量时长69.2%，不是GPU kernel占用率。
- [IFBench失败案例](../results/ninfer-performance/official-r1/quality-failures.json)：词的位置、数字个数、首词改写等；保留全部预测/review/resolved_config。
- [补充功能回归](../results/ninfer-performance/quality-regression/)：46867-token首/中/尾检索、代码隔离测试通过；逻辑题384和2048预算均截断，不计完整通过。
- [备用机最终工具调用](../results/ninfer-performance/ready-baseline/final-tool-probe/)：与正式服务相同MTP4配置，只做短功能检查。

独立GPU采样目标1Hz，共2747点。C1/C8分别存在8.402/6.682秒间隔，严格覆盖状态partial；C2/C4覆盖通过。显存已记录点最大24836MiB，温度最大85°C；所有已记录点中power-cap Active为1957/2747，软/硬thermal均Not Active。只能报告采样点统计，不能声称完整时间平均、连续峰值或已达到带宽roofline。缺口诊断和原始CSV（无损gzip）完整保留；先前113个快照文件SHA均可通过解压字节验证。

## 已有优化与工程证据（vlm-server，24GB 卡）

| 证据 | 可支持的结论 | 不能推出的结论 |
|---|---|---|
| [NInfer/MTP 调优](../docs/reports/2026-10-14-ninfer-mtp.md)，[provenance](../results/tuning/provenance.json) | 固定 v2 权重/引擎的单卡可部署性；小规模代码生成约 185–241 tok/s，收益随输出变化 | UCloud 的 8192/512 正式吞吐、普遍质量不变 |
| [上下文与 Agent](../docs/reports/2026-10-14-context-agents.md)，[summary](../results/context-agents/summary.json) | 253,914 输入有限检索 3/3；长冷预填充约 200.8s；三个 CLI 实际工具调用 | 256K 任意任务均准确、三个客户端均完成所有任务、IFBench 分数 |
| [并发/Roofline](../docs/reports/2026-10-14-concurrency-roofline.md) | 48K–248K 负载下预填充与准入等瓶颈诊断 | 已证明所有负载达到物理带宽极限 |
| [PD/简单策略](../docs/reports/2026-10-15-pd-strategies.md)，[solo comparison](../results/pd-strategies/solo-comparison/comparison.json) | 简单修正明显降低短请求干扰，但存在长短请求代价 | 所有 SOTA 都无法解决、官方负载下也有同倍数收益 |
| [MorphX](../docs/reports/2026-10-16-morphx-control.md)、[EB+](../docs/reports/2026-10-16-ebplus-validation.md)、[QoServe](../docs/reports/2026-10-16-qoserve-validation.md) | 已进行源码审查、适配与各自资格验证，局限见原报告 | 已有完整公平 SOTA 性能排名、我们已提出并验证新调度机制 |

PD 补测的简要结果：

| 策略 | 短请求：冷长混合/单独 P95 | 短请求：续接长混合/单独 P95 | 长请求：冷长混合/单独 | 长请求：续接混合/单独 |
|---|---:|---:|---:|---:|
| 原策略 | 33.58× | 26.00× | +3.6% | +1.9% |
| 简单修正 | 1.78× | 2.00× | +20.9% | +15.4% |
| PD 组合 | 1.50× | 1.52× | +35.8% | +34.9% |

条件是 MTP0、eager、C4；单独基准每项一轮，混合数取此前三轮中位数；78 条补测请求正确。该诊断比较不提供置信区间，也不作为 8192/512 的比赛成绩。

## 贡献与材料边界

上游提供 NInfer 引擎、量化格式和 MTP 机制；本项目已有工作包括版本核对、工具链/部署调优、长上下文和 Agent 集成、压测与故障定位、策略适配及可追溯实验。本轮原版完整分数已补齐，仍无完整新机制胜出证据，答辩时据实说明。部分历史原始大日志仍在旧服务器，仓库只保存摘要和哈希；提交引用的关键结果必须补入可交付原始包，见 TODO。
