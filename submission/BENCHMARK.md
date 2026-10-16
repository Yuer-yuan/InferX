# Benchmark 报告与证据索引

## 单卡正式比赛结果（本轮节点 ucloud1）

**尚未执行。** 不用旧服务器短提示速度、参考基准数字或 Agent 体验填入正式表。

| 正式条件 | 状态 |
|---|---|
| GPU | UCloud 单张 RTX 4090，49,140 MiB；[盘点](evidence/ucloud-inventory.json) |
| 最终代码/引擎/模型 revision、量化、tokenizer | 固定 NInfer v2 + completions 补丁，权重及嵌入 tokenizer 已核验；参数筛选中 |
| 上下文、MTP、并发上限、模板、思考/采样 | 待同机消融与质量验证后冻结 |
| 评测端位置/网络、CPU/RAM、依赖锁、功率、显存 | 同机回环；两机环境与依赖记录见 results/ninfer-performance/，正式运行采样待补 |
| 正式协议 | 待确认 PROTOCOL.md 中差异；按原版 runner 组织目录 |

| 并发 | 成功/100 | 实际 ISL/OSL | TTFT P95(s) | TPOT P95(ms) | E2E P95(s) | 输出 tok/s | 两项 SLO 均通过 |
|---:|---|---|---|---|---|---|---|
| 1 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| 2 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| 4 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| 8 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |

| IFBench，前 100 | prompt strict | instruction strict | prompt loose | instruction loose |
|---|---|---|---|---|
| 最终冻结配置 | 未测 | 未测 | 未测 | 未测 |

正式报告补齐重复次数、原始目录和 SHA256、预热排除方式、百分位计算、失败/超时/截断处理、显存采样范围及长文本/逻辑/代码质量案例。服务或配置有变化时单独编号，不拼接不同配置的最优指标。

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

上游提供 NInfer 引擎、量化格式和 MTP 机制；本项目已有工作包括版本核对、工具链/部署调优、长上下文和 Agent 集成、压测与故障定位、策略适配及可追溯实验。尚无 UCloud 正式分数或完整新机制胜出证据，答辩时据实说明。部分历史原始大日志仍在旧服务器，仓库只保存摘要和哈希；提交引用的关键结果必须补入可交付原始包，见 TODO。
