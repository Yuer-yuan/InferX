# InferX · 赛道 A 提交材料

**当前状态：两机基线及原版完整评测已完成，最终提交验收仍有待办。最高吞吐103.31 tok/s；满足两项延迟SLO的最高吞吐92.09 tok/s；IFBench prompt strict/loose55%。**

参赛方案：单卡 Qwen3.8-27B 推理部署、NInfer/MTP 调优及长上下文并发瓶颈分析。公共工具可用于类似 Linux/CUDA 机器；本轮节点名为 ucloud1/2，各单张 48GB RTX 4090，一台测评、一台备用。此前 vlm-server 的单卡 24GB 实验用于解释方案选择。仓库：https://github.com/Yuer-yuan/InferX ，统一开发分支为 `main`。

| 交付物 | 位置 | 当前状态 |
|---|---|---|
| 原始评分规则、参考基准、评测包 | 仓库根目录两份 MD、eval.zip；[输入校验清单](input-manifest.json) | 已归档，原文保留 |
| 官方评测工具 | [原版说明](organizer/README.md)、[入口](organizer/run.sh)、[runner](organizer/runner.py) | 从 ZIP 原样提取，不修改 |
| 模型选型与部署说明 | [复现说明](REPRODUCE.md)、[模型版本清单](model-manifest.json) | 两机相同冻结配置就绪，消融与完整评测已完成 |
| Benchmark 与性能分析 | [证据及成绩表](BENCHMARK.md)、[评测口径](PROTOCOL.md) | 四档各100及IFBench100已补；GPU覆盖缺口明确保留 |
| 优化方案、消融、失败案例 | [本轮性能与瓶颈](../docs/reports/2026-10-16-ninfer-performance.md)、[长上下文](../docs/reports/2026-10-14-concurrency-roofline.md)、[长短混合策略](../docs/reports/2026-10-15-pd-strategies.md) | 已有研究数据，适用范围见报告 |
| 约 8 页答辩展示 | [离线 HTML 草稿](slides.html) | 正式数据已填；队员分工、投屏彩排待补 |
| 现场 Demo、备用录屏 | [3 分钟演示步骤](REPRODUCE.md#现场-demo) | 步骤已备；实机彩排、录屏待做 |
| 全部源码、配置及必要原始结果 | 本仓库固定提交；发布时生成 `artifacts/` 下源码快照与 Git bundle | 生成方法见复现说明；模型权重用精确下载方式交付 |
| 节点与部署证据 | [两机盘点](../results/ninfer-performance/inventory/)、[构建及接口核验](../results/ninfer-performance/preparation/) | 相同权重/构建、完整客户端环境已就绪 |
| 待办和提交验收 | [TODO](TODO.md)、[本轮检查记录](VALIDATION.md) | TODO 是提交前唯一验收清单 |

按评分维度安排材料：

| 分项 | 分值 | 主要材料与补齐重点 |
|---|---:|---|
| 模型与方案选型 | 10 | 模型来源/量化精确版本、显存预算、48GB 上选择该方案的依据 |
| 推理优化思路与创新性 | 40 | 瓶颈诊断、MTP/并发/上下文取舍、同条件消融；区分上游机制和本项目工程贡献 |
| 推理结果评估 | 25 | UCloud 正式四档压测、IFBench 100、任务质量与失败案例、显存和稳定性 |
| 工程与复现 | 10 | 从干净目录部署、锁定环境、源码快照、权重校验、原始结果 |
| 答辩表现 | 15 | 5 分钟汇报、3 分钟真实 Demo、7 分钟问答与成员分工 |

提交前先完成 [TODO 的 P0](TODO.md#p0正式提交前必须完成)。目录中的未测项目不填估算数；参考基准的 80%/76% 也不是本项目成绩。

评分规则写“本周日上午 10:30”，未给绝对日期。按本次整理日 2026-10-16 推算可能为 **2026-10-18 10:30（Asia/Shanghai）**，该日期仅为推断，须向主办方确认。
