# InferX · 赛道 A 参赛材料

当前进展：100版本C4/C8/C12/C13及两轮C14通过联合延迟SLO；C14使用14客户端、13执行槽。同版独立评测105的原100题严格85/100、宽松88/100。最新[结果与配置](track-a/selected-results.md)和[100入口](../demo/selected100/README.md)独立于下面保留的历史证据索引；从完整源码构建并重新评测见[中文指南](track-a/reproduction.md) / [English guide](track-a/reproduction-en.md)。 独立源码重建664已完成旧缓存C14复验（9.896s / 38.286ms、304.897 tok/s）及原100质量（90/96，1个截断空回答）；历史成绩与新复验独立记录。所有包、结果与获取方式以[唯一交付入口](DELIVERY.md)为准。

**队名：你说啥都队。成员：陈文欣、张志雄、郭坤昌。**

InferX 面向单张 48GB RTX 4090 上的 Qwen3.8-27B 推理，基于 NInfer-4090 改进量化算子、批处理和调度，并结合上游 MTP 投机解码提升性能。测试同时记录请求延迟、输出吞吐与任务质量。

建议先看[答辩 PDF](basic/presentation/main.pdf)了解整体方案，再读按相同八章组织的[技术报告](track-a/README.md)，需要核对结果时查看[Benchmark](track-a/benchmark.md)及[实验索引](track-a/experiment-index.md)。需要重新运行项目时，从[复验步骤](track-a/reproduction.md)开始，按所选结果恢复对应版本和配置。

## 基础材料

| 材料 | 文件 |
|---|---|
| 项目介绍 | [项目README](basic/README.md) |
| 答辩展示 | [中文 PDF](basic/presentation/InferX-ZH.pdf) / [English PDF](basic/presentation/InferX-EN.pdf)、[Typst 与实时预览](basic/presentation/README.md) |
| 运行演示 | [现场性能评测命令](basic/demo.md) |
| 汇报与问答 | [讲稿与答辩准备](basic/defense-notes.md) |
| 代码仓库 | [InferX](https://github.com/Yuer-yuan/InferX)、[代码README](../README.md) |

## 赛道 A 专项材料

| 材料 | 文件 |
|---|---|
| 模型选型说明 | [模型与方案选型](track-a/model-selection.md) |
| 部署与配置说明 | [环境、参数与启动](track-a/deployment.md) |
| Benchmark测试报告 | [测试报告](track-a/benchmark.md)、[成绩及原始证据](track-a/generated-results.md) |
| 优化方案说明 | [核心机制与工程工作](track-a/optimization.md) |
| 性能优化分析 | [前后对比、资源与取舍](track-a/performance-analysis.md) |
| 完整复验说明 | [源码、模型、缓存与复测](track-a/reproduction.md) |
| 评分维度映射 | [评分与证据对应表](track-a/scoring-matrix.md) |

## 评测证据

从[成绩与证据索引](track-a/generated-results.md)选择一项结果，可以找到其汇总与版本标识；[来源及配置清单](track-a/data/sources.json)进一步列出逐请求数据、运行配置和日志位置。[原版评测器](organizer/README.md)用于复测，[文件校验清单](track-a/evidence-manifest.json)用于检查材料引用的文件是否一致。

各项性能与质量结果分别对应实际运行版本、协议和样本量。源码版本、模型版本及具体运行开关见[复验说明](track-a/reproduction.md)和[部署配置](track-a/deployment.md)。
