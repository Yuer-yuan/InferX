# InferX 项目介绍

**队名：你说啥都队｜成员：陈文欣、张志雄、郭坤昌**

InferX面向单张48GB RTX 4090上的Qwen3.8-27B推理。我们基于NInfer接入真实4bit计算，改进多请求状态管理和调度，让固定硬件服务更多请求，同时控制用户等待时间。

当前冻结100在客户端14/引擎13下两轮同时满足延迟目标，总输出306.04、309.95 token/s；同版独立IFBench前100题严格85/100、宽松88/100。同一源码正常重建后的664另有性能304.90 token/s与独立精度90/100、96/100记录。各次版本、配置和失败样本见[Benchmark](../track-a/benchmark.md)。

| 基础材料 | 阅读入口 |
|---|---|
| 答辩展示 | [中文 PDF](presentation/InferX-ZH.pdf) / [English PDF](presentation/InferX-EN.pdf)、[Typst 源码与实时预览](presentation/README.md) |
| 展示结构 | [八章与页码说明](presentation/README.md) |
| 现场演示 | [现场C14性能评测命令](demo.md) |
| 汇报与问答 | [五分钟讲稿及追问](defense-notes.md) |
| 代码仓库 | [GitHub](https://github.com/Yuer-yuan/InferX)、[仓库README](../../README.md) |

完整技术叙事见[技术报告](../track-a/README.md)，实现、Benchmark、部署和复验均从该页进入。精确版本及原始记录见[实验索引](../track-a/experiment-index.md)。
