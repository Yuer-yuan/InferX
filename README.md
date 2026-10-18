# InferX

InferX 面向单张 **48GB RTX 4090** 上的 Qwen3.8-27B 推理，基于 NInfer-4090 优化量化算子、批处理、状态管理和调度。测试同时报告吞吐、请求延迟与任务质量，并保留对应版本和原始结果。

队名：**你说啥都队**。成员：**陈文欣、张志雄、郭坤昌**。

| 你想了解什么 | 从这里开始 |
|---|---|
| 项目与答辩内容 | [材料总览](submission/README.md)、[中文 PDF](submission/basic/presentation/InferX-ZH.pdf) / [English PDF](submission/basic/presentation/InferX-EN.pdf)、[Typst 与实时预览](submission/basic/presentation/README.md)、[技术报告](submission/track-a/README.md) |
| 性能、精度和原始证据 | [Benchmark 报告](submission/track-a/benchmark.md)、[成绩与证据索引](submission/track-a/generated-results.md) |
| 如何部署和复测 | [部署配置](submission/track-a/deployment.md)、[复验步骤](submission/track-a/reproduction.md) |
| 如何现场演示 | [现场性能评测](submission/basic/demo.md) |
| 团队提交前需要准备什么 | [提交清单](submission/records/2026-10-18-submission-checklist.md) |

当前推理服务为 `ninfer-serve`。项目在开源 NInfer-4090 基础上修改；上游实现、团队优化及适用版本见[优化说明](submission/track-a/optimization.md)。复测时应使用成绩对应的源码、权重、量化缓存和配置，不能仅以主仓 `main` 的当前内容代替实验版本。

## 部署与研究记录

公共工具适用于单卡 Linux/CUDA 环境，数据目录、CUDA 和下载源可配置。`ucloud1/2/3` 是实验节点名称，每次测试各使用一张卡，具体硬件和客户端位置随结果记录。早期部署和研究记录按日期保留，适用版本以各报告为准。

- [基础 NInfer 部署与早期评测](submission/REPRODUCE.md)
- [NInfer 基线节点使用与后续实验](docs/ninfer-operations.md)
- [本轮 NInfer/MTP 消融、正式评测进展与瓶颈](docs/reports/2026-10-16-ninfer-performance.md)
- [官方参考配置复现与 vLLM / SGLang 基线](docs/reports/2026-10-16-framework-c4.md)
- [OpenCode、Pi、Claude Code 直接接入 Qwen](docs/agent-quickstart.md)（已配置，GPU1 的 18081 服务，256K 后端）
- [256K 上下文与三个 Agent 的验证结果](docs/reports/2026-10-14-context-agents.md)
- [长上下文、并发与 Roofline 验证](docs/reports/2026-10-14-concurrency-roofline.md)（48K–248K 输入；长预填充准入阻塞与 MTP 对照）
- [NInfer + MTP 单卡调优结果与服务使用](docs/reports/2026-10-14-ninfer-mtp.md)（此前64K调优，代码生成约185–241 token/s）
- [考题原文](docs/competition-requirements.md)
- [实验设计、判断标准及相关工作](docs/superpowers/specs/2026-10-14-feedback-validation-design.md)
- [实施计划](docs/superpowers/plans/2026-10-14-feedback-validation.md)
- [部署、实验状态与复现方法](docs/reports/2026-10-14-pilot.md)
- [仓库补全与多会话实验设计](docs/superpowers/specs/2026-10-14-repository-branches-design.md)
- [仓库实验报告与复现方法](docs/reports/2026-10-14-repository-pilot.md)

## 早期代码生成与验证实验

仓库也保留基于 llama.cpp 的代码生成—CPU 验证反馈实验。HumanEval 阶段使用两张卡独立重复；仓库补全阶段的两组 seed 交错运行于同一服务，每次试验只用一张卡。

这些实验的模型、缓存和完整原始记录曾保存在 vlm-server 的 `/home/guokc/mypro/gnn-mount-sda1/InferX`。这是历史运行路径，当前材料的获取与复测入口见上方表格。

早期实验代码：`inferx/scheduler.py` 管理候选并发、截止时间与取消，`inferx/llama_client.py` 调用 llama.cpp，`inferx/verifier.py` 和 `sandbox_worker.py` 执行受限验证，`scripts/analyze.py` 汇总指标。以下测试命令用于这部分 Python 工具：

```bash
python3 -m unittest discover -s tests -v
```

完整测试需要 `aiohttp` 和 `pytest`；未安装时对应集成测试会跳过，远端 HDD 独立 venv 运行完整套件，依赖见 `requirements-repository.txt`。原验证器需要 Linux、系统 Python 3 和 libseccomp；仓库验证器还要求 x86_64 Linux Landlock ABI ≥3，隔离不可用时拒绝执行。

仓库实验使用 `inferx/episodes.py` 管理多个任务的共享生成许可，`inferx/repo_verifier.py` 在私有源码副本中执行完整测试，`scripts/analyze_repository.py` 检查开发/评估网格并保留失败与取消成本。分析与绘图独立于冻结的推理运行代码。
