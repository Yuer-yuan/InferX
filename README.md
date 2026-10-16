# InferX

**比赛材料入口：[submission/README.md](submission/README.md) · [提交 TODO](submission/TODO.md) · [部署与官方评测](submission/REPRODUCE.md) · [8 页答辩草稿](submission/slides.html)。**

比赛正式评测使用 UCloud 的单张 **48GB RTX 4090**。已有 vlm-server 单卡 24GB 数据用于调优与工程佐证；UCloud 正式性能与 IFBench 尚未运行，不能用历史 token/s 替代正式成绩。开发内容统一集成到 `main`，提交时以固定 commit 和源码快照为准。

单 RTX 4090 上的 Qwen3.8-27B 部署及代码生成—CPU 验证反馈实验。HumanEval 阶段用两张卡独立重复；仓库补全阶段的两组 seed 交错运行于 GPU 1 的同一服务会话，每个试验只使用一张卡。

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

模型、数据、缓存、构建及原始生成记录在 vlm-server 的 HDD 目录 `/home/guokc/mypro/gnn-mount-sda1/InferX`。本仓库保存代码、计划及轻量结果；模型/数据下载使用 hf-mirror。

核心代码：`inferx/scheduler.py` 管理候选并发、截止时间与取消，`inferx/llama_client.py` 调用本机 llama.cpp，`inferx/verifier.py` 和 `sandbox_worker.py` 执行受限验证，`scripts/analyze.py` 检查实验网格并汇总指标。

```bash
python3 -m unittest discover -s tests -v
```

完整测试需要 `aiohttp` 和 `pytest`；未安装时对应集成测试会跳过，远端 HDD 独立 venv 运行完整套件，依赖见 `requirements-repository.txt`。原验证器需要 Linux、系统 Python 3 和 libseccomp；仓库验证器还要求 x86_64 Linux Landlock ABI ≥3，隔离不可用时拒绝执行。

仓库实验使用 `inferx/episodes.py` 管理多个任务的共享生成许可，`inferx/repo_verifier.py` 在私有源码副本中执行完整测试，`scripts/analyze_repository.py` 检查开发/评估网格并保留失败与取消成本。分析与绘图独立于冻结的推理运行代码。
