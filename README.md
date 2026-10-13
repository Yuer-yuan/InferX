# InferX

单 RTX 4090 上的 Qwen3.8-27B 部署及代码生成—CPU 验证反馈实验。两张卡用于独立重复实验，每个试验仍只使用一张卡。

- [考题原文](docs/competition-requirements.md)
- [实验设计、判断标准及相关工作](docs/superpowers/specs/2026-10-14-feedback-validation-design.md)
- [实施计划](docs/superpowers/plans/2026-10-14-feedback-validation.md)
- [部署、实验状态与复现方法](docs/reports/2026-10-14-pilot.md)

模型、数据、缓存、构建及原始生成记录在 vlm-server 的 HDD 目录 `/home/guokc/mypro/gnn-mount-sda1/InferX`。本仓库保存代码、计划及轻量结果；模型/数据下载使用 hf-mirror。

核心代码：`inferx/scheduler.py` 管理候选并发、截止时间与取消，`inferx/llama_client.py` 调用本机 llama.cpp，`inferx/verifier.py` 和 `sandbox_worker.py` 执行受限验证，`scripts/analyze.py` 检查实验网格并汇总指标。

```bash
python3 -m unittest discover -s tests -v
```

完整测试需要 `aiohttp`；本机未安装时 HTTP 测试会跳过，远端客户端环境运行完整套件。验证器需要 Linux、系统 Python 3 和 libseccomp。技能安装于 `.agents/skills`，来源和许可见该目录的 `PROVENANCE.md`。
