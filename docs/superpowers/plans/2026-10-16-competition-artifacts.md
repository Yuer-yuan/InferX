# 比赛提交材料整理计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按两份评分文档与原版 eval.zip 整理可审阅的提交材料、UCloud 评测待办，并将全部开发历史同步到 main。

**Architecture:** 原始规则与原版 runner 保留为不可修改的输入；submission/ 提供材料索引、复现说明、评测口径、答辩草稿和 TODO。已有 vlm-server 证据与尚未运行的 UCloud 正式成绩分别标注，源码快照使用 Git 固定版本。

**Tech Stack:** Markdown、静态 HTML、JSON、原版 Bash/Python 评测入口、Git archive。

**Spec:** 用户提供的根目录 AI比赛评分规则-v3-for现场学生.md、Qwen3.8-27B基准测试.md、eval.zip，以及本轮指定的 UCloud 评测环境。

## Global Constraints

- 赛道 A；单张 RTX 4090；模型固定 qwen3.8。
- 正式成绩来自 UCloud，旧服务器结果只作为调优与工程证据。
- 不修改官方评测脚本，不以短提示/Agent 实验替代正式 8192/512 或 IFBench。
- 不启动 GPU 评测、不下载权重、不修改用户原有测量内容；待办不得冒充完成。
- 用户已明确选择 main，完成检查后正常快进合入并同步 origin，不强推、不删除研究分支。

## Review Focus

- 规则文档与 runner 参数不一致：列出差异和须确认项，保留原始输入哈希。
- UCloud 的 48GB 卡与旧服务器的 24GB 卡：分别留存硬件证据，不混报成绩。
- builtin tokenizer 的 8192 不等于模型端 8192：正式口径显式固定 tokenizer，待主办方确认。
- IFBench 的 32768 输出预算：上下文必须足够，不静默缩短，不把历史质量探针写成 IFBench。
- 复现与发布：标出绝对路径、外部依赖和原始数据缺口；源码快照排除凭据、缓存与权重。

## 执行任务

### Task 1: 规则归档与材料清单

**Files:** submission/README.md、TODO.md、PROTOCOL.md、organizer/、input-manifest.json。

- [x] 原样归档三份输入，校验 ZIP 内文件字节与 SHA256。
- [x] 将 10/40/25/10/15 分项映射到材料和验收标准，标出正式成绩缺口。

### Task 2: UCloud 复现和展示草稿

**Files:** submission/REPRODUCE.md、BENCHMARK.md、slides.html、model-manifest.json、evidence/ucloud-inventory.json。

- [x] 只读采集 UCloud 硬件与工具环境，记录时间和采集命令。
- [x] 固定现有模型/引擎来源，写出迁移和官方 runner 命令、3 分钟 Demo 步骤。
- [x] 生成可离线打开的 8 页 HTML 答辩草稿；未完成成绩留明确文字标记。

### Task 3: 验证与 main 同步

**Files:** README.md、submission/VALIDATION.md；保留既有 PD 补测原文与数据。

- [x] 检查相对链接、归档哈希、runner dry-run、HTML 页数；执行全量 CPU 测试。
- [x] 审查将要同步的文件，不输出秘密值；提交材料及既有补测记录。
- [x] 将所有研究分支的已提交内容快进到 main，验证合入结果并正常推送 origin/main。
- [x] 导出固定提交的完整源码 tar.gz、SHA256 和 Git bundle；记录正式提交前仍需完成的 TODO。

## 执行约定

本轮是材料整理与既有开发内容集成，按用户授权在当前工作区直接执行；不引入新的推理机制或评测工具。用户要求的完整后续 TODO 维护在 submission/TODO.md，本计划只跟踪本轮整理。
