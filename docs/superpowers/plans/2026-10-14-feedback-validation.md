# Feedback Validation Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 vlm-server 单张 4090 上完成可复现瓶颈筛查，决定代码生成—验证反馈调度假设是否值得继续。

**Architecture:** HDD 上源码编译的 llama.cpp 服务运行单卡异步生成；受限 CPU 子进程验证候选，调度器成功即取消。JSONL 原始事件保留在 HDD，轻量汇总和结论回存仓库。

**Tech Stack:** Python 3.12 客户端、llama.cpp v0.6.0、CUDA 12.6、Hugging Face mirror、Linux seccomp、stdlib unittest；Qwen3.8-27B UD-Q4_K_M。

**Spec:** `docs/superpowers/specs/2026-10-14-feedback-validation-design.md`

## Global Constraints

- 每个实验单张 RTX 4090；不使用 TP=2。
- 全部远端下载、环境、缓存和临时文件位于 `/home/guokc/mypro/gnn-mount-sda1/InferX/`，HF_ENDPOINT=https://hf-mirror.com。
- 不修改他人环境；不记录凭据。
- 16 题开发、48 题评估，seed=20261014 划分，2 轮采样；并发 1/2/4/8，候选预算 8，期限 120 秒。
- 候选代码在资源受限且安装 seccomp 成功的子进程中执行；错误、超时和取消均保留记录。
- 先证伪；低于 5% 验证占比且无长尾时，不以反压机制作为本轮贡献。

## Review Focus

- 正确候选完成后仍运行的生成与验证必须被取消，不能污染下一题。
- 截止时间必须覆盖等待、生成、验证，不能丢弃未解决题。
- 沙箱初始化失败不能被计作模型失败或通过，更不能裸执行。
- 取消期间的 token 与时间不能被统计成零成本。
- 开发集选择和评估集指标必须隔离，配置和版本必须可追溯。

## File Map

- `scripts/remote-env.sh`：HDD 环境与缓存；`scripts/download.py`：固定数据版本及 manifest；`scripts/prepare_27b.py`：固定模型下载及哈希验证。
- `inferx/sandbox_worker.py`、`inferx/verifier.py`：seccomp 子进程与异步有界验证。
- `inferx/scheduler.py`：候选并发、预算、deadline、取消、事件。
- `scripts/run_pilot.py`：llama.cpp HTTP 适配器与实验编排；`scripts/analyze.py`：汇总和分集选择。
- `tests/test_verifier.py`、`tests/test_scheduler.py`、`tests/test_analysis.py`：行为测试。
- `docs/reports/2026-10-14-pilot.md`、`results/pilot/`：实际结果、版本、复现方法。

### Task 1: Environment and restricted verifier

**Files:** Create `scripts/remote-env.sh`, `scripts/download.py`, `inferx/__init__.py`, `inferx/sandbox_worker.py`, `inferx/verifier.py`, `tests/test_verifier.py`.

**Interfaces:** Produces `async verify(code: str, tests: str, entry_point: str, timeout: float, cpu: int | None) -> dict` with status, elapsed, test_elapsed, error. Remote env exposes INFERX_ROOT and isolated cache variables. Download writes model directory, tasks.jsonl and manifest.json.

- [ ] Write verifier tests: correct/incorrect function, infinite loop deadline, blocked file/network access, cancellation cleanup, sandbox failure propagates as infrastructure error.
- [ ] Run `python3 -m unittest discover -s tests -p test_verifier.py -v`; expected missing-feature failure before implementation.
- [ ] Implement seccomp allowlist after standard library preload, resource limits, capped stdout, process-group cleanup and explicit infrastructure error. Create HDD-only environment script and pinned download script.
- [ ] Run verifier tests locally and remotely; all pass. Validate canonical solutions on selected tasks to distinguish sandbox incompatibility from model failures.
- [ ] Build pinned llama.cpp v0.6.0 with CUDA 12.6 on HDD; use existing Python client environment read-only, with hf_transfer installed into the HDD project. Download pinned Qwen3.8-27B GGUF/data through mirror, record hashes, relevant package versions and hardware. Confirm model files and caches resolve under HDD.
- [ ] Commit Task 1.

### Task 2: Measured asynchronous baseline

**Files:** Create `inferx/scheduler.py`, `inferx/llama_client.py`, `scripts/run_pilot.py`, `scripts/start_server.sh`, `tests/test_scheduler.py`, `tests/test_llama_client.py`.

**Interfaces:** Consumes Task 1 verify; produces `async run_task(task: dict, generate, verify_candidate, width: int, budget: int, deadline: float, seed: int) -> dict`. Generate receives candidate id, seed and mutable event record, returns code; cancellation must abort underlying request. Per-task result includes candidate events and total/solution latency.

- [ ] Write tests for earliest verified success and loser cleanup, budget exhaustion, deadline during verification, generator failure, nonzero tokens from cancelled generation.
- [ ] Run scheduler tests; expected missing-feature failure.
- [ ] Implement bounded asynchronous workers, shared verification semaphore, monotonic event timestamps and cancellation cleanup; use native llama.cpp SSE streaming, close HTTP response on cancellation, then confirm all server slots idle before the next trial. Persist one result per task immediately.
- [ ] Run `python3 -m unittest discover -s tests -v`; all pass.
- [ ] Run two-task single-GPU smoke; expected valid results, nonzero generated tokens, verifier timing, no residual requests after cancellation.
- [ ] Commit Task 2.

### Task 3: Execute, analyze and decide

**Files:** Create `scripts/analyze.py`, `tests/test_analysis.py`, `docs/reports/2026-10-14-pilot.md`, `results/pilot/`.

**Interfaces:** Consumes Task 2 JSONL; produces JSON/CSV summaries with dev-selected width and evaluation metrics; separates censored failures, inference errors and model failures.

- [ ] Write hand-computed tests for development-only selection, deadline penalty for failures, cancelled-token accounting and verification time decomposition; run and observe expected missing-feature failure.
- [ ] Implement analyzer and run full unittest suite; all pass.
- [ ] Execute baseline sweep on 16 development and 48 evaluation tasks, widths 1/2/4/8, two sampling rounds. Record telemetry; randomize order and exclude warmup.
- [ ] If measurement meets pressure criterion, implement and test minimal backpressure policy then compare; otherwise record the negative result and evidence required for a harder workload.
- [ ] Copy lightweight raw per-task metrics (without large generated text), summary, environment manifest and hashes to results; write Chinese report with limits and exact reproduction commands. Verify result counts and no task overlap.
- [ ] Commit Task 3, run whole-branch independent review and fix important findings with targeted regression tests; rerun full suite.

## Execution choice

用户已要求直接继续，本轮由主 agent 在当前目录的 `research/feedback-validation-pilot` 分支执行，保持计划、技能和结果在用户指定 work-dir 可见；不再为流程确认暂停。实验本身是可丢弃的选题筛查。远端路径已由用户授权，原始大文件不纳入 Git。最终保留本地分支，不自动发布或合并。

## 2026-10-14 model/backend amendment

用户选择 Qwen3.8 的较大 dense 模型。最终采用 Qwen3.8-27B UD-Q4_K_M 单卡全层 offload；16.46 GB GGUF、8 slots、总上下文 16384，具体模型 revision/哈希见 spec。关闭 thinking，采样和题目分割不变。源代码为 llama.cpp v0.6.0/CUDA 12.6；旧模型与 vLLM 安装尝试不产生主实验数据。新增 SSE 取消与完整结果网格检查，统计计入服务端取消清理时间。远端有现成可用的 Python 客户端依赖，读取使用它，不修改已有环境；只将 hf_transfer 下载器安装到项目 HDD 目录。
