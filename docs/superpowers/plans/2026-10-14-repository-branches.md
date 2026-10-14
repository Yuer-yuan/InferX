# Repository Branch Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在真实仓库执行和受控多会话竞争下，验证候选并发浪费及普通反馈/公平基线能否解决它。

**Architecture:** 沿用 llama.cpp 流式客户端与候选调度器，新增 Landlock 仓库验证 worker、确定性数据准备、多会话 admission 层及独立分析器。运行环境和数据全部位于远端 HDD，保留上一轮单卡部署及结果。

**Tech Stack:** Python 3.12、asyncio、llama.cpp v0.6.0、CUDA 12.6、Linux Landlock ABI ≥3、libseccomp、pytest/unittest、Matplotlib。

**Spec:** `docs/superpowers/specs/2026-10-14-repository-branches-design.md`

## Global Constraints

- GPU 1 单卡，8 slots × 8192 token；GPU 0 旧服务保留；不得修改系统 CUDA/驱动/其他环境。
- HDD 根目录 `/home/guokc/mypro/gnn-mount-sda1/InferX`；模型/数据经 hf-mirror；大型文件不进入 Git。
- ExecRepoBench revision `fa61028ce495c9ceff58398b8a7c47b5ae9f5276`，原始 JSONL 行号为 ID，seed `2026101402`，4 仓库各 2 dev / 4 eval。
- 只选正确参考通过、语法有效的 pass 消融使实际测试失败的 block 题；筛选先于生成，所有排除留痕。
- 单题最多 4 候选，arrival 起 120 秒期限，max_tokens 1024，prompt ≤6000 token，thinking off，采样与 spec 一致。
- 全局最多 8 活跃生成，验证 CPU 36/37，引擎/客户端 CPU 28–35；候选验证 30 秒、2 GiB、8 MiB 文件、256 KiB 输出上限。
- 隔离失败是基础设施错误，不计作模型失败，不降级裸执行。成功必须实际运行 canonical 对应数量的测试。
- 8 dev × 3 fixed widths × 2 rounds = 48；eval 4 episodes × 2 arrival patterns × 4 policies × 2 rounds × 4 tasks = 256。仅开发集选择固定基线。
- 原始记录完整保留；观察 token 为下界；任务失败按 120 秒计入解决时间；episode 最后确认引擎空闲。

## Review Focus

- 同时到达、等待 admission、成功取消、deadline 和外部中断交错时，无许可泄漏、孤儿任务或跨会话取消。
- 路径穿越、symlink、原仓库同名已安装包及旧字节码不能导致验证跑错代码或读写范围扩大。
- 没有测试、少收集测试、sandbox 安装失败不能被计为通过。
- JSON 补全保留空白，提示不含答案/测试；任务覆盖筛选不读取任何模型表现。
- 不完整 episode、重复轮次、dev/eval 重叠、失败与取消成本不能被分析器静默丢弃。

---

### Task 1: Repository fixtures and restricted verification

**Files:** Create `inferx/repository.py`, `inferx/repo_restrictions.py`, `inferx/repo_worker.py`, `inferx/repo_verifier.py`, `scripts/prepare_repositories.py`, `scripts/calibrate_repositories.py`, `requirements-repository.txt`, `tests/test_repository.py`, `tests/test_repo_verifier.py`.

**Interfaces:**
- `parse_completion(text: str) -> str`: JSON object containing string code, optional outer JSON fence, preserves exact code whitespace; invalid format raises ValueError.
- `safe_relative(repo: str, filename: str) -> Path`, `extract_repositories(archive: Path, destination: Path, repositories: list[str]) -> None`: dataset leading slash accepted only for matching repo prefix; reject traversal/symlinks.
- `restrict_repository(read_paths: list[Path], scratch: Path) -> None`: fail closed, Landlock plus syscall allowlist; only scratch writable.
- `verify_repo(code: str, task: dict, *, fixture_root: Path, python: str, work_root: Path, cpu: int, timeout: float = 30) -> dict`: fresh private copy, patch only specified source file, subprocess result with status/test_elapsed/elapsed/counts, supports cancellation and InfrastructureError.
- Prepared task dict: task_id/repo_name/file_name/prefix_code/suffix_code/middle_code/split/test_mode/expected_counts; manifest includes source hashes, exclusions, split IDs, dependency freeze.

- [ ] Write tests for path traversal/symlink rejection, whitespace-preserving JSON, valid pass ablation, reference reconstruction and answer-free context filtering. Run `python3 -m unittest discover -s tests -p test_repository.py -v`; expected missing-module failure.
- [ ] Implement preparation helpers and manifest generation; rerun focused tests; expected all pass.
- [ ] Write isolated fixture tests: correct/wrong code, zero or reduced tests, outside-file read/write, network/fork/exec denial, timeout/cancellation reaping and missing worker failure. Run focused verifier tests; expected missing-module failure.
- [ ] Implement Landlock worker and async parent using bounded pipe output, subprocess-group cleanup, read-only repo and writable scratch. Fixture uses unittest so local checks need no pytest. Run tests; expected all pass.
- [ ] Create HDD-only venv with pinned pytest dependencies, safely extract white-listed upstream repos, freeze packages. Run canonical full suites and deterministic coverage filtering; expected 24 eligible tasks or explicit disclosed shortfall. Do not alter test assertions to force compatibility.
- [ ] Save positive/ablation timings and full exclusions, verify selected IDs and source reconstruction. Commit implementation; run full tests remotely; expected all pass.

### Task 2: Live multi-session candidate admission

**Files:** Modify `inferx/scheduler.py` (optional feedback gating, default unchanged); create `inferx/episodes.py`, `scripts/start_repo_server.sh`, `scripts/run_repository_pilot.py`, `tests/test_episodes.py`; extend `tests/test_scheduler.py`.

**Interfaces:**
- `run_task(..., *, await_feedback: bool = False) -> dict`: existing result schema retained; gated mode counts pending verification against the single-candidate limit.
- `run_episode(tasks: list[dict], generate, verify_candidate, *, policy: str, arrivals: list[float], round_id: int, slots: int = 8, budget: int = 4, deadline: float = 120) -> dict`: independent per-task lifecycles, shared FIFO generation permits, task records include scheduled/actual arrival and admission timing; feedback-fair uses width 1 and feedback gating. FIFO with at most one outstanding candidate per task supplies the round-robin admission behavior.
- Runner writes config, runtime source hashes, prompts metadata, warmup, episode JSONL and COMPLETE only after all predeclared episodes; raw outputs stay remote.

- [ ] Write tests for feedback gating, overlap in fixed mode, shared capacity, deterministic arrival handling, one task success without canceling its neighbor, waiting cancellation without permit leakage, deadline including admission and infrastructure error cleanup. Run focused tests; expected missing feature failures.
- [ ] Implement optional gating and shared episode wrapper, recording queued vs admitted generation separately. Preserve old pilot behavior by default. Rerun focused plus prior scheduler tests; expected all pass.
- [ ] Implement model prompt construction with line-boundary truncation/tokenizer check, exact JSON parsing, pipeline adapter, deterministic episode seeds/order, fixed CPU bindings and metadata. No per-task global wait_idle during concurrent execution.
- [ ] Start owned GPU 1 service with 65536 total context; verify model load/memory, prompt bounds, eight active streams cancel to idle, and one canonical repository integration. Expected no OOM and full cleanup.
- [ ] Run development-only smoke before freezing config. Commit; run complete remote suite; expected all pass. Any implementation/config fix before main sweep is documented and smoke repeated, not silently mixed into full results.

### Task 3: Complete measurements and scientific decision

**Files:** Create `scripts/analyze_repository.py`, `scripts/export_repository_results.py`, `scripts/plot_repository.py`, `tests/test_repository_analysis.py`, `results/repository-pilot/`, `docs/reports/2026-10-14-repository-pilot.md`; update README.

**Interfaces:** analyzer consumes episode records, produces dev-selected fixed width, per-mode/policy/round task success/capped time/queue/feedback/token/makespan metrics and paired episode-cluster bootstrap intervals; rejects missing/duplicate grid or split overlap.

- [ ] Write hand-computed analysis tests for development-only selection, failures and pre-admission deadlines, cancellation lower bounds, correlated-episode resampling, incomplete/duplicate grid, split overlap and stated screening gates. Run focused tests; expected missing-module failure.
- [ ] Implement analyzer/export/plots. Run tests; expected all pass.
- [ ] Execute 48 development trials and 64 evaluation episodes (256 task outcomes) with frozen sampling and randomized ordering, two seeds on GPU 1; collect 1-second telemetry. Expected complete coverage; infrastructure failure requires explicit excluded batch and rerun, not removal of a difficult task.
- [ ] Export lightweight evidence, reproduce summaries locally, inspect charts and failed cases, compare gate exactly with spec. Report negative/mixed/insufficient evidence honestly; no engine mechanism unless justified by a separate follow-up plan.
- [ ] Write report with real repository/sandbox limits, ordinary baseline tradeoffs, latest related work, precise reproduction and all deviations. Commit and run complete suite.
- [ ] Independent whole-branch review required by executing-plans. Fix Critical/Important with RED→GREEN and full suite; preserve review/decision record. Keep branch and existing GPU 0 deployment; free owned GPU 1 experiment process at completion.

## Execution choice

主 agent inline 执行，沿用用户已授权方式。最终保留本地研究分支，不发布、不合并；新建 worktree 与流程批准不重复询问。设计检查已经确认没有需要用户补充的重要问题。
