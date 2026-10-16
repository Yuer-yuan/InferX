# 部署、评测与复验说明

**状态：已有 vlm-server 部署可追溯；UCloud 的最终启动命令和干净环境复验尚待完成。** 下列命令中标为待填的项必须用实测值替换，不能据此宣称已完成主办方二次复验。

## 环境和版本

比赛环境为 UCloud 单张 48GB RTX 4090，采集结果见 [盘点](evidence/ucloud-inventory.json)。原实验环境为 vlm-server 单卡 24GB，不能将其预编译二进制、驱动条件、NUMA 绑核参数直接当成 UCloud 环境。

1. 固定 `main` 的最终提交 SHA，保留完整仓库源码快照，核对 `model-manifest.json`。
2. 在 UCloud 明确可写、可持久化的数据目录；核对挂载、空闲容量及实例回收策略后设置以下变量。不要直接 source 旧 `scripts/remote-env.sh`，其 HDD 路径检查是针对 vlm-server 的。

```bash
# 必须改为已经核实的 UCloud 数据目录；以下不是一个默认可用路径。
export INFERX_SUBMISSION_DATA_ROOT=/ABSOLUTE/CONFIRMED/DATA/PATH/inferx
: "${INFERX_SUBMISSION_DATA_ROOT:?}"
mkdir -p "$INFERX_SUBMISSION_DATA_ROOT"/{models,cache,tmp,eval-envs,runs,build}
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$INFERX_SUBMISSION_DATA_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export UV_CACHE_DIR="$INFERX_SUBMISSION_DATA_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$INFERX_SUBMISSION_DATA_ROOT/cache/uv-python"
export PIP_CACHE_DIR="$INFERX_SUBMISSION_DATA_ROOT/cache/pip"
export XDG_CACHE_HOME="$INFERX_SUBMISSION_DATA_ROOT/cache/xdg"
export TMPDIR="$INFERX_SUBMISSION_DATA_ROOT/tmp"
export TIKTOKEN_CACHE_DIR="$INFERX_SUBMISSION_DATA_ROOT/cache/tiktoken"
export MODELSCOPE_CACHE="$INFERX_SUBMISSION_DATA_ROOT/cache/modelscope"
export TORCH_HOME="$INFERX_SUBMISSION_DATA_ROOT/cache/torch"
export TORCH_EXTENSIONS_DIR="$INFERX_SUBMISSION_DATA_ROOT/cache/torch-extensions"
export TRITON_CACHE_DIR="$INFERX_SUBMISSION_DATA_ROOT/cache/triton"
export CUDA_CACHE_PATH="$INFERX_SUBMISSION_DATA_ROOT/cache/cuda"
```

3. 用隔离环境构建引擎，保存完整依赖版本/锁文件和构建日志；Python 评测工具使用 3.12。NInfer 旧部署可参考 [固定版本报告](../docs/reports/2026-10-14-ninfer-mtp.md) 和 `scripts/prepare_ninfer_toolchain.py`、`prepare_ninfer_model.py`、`build_ninfer.sh`。这些脚本包含旧服务器绝对路径，移植前必须逐项适配和验证。原构建为 CUDA redistrib 12.8.1 / nvcc 12.8.93、GCC 13.3、CMake 3.31.6；UCloud 实际采用版本应重新记录。
4. 权重可按下面精确来源获取，不必将 18GB 权重重复放入提交包。路径和 SHA256 见模型清单；不得自动改为仓库 latest。其他候选仅在确定采用后下载。

```bash
curl -fL --retry 3 \
  'https://hf-mirror.com/neroued/Qwen3.8-27B-NInfer/resolve/3526913004b1cf552cb57b88d6a5c6f5e4a89a70/qwen3_8_27b.ninfer' \
  -o "$INFERX_SUBMISSION_DATA_ROOT/models/qwen3_8_27b.ninfer"
printf '%s  %s\n' \
  eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e \
  "$INFERX_SUBMISSION_DATA_ROOT/models/qwen3_8_27b.ninfer" | sha256sum -c -
```

NInfer 源码：`https://github.com/UDPSendToFailed/ninfer-4090`，commit `5c60b7c9b455231795c09da21a9fbb6aa53f08e5`，xgrammar commit `4546988d1cf51670a10eafb8795d37e4a2f0d833`。外部源码、模型许可和精确 acquisition log 一并存档；Q4/Q5 混合权重不能写成 FP8 参考模型。

## 最终服务配置记录（待 UCloud 部署填写）

提交最终版时补全以下内容：实际引擎/构建产物哈希、完整启动 argv、仅含非秘密项的环境变量、GPU UUID/TP=1、CPU 亲和性、模型路径及哈希、tokenizer 来源/commit、上下文、KV 格式、MTP 深度、prefill chunk、最大活动请求数、模板和思考/采样默认值、端口与 `/v1/models` 返回、依赖锁、启动/健康检查/停止命令。

原 Agent 配置为 256K、rk4v4-e8 KV、MTP7、prefill2048、单活动请求；这是旧服务器的可用配置，不是 UCloud 并发评测的预定最优答案。最终上下文须容纳 IFBench 提示词加 32768 输出预算。停止服务时只终止本次启动且身份核验匹配的进程，沿用项目现有 runtime 身份检查方式，不使用批量 `pkill`。

## 官方评测入口

本目录 `organizer/` 三个文件与根目录 eval.zip 字节相同。仅检查和 dry-run 可离线执行，不安装依赖、不创建评测结果、不请求模型：

```bash
bash submission/organizer/run.sh --mode check \
  --env-dir "$INFERX_SUBMISSION_DATA_ROOT/eval-envs"
# check 返回 2 仅表示环境尚未备齐，不是模型性能失败。
bash submission/organizer/run.sh \
  --base-url http://127.0.0.1:8088/v1 --model MODEL_ID_FROM_V1_MODELS \
  --env-dir "$INFERX_SUBMISSION_DATA_ROOT/eval-envs" --dry-run
```

正式严格模型 ISL 模式：先将下列变量填为最终服务一致的 tokenizer 标识及固定 commit。tokenizer 的路径位于评测机；单文件 NInfer 不自动提供匹配的 HF tokenizer，不能猜一个名称替代。builtin 模式是否获准以主办方确认结果为准。

```bash
: "${INFERX_MODEL_ID:?填入实际 /v1/models 的模型 ID}"
: "${INFERX_TOKENIZER_ID:?填入已验证匹配的 tokenizer HF 标识}"
: "${INFERX_TOKENIZER_REVISION:?填入固定 commit}"
bash submission/organizer/run.sh \
  --base-url http://127.0.0.1:8088/v1 --model "$INFERX_MODEL_ID" \
  --tokenizer "$INFERX_TOKENIZER_ID" \
  --tokenizer-revision "$INFERX_TOKENIZER_REVISION" \
  --env-dir "$INFERX_SUBMISSION_DATA_ROOT/eval-envs" \
  --mode all --perf-requests 100 --eval-batch-size 2 \
  --request-timeout 600 --seed 42 \
  --output-dir "$INFERX_SUBMISSION_DATA_ROOT/runs/official-FROZEN_RUN_ID"
```

输出目录必须是全新路径。安装依赖时按原版工具的交互确认流程准备隔离环境；不要在后台评测中跳过确认，不修改全局 Python。实际 API key 只由环境变量提供，不填入文档/命令行/仓库；无鉴权时保持未设置。运行结束后核对退出码、summary 和逐请求原始记录，再分享经清理的结果目录。

性能与精度不重叠运行；服务端配套保存 1 秒 GPU 采样和服务日志。具体协议、分数与异常判定见 [PROTOCOL.md](PROTOCOL.md)。最终还需归档客户端/服务两端环境、数据 SHA256 和原始产物，而非只复制报告图片。

## 现场 Demo

在 UCloud 上事先彩排并准备本地备用录屏：

- 0:00–0:30：展示当前 GPU 型号/数量、服务模型 ID、冻结版本和关键配置。
- 0:30–1:30：真实发送一个短代码或逻辑请求，展示流式输出、usage、端到端时间；代码案例执行固定测试。
- 1:30–2:30：打开冻结的官方四档性能、IFBench 与一个失败案例，定位其原始请求/配置。完整正式评测不塞进 3 分钟。
- 2:30–3:00：说明 MTP/并发取舍、剩余问题和复验入口。可展示 Agent 连接，但不将客户端速度当成官方吞吐。

模型断连时先核对进程/日志/配置与冻结版本；无法恢复时切换已录制备用视频，并主动说明原因与版本。备用视频不能代替所有可现场运行的核心功能。

## 固定版本导出

完成 P0 后以最终 main 提交生成完整源码和必要的原始结果包。以下操作只导出**已提交文件**，运行前确保正式材料/原始结果已纳入该提交，并先扫描秘密、模型许可与第三方数据授权。

```bash
git status --short
git rev-parse main
mkdir -p artifacts
git archive --format=tar.gz --prefix=InferX/ \
  -o artifacts/InferX-source.tar.gz main
git bundle create artifacts/InferX-main.bundle main
sha256sum artifacts/InferX-source.tar.gz artifacts/InferX-main.bundle \
  > artifacts/SHA256SUMS
git bundle verify artifacts/InferX-main.bundle
```

源码包供解压复现，bundle 保存提交历史。冻结提交 ID 与归档校验和一起交付，`artifacts/` 不回填 Git 避免自引用。当前整理版的导出只证明已有开发内容固定，**正式成绩、最终 UCloud 环境和视频待补齐后必须重新导出最终版**。
