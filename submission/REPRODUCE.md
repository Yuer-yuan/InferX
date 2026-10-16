# 部署、评测与复验说明

公共工具按引擎和功能命名，适用于类似 Linux/CUDA 机器。`ucloud1/2` 仅为本轮节点名称；第一台执行测量，第二台只部署和功能核验。结果按节点保存，不能把跨机差异当成优化收益。

本轮已验证 Ubuntu 22.04、单张 49,140 MiB RTX 4090、CUDA 13.2.51、GCC 13.4、Python 3.12.15。两台 CPU 不同，见 [原始盘点](../results/ninfer-performance/inventory/)。24GB 卡需要重新调整 KV/并发预算，本轮配置不能直接当作它的测量结论。正式性能和 IFBench 完成状态见 [BENCHMARK.md](BENCHMARK.md)。

## 数据目录与隔离环境

先选择实际可写的数据目录，设置源码目录和 CUDA。当前节点使用 `/workspace/InferX`，它是容器 overlay，持久性未确认；关键配置和结果必须回传。下载、构建、缓存、临时文件均在指定前缀内。不要 source 带旧服务器 HDD 路径的 `remote-env.sh`。

```bash
# 按机器实际情况修改；这不是已确认持久化的数据盘承诺。
export INFERX_ROOT=/workspace/InferX
export INFERX_CODE_ROOT="$INFERX_ROOT/code"
export INFERX_CUDA_ROOT=/usr/local/cuda-13.2
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p "$INFERX_ROOT"/{config,tools,downloads,cache,tmp,source,models,runs,logs}
cp "$INFERX_CODE_ROOT/scripts/ninfer-env.sh" "$INFERX_ROOT/config/"
```

先提供可用的 `micromamba` 二进制，设置 `INFERX_MAMBA` 指向它；本轮使用 2.9.0，下载包 SHA256 为 `8761c382127e6363bd9e0a2451aa3ef90d071a79133f736e2f759a3bf13040dd`。`bootstrap_ninfer.sh` 不安装驱动，也不自动下载 micromamba 或 CUDA。系统需具备 Bash、可用网络、Linux x86_64 及适配当前 CUDA 的驱动。

```bash
export INFERX_MAMBA="$INFERX_ROOT/tools/micromamba"
# 可选镜像；不设置时分别使用 conda-forge、pypi.org。
export INFERX_CONDA_CHANNEL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
export INFERX_PYPI_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
# 仅在这些公共镜像可直连时设置，避免继承代理导致限速。
export INFERX_DIRECT_MIRRORS=1
bash "$INFERX_CODE_ROOT/scripts/bootstrap_ninfer.sh" "$INFERX_ROOT" \
  > "$INFERX_ROOT/logs/bootstrap.log" 2>&1
source "$INFERX_ROOT/config/ninfer-env.sh"
```

脚本保存 `config/build-conda-explicit.txt`、`build-pip-freeze.txt`。首次安装的宽版本约束用于发现可用依赖；精确复验应重放本轮保存的 explicit lock 和 pip freeze，不能仅凭脚本中宽约束声称二进制一致。构建必须有 FFmpeg≥6、libcurl≥7.85 及 zlib 的开发元数据；系统旧版 FFmpeg/curl 不能替代隔离依赖。

## 固定源码、权重和构建

```bash
git clone --recursive https://github.com/UDPSendToFailed/ninfer-4090 \
  "$INFERX_ROOT/source/ninfer-4090"
git -C "$INFERX_ROOT/source/ninfer-4090" checkout --detach \
  5c60b7c9b455231795c09da21a9fbb6aa53f08e5
git -C "$INFERX_ROOT/source/ninfer-4090" submodule update --init --recursive
git clone --recursive https://github.com/mlc-ai/xgrammar.git \
  "$INFERX_ROOT/source/xgrammar_source-src"
git -C "$INFERX_ROOT/source/xgrammar_source-src" checkout --detach \
  4546988d1cf51670a10eafb8795d37e4a2f0d833
git -C "$INFERX_ROOT/source/xgrammar_source-src" submodule update --init --recursive
git -C "$INFERX_ROOT/source/ninfer-4090" apply --check \
  "$INFERX_CODE_ROOT/patches/ninfer-v2-completions.patch"
git -C "$INFERX_ROOT/source/ninfer-4090" apply \
  "$INFERX_CODE_ROOT/patches/ninfer-v2-completions.patch"
export INFERX_CUDA_ARCH=89 INFERX_BUILD_JOBS=4
bash "$INFERX_CODE_ROOT/scripts/build_ninfer_portable.sh" "$INFERX_ROOT" \
  > "$INFERX_ROOT/logs/build.log" 2>&1
```

补丁增加真实 raw completions/token-ID/ignore_eos/usage 接口；raw 请求关闭前缀复用以保证缓存隔离，chat 仍保留原有缓存机制。它不修改量化、GPU 内核、MTP 接受规则或调度算法。补丁 SHA256、模型与 tokenizer 清单见 [model-manifest.json](model-manifest.json)。

```bash
hf download neroued/Qwen3.8-27B-NInfer qwen3_8_27b.ninfer \
  --revision 3526913004b1cf552cb57b88d6a5c6f5e4a89a70 \
  --local-dir "$INFERX_ROOT/models/qwen38-ninfer"
printf '%s  %s\n' \
  eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e \
  "$INFERX_ROOT/models/qwen38-ninfer/qwen3_8_27b.ninfer" | sha256sum -c -
bash "$INFERX_CODE_ROOT/scripts/prepare_judge_env.sh" "$INFERX_ROOT" \
  > "$INFERX_ROOT/logs/prepare-judge.log" 2>&1
```

NInfer 权重是固定 v2 的 Q4/Q5 混合格式，不能替换成 latest/v3 或写成 FP8。下载限流可以从已核验节点复制同一个公开权重，接收后仍需完整 SHA256；保存失败下载和复制来源记录。本轮两台模型哈希一致，构建产物也逐字节一致；跨 OS、CUDA 或 GPU 架构的机器应重新构建。

评测环境为 AIPerf 0.13.0 / EvalScope[ifbench] 1.13.0，Python 3.12，分别隔离安装。本轮 API 客户端没有安装 torch；安装脚本为可能需要它的依赖指定 CPU 后端，不占推理 GPU。若 NLTK 下载因网络策略失败，可从官方 nltk_data 下载 `punkt_tab`、`stopwords`、`averaged_perceptron_tagger_eng` ZIP，按 [资源清单](../results/ninfer-performance/preparation/nltk-resources.json) 核验来源/哈希后分别解压到 `eval-envs/nltk_data/{tokenizers,corpora,taggers}`。脚本复用已存在的资源，最后必须通过原版 `--mode check`，不修改评测器或绕过资源校验。

匹配 tokenizer 直接从已核验权重提取，避免猜测外部 HF tokenizer 名称。v2 容器格式依据固定源码的 `tools/artifact/container.py`：16字节前缀、JSON对象目录、4096对齐的数据起点、资源相对偏移。上游完整读取器依赖 torch；这里仅用标准库读取六个资源，并逐个核对已冻结的 SHA256。目标目录必须不存在。

```bash
python - <<'PY'
import hashlib, json, os, struct
from pathlib import Path
r = Path(os.environ['INFERX_ROOT'])
code = Path(os.environ['INFERX_CODE_ROOT'])
model = r / 'models/qwen38-ninfer/qwen3_8_27b.ninfer'
expected = json.loads((code / 'submission/model-manifest.json').read_text())['final_tokenizer']['files']
out = r / 'models/tokenizer'
out.mkdir(exist_ok=False)
manifest = []
with model.open('rb') as f:
    magic, size = struct.unpack('<8sQ', f.read(16))
    assert magic == b'NINFER\x00\x02' and 0 < size < 32 * 1024 * 1024
    directory = json.loads(f.read(size))
    payload = ((16 + size + 4095) // 4096) * 4096
    for item in expected:
        obj = next(x for x in directory['objects'] if x['name'] == item['resource'])
        assert obj['kind'] == 'resource' and obj['encoding'] == 'raw-bytes-v1'
        assert obj['bytes'] == item['bytes'] and obj['offset'] >= 0
        assert payload + obj['offset'] + obj['bytes'] <= model.stat().st_size
        f.seek(payload + obj['offset'])
        data = f.read(obj['bytes'])
        assert hashlib.sha256(data).hexdigest() == item['sha256']
        name = Path(obj['name']).name
        (out / name).write_bytes(data)
        manifest.append({'file': name, 'bytes': len(data), 'sha256': item['sha256']})
(r / 'config/tokenizer-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
PY
```

## 可迁移配置与运行

公共 [pilot 模板](../configs/ninfer/qwen38-27b-pilot.template.json) 包含 MTP0/4/7、活动数4、64K上下文和 int8 KV，属于消融起点。模板按 JSON 字符串递归展开，路径含空格、引号或反斜杠也能正确编码；不要直接对整个 JSON 使用 `envsubst`。模板中的前缀和日志输出目录都必须展开。生成的配置放在 `config/`，不要提前创建 sweep 的结果目录。

```bash
export INFERX_RUN_DIR="$INFERX_ROOT/runs/pilot-NEW_RUN_ID"
python - <<'PY'
import json, os
from pathlib import Path
from string import Template
source = Path(os.environ['INFERX_CODE_ROOT']) / 'configs/ninfer/qwen38-27b-pilot.template.json'
def render(value):
    if isinstance(value, str): return Template(value).substitute(os.environ)
    if isinstance(value, list): return [render(x) for x in value]
    if isinstance(value, dict): return {k: render(v) for k, v in value.items()}
    return value
config = render(json.loads(source.read_text()))
target = Path(os.environ['INFERX_ROOT']) / 'config/pilot.json'
with target.open('x') as f: json.dump(config, f, indent=2)
PY
python "$INFERX_CODE_ROOT/scripts/run_ninfer_sweep.py" \
  --config "$INFERX_ROOT/config/pilot.json" --output-dir "$INFERX_RUN_DIR" \
  --runner "$INFERX_CODE_ROOT/submission/organizer/runner.py"
```

每次使用新 RUN_ID 和新配置文件名。driver 拒绝已有结果目录/占用端口，完整保存原生 AIPerf 记录、配置哈希和失败；它是小样本 PILOT，不替代正式结果。模型输入使用相同嵌入 tokenizer，精确8192/512；MTP 启用时同时使用 `--lm-head-draft`。配置只允许单卡，第二台不重复运行这个筛选。

直接部署时使用最终冻结配置中的 `command` argv 在前台启动，环境 `CUDA_VISIBLE_DEVICES=0`；终端 Ctrl-C 停止该进程。本轮后台服务保存于各节点 `config/active-service.json`，记录 PID、start_ticks、UID、完整 argv 和构建哈希；停止必须先用现有 `inferx.service_ownership` 核对身份，再通过 pidfd 发信号，不用批量 `pkill`。本轮 Conda Python 未提供 `os.pidfd_open`，控制操作使用提供该能力的系统 `/usr/bin/python3`；推理客户端仍使用隔离 Python3.12。

服务核验入口（实际请求，不是性能测量）：

```bash
python "$INFERX_CODE_ROOT/scripts/probe_ninfer_api.py" \
  --base-url http://127.0.0.1:8088/v1 --model qwen38-27b \
  --tokenizer "$INFERX_ROOT/models/tokenizer" \
  --prompt-tokens 8192 --max-tokens 512 \
  --output "$INFERX_ROOT/runs/api-probe.json"
```

从本地访问节点可使用 `ssh -L 8088:127.0.0.1:8088 MACHINE_ALIAS`，客户端地址保持 `http://127.0.0.1:8088/v1`。机器别名由本地 SSH 配置决定，不写进引擎/评测脚本。

## 原版正式评测

`organizer/` 三个文件保持 eval.zip 原样。先执行只读依赖检查；check 失败不是模型成绩。

```bash
bash "$INFERX_CODE_ROOT/submission/organizer/run.sh" --mode check \
  --env-dir "$INFERX_ROOT/eval-envs"
bash "$INFERX_CODE_ROOT/submission/organizer/run.sh" \
  --base-url http://127.0.0.1:8088/v1 --model qwen38-27b \
  --tokenizer "$INFERX_ROOT/models/tokenizer" \
  --env-dir "$INFERX_ROOT/eval-envs" \
  --mode all --perf-requests 100 --eval-batch-size 2 \
  --request-timeout 600 --seed 42 \
  --output-dir "$INFERX_ROOT/runs/official-NEW_RUN_ID"
```

本地 tokenizer 不传 `--tokenizer-revision`；来源为模型固定 revision 的嵌入资源，文件哈希另存。客户端和服务同机回环网络。四档 C1/C2/C4/C8、每档8预热+100正式、ISL8192/OSL512；IFBench前100、batch2、只显式覆盖max_tokens32768。64K上下文留足输出预算。性能与精度顺序执行，另开1秒 GPU 采样并保存日志和退出码；不在两机同时拼合成绩。SLO及 ZIP/参考 MD 差异见 [PROTOCOL.md](PROTOCOL.md)。

依赖已由独立安装脚本准备，不修改原版 runner 的交互安装规则。API key 仅使用环境变量，不写命令行或仓库。正式输出目录必须不存在；结果须核验逐请求长度、错误、原始分数和展开配置后回传。两台 GPU 相同、CPU不同，应以各自同机基线验证后续机制。

## 现场 Demo

在实际评测节点上事先彩排并准备本地备用录屏：

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

源码包供解压复现，bundle 保存提交历史。冻结提交 ID 与归档校验和一起交付，`artifacts/` 不回填 Git 避免自引用。当前整理版的导出只证明已有开发内容固定，**正式成绩、最终实测环境和视频待补齐后必须重新导出最终版**。
