# Qwen3.8-27B：单张 RTX 4090 的 NInfer / MTP 调优

已完成部署和分阶段调优。代码助手默认选择 **NInfer + MTP draft=7、INT8 KV、prefill chunk=2048、65,536 总上下文、单活动会话**，开启内存中的 prefix reuse。它是本轮实际测试范围内的代码生成优胜配置；说明文字任务中 draft=4 略快，保留为备选。没有修改 NInfer 推理源码、系统驱动、功率上限或全局 CUDA。

## 测量结果

每项只使用 `vlm-server` 的 GPU 1（24 GiB RTX 4090）。GPU 0 的原有服务保留。关闭 thinking，temperature=0、seed=42。下表是独立于开发集的四项提示，各重复三次的中位数；全部为禁用前缀复用的冷请求。

| 独立提示 | NInfer 无 MTP | NInfer MTP=4 | NInfer MTP=7 | MTP=7 端到端 |
|---|---:|---:|---:|---:|
| LRU 实现 | 50.98 | 182.95 | **240.98** | 208.13 |
| 拓扑排序实现 | 50.95 | 160.29 | **184.82** | 174.10 |
| 概率推导 | 50.87 | 170.81 | **212.82** | 207.95 |
| API 迁移说明 | 50.93 | **105.54** | 100.35 | 98.05 |

单位均为 token/s。生成速率使用客户端的 `(completion_tokens-1)/(wall_time-TTFT)`；首个 SSE chunk 可能包含多个 token，因此这是观测估计，不能当作逐 token 硬件计时。端到端速率为 `completion_tokens/wall_time`。没有按 chunk 数估算 token；缺失 usage、API 错误、断流均不计为成功测量。服务器内部计时另存，供交叉检查。

MTP=7 相对同一 NInfer 权重无 MTP 的代码加速为 **4.73× / 3.63×**。三次生成速率范围：LRU 239.71–241.04、拓扑排序 170.90–187.95、概率 200.26–215.49、说明文字 100.30–106.39 token/s；小样本范围不是置信区间。短提示 TTFT 中位数约 0.116–0.119 秒，无 MTP 约 0.109–0.111 秒。MTP=7 的草稿接收率中位数分别为83.5%、60.0%、70.2%、25.9%，说明文字低接收率与其更偏好短草稿的结果一致。

NInfer 的 0/4/7 三种配置共 **18 份独立代码输出**通过隔离执行检查；检查覆盖 LRU 淘汰、更新、零/负容量，拓扑排序重复边、缺省顶点、字典序、环及输入不变。概率性能测量使用 1,024 token 上限并发生截断；单独以 2,048 上限运行的三种配置均自然结束，结果为正确的 2/5、3/5。截断记录不计作质量通过。

这些是有限的功能门槛，不代表 HumanEval、SWE-bench 或完整 Benchmark 准确率。开发集排程题曾出现错误首结论。NInfer 在无 MTP、相同 greedy/seed 下也有逐字差异，因此不能把所有文本差异归因于 MTP，更不能宣称本部署已证明逐 token 等价。跨引擎量化格式不同，也不能把跨引擎提升全部归因于内核。

## 长提示、缓存和并发

| NInfer MTP=7 冷提示长度 | TTFT | 生成 token/s | 端到端 token/s |
|---|---:|---:|---:|
| 2,026 tokens | 0.900 s | 207.44 | 142.01 |
| 8,156 tokens | 3.450 s | 208.20 | 74.09 |
| 32,757 tokens | 14.732 s | 容量探针 | 容量探针 |
| 63,990 tokens | 31.472 s | 容量探针 | 容量探针 |

2K/8K 各重复三次；32K/近64K 各一次、输出 11 token，成功返回末尾标记，无 OOM。这只验证上下文容量，标记在末尾，不能当作长距离理解测试。65,536 是输入和输出的总上限，不能发送 65K 输入后再生成 8K。

开启 prefix reuse 后，相同 2K/8K 提示在每次完整预热后测三次，TTFT 中位数为 **0.040 / 0.057 秒**，明确报告复用 2,022 / 8,152 token。带前一轮模型回复的真实追加对话各测一次，第二轮 TTFT 为 **0.297 / 0.320 秒**，分别复用 2,022/2,486 和 8,152/8,594 token；此次路径是 restore_turn_checkpoint，需要重算前轮回复及新增内容。warm 收益与 MTP 生成加速分别报告，未启用磁盘 prompt cache。

下表为三个开发提示各一波同时提交，**总输出 token 数 / 整波完成时间**；不能把单流 decode 速率相加替代总吞吐。这不是完整的排队或负载稳定性 Benchmark。

| NInfer MTP=7 | 代码总吞吐 | 推理总吞吐 | 说明总吞吐 | 代码单流生成中位数 |
|---|---:|---:|---:|---:|
| 并发 2 | 240.55 | 215.39 | 142.46 | 128.86 |
| 并发 4 | 264.84 | 245.95 | 166.67 | 71.67 |

四路的总吞吐有所增加，单流响应速度明显下降，因此代码助手默认保持一个活动请求，其余请求排队；排队超时设为 120 秒。多用户吞吐目标需要重新选择并发配置。KV 总容量仍为 65,536，不能理解成每个并发请求各自获得独立 64K KV。

MTP=7 在最终试验后采样为 **20,421 MiB**；MTP=4 BF16 KV 为 22,339 MiB，INT8 为 20,413 MiB。这些是 `nvidia-smi` 采样，不是峰值监测。BF16 没有稳定收益，INT8 留出的余量更大。HDD 冷启动读权重约 59–63 秒：此分支在 Linux 上用 O_DIRECT，重复启动也不能假定由文件页缓存提速。

## 扫描范围与对照

NInfer 测试了 draft=0/2/4/6/7/8、INT8/BF16 KV、prefill chunk=512/1024/2048。分阶段选择，未穷举全部组合。draft=8 读完权重后在 CUDA Graph 初始化时失败：`cudaErrorGraphExecUpdateFailure (update result 2)`；不赋予它速度结果。draft=7 在短代码、独立代码、长输入、并发与实际 API 探针通过后才选作默认。

原有 GGUF 的 llama.cpp 对照也重新以单卡单槽运行；最终为 MTP=4、fp16 KV、ubatch=512、64K。独立提示同样各三次：

| 独立提示 | llama 无 MTP | llama MTP=4 | llama MTP=4 端到端 |
|---|---:|---:|---:|
| LRU 实现 | 49.26 | 138.72 | 111.87 |
| 拓扑排序实现 | 49.26 | 132.17 | 116.64 |
| 概率推导 | 49.21 | 130.32 | 125.79 |
| API 迁移说明 | 49.24 | 83.26 | 78.72 |

llama 对照 12 份独立代码输出全部通过相同执行检查。其 MTP=4 的冷 8K TTFT 约 3.34 秒，warm 约 0.189 秒、复用 8,152 token；近64K TTFT 29.53 秒。NInfer 主要优势在生成阶段，冷长提示预填充并未全面超过 llama。llama 开发阶段还测试 draft=2/6/8、q8 KV、ubatch1024/2048、draft8+pmin0.5/0.8，未超过所选综合配置。

## 服务使用

服务绑定远端回环地址 `http://127.0.0.1:18081`，模型名 **`qwen38-27b`**；支持 `/v1/chat/completions` 和 `/v1/messages`。未配置开机自启。日常启动脚本：

```bash
# 在 vlm-server 上；已有服务运行时无需重复启动
source /home/guokc/mypro/gnn-mount-sda1/InferX/env.sh
bash "$INFERX_ROOT/code-tuning/scripts/start_best_server.sh"
```

从本机使用先建立隧道：

```bash
ssh -N -L 18081:127.0.0.1:18081 vlm-server
```

然后请求：

```bash
curl http://127.0.0.1:18081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-27b","messages":[{"role":"user","content":"用 Python 实现一个 LRUCache"}],"max_tokens":1024,"temperature":0,"stream":true}'
```

默认输出上限 8,192，但客户端显式预算优先、总上下文限制仍生效。启动时关闭 thinking；不要发送 llama 专属 `chat_template_kwargs`，该 NInfer 版本会返回 HTTP400。原有 GPU 0 服务仍在 18080。

Anthropic 工具协议已实际测试：强制调用 `add_numbers(a=2,b=3)`，再发送匹配 `tool_use_id` 的结果 5，模型自然结束并只回答 5；两次 HTTP 请求合计 **0.494 秒**。这验证了本例的 `/v1/messages` 工具往返，未测试完整 Claude Code 会话、复杂多工具选择或流式工具参数。

备选参数由通用启动器显式提供：

```bash
# 均要求先安全停止当前 GPU 1 服务；启动器会拒绝占用中的卡
bash "$INFERX_ROOT/code-tuning/scripts/start_tuned_server.sh" ninfer 4 int8 2048 65536 1 warm
# llama 回退，保留已验证的 integrated MTP
bash "$INFERX_ROOT/code-tuning/scripts/start_tuned_server.sh" llama 4 f16 512 65536 1 warm
```

停止时使用 `tuning/best-service.json` 记录的 PID，先核对 `/proc/PID/stat` 的 start ticks、`/proc/PID/exe` 指向本目录的 `ninfer-serve`、进程归属及 GPU 1；一致后才发送 SIGTERM。PID 已消失或身份不匹配时不要 kill，更不能 `pkill llama` 或按 GPU 杀进程。不要动 GPU 0 的 PID 2504785。服务元数据与实际命令行归档在 `results/tuning/best-service.json`，PID 只反映本次部署。

## 构建、来源与复现

使用 [NInfer SM89 分支的固定提交](https://github.com/UDPSendToFailed/ninfer-4090/tree/5c60b7c9b455231795c09da21a9fbb6aa53f08e5)，源码没有修改。模型为 [兼容 v2 版本](https://huggingface.co/neroued/Qwen3.8-27B-NInfer/tree/3526913004b1cf552cb57b88d6a5c6f5e4a89a70)，18,210,531,328 bytes，SHA256 `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e` 已验证。不能用该仓库最新 main 的 v3 容器直接替换。NInfer 权重为混合 groupwise Q4/Q5 等格式，GGUF 为 UD-Q4_K_M；精确版本和哈希见 `results/tuning/provenance.json`。

服务器是双 Xeon Gold 6348、约503GiB RAM；推理绑 CPU 28–35。GPU 使用原有驱动 565.57.01，隔离 CUDA 12.8.1 redistrib（nvcc12.8.93）、GCC13.3、CMake3.31.6。运行日志确认 compile/runtime12.8、driver API12.7、SM89 和 GPU1 UUID。CUDA12.x minor compatibility 有功能限制，实际运行验证是必要条件。[NVIDIA 兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

下列步骤用于在这台服务器的 HDD 前缀重建；不是跨发行版通用安装器。`env.sh` 由仓库 `scripts/remote-env.sh` 部署，校验 HDD 挂载并把模型、下载、编译、临时文件和所有缓存指向 HDD。已有目录勿重复 clone；先核对固定提交和干净状态。

```bash
source /home/guokc/mypro/gnn-mount-sda1/InferX/env.sh
cd "$INFERX_ROOT"
git clone --branch feat/rtx-4090-sm89-native \
  https://github.com/UDPSendToFailed/ninfer-4090 source/ninfer-4090
git -C source/ninfer-4090 checkout 5c60b7c9b455231795c09da21a9fbb6aa53f08e5
python3 -m venv ninfer-build-venv
ninfer-build-venv/bin/pip install cmake==3.31.6
python3 code-tuning/scripts/prepare_ninfer_toolchain.py
mkdir -p toolchains/ninfer-debs toolchains/ninfer-deps
cd toolchains/ninfer-debs
apt-get download libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
  libcurl4-openssl-dev libswresample-dev
for pkg in ./*.deb; do dpkg-deb -x "$pkg" "$INFERX_ROOT/toolchains/ninfer-deps"; done
cd "$INFERX_ROOT"
bash code-tuning/scripts/build_ninfer.sh >tuning/ninfer-build-reproduce.log 2>&1
```

本次提取的 FFmpeg 开发包为 Ubuntu `6.1.1-3ubuntu5`，curl 为 `8.5.0-2ubuntu10.15`，与已安装 runtime 完全匹配。以上 apt 命令只下载，不安装；若仓库包版本变化，需先核对 runtime，不能盲用不同 ABI。Ninja、pkg-config、GCC13 使用现有系统工具。构建脚本修正隔离 `.pc` 路径和开发 `.so` 链接，没有写系统目录；UI关闭，xgrammar 为 v0.2.5.post1 / `4546988d1cf51670a10eafb8795d37e4a2f0d833`。首次 GQA CUDA 编译耗时十余分钟。

模型下载脚本需要 Python3.11+、requests 和 hf_transfer；可以安装到 HDD 独立 venv，然后运行 `scripts/prepare_ninfer_model.py`。它固定 `hf-mirror.com`、版本、大小和 SHA256。已有同大小文件也会完整校验；哈希失败应先隔离坏文件再重试。不要将凭据写入脚本或报告。

压测在 HDD 的 `repo-venv`（Python3.12、aiohttp、pytest）运行。仓库保存提示和配置；原始响应及编译日志留在 HDD `tuning/`。

```bash
source /home/guokc/mypro/gnn-mount-sda1/InferX/env.sh
cd "$INFERX_ROOT/code-tuning"
# 需要空闲 GPU1，且 tuning/ 中已放入 results/tuning 的 corpus/config JSON
INFERX_GRID="$INFERX_ROOT/tuning/ninfer-final.json" \
  /usr/bin/python3 scripts/run_tuning_grid.py
# 运行中的服务可用于 warm 测试；输出名必须是新文件
"$INFERX_ROOT/repo-venv/bin/python" -m scripts.bench_tuning \
  --corpus "$INFERX_ROOT/tuning/context.json" \
  --output "$INFERX_ROOT/tuning/reproduce-warm.jsonl" \
  --config reproduce-ninfer7 --cache warm --repeat 3
"$INFERX_ROOT/repo-venv/bin/python" -m scripts.check_tuning_tools \
  --output "$INFERX_ROOT/tuning/reproduce-tools.json"
"$INFERX_ROOT/repo-venv/bin/python" -m scripts.export_tuning_results
"$INFERX_ROOT/repo-venv/bin/python" -m pytest -q
```

冷测须使用冷启动参数 `... 1 cold`（禁用 prefix reuse），不能在已有 warm 服务上只改标签。`run_tuning_grid.py` 独占它自己创建的进程；现有 winner 需按前述身份核对停止后再运行。导出结果按源文件及 output budget 分组，避免将截断性能样本与完整质量样本混算。失败原始记录仍保留在导出数据里。

归档共 **247 次请求记录、155 个分组**：243 次协议成功，其中 80 次达到输出预算；4 次失败保留（旧客户端 keep-alive 断连1次，NInfer不支持的模板字段HTTP400三次）。draft8启动失败单列，未混入请求吞吐。独立代码检查30/30、完整质量补测代码10/10通过。`results/tuning/trials.jsonl` 保留请求/输出哈希、缓存计数和单次指标；`summary.json` 保留每组 n/min/median/max；`ninfer-final-telemetry.json` 保留原生接收计数与 timing，并按请求顺序与生成 token 数核对。`*-launch.json`、corpus、网格和版本清单支持复核；完整文本及服务器日志留在 HDD。

工程执行决策、修复证据和审查记录见 [执行记录](2026-10-14-ninfer-execution-record.md)。后续可研究按任务和接收率调整 draft 长度，但这次没有实现自适应调度，也不把调参结果包装成新的研究机制。
