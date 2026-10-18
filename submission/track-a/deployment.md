# 部署与关键配置

当前采用冻结100的[便携入口](../../demo/selected100/README.md)，性能与质量分别使用 `perf-c14` 和 `quality100`。部署包含推理引擎及依赖、固定模型与tokenizer、所选版本的量化缓存。完整操作顺序见[复验说明](reproduction.md)。

| 当前100配置 | 性能 `perf-c14` | 质量 `quality100` |
|---|---:|---:|
| 客户端 / 引擎活动容量 | 14 / 13 | 13 / 13 |
| KV容量 | 121856 | 532480 |
| 最大上下文 / 输入块 | 65536 / 2048 | 65536 / 2048 |
| 普通MTP窗口K / KV格式 | 3 / int8 | 3 / int8 |
| 生成协议 | 冷8192输入、固定512输出 | 默认thinking与采样、自然EOS、最大32768 |
| 测试规模 | 128正式 + 8预热 + 1探测 | 原IFBench前100题、111条指令 |

客户端与服务显式使用同一recipe；历史u3性能选择 `100-k3-u3-47.json`，评测105选择 `100-k3-full64.json`。两种资产目录集合有区别，不能混用；其共有服务权重的对应关系见便携入口。新构建必须使用绑定其实际二进制身份的配方，并独立保存新结果。

下文保留硬件说明和早期输出投影版的部署细节。第3—4节的历史启动参数不用于复现100的成绩。

## 1. 硬件与环境

每次服务只使用一张48GB RTX 4090，实测显存49,140MiB，功率上限450W。不同节点用于独立研发和复测，不参与多卡合并推理。

已验证工具链包括Ubuntu 22.04、Linux x86_64、SM89、CUDA 13.2.51、GCC 13.4和Python 3.12。实际驱动、CPU、RAM和动态库版本以该次运行身份为准，见[机器盘点](../../results/ninfer-performance/inventory/)与原始服务启动日志。环境构建顺序见[复验说明](reproduction.md)。

## 2. 先分清四个参数

| 参数 | 含义 | 由谁设置 |
|---|---|---|
| 客户端并发 | 评测器同时发出的请求数 | runner |
| 引擎活动容量 | 服务内部允许同时执行的请求数 | `--max-concurrency` |
| KV容量 | 给请求历史分配的缓存预算 | `--kv-capacity` |
| 投机窗口K | 一次目标验证的候选窗口配置 | `--draft-tokens`及该版本实际实现 |

例如“客户端16、活动容量12”仍是16并发负载，多出的请求需要等待。混合提议版的目标窗口5与MTP提议3也需要分别记录。

## 3. 历史输出投影版与验证扩展版配置

| 配置 | 输出投影版12并发 | 输出投影版16并发 | 输出投影版32并发 | 验证扩展版质量 |
|---|---:|---:|---:|---:|
| 客户端 / 活动容量 | 12 / 12 | 16 / 16 | 32 / 32 | 16 / 16 |
| KV容量 | 104448 | 139264 | 278528 | 532480 |
| 最大上下文 | 65536 | 65536 | 65536 | 65536 |
| MTP窗口K | 3 | 3 | 3 | 3 |
| 输入块大小 | 2048 | 2048 | 2048 | 2048 |
| KV格式 | int8 | int8 | int8 | int8 |
| 生成协议 | 固定512输出 | 固定512输出 | 固定512输出 | 自然结束，最大32768 |
| 所在节点 | ucloud3 | ucloud3 | ucloud1 | ucloud2 |

前三列是同一binary；质量列是另一个版本，并启用输出投影验证路径及GDN状态融合。同一验证扩展版另有性能测试，KV为139264，也不能和质量配置互换。

其他候选的差异包括融合开关、K5、提示词查找、混合提议和生成进度调度，详见[逐项配置和证据](experiment-index.md)。精确env/argv必须随版本恢复，不能只更换文件名。

## 4. 历史输出投影版12并发启动命令

下面是实测启动参数的路径参数化写法。`INFERX_ROOT`指向数据目录；`runtime9270`需包含对应引擎及匹配依赖，三个weights目录是该版本的量化缓存。运行前按runtime实际布局设置库路径，并核对依赖清单。

```bash
export INFERX_ROOT=/path/to/inferx-data
export INFERX_RUNTIME="$INFERX_ROOT/runtime9270"
export CUDA_VISIBLE_DEVICES=0
export NINFER_PREFILL_A8=all
export NINFER_VERIFY_Q5_A8=1
export NINFER_SCHED_POLICY=completion_queue
export NINFER_SCHED_TTFT_MARGIN_MS=300
export NINFER_PREFILL_FULLK=s4gptq64ffn
export NINFER_S4_GPTQ_CACHE="$INFERX_ROOT/weights025"
export NINFER_S4_TARGET_FFN=1
export NINFER_S4_OMIT_LEGACY_FFN=1
export NINFER_INPUT_S4=1
export NINFER_INPUT_S4_CACHE="$INFERX_ROOT/weights041"
export NINFER_OUTPUT_S4=1
export NINFER_OUTPUT_S4_CACHE="$INFERX_ROOT/weights047"
mkdir -p "$INFERX_ROOT/logs"
"$INFERX_RUNTIME/ninfer-serve" \
  "$INFERX_ROOT/models/qwen38-ninfer/qwen3_8_27b.ninfer" \
  --host 127.0.0.1 --port 8088 --model-id qwen38-27b \
  --max-context 65536 --kv-capacity 104448 --max-concurrency 12 \
  --max-pending-requests 32 --pending-timeout-ms 600000 \
  --prefill-chunk 2048 --kv-dtype int8 \
  --presence-penalty 0 --frequency-penalty 0 --seed 42 --no-ui \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --request-log-jsonl "$INFERX_ROOT/logs/native.jsonl" \
  --no-prefix-reuse --no-disk-cache
```

参数来源：[实际运行环境](../../results/three-node-optimization/ucloud1/c32-tonight/C16-C12-handoff-to-ucloud3/environment.json)、[12并发运行身份](../../results/three-node-optimization/ucloud3/output47-C16-C12-delivery-001/raw/C12/candidate-identity.json)。实际构建关联55项依赖；复制单个可执行文件不能代替匹配运行环境。

## 5. 接口与负载

先查询 `/v1/models` 确认模型，再用[API probe](../../scripts/probe_ninfer_api.py)检查raw、chat、流式、usage与ignore_eos行为。现场可使用[Demo中的短请求](../basic/demo.md)展示真实输出。

性能评测用 `/v1/completions` 和精确token IDs，temperature=0、ignore_eos=true、固定输出512。服务关闭前缀复用与磁盘缓存，native日志应显示每条实际计算8192输入token、前缀命中0、full_reset。

IFBench用chat接口、原前100、默认thinking和模型采样、最大32768、自然EOS。当前100版本评测105的客户端/引擎容量均为13；历史076为16，更早的基础版为2。不能给服务全局强制greedy/no-thinking后仍声称使用同一质量协议。

## 6. 日志与异常排查

| 现象 | 首先查看 | 处理与记录 |
|---|---|---|
| 服务无法启动 | 端口、启动stderr、动态库依赖 | 记录冲突或依赖错误，确认资源后重启 |
| 显存不足 | startup内存预算、KV、活动容量、Graph | 保留OOM与原参数；调整配置后作为新配置测试 |
| 首字或生成慢 | 客户端时序、native输入量/提交数、GPU采样 | 区分排队、输入处理与生成等待 |
| 超时、空输出、截断 | HTTP记录、finish_reason、生成预算 | 保留在相应测试结果中，不剔除失败样本 |
| 复测结果不同 | 实际binary/source、缓存、env/argv、采样参数 | 逐项对照原记录，再分析性能或质量差异 |

保存实际启动身份、逐请求JSONL、原版导出、GPU采样和退出记录。启动命令在前台运行，可用当前终端Ctrl-C停止；后台服务按PID、启动时刻和可执行文件核验归属后管理。端口冲突不应通过停止未知服务解决。

数值/状态检查、接口检查和CPU测试各有范围，不能相互替代。完整复测流程见[复验说明](reproduction.md)。
