# 100 性能与质量复现入口

从完整源码重新构建、校验输入并实际评测，使用[中文复验指南](../../submission/track-a/reproduction.md) / [English reproduction guide](../../submission/track-a/reproduction-en.md)。指南绑定新构建的独立配方；本目录下原配方继续保留历史100二进制身份。

本目录提供当前通过 C4、C8、C12、C13、C14 性能测试的冻结 100 版本入口；旧版本仍在 `demo/selected_recipes/` 保留。同一版本已完成原 100 题质量评测105：strict 85%、loose 88%，指令评分96/111、99/111；零请求错误、零截断，1条空最终答案保留计分。两个资产配方共用同一模型实现：

- version：`100-k3`，普通 MTP K3；bounded decode membership 开启。
- binary SHA：`e4ed1bf4baebf7da3a81e4aed01307c7e3e7ae97faef7672c507ed21bdaeb38c`，243219440 bytes。
- source SHA：`c976692d9657408308193b50772c700382a6e243d9da353165205dc5252f76a8`。
- 19 个 `NINFER_*` 变量随配方固定；启动环境另含 OMP/OPENBLAS 单线程及库搜索路径。外部实验开关不继承。

已有 u3 C14 两轮原128评测同时通过 TTFT/TPOT SLO，分别为 **9.893643 s / 39.929242 ms** 和 **9.821661 s / 39.141798 ms**。首轮 TPOT 余量仅 0.070758 ms，不能据两轮结果声称有充分余量或适用于所有硬件。同一程序的 C4/C8/C12/C13 也已完成各 128 条测量并通过。完整数据见仓库 `submission/track-a/data/selected-campaign.json`。本入口的可移动性经过 CPU 检查；不将脚本检查当作新的 GPU 复验或精度结果。

## 显式选择资产

| `--recipe` | 025层/文件 | 041层/文件 | 047层/文件 | 总服务权重文件 |
|---|---:|---:|---:|---:|
| `candidate_recipes/100-k3-u3-47.json` | 64/256 | 64/128 | 47/94 | 478 |
| `candidate_recipes/100-k3-full64.json` | 64/256 | 64/128 | 64/128 | 512 |

u3 C14 实测使用 u3-47；u1 部署使用 full64。两个 manifest 中共有的 478 个权重文件 SHA/大小一致；100 的固定 output mask 与 080 相同，full64 额外的 17 层 output 文件不被 loader 使用。依据是 `evidence/assets-equivalence.json` 和 `100-loader-verification.json`。这只证明已核对的资产映射，**未证明 100 与原 080 GPU cubin 相同**；100 是正常重建的独立二进制。

服务与客户端必须选择同一份 recipe；启动记录会保留其完整 SHA 和选中的 manifest。检查所选资产的精确层目录集合、`packed.bin`/`scales.f32` 集合及大小；full64 与 u3-47 互换、额外层、缺文件、截断均拒绝。辅助 JSON/日志不属于服务权重清单。两个配方不会静默合并资产集合。

服务会重新 hash 二进制；大模型/cache 内容 SHA 继承冻结回执，仅核验当前文件大小和目录集合，`content_sha_rehashed=false`。因此不检测同尺寸内容损坏。应部署来自冻结版本的不可变资产；历史动态依赖清单在 `evidence/actual-runtime-dependencies.json`，本包未重新验证实际 GPU 环境。

## Profile

| profile | 客户端并发 | native 容量 | KV容量 | 状态 |
|---|---:|---:|---:|---|
| perf-c4 | 4 | 4 | 65536 | 单轮实测通过 |
| perf-c8 | 8 | 8 | 69632 | 单轮实测通过 |
| perf-c12 | 12 | 12 | 104448 | 单轮实测通过 |
| perf-c13 | 13 | 13 | 113152 | 单轮实测通过 |
| perf-c14 | **14** | **13** | **121856** | 两轮实测通过 |
| quality100 | 13 | 13 | 532480 | 原100题评测105已完成：strict 85%，loose 88% |

性能 KV 为 `max(65536,8704*client_C)`。CLI 要求 KV 不少于 `max-context=65536`；原 C4/KV34816 在 GPU/请求前被拒绝，该配置错误不是性能 FAIL。入口会提前明确拒绝 KV<context。特别注意 C14 的 KV 按客户端 14 保留，不是 113152；评测客户端仍发 C14。每换一个 profile 都需独立启动对应服务。C15/C16/C32 不在本候选包的范围内。

## 部署与性能入口

整个目录可移动到其他 Linux 主机任意路径；只需 Python 标准库读取本包。包内自带冻结 d80 runner、原137条 dataset/meta、两份资产清单，不依赖原仓库绝对路径。模型、二进制、GPU 运行库和已安装的 AIPerf/EvalScope 环境不包含在包中。

默认 `--root /workspace/InferX` 布局：

```text
artifacts/100-k3/ninfer-serve
assets/model/qwen3_8_27b.ninfer
assets/tokenizer/
assets/weights025/   assets/weights041/   assets/weights047/
envs/build/lib/      eval-envs/
```

可用 `--binary`、`--model`、`--assets`、`--tokenizer`、`--build-lib-dir`、`--cuda-lib-dir` 显式搬迁路径。`--assets` 指含三个 weights 子目录的目录。默认 CUDA 路径是 `/usr/local/cuda-13.2/lib64`；GPU 选择由启动时 `CUDA_VISIBLE_DEVICES` 控制。本包不设置功耗、不选择其他机器、不编译、不安装依赖。

在本包目录内，终端一前台启动指定资产配方：

```sh
python3 serve_candidate.py \
  --recipe candidate_recipes/100-k3-u3-47.json --profile perf-c14 \
  --root /workspace/InferX --port 18499 \
  --log-dir /workspace/InferX/repro-candidate100/service-c14-001
```

服务加载并监听后，同机终端二运行：

```sh
OPENAI_API_KEY=EMPTY python3 perf_candidate.py \
  --recipe candidate_recipes/100-k3-u3-47.json --concurrency 14 \
  --root /workspace/InferX \
  --service-launch /workspace/InferX/repro-candidate100/service-c14-001/launch.json \
  --base-url http://127.0.0.1:18499/v1 \
  --output-dir /workspace/InferX/repro-candidate100/perf-c14-001
```

full64 部署须在两条命令中均显式改选 `100-k3-full64.json`。低 C 同时修改服务 profile 和客户端 concurrency，使用新输出目录。端口只能连接同机已验证进程；核验 PID/start、exe SHA、实际 argv、完整环境和监听 socket，避免测试到旧服务。已有输出目录不覆盖。

仍执行原128正式+8预热+1probe、8192/512、temp0、ignore_eos，保留初始请求；原 namespace 固定为 `8a90faa7364b4a6aad73eac63d71d369:candidate:c12`，不会随 C 改写。运行前核验冻结 dataset/meta，运行后核验全部137条 inputs 的 canonical SHA，再调用原 runner 统计。合法 SLO FAIL 会保留原始结果；完整性错误以非零状态退出。

## 独立原100题精度

`quality100` 和 `accuracy_candidate.py` 采用独立精度配置：native13/client13/KV532480，IFBench 前100题，生成配置只覆盖 `max_tokens=32768`，自然 EOS、默认 thinking/采样。包内没有 thinking512 变体，也不继承其他二进制的准确率。

评测105已经完成；以下命令发起独立的新一轮评测，不会复用105的分数。性能评测结束后，在服务终端用 `Ctrl+C` 正常停止性能服务，再启动质量服务；两种配置不同时占用同一张卡。

在本包目录内，终端一启动质量服务：

```sh
python3 serve_candidate.py \
  --recipe candidate_recipes/100-k3-full64.json --profile quality100 \
  --root /workspace/InferX --port 18500 \
  --log-dir /workspace/InferX/repro-candidate100/service-quality-001
```

服务就绪后，在终端二执行原100题评测：

```sh
OPENAI_API_KEY=EMPTY python3 accuracy_candidate.py \
  --recipe candidate_recipes/100-k3-full64.json \
  --root /workspace/InferX \
  --service-launch /workspace/InferX/repro-candidate100/service-quality-001/launch.json \
  --base-url http://127.0.0.1:18500/v1 \
  --output-dir /workspace/InferX/repro-candidate100/quality100-001
```

若部署的是 u3-47 资产，两条命令均改用 `candidate_recipes/100-k3-u3-47.json`。先核对 `evaluation-identity.json` 的退出状态，随后打开 `evaluation/accuracy/` 中的报告、逐题预测与评分记录。保留全部100题，包括空最终答案和格式失败；`evaluation-identity.json` 的 `complete` 表示流程完成，不等于题目全部答对。

所有入口均支持 `--dry-run`，可在上述命令末尾追加，先检查参数而不运行推理。dry-run 验证配方和本包文件，不证明实际模型/cache已部署或服务已验证。本目录原有CPU可移植性检查也不代表上述新目录已经实际运行。

## CPU 验证与交付边界

```sh
python3 -B -m unittest discover -s tests -p 'test_candidate.py' -v
```

测试封锁 exec、HTTP 和评测子进程，模拟 Linux `/proc` 与 socket。检查不相等的 client/native、错误 KV、服务身份变化、精确资产选择、路径含空格、原 payload 和精度 batch13。`checks/` 保存独立搬迁 dry-run、原 C14 参数等价检查与受保护 demo 文件 SHA；`MANIFEST.json` 固定包文件。原 runner 保持字节不变。

本目录的 index 已绑定当前冻结100的配方；名称 `candidate` 是脚本既有schema，不表示性能或质量仍待测。`evidence/status.json` 记录实际六轮性能与105精度来源。原准备包及其CPU回执保留原样；本目录的状态更新另记在部署回执中。入口的CPU检查不能替代搬迁到新硬件后的GPU复现。
