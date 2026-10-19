# 24GB RTX 4090：C16 性能与 IFBench 精度

2026-10-19 完成，版本 `projected-prefill-grace`，节点 `8876`，单张标准 24GB RTX 4090。最终状态与完整精度见 [selected-result.json](../projected-prefill-grace-confirm/selected-result.json)，本地原始文件与源码复核见 [final-local-verification.json](../projected-prefill-grace-confirm/final-local-verification.json)。

## 性能

原协议：client16/native16，实际 8192 输入 tokens → 512 输出 tokens，greedy/no-thinking、ignore EOS；每轮 128 条正式测量 + 8 条预热 + 1 条探测。探测输出为 1 token，其余 136 条各输出 512 tokens。关闭 prefix/disk cache 复用和诊断 trace。三轮使用同一二进制和配置，后两轮为固定的同输入复测及原生成器新 seed 测试。

| 轮次 | seed | TTFT P95（ms） | TPOT P95（ms） | 输出 tokens/s | 独立复核 |
|---|---:|---:|---:|---:|---|
| 首轮 | 20261016 | 9888.901213 | 39.637684 | 332.863546 | [audit](../projected-prefill-grace/independent-audit.json) |
| 相同输入复测 | 20261016 | 9856.828430 | 39.291928 | 329.717567 | [audit](../projected-prefill-grace-confirm/independent-same128-r2-audit.json) |
| 新 seed | 20261019 | 9871.622181 | 39.221759 | 333.410577 | [audit](../projected-prefill-grace-confirm/independent-fresh128-20261019-r3-audit.json) |

三轮均满足 TTFT P95 ≤ 10000 ms、TPOT P95 ≤ 40 ms。合计 384 条正式请求、411 条含预热和探测的冷请求，无错误。峰值 GPU 使用量均为 23539 MiB，最低剩余 569 MiB。最差 TTFT / TPOT 余量为 111.099 ms / 0.362316 ms；P95 达标不代表所有单条请求均低于阈值，也不代表其他负载有相同结果。

TPOT 按每条正式请求 `(请求总延迟 − TTFT) / 511` 计算，再取线性插值 P95；吞吐为正式阶段的 65536 输出 tokens 除以该阶段时长。[复核脚本](../audit_performance.py)校验原始请求长度、冷请求记录、客户端与引擎配置和输入身份。

## 精度

本版本实际运行原始 IFBench 前 100 题，共 111 条指令。使用同一源码、二进制、权重、INT8 KV 与算子/调度环境；独立精度配置为 client2/native2、KV 131072、context 65536、prefill chunk 2048、thinking 开启、max_tokens 32768。采样采用模型默认值：temperature 1、top_k 20、top_p 约 0.95、seed 42。

| 指标 | 通过 / 总数 | 得分 |
|---|---:|---:|
| Prompt-level strict | 87 / 100 | 87.00% |
| Instruction-level strict | 98 / 111 | 88.29% |
| Prompt-level loose | 91 / 100 | 91.00% |
| Instruction-level loose | 102 / 111 | 91.89% |

100 条均为冷请求并自然结束；零请求错误、零截断、零空最终答案。所有题目的 metadata 与冻结前 100 题一致；原版 EvalScope 1.13.0 评分器在 CPU 上重算全部题目，逐项匹配原评分。详见[独立复核汇总](../projected-prefill-grace-confirm/quality-independent/summary.json)、[逐题评分](../projected-prefill-grace-confirm/quality-independent/all100.jsonl)、[题目 metadata](../projected-prefill-grace-confirm/quality-independent/actual-first100-metadata.json)及[复核脚本](../audit_quality100.py)。精度为本次 C2 实测，不代表 C16 精度测试。

## 原始记录

[raw-results.tar.gz](raw-results.tar.gz) 保存三轮性能和一轮精度的全部原始文件：输入、逐请求数据、GPU 采样、服务日志、启动参数、配方、AIPerf 导出、EvalScope 预测和评分。归档内保留仓库相对目录；[RAW-SHA256SUMS](RAW-SHA256SUMS)逐文件记录原始字节校验值，[MANIFEST.json](MANIFEST.json)记录归档与本次公开附件的身份。

在仓库根目录执行，解压到一个新的独立目录：

```bash
export INFERX_REPO="$PWD"
export INFERX_EVIDENCE="$PWD/artifacts/8876-published-evidence"
mkdir -p "$PWD/artifacts"
mkdir "$INFERX_EVIDENCE"
tar -xzf results/8876-24gb/published/raw-results.tar.gz -C "$INFERX_EVIDENCE"
(cd "$INFERX_EVIDENCE" && sha256sum -c "$INFERX_REPO/results/8876-24gb/published/RAW-SHA256SUMS")
```

归档包含原运行时的绝对路径和启动时状态。原配方的 `pending` / `unqualified` 以及早期性能审计的 `quality=unmeasured` 是运行当时的快照，未回写改动；最终完成状态以 `selected-result.json` 和独立精度复核为准。性能原始目录为：

- `results/8876-24gb/projected-prefill-grace/C16-projected-prefill-grace-int8-margin150-original128-r1`
- `results/8876-24gb/projected-prefill-grace-confirm/projected-prefill-grace-confirm-same128-r2`
- `results/8876-24gb/projected-prefill-grace-confirm/projected-prefill-grace-confirm-fresh128-20261019-r3`

精度原始目录为 `results/8876-24gb/projected-prefill-grace-confirm/projected-prefill-grace-confirm-quality100`。

## 源码与复现配置

完整变更以[相对于 kit100 的源码补丁](../projected-prefill-grace/full-source-vs-kit100.patch)交付。基线完整源码已包含在仓库现有的 [reproduction-kit-v1.tar.gz](../../three-node-optimization/ucloud3/selected100-single-node-reproduction-20261018/reproduction-kit-v1.tar.gz) 中；其获取与模型资产来源见[原复验指南](../../../submission/track-a/reproduction.md)。该指南中的 48GB/C14、CUDA 13.2 配置属于历史版本；本次 24GB 实测使用 CUDA 12.8.1、GCC 13、sm_89。

在仓库根目录执行以下步骤重建源码，使用新的输出目录：

```bash
export INFERX_REPO="$PWD"
export INFERX_24GB_SOURCE="$PWD/artifacts/8876-source-rebuild"
export INFERX_BASE_KIT="$PWD/results/three-node-optimization/ucloud3/selected100-single-node-reproduction-20261018/reproduction-kit-v1.tar.gz"
printf '%s  %s\n' 62fc432a278f2b7f8c60472c173929fc1f537ea2785908207e8b1ce7703ed545 "$INFERX_BASE_KIT" | sha256sum -c -
mkdir -p "$PWD/artifacts"
mkdir "$INFERX_24GB_SOURCE"
tar -xzf "$INFERX_BASE_KIT" -C "$INFERX_24GB_SOURCE" --strip-components=1
patch --batch --fuzz=0 -p3 -d "$INFERX_24GB_SOURCE/source" < results/8876-24gb/projected-prefill-grace/full-source-vs-kit100.patch
python3 - <<'PY'
import hashlib, json, os
from pathlib import Path
root = Path(os.environ['INFERX_24GB_SOURCE']) / 'source'
manifest = json.loads((Path(os.environ['INFERX_REPO']) / 'results/8876-24gb/projected-prefill-grace/source-identity.json').read_text())
actual = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(root.rglob('*')) if p.is_file()}
assert actual == manifest['files']
assert hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == manifest['aggregate_sha256']
print('Verified', len(actual), 'source files:', manifest['aggregate_sha256'])
PY
```

已验证：补丁重建后全部 1045 个源码文件与实测版本逐字节一致，见[原补丁验证](../projected-prefill-grace/patch-dry-run.json)。新增调度策略通过显式环境变量开启，默认关闭；同时交付显存压缩、INT8 attention 及有序 Q5 residual 等本轮累计变更。

[reproduction.json](../projected-prefill-grace/reproduction.json)记录实际性能 argv、完整 NINFER 环境、资产合同和身份；精度配置与实际 argv 见[原 quality recipe](../projected-prefill-grace-confirm/projected-prefill-grace-confirm-quality100/recipe.json)和[launch.json](../projected-prefill-grace-confirm/projected-prefill-grace-confirm-quality100/launch.json)。迁移时替换模型、权重缓存、库、日志、GPU 和端口路径，并保留协议与策略设置。性能使用 KV 139264 / native16，精度使用 KV 131072 / native2。资产检查的内容 SHA 继承冻结回执，启动时核验文件集合与大小，具体范围在资产合同中记录。

正常构建及 CTest 的[资格回执](../projected-prefill-grace/qualification.json)、[JUnit 结果](../projected-prefill-grace/ctest.xml)和[构建复核](../projected-prefill-grace/independent-build-receipt-check.json)记录 151 项结果：147 通过、4 项原有其他模型测试跳过、0 失败。源码补丁和配置支持重建；模型、量化缓存、GPU 运行库及已构建二进制不包含在本次附件中，新环境构建和新运行需记录自己的身份与成绩。

| 对象 | SHA-256 |
|---|---|
| 源码聚合 | `44c210050cd1846834259da92f05f15d4aa1fbe0b8daaa27e89dd6365c15b7a6` |
| 实测二进制 | `9794763639fd2f102c8cd23d457006699eb5b45ae388d4cb167423d26be588b0` |
| 完整源码补丁 | `0459f360f77ea3db69ab2a92409397a6d332e821eb32862232ef00aa78724239` |
| 模型容器 | `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e` |
| 原性能评测器 | `d80df390cf6d1d50129796b31fdd7f4d1e95d0043d3b942d47ac582118702fb0` |
