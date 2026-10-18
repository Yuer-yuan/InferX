# 实验索引：名称、版本与原始证据

正文使用可读名称，本页负责定位精确版本、参数和原文件。binary是可执行文件的SHA-256；source是该次源码文件清单的摘要。摘要用于核对身份，不能代替实际源码。

## 主要结果怎么找

| 要查的结果 | 文中名称 | binary前缀 | 关键配置 |
|---|---|---|---|
| 当前冻结100：C4/8/12/13及C14两轮 | 冻结100 | `e4ed1bf4` | C14客户端14/活动13，KV121856；[原始配置与证据](selected-results.md) |
| 当前冻结100：原100题质量105 | 冻结100 | `e4ed1bf4` | 客户端/活动13，KV532480，strict85/100、loose88/100 |
| 完整源码重建：性能与原100题质量002 | 重建664 | `664cfbb9` | C14性能304.8967 tok/s；独立质量90/100、96/100；[复验记录](reproduction.md) |
| 新量化239矩阵／478文件与独立C14 | 复验116 | `e919f180` | C14性能305.5329 tok/s、两项SLO通过；原100题质量待终态；[完整流程](quantization-reproduction.md) |
| 历史12并发单轮通过 | 输出投影量化版 | `9270f73e` | u3，活动12，K3，KV104448 |
| 同版16/32并发 | 输出投影量化版 | `9270f73e` | 16在u3、32在u1；KV139264/278528 |
| 历史验证扩展版100题质量 | 验证阶段扩展版 | `66328f67` | u2，客户端/活动16，K3，KV532480 |
| 活动容量16与12取舍 | 递推状态融合版 | `2362fc01` | 固定客户端16，u3，K3，KV139264 |


## 116新量化复验身份

- 正常构建二进制：`e919f18084f3cc09718158cfae071cdf8c58d5660715648db375643016c878f5`；源码聚合：`c976692d9657408308193b50772c700382a6e243d9da353165205dc5252f76a8`。
- 独立[116-fresh47配方](../../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/deployment116/entrypoint/candidate_recipes/116-fresh47.json) SHA：`4b7d1b7542291948bbec1af23fe99973189d48d6e171bcabc4a40e0b6c123a59`。
- 025／041／047实际拟合128／64／47矩阵，生成256／128／94文件；[478文件SHA及大小对照](../../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/fresh-vs-original478-SHA.json)全部一致、差异0。
- C14：client14／native13／KV121856／K3；原128＋8＋1，全部冷8192／512。[汇总](../../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/fresh-C14-raw-001/deployment116/performance-C14/summary.json)、[原客户端数据](../../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/fresh-C14-raw-001/deployment116/performance-C14/c14/profile_export.jsonl)、[观察时间窗](../../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/fresh-C14-raw-001/deployment116/performance-C14/c14/phase_manifest.json)。TTFT 9.86221024060s、TPOT 39.42860192427ms、305.53286821935 tok/s，0错、两项SLO通过。
- 原100题质量：另启client/native13、KV532480、K3，保留默认thinking、采样及自然EOS，仅显式覆盖max_tokens32768；实际终态与四项原分数待单独记录，不沿用664或105。

## 每次性能测试

旧基线四行是100正式+8预热，其余为128+8。均单卡、冷8192输入/512输出。指标见[完整成绩表](generated-results.md)。

| 版本与并发 | 节点 | 客户端/活动容量 | K / KV | 证据 |
|---|---|---|---|---|
| 旧NInfer基线 · 1并发 `b604d050` | ucloud1 | 1/1 | 4 / 131072 | [汇总](../../results/ninfer-performance/official-r1/runs/official-r1/performance/c1/summary.json) · [客户端](../../results/ninfer-performance/official-r1/runs/official-r1/performance/c1/profile_export.jsonl) |
| 旧NInfer基线 · 2并发 `b604d050` | ucloud1 | 2/2 | 4 / 131072 | [汇总](../../results/ninfer-performance/official-r1/runs/official-r1/performance/c2/summary.json) · [客户端](../../results/ninfer-performance/official-r1/runs/official-r1/performance/c2/profile_export.jsonl) |
| 旧NInfer基线 · 4并发 `b604d050` | ucloud1 | 4/4 | 4 / 131072 | [汇总](../../results/ninfer-performance/official-r1/runs/official-r1/performance/c4/summary.json) · [客户端](../../results/ninfer-performance/official-r1/runs/official-r1/performance/c4/profile_export.jsonl) |
| 旧NInfer基线 · 8并发 `b604d050` | ucloud1 | 8/8 | 4 / 131072 | [汇总](../../results/ninfer-performance/official-r1/runs/official-r1/performance/c8/summary.json) · [客户端](../../results/ninfer-performance/official-r1/runs/official-r1/performance/c8/profile_export.jsonl) |
| 早期前馈量化版 · 8并发 `a3326323` | ucloud3 | 8/8 | 3 / 139264 | [汇总](../../results/three-node-optimization/ucloud3/c32-tonight/C8-target-s4-K3-n128-r1/results/raw/candidate/c8/summary.json) · [客户端](../../results/three-node-optimization/ucloud3/c32-tonight/C8-target-s4-K3-n128-r1/results/raw/candidate/c8/profile_export.jsonl) · [服务日志](../../results/three-node-optimization/ucloud3/c32-tonight/C8-target-s4-K3-n128-r1/results/raw/candidate/requests.jsonl) |
| 前馈量化版 · 32并发 `c3f48207` | ucloud1 | 32/32 | 3 / 278528 | [汇总](../../results/three-node-optimization/ucloud1/c32-tonight/C32-target-s4-omit-K3-n128-r3/raw/run/candidate/c32/summary.json) · [客户端](../../results/three-node-optimization/ucloud1/c32-tonight/C32-target-s4-omit-K3-n128-r3/raw/run/candidate/c32/profile_export.jsonl) · [服务日志](../../results/three-node-optimization/ucloud1/c32-tonight/C32-target-s4-omit-K3-n128-r3/raw/run/candidate/requests.jsonl) |
| 输入投影量化版 · 32并发 `40b3e867` | ucloud1 | 32/32 | 3 / 278528 | [汇总](../../results/three-node-optimization/ucloud1/c32-tonight/C32-input-allphase-K3-n128-r1/summary.json) |
| 输出投影量化版 · 32并发 `9270f73e` | ucloud1 | 32/32 | 3 / 278528 | [汇总](../../results/three-node-optimization/ucloud1/c32-tonight/C32-output47-on-allphase-K3-n128-r1/raw/run/candidate/c32/summary.json) · [客户端](../../results/three-node-optimization/ucloud1/c32-tonight/C32-output47-on-allphase-K3-n128-r1/raw/run/candidate/c32/profile_export.jsonl) · [服务日志](../../results/three-node-optimization/ucloud1/c32-tonight/C32-output47-on-allphase-K3-n128-r1/raw/run/candidate/requests.jsonl) |
| 输出投影量化版 · 12并发 `9270f73e` | ucloud3 | 12/12 | 3 / 104448 | [汇总](../../results/three-node-optimization/ucloud3/output47-C16-C12-delivery-001/raw/C12/candidate/c12/summary.json) · [客户端](../../results/three-node-optimization/ucloud3/output47-C16-C12-delivery-001/raw/C12/candidate/c12/profile_export.jsonl) · [服务日志](../../results/three-node-optimization/ucloud3/output47-C16-C12-delivery-001/raw/C12/candidate/requests.jsonl) |
| 输出投影量化版 · 16并发 `9270f73e` | ucloud3 | 16/16 | 3 / 139264 | [汇总](../../results/three-node-optimization/ucloud3/output47-C16-C12-delivery-001/raw/C16/candidate/c16/summary.json) · [客户端](../../results/three-node-optimization/ucloud3/output47-C16-C12-delivery-001/raw/C16/candidate/c16/profile_export.jsonl) · [服务日志](../../results/three-node-optimization/ucloud3/output47-C16-C12-delivery-001/raw/C16/candidate/requests.jsonl) |
| down投影调整版 · 16并发 `dc99ec83` | ucloud3 | 16/16 | 3 / 139264 | [汇总](../../results/three-node-optimization/ucloud3/down053-C16-official128-001/delivery/raw/run/candidate/c16/summary.json) · [客户端](../../results/three-node-optimization/ucloud3/down053-C16-official128-001/delivery/raw/run/candidate/c16/profile_export.jsonl) · [服务日志](../../results/three-node-optimization/ucloud3/down053-C16-official128-001/delivery/raw/run/candidate/requests.jsonl) |
| 高批注意力版 · 32并发 `8b20de4d` | ucloud3 | 32/32 | 3 / 278528 | [报告](../../results/three-node-optimization/ucloud3/gqa-highbatch-C32-official128-001/REPORT.md) · [汇总](../../results/three-node-optimization/ucloud3/gqa-highbatch-C32-official128-001/delivery/content/raw/candidate/c32/summary.json) · [客户端](../../results/three-node-optimization/ucloud3/gqa-highbatch-C32-official128-001/delivery/content/raw/candidate/c32/profile_export.jsonl) · [服务日志](../../results/three-node-optimization/ucloud3/gqa-highbatch-C32-official128-001/delivery/content/raw/candidate/requests.jsonl) |
| 递推状态融合版 · 32并发 `2362fc01` | ucloud3 | 32/32 | 3 / 278528 | [报告](../../results/three-node-optimization/ucloud3/gdn-fusion-C32-official128-001/REPORT.md) · [汇总](../../results/three-node-optimization/ucloud3/gdn-fusion-C32-official128-001/delivery/content/raw/candidate/c32/summary.json) · [客户端](../../results/three-node-optimization/ucloud3/gdn-fusion-C32-official128-001/delivery/content/raw/candidate/c32/profile_export.jsonl) · [服务日志](../../results/three-node-optimization/ucloud3/gdn-fusion-C32-official128-001/delivery/content/raw/candidate/requests.jsonl) |
| 递推状态融合版 · 16并发 `2362fc01` | ucloud3 | 16/16 | 3 / 139264 | [报告](../../results/three-node-optimization/ucloud3/gdn-fusion-C16-official128-001/REPORT.md) · [汇总](../../results/three-node-optimization/ucloud3/gdn-fusion-C16-official128-001/delivery/content/raw/candidate/c16/summary.json) · [客户端](../../results/three-node-optimization/ucloud3/gdn-fusion-C16-official128-001/delivery/content/raw/candidate/c16/profile_export.jsonl) · [服务日志](../../results/three-node-optimization/ucloud3/gdn-fusion-C16-official128-001/delivery/content/raw/candidate/requests.jsonl) |
| 验证阶段扩展版 · 16并发 `66328f67` | ucloud1 | 16/16 | 3 / 139264 | [报告](../../results/three-node-optimization/ucloud1/output047-verify-integration-076/REPORT.md) · [汇总](../../results/three-node-optimization/ucloud1/output047-verify-integration-076/C16-r1/delivery/raw/candidate/c16/summary.json) · [客户端](../../results/three-node-optimization/ucloud1/output047-verify-integration-076/C16-r1/delivery/raw/candidate/c16/profile_export.jsonl) · [服务日志](../../results/three-node-optimization/ucloud1/output047-verify-integration-076/C16-r1/delivery/raw/candidate/requests.jsonl) |
| 递推状态融合版 · 16客户端 / 活动容量12 `2362fc01` | ucloud3 | 16/12 | 3 / 139264 | [报告](../../results/three-node-optimization/ucloud3/gdn-fusion-client16-native12-001/REPORT.md) · [汇总](../../results/three-node-optimization/ucloud3/gdn-fusion-client16-native12-001/delivery/content/raw/candidate/c16/summary.json) · [客户端](../../results/three-node-optimization/ucloud3/gdn-fusion-client16-native12-001/delivery/content/raw/candidate/c16/profile_export.jsonl) · [服务日志](../../results/three-node-optimization/ucloud3/gdn-fusion-client16-native12-001/delivery/content/raw/candidate/requests.jsonl) |
| 递推状态融合版 · K5 / 状态融合关闭 `2362fc01` | ucloud2 | 16/16 | 5 / 139264 | [报告](../../results/three-node-optimization/ucloud2/native16-K5-C16-075-r2/REPORT.md) · [汇总](../../results/three-node-optimization/ucloud2/native16-K5-C16-075-r2/delivery/content/C16-n128-w8/summary.json) · [客户端](../../results/three-node-optimization/ucloud2/native16-K5-C16-075-r2/delivery/content/C16-n128-w8/profile_export.jsonl) · [服务日志](../../results/three-node-optimization/ucloud2/native16-K5-C16-075-r2/delivery/content/service/native.jsonl) |
| 提示词查找优先版 · 16并发 `4b6484e0` | ucloud3 | 16/16 | 3 / 139264 | [报告](../../results/three-node-optimization/ucloud3/prompt-lookup-N5-C16-official128-recovery002/REPORT.md) · [汇总](../../results/three-node-optimization/ucloud3/prompt-lookup-N5-C16-official128-recovery002/delivery/content/raw/candidate/c16/summary.json) · [客户端](../../results/three-node-optimization/ucloud3/prompt-lookup-N5-C16-official128-recovery002/delivery/content/raw/candidate/c16/profile_export.jsonl) · [服务日志](../../results/three-node-optimization/ucloud3/prompt-lookup-N5-C16-official128-recovery002/delivery/content/raw/candidate/requests.jsonl) |
| 混合提议版 · 16并发 `38f4edfc` | ucloud1 | 16/16 | 5 / 139264 | [报告](../../results/three-node-optimization/ucloud1/targetK5-proposal3-combination-079/C16-r1/summary.json) · [汇总](../../results/three-node-optimization/ucloud1/targetK5-proposal3-combination-079/C16-r1/delivery/raw/candidate/c16/summary.json) · [客户端](../../results/three-node-optimization/ucloud1/targetK5-proposal3-combination-079/C16-r1/delivery/raw/candidate/c16/profile_export.jsonl) · [服务日志](../../results/three-node-optimization/ucloud1/targetK5-proposal3-combination-079/C16-r1/delivery/raw/candidate/requests.jsonl) |
| 混合提议与调度改进版 · 16并发 `c72ad839` | ucloud1 | 16/16 | 5 / 139264 | [报告](../../results/three-node-optimization/ucloud1/hybrid079-request-yield-max-080/C16-r1/summary.json) · [汇总](../../results/three-node-optimization/ucloud1/hybrid079-request-yield-max-080/C16-r1/delivery/raw/candidate/c16/summary.json) · [客户端](../../results/three-node-optimization/ucloud1/hybrid079-request-yield-max-080/C16-r1/delivery/raw/candidate/c16/profile_export.jsonl) · [服务日志](../../results/three-node-optimization/ucloud1/hybrid079-request-yield-max-080/C16-r1/delivery/raw/candidate/requests.jsonl) |
| 生成进度调度改进版 · 16并发 `837f0f1f` | ucloud3 | 16/16 | 3 / 139264 | [报告](../../results/three-node-optimization/ucloud3/request-yield-max-C16-official128-recovery002/REPORT.md) · [汇总](../../results/three-node-optimization/ucloud3/request-yield-max-C16-official128-recovery002/delivery/content/raw/candidate/c16/summary.json) · [客户端](../../results/three-node-optimization/ucloud3/request-yield-max-C16-official128-recovery002/delivery/content/raw/candidate/c16/profile_export.jsonl) · [服务日志](../../results/three-node-optimization/ucloud3/request-yield-max-C16-official128-recovery002/delivery/content/raw/candidate/requests.jsonl) |

配置差异还包括：

- 混合提议版：目标窗口5、MTP提议3，提示词查找混合、状态融合、输出投影验证开启。
- 混合提议与调度改进版：上述基础上增加 `NINFER_SCHED_REQUEST_YIELD_MAX=1`。
- 生成进度调度改进版：K3、状态融合开、提示词查找关，只更新生成进度估计。
- K5融合关闭、提示词查找优先、活动容量限制也各有独立配置。完整env/argv见各报告及native启动记录。

## 完整质量测试

| 版本 | 实际配置 | 原始证据 |
|---|---|---|
| 前馈量化版 `c3f48207` | C2，默认thinking/采样；完整服务配置见原报告 | [评分汇总](../../results/three-node-optimization/ucloud3/c32-current-official100-001/final-raw/CPU-score-001/summary.json) |
| 输入投影量化版 `40b3e867` | C2，默认thinking/采样；完整服务配置见原报告 | [评分汇总](../../results/three-node-optimization/ucloud2/true-int4-input-allphase-quality-preparation-050/terminal-evidence-001/CPU-score-001/summary.json) |
| 验证阶段扩展版 `66328f67` | C16/native16/K3，fusionON/Verify1，KV532480 | [报告](../../results/three-node-optimization/ucloud2/outputVerify-K3-C16-quality-076/REPORT.md) · [评分汇总](../../results/three-node-optimization/ucloud2/outputVerify-K3-C16-quality-076/delivery/content/CPU-score-001/summary.json) · [实际运行](../../results/three-node-optimization/ucloud2/outputVerify-K3-C16-quality-076/runtime-summary.json) |

均为原前100题、111条约束，默认thinking/采样、最大32768。质量配置与性能配置分别核对，逐题失败与空输出/截断完整保留。

## 完整执行文件与源码身份

### 旧NInfer基线

- binary：`b604d0507e753de76e6b3b288011434b2bb07f9038738f8c73768ddac8dfba86`。
- 源码基线或运行身份按原报告恢复；来源清单未单列完整源码摘要。

### 早期前馈量化版

- binary：`a33263234e03c0b3607136d9a4e8722fe71cb6978e4c1171ddaf9a0216ae6129`。
- source：`ce8fa3c8ba401b1e4d9fb0f87e2ecf09c3569f439afb7fe9bf758f787ab40a9d`。
- [原运行身份](../../results/three-node-optimization/ucloud3/c32-tonight/C8-target-s4-K3-n128-r1/results/raw/candidate-identity.json)。

### 前馈量化版

- binary：`c3f482079962708672f72c0716d1730f440c1a4fdfe99994073a3be88be1d209`。
- source：`eb0a8bb9194d8d54fa79d613780cc3c11df31bf6386ea18d41f9748d15bc77fc`。
- [原运行身份](../../results/three-node-optimization/ucloud1/c32-tonight/C32-target-s4-omit-K3-n128-r3/raw/run/candidate-identity.json)。

### 输入投影量化版

- binary：`40b3e867ef2b78da801d6f39613a1722619c06a1c3932fa87cb021f36d0f6853`。
- 源码基线或运行身份按原报告恢复；来源清单未单列完整源码摘要。

### 输出投影量化版

- binary：`9270f73e92409bc94e4b8223adb755c8b1527e611eec21e1ef8778482d0b74c2`。
- source：`91621a3c46a4c901a51863442704ea7c398ebdfb73ade9f3b1c4afac4ab235cc`。
- [原运行身份](../../results/three-node-optimization/ucloud1/c32-tonight/C32-output47-on-allphase-K3-n128-r1/raw/control/runtime-identity.json)。

### down投影调整版

- binary：`dc99ec83f504723667f03e06c732e01cae92d36e62b63d0e49f1ac79ad694424`。
- source：`f592cb8d4278c0cabd6a7044479d2aee129a4c2b4b6336b430ce297779af6198`。
- [原运行身份](../../results/three-node-optimization/ucloud3/down053-C16-official128-001/delivery/raw/run/candidate-identity.json)。

### 高批注意力版

- binary：`8b20de4daa8c0556ab9e2b69f60f5ea6ae3b1598680ee6f04a33d193b7e3a1c7`。
- source：`61a15f2f08d8efd9f185c4edabbad6370104584f522446537c765617b94a045e`。
- [原运行身份](../../results/three-node-optimization/ucloud3/gqa-highbatch-C32-official128-001/delivery/content/raw/candidate-identity.json)。

### 递推状态融合版

- binary：`2362fc019db65e2704ed9a8477116ea8630dd5896f5762be35a673e568759142`。
- source：`4fcbcf2cee012a4ec03e252cfc4bf54aa8795b72978c038d25b08396938721ad`。
- [原运行身份](../../results/three-node-optimization/ucloud3/gdn-fusion-C32-official128-001/delivery/content/raw/candidate-identity.json)。

### 验证阶段扩展版

- binary：`66328f6735061654d31a2a67e1b92096ff38c7820ffd2c1f54febcf4a7ef2691`。
- source：`3a7ea905ba0774a4ddce4f007cc98836cabc9feac2e5efcc5605567aae0cb47d`。
- [原运行身份](../../results/three-node-optimization/ucloud1/output047-verify-integration-076/C16-r1/delivery/raw/candidate-identity.json)。

### 提示词查找优先版

- binary：`4b6484e0ab49ac9639ccc51839ab740088f254202587ec28b4df5b599e330ffa`。
- source：`328fafc3a4c1730d9dc9212b23b1e6b7bad90d757f6748566c1348fb37cc10df`。
- [原运行身份](../../results/three-node-optimization/ucloud3/prompt-lookup-N5-C16-official128-recovery002/delivery/content/raw/candidate-identity.json)。

### 混合提议版

- binary：`38f4edfcdeadb4467cb6f64733a5c479250c5396ef0ca86ddf7e68fa5ef0c10e`。
- source：`5397eb4c41e3077589bcb1aa5eeef750b56824a92259cd5c2bd89a498a9576a4`。
- [原运行身份](../../results/three-node-optimization/ucloud1/targetK5-proposal3-combination-079/C16-r1/delivery/raw/candidate-identity.json)。

### 混合提议与调度改进版

- binary：`c72ad839ab3de47b5a4e19a69309e7eeff911e194b3e57516b644b4211f9c71c`。
- source：`a6acc1d26edfb10e98d294ecc1720e9ae9a5c0cc67aa0c154936b1322f52d894`。
- [原运行身份](../../results/three-node-optimization/ucloud1/hybrid079-request-yield-max-080/C16-r1/delivery/raw/candidate-identity.json)。

### 生成进度调度改进版

- binary：`837f0f1f98e4d45901a2c521e145d00e01716c3880e0679b530854c827f2d925`。
- source：`2bf64891c77d0c2e478d7f1648a76ae794b8b1c16852b3533a14ffc1134cdd01`。
- [原运行身份](../../results/three-node-optimization/ucloud3/request-yield-max-C16-official128-recovery002/delivery/content/raw/candidate-identity.json)。

## 模型与缓存身份

| 输入 | 固定身份或说明 |
|---|---|
| 基础模型 | `neroued/Qwen3.8-27B-NInfer` |
| revision | `3526913004b1cf552cb57b88d6a5c6f5e4a89a70` |
| 模型文件 | `qwen3_8_27b.ninfer`，18,210,531,328 bytes，固定v2混合Q4/Q5 |
| 模型SHA-256 | `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e` |
| 基础引擎commit | `5c60b7c9b455231795c09da21a9fbb6aa53f08e5` |
| xgrammar commit | `4546988d1cf51670a10eafb8795d37e4a2f0d833` |
| 前馈缓存 weights025 | 8,690,073,600 bytes，[校准与manifest说明](../../results/three-node-optimization/ucloud2/true-int4-all-layer-GPTQ-025/REPORT.md) |
| 输入缓存 weights041 | 2,641,100,800 bytes，[实际加载环境](../../results/three-node-optimization/ucloud1/c32-tonight/C16-C12-handoff-to-ucloud3/environment.json) |
| 输出缓存 weights047 | 1,022,361,600 bytes；64个生成矩阵与47层实际GDN预填充路由分别核对 |

缓存包含packed codes、scales及manifest，字节数不代替逐文件校验。获取、tokenizer提取和重建条件见[复验说明](reproduction.md)与[模型清单](../model-manifest.json)。

## 机器与材料身份

ucloud1/2/3是节点标识，每次成绩只用其中一张GPU。GPU UUID、驱动和内存预算以native的server_start为准。原文件中的绝对路径表示当时运行位置，迁移时按[部署说明](deployment.md)展开到新目录。

[来源清单](data/sources.json)记录精确路径，[结果JSON](data/results.json)保留复算与资源字段，[材料manifest](evidence-manifest.json)固定本次引用文件和生成产物。
