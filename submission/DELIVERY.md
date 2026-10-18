# 提交与复现入口

本次提交提供完整源码、固定公共输入的获取方式和量化缓存重建流程，**不附大型模型权重及12 GB量化缓存**。先读[量化重建指南](track-a/quantization-reproduction.md)，按其中链接安装环境、获取固定基础模型与FP8输入，再从源码生成缓存、构建服务并运行原评测。已有旧缓存时也可按[664复验指南](track-a/reproduction.md)复查对应历史结果。

[答辩材料](basic/presentation/README.md)、[技术报告](track-a/README.md)、[性能评测现场命令](basic/demo.md)和[采用结果](track-a/selected-results.md)分别对应展示、实现、运行与证据。[提交快照说明](SNAPSHOT.md)列出归档范围、版本身份和校验方式。

## 已有实际运行证据

| 范围 | 实际结果 | 边界 |
|---|---|---|
| 664：完整源码重新构建 | 1023 个源文件，源码聚合 SHA `c976692d9657408308193b50772c700382a6e243d9da353165205dc5252f76a8`，新二进制 `664cfbb9…f7518` | 完整源码与精确二进制 SHA 以正式包及其清单为准；没有使用历史二进制替代构建 |
| 664：C14 性能 | 128 个测量请求，TTFT P95 **9.8964 s**，TPOT P95 **38.2858 ms**，吞吐 **304.8967 token/s** | 客户端 14、执行槽 13；保留初始请求和队列等待；本轮同时通过 10 s / 40 ms 门槛 |
| 664：原 100 题质量 | 题目严格 **90/100**、宽松 **96/100**；指令严格 **100/111**、宽松 **106/111** | 100 个请求均完成；第 88 题空最终答案且触及输出长度上限，保留在分母中 |
| 新前缀环境安装 | 27 步安装与检查均成功；原入口与新 EvalScope 组合通过固定数据全部 300 条及前 100 题 / 111 指令检查 | 以上性能/质量复验使用原有环境；新安装验证没有发模型推理请求 |
| 116：从 FP8 重建量化缓存 | 025/041/047共239个矩阵拟合完成，生成478个服务文件；逐文件SHA与参考缓存一致，239项独立CPU抽样核对无差异 | [SHA核对](../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/fresh-vs-original478-SHA.json)、[拟合核对](../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/fresh239-sampled-fit-numeric-summary.json)与[原始证据包](../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/fresh239-fit-light-evidence-001.tar.gz) |
| 116：新缓存与新构建的C14 | binary `e919f180…`；128条正式请求全部成功；TTFT P95 **9.8622 s**、TPOT P95 **39.4286 ms**，吞吐 **305.5329 token/s** | [原始汇总](../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/fresh-C14-raw-001/deployment116/performance-C14/summary.json)；独立实测同时通过10 s / 40 ms，原100题质量仍待终态回执 |

664复验使用了固定公共基础模型和旧478文件量化缓存；116复验使用同一源码重新构建的二进制及本轮重新生成的478文件缓存，两轮结果分别记录。[性能独立核对](../results/three-node-optimization/ucloud3/selected100-single-node-reproduction-20261018/root-C14-independent-audit.json)与[质量独立核对](../results/three-node-optimization/ucloud3/selected100-single-node-reproduction-20261018/root-quality-independent-audit.json)记录了原始数据、统计口径和异常保留情况。历史 `e4ed1bf4…` 二进制的 100/105 结果另行保存；没有把历史成绩移交给新二进制。

## 固定交付物

| 文件 | 用途与校验 |
|---|---|
| [reproduction-kit-v1.tar.gz](../results/three-node-optimization/ucloud3/selected100-single-node-reproduction-20261018/reproduction-kit-v1.tar.gz) | 完整源码、构建及运行入口、依赖清单、C14 和原 100 题记录。40,181,061 字节；SHA256 `62fc432a278f2b7f8c60472c173929fc1f537ea2785908207e8b1ce7703ed545` |
| [完整内容清单](../results/three-node-optimization/ucloud3/selected100-single-node-reproduction-20261018/reproduction-kit-v1-complete-content-manifest.json) | 覆盖该压缩包实际 1212 个文件，包括包内其他清单；SHA256 `9001c7adbb05d2ce63aa43b15e9590becfafc4eaf63ef4ecc8dbd97a3571638a` |
| [IFBench 固定数据包](../results/three-node-optimization/ucloud1/ifbench-fixed-dataset-acquisition-CPU/ifbench-fixed-dataset-cache-r1.tar.gz) | 固定公共数据、缓存准备及原始加载器检查脚本。SHA256 `c267da418e8c0a79b402e37237b14fde4a17233f60b8d5f866b85493fda757aa` |
| [FP8 获取脚本包](../results/three-node-optimization/ucloud1/fp8-pinned-acquisition-r2/fp8-pinned-acquisition-r2.tar.gz) | 按 72 个内容文件的固定来源和 SHA 获取量化输入；不假定统一仓库 revision。SHA256 `b115a3276e1561a68350f173db9d58a2a642846af9fc88f6e5ea6596a41c721f` |
| [全流程量化源码包 002](../results/three-node-optimization/ucloud2/full-quantization-reproduction-116-portable-source-002.tar.gz) | 四套完整父版本源码、采集与拟合脚本、固定输入、环境锁及新缓存评测入口。38,965,052 字节；SHA256 `7300c70a5857221537b878046d8c869ce640c788de4773d91d4e523b6bb41f9e`。包内 `portable116/README.md` 为入口；实际复验结果见上表及独立证据 |
| [环境与数据实际复验补充包](../results/three-node-optimization/ucloud3/selected100-single-node-reproduction-20261018/reproduction-addenda-r1.tar.gz) | 安装、原入口组合、包版本及动态库路径对照的原始回执。10,501,827 字节；SHA256 `4d28291bac9a0ce1ba9cc21d9a6654a12dcd7206c2aa1f5835c223c5e37867a8` |

已完成的新环境和数据检查以独立附件保存；116新缓存的测量按其实际完成范围报告。既有冻结包保持原样，归档包含本表所列附件。

## 输入资产与交付方式

公开基础模型从 `neroued/Qwen3.8-27B-NInfer` 的固定 revision `3526913004b1cf552cb57b88d6a5c6f5e4a89a70` 获取，文件为 `qwen3_8_27b.ninfer`，SHA256 为 `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e`。下载及 tokenizer 提取命令见复现指南。

精确复查上述664历史成绩时，旧缓存的身份为`selected100-u3-47-caches.tar`，12,082,360,320字节，SHA256 `a8936aa7d39e9f971256a30b415f6f271f4f70418d0705233c0c84bfabfc3fb7`。这项资产用于说明历史测量条件，不是本次提交的必要附件；本次默认执行002源码包重建缓存。重建后保存新缓存与新二进制身份，并运行自己的性能和精度测试。

最终材料按当前工作区的显式文件清单归档并核验解压内容。不要直接用 `git archive main` 代替最终交付：当前实际源码包、量化脚本和新记录中存在尚未纳入该提交的文件。归档、可访问的上传地址和主办方提交回执是不同步骤；此页不表示已经上传或提交。
