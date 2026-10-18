# 提交快照与读取顺序

本次归档标识为 `InferX-TrackA-20261018-r2`。归档以当前工作区的显式文件清单生成，包含尚未进入Git提交的源码包、文档和测试证据；根目录的`SNAPSHOT.json`与`MANIFEST.json`标识实际内容，不能用旧Git HEAD代替本次快照。

1. 阅读[材料总览](README.md)与[答辩PDF](basic/presentation/InferX-ZH.pdf)。
2. 查看[Benchmark](track-a/benchmark.md)和[原始证据索引](track-a/experiment-index.md)。
3. 按[源码重建量化缓存](track-a/quantization-reproduction.md)获取固定公共输入、安装环境、构建源码、生成缓存并评测；该路线不要求随包交付12 GB缓存。
4. 现场已有部署时，直接使用[性能评测命令](basic/demo.md)。

## 固定源码和结果身份

| 对象 | 本快照采用的身份 |
|---|---|
| 引擎完整源码 | 1023文件，聚合SHA `c976692d9657408308193b50772c700382a6e243d9da353165205dc5252f76a8` |
| 完整源码与历史复验包 | `reproduction-kit-v1.tar.gz`，SHA `62fc432a278f2b7f8c60472c173929fc1f537ea2785908207e8b1ce7703ed545` |
| 缓存重建完整源码包 | `full-quantization-reproduction-116-portable-source-002.tar.gz`，SHA `7300c70a5857221537b878046d8c869ce640c788de4773d91d4e523b6bb41f9e` |
| 冻结100历史运行 | binary `e4ed1bf4…`；C14两轮306.04/309.95 token/s，独立质量105为85/100、88/100 |
| 同一源码重建664 | binary `664cfbb9…`；旧缓存C14为304.8967 token/s，独立质量002为90/100、96/100 |
| 从FP8重新生成缓存的116复验 | 239个矩阵生成478个文件，逐文件SHA与参考缓存一致；binary `e919f180…` 的独立C14为305.5329 token/s，原100题质量待终态回执 |

包文件的相对位置见[交付入口](DELIVERY.md)。公共基础模型、FP8输入和环境依赖按固定来源获取；大型权重、量化缓存、临时传输文件、历史编译产物及本地账号配置不放入本次归档。

## 校验

发布文件为`InferX-TrackA-20261018-r2.tar.gz`及同名`.sha256`。取得两者后：

```bash
sha256sum -c InferX-TrackA-20261018-r2.tar.gz.sha256
tar -xzf InferX-TrackA-20261018-r2.tar.gz
cd InferX-TrackA-20261018-r2
python3 VERIFY.py
```

`VERIFY.py`检查全部文件的路径、大小、SHA以及是否有额外文件。归档中的`SHA256SUMS`也可用于标准工具核验。校验成功后再进行构建或评测，运行产物放在归档目录之外。

现场如果更换代码、模型、缓存或配置，应声明相对于本快照的变化。
