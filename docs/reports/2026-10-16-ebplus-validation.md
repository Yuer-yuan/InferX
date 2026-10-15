# EB+ 适配进度：尚无性能结论

目标是保留公开实现的 EB/MB crossover、CFR/IFR 阈值和三阶段调度，接到已通过完整 27B/128K 验证的 vLLM 0.31 引擎。当前已完成源码约束、作者策略执行、适配候选生成和真实当前引擎CPU合约；本机校准、完整模型门槛和正式性能测量仍待完成。

源码固定为 `weifang231/eb-vllm@5b6a619b732b8eeaea93c8c726dbe13948920599`。作者调度器、校准器和实际安装的现代调度器已逐字节固定到 [源码胶囊](../../results/ebplus-validation/source-capsule-v1.tar.gz)，包含作者许可证。服务器原件、后续环境、构建、校准和原始数据均放 HDD。

| 旧实现与当前引擎的差异 | 候选适配方式 |
|---|---|
| `compute_encoder_budget` 已变更 | 使用当前原生初始化，不导入或覆盖旧引擎 |
| 当前增加持有 KV 的等待队列、延后队列和流式请求状态 | 保留完整原生队列；在运行和准入扫描中增加阶段资格检查 |
| 分配接口增加 Mamba 对齐、prefix replay、CoW 和检查点输出 | 调用原生分配与输出路径，不复制旧版直接分配代码 |
| 独占阶段需暂停另一阶段的执行 | 暂停执行资格，所有请求仍计入原生全局 C4 限制并持有其状态 |
| 同一步中多个 prefill 完成影响 refill 目标 | 在准入检查中计入本步计划完成量，在原生更新后更新策略计数 |
| 旧 preemption 直接重置部分字段 | 使用现代原生 preemption 保留新增状态清理语义 |
| 作者自适应 N 的启发式下界为 16 | 在边界限制到硬件规定的 C4；不扩充内存池 |
| 当前支持异步并行批次及 DP throttle | 当前候选显式限定同步、单 GPU；后续须与同配置原生对照重新核验 |

作者数学函数、在线更新、crossover 和阶段转换函数从固定源码提取。原生独占扫描也从实际安装源码提取，仅加入资格、准入和 prefill 不驱逐 decoder 的限制。原生 MB 路径保持调用当前调度器。候选通过源码 SHA 拒绝未知版本，输出目录存在即拒绝覆盖；这不是已完成的部署声明。

本地标准库测试已观察缺失功能失败，再通过 10 项测试，覆盖源码/校准器变化拒绝、旧目录保护、产物与输入字节绑定、阶段资格、本步 refill 完成计数，以及实际作者数学和 crossover。随后在实际vLLM0.31环境初始化当前Scheduler，并使用真实Request、ModelRunnerOutput和小型原生FullAttentionSpec+MambaSpec逻辑缓存池，5项合约全部通过：全局C4、暂停prefill时保留所有权、真实分配器异常、完成请求驱动在线切换、原生preemption/resume。这是接口与状态合约，尚未装载模型权重验证GDN数值。

实际作者代码显示，采用作者建议的IFR模式时，参数默认每100个完成请求更新一次，hazard最少30个样本；25请求波次中仍处于MB冷启动。若完全保留未指定的direct模式且没有比例/阈值覆盖，ratio可保持0，作者也明确警告crossover不会触发。本地测试确认第 100 次完成会运行校准代价驱动的 crossover。此处单元测试的系数仅用于分支验证，绝不用于 GPU 部署或性能结论。正式实验需本机校准，并分别标明默认冷启动、充分独立预热或响应更新率消融，不能把未触发机制算作 SOTA 失效。

MorphX正式测量与单独基准已终态恢复后，才在HDD隔离代码树运行CPU检查。没有并行GPU controller，原256K/MTP7服务和双manifest保持恢复、GPU0停止。实际preemption的已计算token统计另加精确回归，先观察5000≠2048失败，再保存重置前计数修正，5项真实原生合约全部通过。

[兼容性审计](../../results/ebplus-validation/compatibility-audit.json)记录边界、33个作者方法哈希、生成模块哈希及限定；[准备清单](../../results/ebplus-validation/qualification/preparation-v2.json)对应HDD `ebplus-validation/overlays/adapter-v2`。下一步必须使用本模型/GPU测量硬件与MB代价，才可通过完整代码/24K/128K门槛并比较效能。
