# EB+ 适配：已完成一版硬件校准，联合性能待验证

目标是保留公开实现的 EB/MB crossover、CFR/IFR 阈值和三阶段调度，接到已通过完整 27B/128K 验证的 vLLM 0.31 引擎。当前已完成源码约束、作者策略执行、适配候选生成、真实当前引擎CPU合约和本机分离代价校准；独立MB网格正在运行，完整模型门槛和正式性能测量仍待完成。

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

Task1最终全量CPU检查285项通过、5项专用检查在通用环境跳过后已单独5/5通过；[日志](../../results/ebplus-validation/qualification/full-cpu-suite-v2.log)。当前原服务再次核验健康空闲、原argv及双manifest一致、GPU0停止；[终态](../../results/ebplus-validation/qualification/after-v2-service.json)。

本机分离校准已经实际装载27B INT4权重，在GPU1执行作者CUDA Event计时与拟合流程。每个几何点先5次预热、再20次测量，共200条原始计时；并未使用混合负载结果拟合。

| 实际几何 | GPU时间中位数（ms） |
|---|---:|
| Prefill 256 token | 106.497 |
| Prefill 512 token | 179.502 |
| Prefill 1024 token | 346.141 |
| Prefill 1568 token | 545.231 |
| Prefill 3136 token | 1165.458 |
| Decode 1请求 | 96.184 |
| Decode 2请求 | 99.540 |
| Decode 4请求 | 101.579 |

作者线性拟合得到 `alpha_p=0, beta_p=0.000370879, alpha_d=0.095164660, beta_d=0.001687077`，单位秒；prefill/decode的R²分别0.99748/0.89495。原始prefill截距为−0.01328秒，按作者原实现截为0；未隐藏负截距或调整原公式。Decode上下文仍按作者协议512 token；这些系数不能独自证明24K／128K历史的代价模型准确，也不能用来宣称达到带宽上限。[完整校准与未截断拟合](../../results/ebplus-validation/calibration/hardware-v3/hardware-result.json)。

适配中保留两个失败记录。v1已装载模型，但旧`EngineCoreRequest.eos_token_id`字段被当前接口拒绝；v2修复字段后，发现请求2048 token实际只执行1568，主动终止了自己的controller。两个尝试均恢复原服务，均无SOTA效能分数。v3预先改用与实际1568-token检查点对齐的几何，所有200条实际执行量逐条匹配。

作者关闭prefix caching以避免校准命中，但当前混合引擎会因此把Mamba模式强制改为`none`。本适配保留`align`并给每个请求独立`cache_salt`，避免跨请求命中，同时测量真实部署的状态路径。原生KV组包含BF16卷积状态及FP32 GDN状态；不能把它写成先前SGLang的16槽全BF16布局。正式因果对照将固定同一vLLM实际98块混合池；引擎启动日志的143001-token估计与`98×1568`不是同一种容量定义，不能忽略状态页占用。[运行时原件](../../results/ebplus-validation/calibration/hardware-v3/hardware-result.json)。

独立MB校准预先固定3个token预算、2个prefill cap、1/2/4同时请求及作者的3组输入输出比例，共54点；每点完整生成，使用总输入加输出token吞吐，分别选各比例最佳点后拟合倒数的二次式。三点恰好拟合不能作为泛化验证。MB拟合完成并恢复后，排队的CPU门槛、匹配采样校准和原生／强制EB／自动EB+完整模型门槛才会开始。自动模式会额外完成100个预热工具请求（重复冻结短历史，独立于正式波次），确认初始化与crossover实际执行；不得把25请求的MB冷启动算作完整EB+表现。

vLLM后端的相同冻结输入实际为短24557、冷长131049、续接131038 token。已有0.31原始探针与三轮记录确认了这些值，现新增固定后端合约；原SGLang精确计数保持原值，没有放宽输入门槛。后续新门槛还会逐条从SSE重建工具语义，不能只相信成功标志。

v3保留作者`SamplingParams`默认的随机采样，而当前部署与MB网格为贪心采样。这是明确的标定配置差异，不能直接把v3系数用于最终结论。已增加真实当前请求接口回归，先观察`temperature=1.0≠0.0`失败、再通过；v4将用贪心配置重复相同的200条CUDA Event流程，保留v3作诊断，不修改原始拟合公式。尚未取得成对GPU结果，不能把采样设置差异宣称为已证实的性能原因。

当前本地完整CPU检查297项通过，9项专用环境检查跳过；真实vLLM环境已单独通过9项原生合约，采样匹配回归也已单独通过。新的实机链路会在MB计时完全结束后再重复完整CPU和真实原生检查，然后按拥有者校验串行运行v4与模型门槛。
