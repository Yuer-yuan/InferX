# EB+ 适配：校准与完整模型验证完成，联合性能待比较

目标是保留公开实现的 EB/MB crossover、CFR/IFR 阈值和三阶段调度，接到已通过完整 27B/128K 验证的 vLLM 0.31 引擎。当前已完成源码适配、本机分离代价与独立MB代价校准，原生／强制EB／自动EB+三种模式均通过完整模型检查：118条严格工具请求、3次代码检查，含每组4条24K预热与完整冷长／续接长输入。正式混合性能结论仍待完成。

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

独立MB网格现已完成54点、126个测量请求，另有24个预热请求。每个测量请求的输入及输出长度均逐项吻合；从原始点重新计算，精确重建作者倒数最佳总token吞吐拟合。三个比例下最佳吞吐分别61.731／108.358／426.414总token/s，对应decode-heavy／balanced／prefill-heavy，均为4并发。得到 `a=0.000394394, b=0.017524766, c=0.000287620` 秒/token。[完整网格及拟合](../../results/ebplus-validation/calibration/mb-grid-v1/mb-grid-result.json)。校准域为输出占总token比例1/9至8/9；例如输入24557、输出64时比例约0.0026，属于域外推断，不能把三点拟合的零残差当作该场景准确性证明。

贪心版v4也已完成独立200条GPU事件计时，参数为 `alpha_p=0, beta_p=0.000370452, alpha_d=0.094891817, beta_d=0.001281345`；R²为0.99726／0.87821。主要代价与v3接近，采样对齐保证了部署一致性，未证实显著提速。[v4原始记录](../../results/ebplus-validation/calibration/hardware-v4/hardware-result.json)。

MB测量结束后，控制输出管道断开暴露了恢复日志处理缺陷：服务已健康并写出新身份，但打印异常被当作健康检查失败，双manifest留下旧PID。独立核对新PID、完整原argv、256K上下文、空闲队列和GPU0停止后，只修复登记，未重启服务；原failed campaign和54点结果均保留。[修复登记证据](../../results/ebplus-validation/recovery/mb-registration-v1/verification.json)。真实关闭管道回归先失败，再通过；状态事件改为非致命的无缓冲输出，后续任务改为HDD文件接收日志。

第一版模型检查另因当前引擎不公开`/server_info`诊断接口而在推理检查前停止，原服务已自动恢复。候选已在原生Scheduler初始化时独立记录真实配置和混合缓存布局；新版本对404保留该实测证据，其他HTTP错误仍中止。未启用额外管理接口，也未将诊断路由缺失作为SOTA效能失败。新版本远端完整CPU检查299项通过、9项专用检查跳过；相同核心的9项真实原生合约已另行通过。

最新完整模型验证已完成并独立复核原始SSE：原生、强制EB、自动EB+三组均通过代码、4条24K预热、冷长131049和续接131038 token工具检查；自动组额外完成100条真实工具请求。总计118条工具请求、3次代码门槛全部有效。实际相同的C4／98块混合池／1568块大小／FP8 KV／BF16卷积+FP32 GDN／MTP0／eager布局逐字段一致。[完整审计](../../results/ebplus-validation/qualification/model-audit-v3.json)、[原始数据包](../../results/ebplus-validation/qualification/model-gates-v3.tar.gz)、[逐文件哈希](../../results/ebplus-validation/qualification/model-gates-v3-inventory.json)。

实测强制EB执行245次prefill与727次decode调度；自动模式在100次完成时初始化并实际评估crossover，选择保留MB，最终107次完成、1次评估。该结果证明机制执行过，尚不证明保留MB对长短混合负载最优。零等待关停曾在写出策略证据前强杀EngineCore；统一使用30秒等待后，证据成功写出，原256K/MTP7服务和双manifest恢复并再次核对。关停设置不改变推理算法。

另修复中途观测计数：作者更新事务先把100加进总量、再评估、最后清空批次，旧快照在评估中会暂记200，最终关停记录107原本正确。真实当前引擎回归先观察200≠100失败，再通过；新增决策输入记录便于复核成本判据。该修正仅涉及3个观测方法，33个作者方法逐字节相同，11个调度／分配／状态边界方法AST相同。[变更范围核对](../../results/ebplus-validation/qualification/observer-comparison-v3-v4.json)、[精确差异](../../results/ebplus-validation/qualification/observer-comparison-v3-v4.diff)。新诊断仍逐组重做完整输入语义门槛，不把源码等价当作混合效能证据。

当前完整CPU检查299项通过、10项专用检查跳过；真实vLLM环境10项专用合约已另行全部通过。[完整CPU](../../results/ebplus-validation/qualification/task2-final-cpu.log)。原服务PID3989983、原argv、262144窗口、空闲及GPU0停止均已独立核对；恢复原件与审计包含于上述数据包。
