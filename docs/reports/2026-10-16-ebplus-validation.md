# EB+ 修正版混合诊断：当前配置未改善联合表现

在同一张4090、27B INT4、C4、98块混合缓存池、MTP0、eager、4096 token预算及2048 prefill cap下，修正版原生／强制EB／作者推荐滞回1e-6的自动EB+均已完整完成冷长与续接长混合诊断。每组包含24条24557-token短请求和1条约131K-token长请求，150条计时工具请求全部通过；各组另有独立代码检查、4条预热和100条真实策略预热。自动组实际选择EB，并执行独占prefill和decode。

| 策略 | 冷长：短P95 / 长耗时（秒） | 续接：短P95 / 长耗时（秒） |
|---|---:|---:|
| 同配置原生 | 103.94 / 89.58 | 93.54 / 78.99 |
| 强制EB | 190.15 / 253.68 | 221.14 / 274.60 |
| 自动EB+，滞回1e-6 | 191.51 / 254.97 | 221.35 / 274.82 |

这些是每组一轮的诊断绝对值，尚无正式三轮或匹配单独基准，结论限定于这一适配与配置。不能据此宣称作者实现上限、所有SOTA的能力或已达到带宽下限。原生冷长／续接组分别发生3／2次抢占，强制EB为5／7次；现有token指标不累计重放GPU工作量，额外计算代价仍待核对。原始181文件完整留存，[原始包](../../results/ebplus-validation/comparison/diagnostic-v2/raw.tar.gz)、[严格导出](../../results/ebplus-validation/comparison/diagnostic-v2/export.json)、[逐文件哈希](../../results/ebplus-validation/comparison/diagnostic-v2/inventory.json)。两次独立导出完全相同，新增导出器也能逐字节重现旧结果。

下一步已预先声明关闭vLLM官方完整输入容量检查的原生／自动EB+对照，模型、池、分块与其他设置一致。该开关只改变准入检查，不表示在准入时物理预留整个请求。先在新设置下重新运行作者完整54点MB标定，再做4个完整混合诊断；相同模型与GPU kernel的200条分离事件标定保留。两个对照必须记录实际初始化后的开关值，并拒绝与其不一致的MB代价文件。[标定配置](../../configs/ebplus-admission-calibration.json)、[诊断配置](../../configs/ebplus-admission-diagnostic.json)。这是现成策略验证，尚未证明需要新机制。

新增观测回归在真实vLLM中先因缺失开关字段失败，再修复；14项真实原生合约全部通过，完整CPU测试322项通过、14项专用检查另行执行。[真实原生日志](../../results/ebplus-validation/comparison/admission-flag-v1/native-admission-green.log)、[完整CPU](../../results/ebplus-validation/comparison/admission-flag-v1/full-cpu-green.log)。33个作者方法和策略文件字节相同，全部调度类AST相同；仅运行时观测函数增加字段。[差异核对](../../results/ebplus-validation/comparison/admission-flag-v1/overlay-v5-v6-comparison.json)。上一轮结束后已独立核验原256K/MTP7服务、原argv、双manifest、空闲与GPU0停止；后续拥有者仍负责每次退出恢复。

另已确认更贴近延迟目标的QoServe/Niyama公开作者分支：[microsoft/sarathi-serve，niyama_asplos2026](https://github.com/microsoft/sarathi-serve/tree/niyama_asplos2026)，对应[正式公开工件](https://zenodo.org/records/18218177)。源码固定72339db5b120614033b317880ad3a05a5e4c9415并落于8876 HDD，已读调度、期限／优先级和分层预测器；其预测系数需要针对本机重新标定，当前尚无QoServe适配效能数据。最终研究判断仍须认真处理这类直接面向SLO的基线。

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

Task3已启动独立诊断，运行路径为HDD `ebplus-validation/raw/diagnostic-v1`，使用只增加观测的adapter-v4。预先声明三种运行配置：同引擎原生、作者IFR自动EB+、作者强制EB。每个冷长／续接长组重启引擎，依次完成代码检查、4条严格24K预热、100条实际冻结短工具请求，再发送24条短请求（0至46秒每2秒一条）和第16秒的完整128K长请求。所有配置采用相同历史；强制EB补充自动策略的另一条实际执行路径。零滞回条件用自动模式的真实决策输入独立复算，暂不当作第四组性能数据。诊断后固定三组，再做三轮全新混合波次与各自C4服务器上的串行客户端单独基准；当前尚无正式联合性能结论。

新导出器保持原SGLang审计接口不变，单独核验vLLM的24557／131049／131038精确输入、原始SSE工具语义、100条预热历史、实际混合缓存组／98块池／MTP0／eager／C4，以及校准和测量源码。自动模式的原公式由观测输入独立复算；保留MB属于合法选择。计数器包含未计分的预热，API缓存命中不是实际GPU工作量，均不据此推算物理带宽。

第一组自动模式的实际100次完成记录揭示了需要修正的参数尺度：原公式`LHS=6.26307e-5`、`RHS=5.59401e-5`，差值`6.69061e-6`秒／总token，默认滞回`1e-4`因此保留MB；相同观测输入下，`1e-6`会选择EB。[观测原件与独立复算](../../results/ebplus-validation/comparison/qualification/hysteresis-scale-diagnostic-v1.json)。作者[复现说明](https://github.com/weifang231/eb-vllm/blob/main/reproduce/REPRODUCE.md)明确建议无切换时尝试`1e-6`。将先补测这一作者已有参数，不能把未校正的默认设置作为SOTA上限或负面结论；原公式、实测系数、IFR、100次更新和内存限制均不变。

因此内部诊断上限从3组增为4组，原生／默认自动／强制EB原始数据继续保留；新自动参数组将另起目录，在前轮全部终态恢复后独立通过相同完整输入及实际EB执行检查。此前冻结的三组正式配置暂不启动，补测资格通过后再固定包含已调好EB+的正式配置，所有正式数据仍未产生。调整依据是实测决策尺度与作者指导，未用正式结果优化参数。当前完整CPU检查314项通过、10项专用检查跳过；原生调度核心与已实际通过的10项专用合约及adapter-v4保持一致。


第一轮混合诊断已保留为失败记录：原生与默认自动MB的4个波次完整完成，但强制EB在100条预热后、混合组仅完成6条短请求便停止推进。现场持续有排队请求、GPU为0%且EngineCore忙等。只向已核对身份的自有controller发送SIGINT，原256K/MTP7服务、原argv和双manifest恢复并独立核验；GPU0保持停止。[中断原始包](../../results/ebplus-validation/comparison/diagnostic-v1-interrupted.tar.gz)、[逐文件清单与排除说明](../../results/ebplus-validation/comparison/diagnostic-v1-debug/diagnostic-v1-inventory.json)。这组不产生SOTA效能分数。

真实原生CPU合约复现了同类边界：30720-token独立长请求在128块小型原生混合池中可单独完成；与3个2560-token解码请求并存时，原生MB继续解码，旧EB适配连续返回空调度。现代原生`full_sequence_must_fit`可在空闲比例仍高时拒绝完整请求，旧阶段机只看到空闲比例，因而看不到该准入压力。桥接层现将真实分配拒绝接到作者原有内存压力退让分支，在释放容量或队头改变后重试；不隐藏请求、不伪造缓存API，也不改变C4或作者数学。原生合约先观察两次调度总token仍为0的失败，再修复；13项真实原生CPU合约全部通过，包括完整完成、原生对照及无需ptrace的空调度快照。[RED](../../results/ebplus-validation/comparison/diagnostic-v1-debug/cpu-reproduction-red-v5.log)、[13项GREEN](../../results/ebplus-validation/comparison/diagnostic-v1-debug/native-full-green-v2.log)。

作者策略文件与33个原方法哈希保持相同；实际桥接阶段分派明确扩展了一个资源压力谓词，不能把它描述为所有上游阶段代码原样执行。精确AST比较验证其余原分支、现代分配与状态转换主体和原生控制均保持一致。[适配差异核验](../../results/ebplus-validation/comparison/diagnostic-v1-debug/capacity-boundary-comparison-v4-v5.json)、[adapter-v5](../../results/ebplus-validation/reproduction/adapter-v5/manifest.json)。完整CPU315项通过，13项专用检查另在真实环境全部通过。[完整CPU](../../results/ebplus-validation/comparison/diagnostic-v1-debug/full-cpu-v12.log)。

新的HDD `raw/diagnostic-v2` 已启动，使用不可变`code-task3-v12`与`adapter-v5`，重新检查原生、强制EB、作者推荐`1e-6`滞回的自动EB+，每种冷长／续接长各一组。新的完整模型及实际阶段证据通过后才冻结正式三轮与匹配单独基准。此前中断不是作者性能上限，准入映射修复也不是研究创新。
