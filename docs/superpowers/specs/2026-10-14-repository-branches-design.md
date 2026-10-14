# 单卡仓库补全：多会话候选预算验证

## 目标与授权

用户要求继续上一轮研究验证。上一轮 HumanEval 的 CPU 验证压力被证伪，但观察到候选并发提高 token 吞吐、增加任务解决时间。本轮验证这个现象在真实仓库测试及多会话竞争下是否仍成立，且是否被普通公平调度和反馈门控解决。接受否定或不确定结果，不预设论文机制。

沿用用户要求的当前 work-dir，在 `research/repository-branch-pilot` 分支执行，起点 `89b6461`；原分支和已完成结果保留。用户此前多次要求直接继续，按既有 inline 执行方式完成计划，不重复询问流程授权。以下实验取舍由执行者作出，并非用户指定参数。

## 选型与相关工作

选用公开 [ExecRepoBench](https://huggingface.co/datasets/CSJianYang/ExecRepoBench)，revision `fa61028ce495c9ceff58398b8a7c47b5ae9f5276`，实际 1164 条、50 个仓库。代码补全有明确位置，避免第一阶段同时混入 Agent 文件定位、工具选择能力；本轮不声称完整 SWE-bench Agent 性能。SWE-bench 官方流程依赖 Docker，当前用户不能使用 Docker；没有修改全局权限或扩大访问范围。

数据与仓库归档已通过 hf-mirror 获取并完整校验，位于 HDD `data/exec-repo-v1/`：JSONL 257233445 字节，SHA256 `b614124d827b5e7290e9f5e3e96522583a6bb9a4b0131ff38e35a7b29716c4c5`；repos.zip 1373686190 字节，SHA256 `46de2f73d6f3f2c18758c826bf6a07f42250ba455bb39179b34791b581a921bb`。

必须对照的已有思路：

- [Adaptive Generate–Rank–Verify](https://arxiv.org/abs/2605.17609)：动态生成与验证预算已有研究。
- [Autellix](https://arxiv.org/abs/2502.13965)、[SAGA](https://arxiv.org/abs/2605.00528)：程序/工作流级优先级、公平性和状态管理已有研究。
- [SpecScale，2026-09-30](https://arxiv.org/abs/2609.39334)：已研究推测搜索的剪枝、重复计算合并和延迟验证；其 GPU PRM 验证与本轮 CPU 仓库执行不同，但“分支过多”本身不是新发现。
- [Sample Count Is Not Enough，2026-09-16](https://arxiv.org/abs/2609.19499)：固定候选数下，串行/批量执行成本已有实测。

本轮的对照是通用机制的受控实现，不宣称复现上述完整系统。若普通机制已足够，就不包装成新策略。

## 数据与验证环境

优先仓库为 `more-itertools`、`sqlparse`、`json_repair`、`python-fastjsonschema`，均为 Python 库。只使用 `grammar-based: block`，屏蔽区对应源文件，禁止修改测试或执行入口。每条按原始 JSONL 行号作为稳定 ID，顺序由 seed `2026101402` 固定；选题在模型生成前完成。

先检查源文件等于 prefix + middle + suffix；屏蔽区前缀当前行只含缩进、后缀从换行开始，并且把 middle 替换为 `pass` 后整个文件仍能通过 AST 解析。原始代码必须通过完整仓库测试，`pass` 消融必须实际测试失败，不能只因语法错误被判失败。这样排除文档补全和没有测试覆盖的样本。每仓库选最先通过环境/覆盖检查的 6 题，按固定随机顺序分成 2 dev、4 eval，共 8 dev / 16 eval。所有筛除及原因留存。若某仓库不足 6 条，不看生成表现，由同一算法使用 `compressed_rtf` 作为后备；仍不足时完整报告可用题目数量，不伪装成原定网格。

依赖在 HDD 单独 venv 安装，冻结依赖版本，不修改现有 conda 环境。仓库包通过绝对仓库路径及 src 加入 sys.path，cwd 使用该候选 scratch，避免从已安装同名包执行“正确版本”。归档仅提取白名单仓库，拒绝路径穿越和 symlink；跳过旧缓存、字节码、.git、构建目录。保留原始源码，不更改生产库行为或测试断言来适配环境。

新的独立仓库 worker 使用 [Landlock](https://docs.kernel.org/userspace-api/landlock.html)（服务器 ABI 4）与 seccomp allowlist：只读候选仓库、Python 标准库/独立 venv 依赖，只有该候选的 scratch 可写；网络、进程创建/exec、ptrace、向其他进程发信号等均禁止。不允许读取其他用户文件。过滤器安装失败即基础设施错误，禁止降级裸执行。清除秘密环境变量及额外 FD，固定单个验证 CPU，限制地址空间 2 GiB、文件大小 8 MiB、输出 256 KiB、验证 wall time 30 秒并 kill/reap 整个进程组。仍只面向非对抗性基准输出，不是通用恶意代码服务。

运行完整 `tests/`：more-itertools、compressed_rtf、PTable 使用 unittest discovery，其余使用 pytest，关闭外部插件自动加载与缓存写入。记录收集/通过/失败/跳过计数，成功必须有测试运行，并符合该仓库 canonical 的收集数量和跳过数量上限，避免零测试成功。使用隔离目录运行正确参考、错误消融、文件/网络/进程访问及死循环检查。若已知正确版本在环境限制下失败，先排查环境，不能算模型失败。保留原始 evaluate_repo.py 及本轮测试入口的差异说明。

## 生成前校准修订

calibration-v1～v3 记录初始环境排查。more-itertools 需要线程，sqlparse 需要子进程，fastjsonschema 超过 2 GiB 且缺少 benchmark 插件；compressed_rtf 只有 5 条满足覆盖条件，不能凑足 6 题。没有删除测试、放宽资源上限或观察模型表现来挑题。生成前将后备顺序扩展为 xmltodict、PTable、algorithms、PyPattyrn、python-pinyin、boltons。pytest 日志定向 scratch，所有测试 cwd 也使用 scratch，使相对路径输出无需写入只读源码目录。完整校准 v4 最终选定 json_repair、xmltodict、PTable、algorithms，各 2 dev / 4 eval，维持 24 题及原有实验网格。消融资格还要求实际失败计数大于零，且收集数不变；结果清单 hash 见 calibration-v4/manifest.json。此兼容性筛选限制外推范围，不能代表任意企业仓库。

## 模型与提示

沿用同一已校验 Qwen3.8-27B UD-Q4_K_M 与 llama.cpp v0.6.0 / CUDA 12.6。GPU 0 旧服务保留；GPU 1 承担新试验，两轮均在 GPU 1 顺序完成。单卡 8 slots × 8192 context，总 ctx 65536，f16 KV，全部层 GPU offload；预估新增 KV 3 GiB，必须先实测可用容量及八路取消。若不合适，记录原因后收缩配置再冻结，不暗中 CPU offload。

提示要求补全 prefix/suffix 之间的代码，返回 JSON `{"code": "精确保留缩进与换行的补全文本"}`。使用 chat template、thinking 关闭，temperature 0.8、top_p 0.95、top_k 0、min_p 0、repeat_penalty 1、max_tokens 1024。解析 JSON（可容忍外层 JSON fence），不自动修改源码缩进；格式错误是候选失败。只有 prefix/suffix 和源代码上下文进入提示，middle_code、测试内容、验证输出不进入提示。本轮候选相互独立，不包含迭代修复。

按数据集提供的相关文件顺序选择非测试、非目标文件的上下文；初始 prefix 末 8000 字符、suffix 首 8000 字符、相关源码最多 6000 字符。通过引擎 tokenizer 检查 chat prompt ≤6000 token，超出则先减相关上下文、再成比例缩短 prefix/suffix。截取在行边界；算法固定，不按生成质量调参。记录实际提示 hash、token 数、保留范围。测试和 canonical 只在验证端使用。

## 调度与实验

每题最多 4 个候选，从预定到达时刻开始的绝对期限 120 秒，覆盖调度入队、生成、验证和反馈；deadline 后完成不算通过。全局最多 8 个活跃 HTTP 生成，两个验证 worker，分别绑定 CPU 36/37；引擎和客户端 CPU 28–35。取消任务时只取消它自身候选，不能清空其他会话；仅在 episode 结束后查询全部 slots 空闲，并计入 drain。

四个对照：

1. fixed1 / fixed2 / fixed4：每题活跃生成上限 1/2/4，生成完成即异步验证并补充，候选提交按 FIFO 争用全局 8 槽。
2. feedback-fair：每题同时最多一个未获得反馈的候选，验证失败后才允许下一候选；生成空槽按会话 round-robin 分配。它是保守的反馈门控加公平调度基线，不声称新机制，也不根据未来通过概率决策。

保守基线可能降低 GPU 批量吞吐，必须同时报告质量、期限内通过率、完成时间、实际 makespan、排队、生成 token 下界、验证反馈及取消。不能只报告减少 token 的百分比。

开发阶段先对 8 dev 题分别运行固定宽度 1/2/4，两轮、独立逐题执行（48 次）。按通过数优先、其次失败按 120 秒计入的平均解决时间、最后较小宽度选择固定基线。每题 seed 由 round、原始行号、候选序号组成，跨策略一致；不保证批处理下相同文本。

评估阶段将 16 eval 题用固定随机顺序划分为 4 个 episode，每个 episode 4 个不同仓库/任务，尽量每仓库一题。到达模式：burst `[0,0,0,0]` 与 stagger `[0,5,10,15]` 秒。这是公开、可复现的受控到达负载，不声称生产 trace。4 episodes × 2 到达模式 × 4 policies × 2 rounds = 64 episodes / 256 个任务结果。每轮 episode/policy/mode 顺序随机化；两轮不用同一随机采样 seed。清理完成后才切换 episode。热身排除；遥测每秒记录 GPU 显存、温度、功率和负载。

## 判定标准

主要比较反馈公平基线与 dev 选定的 fixed 基线，并同时展示所有固定宽度。分别对两种到达模式、两轮报告结果；保持 episode 内相关性，bootstrap 重采样 episode 模板、保留配对策略和两轮。4 个模板只能支持筛查，明确置信区间有限。

- 只有两种到达模式均出现同向变化、两轮总通过数不减少，且平均截断解决时间改善至少 20% 或 token 降低至少 30% 且时间不恶化超过 5%，才认为“普通反馈门控/公平调度已解释主要浪费”，不继续包装新机制。
- 若上述普通策略不能解决，且两轮均有候选/验证排队中位数占任务耗时至少 20%，或未采用 token 占比至少 50% 并伴随明确延迟恶化，才保留更复杂机制的研究动机。未采用 token 并非都可无损消除。
- 若每模式每轮任一主要对照成功不足 8/16，结论为质量/样本不足，不能把模型失败造成的排队当作已有策略失效证据。
- 不满足以上条件则记为混合或不支持，不根据评估结果临时改规则。候选可以全部失败，基础设施错误必须中止并单列。

本轮交付：可复现脚本、环境/数据/代码哈希、所有失败和排除原因、完整轻量事件、图表及中文研究判断。只有测量支持，才另行研究多任务候选预算、取消成本与状态保留的联合机制；本轮不改推理引擎内部，不绑定三级存储。
