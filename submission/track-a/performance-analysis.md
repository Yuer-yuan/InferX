# 性能、资源与优化取舍

当前100版本在14个客户端并发请求、13个引擎活动槽配置下，两轮满足延迟目标。请求可以在准入阶段等待，但等待必须计入首字延迟；容量选择依据是TTFT与TPOT同时达标。以下将当前配置取舍与历史算子、调度及资源诊断分开。

## 14并发请求：准入等待与生成延迟的取舍

三轮均使用100版本的同一二进制与源码、K3、KV121856、上下文65536、冷8192输入/512输出、128正式+8预热+1探针、相同完整payload（含salt），零错误。节点与引擎活动上限如下，吞吐沿用原profiling时间窗；P95从原128条请求独立复算。

| 节点 / 轮次 | 客户端并发 | 引擎活动上限 | TTFT P95 / s | TPOT P95 / ms | 总吞吐 / tok/s | 联合SLO |
|---|---:|---:|---:|---:|---:|---|
| ucloud1 / r1 | 14 | 14 | 9.724916 | 41.022342 | 306.063528 | 未通过 |
| ucloud3 / r1 | 14 | 13 | 9.893643 | 39.929242 | 306.038803 | 通过 |
| ucloud3 / r2 | 14 | 13 | 9.821661 | 39.141798 | 309.951742 | 通过 |

**选择理由：**保留14个未完成客户端请求，将引擎同时活动请求限制为13，其余请求等待空闲活动槽。活动上限约束内存中的请求状态及参与调度的请求数，不意味着GPU每轮总是处理13个请求。控制活动集合旨在减轻Prefill/Decode和批处理竞争，代价是消耗首字延迟预算。

**排队没有被排除：**客户端TTFT从发起请求计时，包含准入等待、Prefill及首个Token产生；TPOT按`(E2E − TTFT) / (512 − 1)`统计首个Token之后的生成阶段。允许排队的条件是含等待的TTFT P95仍不超过10s，同时TPOT P95不超过40ms。把等待移至准入阶段不自动提高SLO容量。

**结论范围：**引擎14在ucloud1的单轮TPOT超限，13个活动槽在ucloud3两轮联合达标，因此当前选择后者作为14并发服务配置。两组来自不同GPU节点，未隔离准入上限的单独影响；不报告净增益百分比，不将“预期减少竞争”当作已完成消融证明。首轮TPOT余量仅0.070758ms，两轮通过不构成长期稳定容量保证。其他页面的C15/C16失败结果保持不变。

原始来源、完整配置与逐文件哈希见[selected-campaign.json](data/selected-campaign.json)及[固定来源登记](data/selected-campaign-sources.json)；`build_selected_campaign.py`进行CPU审计，`cost_charts.py`导出双语矢量图。历史同机16客户端/16或12活动槽对照见后文，它提供准入等待代价的独立观察，不与100版本混算收益。

## 新增诊断：Prefill对Decode的直接干扰证据

085版本的同机C16/K5诊断使用原128个正式请求、8预热和1探针，冷输入8192/输出512。启用既有scheduler trace后，128/128正式请求通过绝对首字/结束时间唯一关联到执行器；550个Prefill和1275个Decode区间没有重叠。独立复算保留原128请求P95的两个贡献者及0.35/0.65插值权重，没有对子集重新取P95。

| TPOT P95组成 | ms/token | 占比 |
|---|---:|---:|
| 其他请求的Prefill | 31.9341 | 64.65% |
| 包含当前请求的共享Decode | 17.4523 | 35.33% |
| 不包含当前请求的Decode | 0 | 0% |
| 主机控制/未记录余项 | 0.0117 | 0.02% |
| 合计 | 49.3980 | 100% |

每个8192-token请求自身Prefill的中位耗时为0.90893秒。真实尾请求session_000052从首字到结束约25.667秒，其中其他请求Prefill约17.278秒、共享Decode约8.384秒。该轮尾延迟主要被其他请求的Prefill执行占据，支持同时缩短Prefill服务成本并控制其插入时机。

按当前归因做延迟预算：若保持Decode分量、余项和P95贡献者不变，40ms目标只给Prefill留下22.536ms/token，相比当前31.934ms需要减少约29.4%。这是固定其他项的算术预算，用于说明Prefill服务成本距离目标还有多远；不是对优化后P95的预测，也不能据此推出GPU利用率。

这是**同步后的executor主机墙钟区间**，不是GPU kernel独占时间、SM占用率或硬件效率测量；不能据此宣称Prefill硬件利用率低。traceON改变时序，因此49.398ms只用于诊断，不替代085正式traceOFF的50.189ms，也不直接归因到另一版本。其余条件固定时需要多少收益属于条件预算，不是实测预测。

来源：[诊断报告](../../results/three-node-optimization/ucloud1/verify-w2-gate-085/trace-r1/REPORT.md)、[原始时序](../../results/three-node-optimization/ucloud1/verify-w2-gate-085/trace-r1/delivery/control/scheduler.jsonl)、[完整归因](../../results/three-node-optimization/ucloud1/verify-w2-gate-085/trace-r1/attribution.json)。[展示用独立复算与来源登记](data/cost-evidence.json)由`build_cost_evidence.py`生成，原始证据不被改写。

## 同轮serving显存：173秒窗口

同一085诊断轮，正式请求窗口为UTC 2026-10-17 18:13:13.906至18:16:07.241，长173.335秒。GPU UUID与引擎记录一致，GPU日志时间与原生startup epoch核对一致；约1Hz的173个采样均为32738MiB，即31.9707GiB。

| 构成 | GiB | 依据 |
|---|---:|---|
| 权重和量化缓存 | 21.53296 | 原权重7.54149；FFN S4 8.09326、输入S4 2.45972、输出S4 0.69923、Verify W2 2.73926；同轮加载字节记录 |
| KV payload | 4.65875 | 原生`kv_payload_bytes`，包含模型与MTP KV |
| 非KV持久序列状态 | 4.90949 | `sequence.capacity_bytes − kv_payload_bytes`，含GDN、回放及其他序列缓冲；不把差值全称为GDN |
| 工作区和实际Graph | 0.47182 | 工作区0.33705 + Graph实际0.13477（138MiB） |
| 未细分余项 | 0.39768 | 设备采样总量减以上已核对分配；不强行归入某个池 |
| 合计 | 31.97070 | 同轮GPU采样总量 |

组件是同轮startup的固定分配展开；动态总量来自GPU采样。这不是逐次allocation追踪，图中组件保持常数不表示额外测得了各池的实时曲线。5GiB Graph allowance为预算，不能当实际占用；runtime reservation包含其他项，不能重复相加。启动后CUDA报告可用15.43140GiB。采样间的瞬时峰值仍可能漏采。

来源：[GPU采样CSV](../../results/three-node-optimization/ucloud1/verify-w2-gate-085/trace-r1/delivery/control/C16-gpu.csv)、[原生启动/请求记录](../../results/three-node-optimization/ucloud1/verify-w2-gate-085/trace-r1/delivery/raw/candidate/requests.jsonl)、[各缓存加载及状态池日志](../../results/three-node-optimization/ucloud1/verify-w2-gate-085/trace-r1/delivery/raw/candidate/server.log)。

## 调度选择：同一计算路径下的延迟交换

085、086和088的正式C16测试使用同一GPU、完全相同的137个payload（含salt）、K5/KV139264，128正式+8预热+1探针，零错误/全冷输入，traceOFF；各为一次顺序运行。086和088分别继承085，不是累计叠加。展示数据独立核对逐请求P95、节点、并发、K、KV和输入。

| 调度 | TTFT P95 / s | TPOT P95 / ms | tok/s | 联合SLO |
|---|---:|---:|---:|---|
| 085 原策略 | 10.00548 | 50.18877 | 375.65448 | 未通过 |
| 086 放松PF硬边界 | 13.28037 | 45.52598 | 377.76598 | 未通过 |
| 088 保留硬边界并计入PF债务 | 11.70530 | 48.22187 | 381.95400 | 未通过 |

086降低生成等待但显著延长首字等待。088计算队列中在Decode截止前必须完成的Prefill服务下界，保留原PF硬边界；执行当前块后的预算扣除`max(已知PF债务, 当前PF块)`，避免重复计算当前块。结果仍未联合达标。选择应围绕TTFT与TPOT共同约束，继续降低Prefill执行成本，并以队列与逐请求进度调节P/D顺序。不能仅按吞吐或单项TPOT选策略。

来源：[085](../../results/three-node-optimization/ucloud1/verify-w2-gate-085/C16-r1/summary.json)、[086](../../results/three-node-optimization/ucloud1/reachable-prefill-queue-086/C16-r1/summary.json)、[088](../../results/three-node-optimization/ucloud1/known-prefill-debt-088/C16-r1/summary.json)；完整binary/source及raw文件哈希见[本节登记](data/cost-evidence.json)。

## 总吞吐与用户等待是两个目标

同一输出投影量化版从12增加到16并发，总吞吐由284.75升至307.15 token/s，生成P95却从37.34升至51.70毫秒。两轮在同一节点执行，KV随并发调整；这个结果说明容量配置本身有取舍，不能当作单算子对照。

已注册的32并发最高总吞吐为递推状态融合版的375.71 token/s，但首字23.09秒、生成96.32毫秒，均未通过延迟要求。对外说明“能多快”时，需要同时给出负载、并发和延迟。

## 对照一：减少活动请求，排队变长

固定16个客户端请求，采用同一递推状态融合版本、同一节点、相同请求与KV预算，只把引擎活动容量从16降为12。

| 活动容量 | 首字P95 / 秒 | 生成P95 / 毫秒 | 总输出 token/s |
|---:|---:|---:|---:|
| 16 | 9.856 | 48.784 | 311.768 |
| 12 | 14.426 | 43.720 | 306.842 |

生成P95下降10.38%，首字P95增加46.37%，吞吐下降1.58%。两配置均未通过。这与“生成阶段竞争减少、更多请求先排队”的预期相符，说明单独降低活动容量不足以解决两项延迟问题。

两轮均128正式+8预热、另1探测，137个实际payload去掉cache_salt后相同，预热/正式分组也相同。它是一次顺序对照，未证明跨轮稳定收益。[完整报告](../../results/three-node-optimization/ucloud3/gdn-fusion-client16-native12-001/REPORT.md)。

## 对照二：生成进度调度改善尾部，尚差一段距离

生成进度调度改进保留模型算术和Graph，使用每请求提交产出修正剩余工作估计。同节点、相同负载的一次顺序对照如下：

| 16并发方案 | 首字P95 / 秒 | 生成P95 / 毫秒 | 总输出 token/s |
|---|---:|---:|---:|
| 递推状态融合版 | 9.856 | 48.784 | 311.768 |
| 增加逐请求进度估计 | 9.838 | 45.745 | 315.734 |

生成P95降低约6.23%，吞吐提高约1.27%，但仍未达到40毫秒。该结果支持继续研究调度，尚不足以宣布16并发达标。[完整报告](../../results/three-node-optimization/ucloud3/request-yield-max-C16-official128-recovery002/REPORT.md)。

另一节点的混合提议方案从52.460降到48.415毫秒，吞吐从344.055升到346.088 token/s，但首字由9.898升至10.070秒。两轮采用target K5/MTP proposal3组合，不能与上表混为单一优化路径；其报告和配置见[实验索引](experiment-index.md)。

## 对照三：局部执行更快，整轮尾延迟可能更差

| 同机顺序对照 | 总吞吐变化 | 生成P95变化 | 结论 |
|---|---:|---:|---|
| down投影实现调整，16并发 | 307.148→311.478 | 51.703→54.113 ms | 吞吐略增，尾延迟恶化 |
| GDN状态融合，32并发 | 374.046→375.714 | 92.109→96.319 ms | 平均生成执行改善，尾部仍变差 |
| 提示词查找优先，16并发 | 311.768→319.120 | 48.784→53.891 ms | 更多输出未转化为延迟达标 |

对应报告：[down投影](../../results/three-node-optimization/ucloud3/down053-C16-official128-001/REPORT.md)、[状态融合](../../results/three-node-optimization/ucloud3/gdn-fusion-C32-official128-001/REPORT.md)、[提示词查找](../../results/three-node-optimization/ucloud3/prompt-lookup-N5-C16-official128-recovery002/REPORT.md)。这些结果均保留，未据局部速度自动替换主性能版本。

状态融合对照中，native记录的共享Decode参与时间每提交token均值从22.576降到21.271毫秒，109/128配对请求下降。但这些计时包含批执行参与，不能相加成每个请求独占GPU时间；剩余等待也不能在缺少物理trace时全部归为Prefill。

## 为什么没有继续采用部分路线

| 路线 | 实测观察 | 本次选择 |
|---|---|---|
| 2bit权重算子 | 小形状T64/T128约1.42/1.37倍；T2048耗时反增10.63% | 保留小形状研究，未接入大预填充 |
| 2bit全层拟合 | 留出误差更大；参考含固定4bit down路径 | 保留重建诊断，未作为完整模型精度结论 |
| FP16递推状态 | 局部Graph执行约230.40→117.76微秒 | 8192步慢衰减精度检查失败，未推广 |
| 固定半宽低秩 | 训练拟合改善，留出误差显著恶化 | 停止该候选 |
| 高批注意力分片 | B16/T4约1.167倍、B32/T4约1.221倍 | 未同时达到原筛选要求的1.2倍 |
| FA2预填充 | Graph几何平均约1.010倍；原数值检查一例两臂超界 | 未达到速度/数值晋级条件 |

2bit例子说明：位数更低还需要合适的算子、形状和数据路径，减少存储不自动加速。以上算子结果也不代表整模型质量或端到端收益。证据：[W2速度](../../results/three-node-optimization/ucloud1/w2a4-gate-register-probe-066/RESULT.md)、[W2误差](../../results/three-node-optimization/ucloud2/true-int2-full-L31-GPTQ-063/REPORT.md)、[历史机制记录](../../docs/reports/2026-10-16-three-node-optimization.md)、[注意力筛选](../../results/three-node-optimization/ucloud3/gqa-highbatch-split-probe-001/REPORT.md)、[FA2筛选](../../results/three-node-optimization/ucloud3/fa2-head256-prefill-exploratory004/REPORT.md)。


## 116新量化与新二进制的独立复验

116实际重新拟合239个矩阵／478个服务文件，三类产物全部SHA及大小与原selected478相同；另从c976源码正常构建e919f180…，在新缓存上独立测量。其客户端14／活动13、KV121856、普通K3配置，原128测量＋8预热＋1探测，冷8192／512，TTFT P95为9.86221024060s、TPOT P95为39.42860192427ms，吞吐305.53286821935 tok/s；128个测量请求成功，错误0，联合SLO通过。

吞吐使用原profiling观察窗214.497380861s（整数end_ns−start_ns），输出65536 Token；不将另一benchmark_duration字段214.49216998400001s换作分母。所有初始测量和准入等待保留，性能服务正常回收。来源：[原始汇总](../../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/fresh-C14-raw-001/deployment116/performance-C14/summary.json)、[逐请求数据](../../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/fresh-C14-raw-001/deployment116/performance-C14/c14/profile_export.jsonl)、[独立复算](../records/typst/clean-reproduction/fresh-performance-audit.json)。

这是一轮独立复验，不能由e919、664与历史e4ed之间的数值差异推断某项优化收益，也不能据此建立长期稳定上限。独立原100题质量已完成：strict85/100、loose88/100，指令95/111与99/111，0错误；key50同时为空final及32768截断，全部100题计分且经原verifier独审，服务正常回收。性能与精度分别使用KV121856和532480；完整终态见[复验指南](quantization-reproduction.md#116新二进制的原100题质量)。

## 资源与成本

每轮使用一张48GB RTX 4090，实际显存49,140MiB，功率上限450W。研发使用不同节点分别实验，没有把三张GPU合并计算一份成绩。

运行时显存包含权重、KV、递推状态、工作区和CUDA Graph。状态融合增加约103MB持久状态；模型文件和缓存的磁盘大小也不等于显存峰值。具体启动内存字段保留在[结果JSON](data/results.json)及原始server_start日志。

GPU采样用于观察资源变化；一秒采样不能保证捕获连续峰值，旧轮还存在采样缺口。当前没有完整能耗积分、租用账单或电价，因此不报告节能百分比、平均功率或每token成本。450W仅是上限。

## 下一步

优先减少16并发的生成尾延迟，结合实际GPU时间线区分算子执行、批次竞争和调度等待。随后固定同一版本与配置完成性能和IFBench联合验证，并增加不同seed与重复轮次。当前冻结100已在客户端14/引擎13下两轮通过，完整源码重建664另有一次通过；进一步验证重复性和更高并发的尾延迟。
