# 简单调度与开源 PD 机制的验证

实验进行中；以下筛选结果不能替代正式三轮重复。当前证据已表明简单调度有很大收益，因此不能预设“简单方法无效”或已经形成研究空白。开源机制的移植失败与方法本身的效果分开记录。

## 负载与边界

单张 RTX4090、既有27B INT4权重，实际架构为Qwen3_5ForConditionalGeneration；64层含48层GDN和16层完整Attention。SGLang0.5.19、C4、MTP0、FP8 Attention KV、BF16循环状态。使用冻结的约24K短历史、约128K长输入，48秒内短请求0.5RPS，长请求在第16秒加入。每组独立重启、代码检查、四路24K严格工具预热；混合组必须完整完成长输入和工具校验。

5秒短请求P95、长完成代价不超过25%、吞吐损失不超过10%是本研究的判断标准，尚非测得的企业SLA。每波只有24个短请求，三轮独立重启用于检查重复性，不能据此宣称生产P99或真实流量发生频率。

## 简单策略筛选

第一组匹配资源为152000 KV token、12个BF16循环状态槽。所有策略保持总512-token轮预算。queue128/256在有等待请求时限制正在进行的长块，并将其余预算交还原生准入；responsive128/64还在已有活跃decode时保持小块。不会绕过完整候选的容量检查，也不允许同时存在第二个未完成prefill块。

| 单轮筛选策略 | 纯短P95(s) | 冷长混合短P95(s) | 续接模板短P95(s) |
|---|---:|---:|---:|
| 原策略 |3.147|97.002|71.955|
| queue128 |3.049|12.483|25.540|
| queue256 |2.987|38.494|51.670|
| responsive64 |3.093|10.436|14.858|

queue128明显改善尾延迟，但仍高于5秒。实际源码还存在准入计数的问题：一个正在继续prefill的请求已持有请求行，却在总候选数与空闲行比较时再次占用了空闲计数。对已持有行计入credit、同时保留C4硬上限后，真正放行更多请求；12状态槽随即出现Mamba缓存分配不足。这些崩溃不是性能结果。

因此第二组静态资源调整为140000 KV token、16循环状态槽，其余精度、上下文、C4和显存比例相同。KV仍容纳完整128K；所有比较使用同资源的新对照。它属于容量配置与准入修正，没有引入新的KV管理方法。

目前已完成的冷混合筛选：原策略短P95=97.024秒、长88.380秒；responsive128+owned-row credit短P95=5.263秒、长101.670秒，长代价约15.0%，两组均完整工具正确。后续续接、更小预算与正式重复待补齐。

## 开源机制的实际适配

适配的是SGLang内置[PD-Multiplexing](https://github.com/sgl-project/sglang/issues/10813)，保留其SM分区、Green Context流和层间交错调度。增加当前Dense Qwen的split-prefill入口，层间保留hidden/residual，最终norm和logits只执行一次；仅支持TP1/PP1、纯文本、MTP0。每个overlay独立复制于HDD，并记录原文件、修改文件和完整源码SHA。

原路径要求关闭token分块和普通overlap；在最小131328 KV池时，完整输入prefill仍未通过四路24K预热的显存门槛。另行标注的token512分块组合复用正常stash/filter转移，但第一次完整模型预热遇到FlashInfer decode计划中的CUDA invalid device context。不能据此说论文无效。

在16状态槽、131328 KV池、每个FlashInfer workspace128MiB的完整模型A/B中，只切换显式Green Context绑定。不绑定再次在四路预热时出现同一错误；绑定路径通过代码检查、四路24K工具预热和完整128K工具请求，测量段约87.750秒，0错误。独立小数值探针的三种模式都通过，所以数值小探针本身没有复现根因；完整模型A/B才建立本配置中的资格差异。它仍不能替代混合负载效果或一般模型质量评估。见[完整资格证据与输入SHA](../../results/pd-strategies/compatibility/green-context-model-qualification.json)。所有失败和未完成组保留，未纳入有效吞吐。

独立资格检查通过后，8192默认层预算的PD+responsive128在真实冷混合波次中停滞：9个短请求完成后无进展，GPU利用率100%、显存带宽利用率0%，最后原生事件是在长请求旁放行新短请求。该组保留为不完整失败，不能产生有效PD吞吐；拥有该实验的控制器收到SIGTERM后恢复了原服务。原生指标不能单独定位GPU锁、kernel或依赖的根因。

源码公式forward_count=split_forward_token_budget/extend_tokens说明，128-token小块配8192预算时可一次跑完64层。另用2048预算限制活跃P/D阶段每段最多16层。带定时Python栈输出的首次诊断在代码门槛即SIGSEGV退出，不能据此定位原混合停滞。移除该诊断后的干净配置通过冷混合波次：短P95=4.688秒，长119.163秒，25/25工具正确、完整128K；仅一轮，还需要长任务代价和重复验证。

同时发现量化实现中的独立并发风险：当前compressed-tensors WNA16的每个scheme持有一份Marlin workspace，并将它原样传给P/D流；实际CUDA模板把它解释为全局barrier锁。新增显式可选的每流独立锁区，保持权重、精度和实际KV池不变，准备三个SM分区资格检查。CPU回归证明workspace按流和scheme独立、同流复用、分配失败可重试。该源码风险与当前停滞的因果关系尚未建立，干净l16未隔离锁区也曾完成一轮。见[Marlin源码审计和哈希](../../results/pd-strategies/compatibility/marlin-concurrency-audit.json)。

## 重复结果与可复现证据

PD通过后，正式三路对照统一使用131328 KV、16状态槽、128MiB workspace、eager extra_buffer和普通overlap关闭；原策略和普通128策略保留正常mixed batching，PD使用其独立P/D流。三路均相同精度、上下文与C4。比较对象为原策略、responsive128+owned-row credit、以及PD96/32SM+同样的responsive128和credit。此前140000/lazy筛选是另一个资源配置，不混算收益。正式前尚未开始的三条state16队列已取消，没有产生正式样本。

待正式重复完成后由原始响应重新导出；原始数据位于服务器HDD的pd-baselines/raw/pd-strategies-*。代码、配置、失败和正式测量分别保留，不用选择轮次冒充三轮验证。
