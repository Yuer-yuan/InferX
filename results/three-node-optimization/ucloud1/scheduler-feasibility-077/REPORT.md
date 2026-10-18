# 077：限时调度可行性复核

**结论：没有发现值得立即启动 GPU、且有证据支持能同时保住 TTFT 的最小调度改动。** 不重做累计/逐请求 EWMA yield，不扫 cap、K、chunk 或 margin。此结论针对眼下的小改动，不是“所有调度策略不可能”的数学证明。生产源、二进制均未改变，无编译或 GPU 请求。

- 当前并非整8192不可打断：`concurrent_executor.h:1356–1375` 每个 Prefill unit 都重新构造 decode membership/选择；`program_impl.h:1917–1957` 每次只推进最多2048 tokens，然后返回。块末 `text_context_impl.h:1309` 实际 synchronize，已是安全的调度边界。
- **2048块内部不可恢复**：`text_context_impl.h:990–1035` 在一个函数内执行64层，`1107` reset共享work；没有层cursor/独立Prefill和Verify work/io状态。插入一次完整decode会复用这些buffer/control，不能只在循环中加callback。可靠的层间抢占需保留Prefill hidden/位置/arena生命周期和控制恢复，与“立即落地小调度patch”不符。
- 完整队列确已计入：`concurrent_executor.h:1265–1281` 将active partial和全部pending一次计入；`slo_prefill_queue_policy.h:21–41` 取FCFS累计最晚开始边界；`slo_completion_policy.h:65–70` 在硬PF边界冲突时保PF。没有遗漏后排需求的旧bug可再次修。

076原128结果的实际证据（脚本与全部原数据SHA见result.json）：

1. 首波第11个first token在测量首到达后 **9.921870s**，全部16个在 **14.443484s**。原native两个完整5s测量窗口均 **0 decode rounds**；末端分别decode_ready10/prefilling1/waiting5，以及decode_ready15/prefilling1/waiting0。客户端首token后最长连续空档12.701796s，与该现象一致。这证明队列优先级确在持续阻塞decode，不是单个2048块遗漏yield。
2. TTFT P95 **9982.084566ms**，余量仅 **17.915434ms**。实际native shared round参与耗时中位 **51.855357ms**；固定其他轨迹，仅在原P95低贡献者arrival13之前插一轮，P95增量 `.35×51.855357=18.149375ms`，已超现有余量。这只是局部成本敏感度，未来arrival/acceptance/替补变化不能按此精确预测。
3. 原P95 postfirst remainder37.975050ms不是已证纯PF GPU时间。要从56.797723ms降到40ms，同两个贡献者需省 **8.583637s** postfirst时长。median ownPF909.042299ms，除以四块得227.260575ms/块，仅是粗均值；没有新trace，不把它当每块直接计时。

对三个即时设想的取舍：

- **固定每块插一decode / 强制公平轮转**：会占用已经几乎用尽的first-token预算；全PF工作不减少，不能据此承诺减少8.58s postfirst等待，不建议现在投入一轮整模实验。
- **遇到过期TTFT就让decode优先**：会改变原FCFS硬边界；现有稳态arrival69的TTFT10017.520ms正是原P95高贡献者，不能把“已超10s”当作延迟免费，延后它仍抬升P95。缺少支持安全晋级的证据，不偷偷放宽margin/尾部规则。
- **块内按层切换**：可作为后续工程机制，但必须分离状态/工作区并做真实状态/Graph检查；不是本轮十分钟可交付的小patch。仅缩token块还会改变算子形状/摊销，之前mixed小块结果已证明可能抵消局部收益。

原traceOFF没有逐chunk决策日志，无法精确回放每一时刻的executor pending队列。result.json保留全128生命周期“已到达但未first”的集合及原native5s快照，二者明确分开，不伪造精确调度时间线。建议保持当前失败完整，优先已有N5/实际服务成本下降候选；这次不新增分析/GPU队列。
