# MorphX 控制传输与 CPU 放置验证

当前是进展快照：算子、完整模型和 CPU 本地绑定的资格验证已完成，混合负载诊断进行中，尚无新三轮性能结论。目标是排除移植成本后评价 SOTA，当前改动属于兼容与部署工程。

此前 [MorphX-on-SGLang 对照](2026-10-15-sota-validation.md) 未改善联合权衡，但另轮插桩时间线有大量 8 字节控制拷贝、少量 P/D 生命周期重叠。这是待验证的线索，不能直接当作未插桩场景的根因。当前分别检查两个因素：

| 版本 | 修改 | 保留 |
|---|---|---|
| ABI 对照 | 修正旧/新查询接口的四/五参数 ABI | 原 CUDA-copy 初始化、进度轮询、profiler、动态 SM 策略 |
| fill | 将 agent 整数初始化改为同流 `cuMemsetD32Async` | 精确整数值、先分配后初始化再启动的顺序 |
| event | 用 kernel 后记录的完成事件判断调度线程是否可回收资源 | profiler 仍读取实际进度，`schedule_2` 继续在线调整 SM |
| both | 同时应用 fill 与 event | 同一编译器、模型、准入和 P/D 流 |

完成事件未记录时不得查询；未完成不得回收；错误必须退出。两次同步的计数/边界初始化仍保留，避免异步初始化覆盖调度线程的新边界。当前未增加 KV 管理或改变模型精度。

**实际资格证据。** 第一轮四个构建分别通过向量的 profile 开/关、五种 Marlin 几何的 profile 开/关、双流 Marlin 压力检查，共 20 个 GPU 任务、264 次数值比对，最大误差为 0。双流检查观察到实际非零最小 SM 边界写入，不能由此推断瞬时 SM 利用率。[算子审计](../../results/morphx-control/qualification/kernel-audit-v1.json)。

四个版本又分别通过代码检查、四个 24568-token 严格工具预热和完整 131060-token 长输入；全部工具正确，实际 C4、131328 KV token、16 BF16 SSM 槽、FP8 KV、MTP0、eager 均核对。[完整模型审计](../../results/morphx-control/qualification/model-audit-v2.json)。启动前曾有一次配置路径错误，未停止原服务、未执行模型，保留为 [preflight 失败](../../results/morphx-control/qualification/model-preflight-failure-v1.json)，修正后使用新目录。

**硬件放置修正。** 实际 GPU1 位于 NUMA1，CPU 28–55/84–111 本地；作者默认调度线程使用 CPU0（NUMA0），此前实验客户端还允许使用 CPU0/56。因此为每个版本增加仅影响主机绑定函数的 `SMSCHED_AGENT_CPU`，默认行为保留；本轮明确使用 agent36、模型线程28–35、客户端40–43/96–99，物理核心互不重叠。此设置不承诺整机独占核心。[拓扑与核心证据](../../results/morphx-control/qualification/host-placement.json)。

新构建只修改主机绑定函数，GPU kernel、profiler、`schedule_2` 和启动路径保持原字节。CPU 回归执行真实绑定，检查默认值、合法绑定和非法输入拒绝，完整 CPU 测试 272 项通过。四个新构建已分别重新通过代码/四路24K/完整128K门槛，并从实际后端日志确认 `requested=36 observed=36`；其完整审计已保留在本次结果目录。

**下一步测量。** 在相同 CPU 放置下，先运行编译器对照及四个传输版本的完整冷混合诊断；传输比较固定 split96/mode16（保留“关闭启动 CI 限制”消融标签）。随后对选定传输检查 split60/mode0、96/mode0、96/mode16，在正式三轮前冻结。正式实验需要冷长/续接各三轮和各自同 C4 服务上的串行单独基准。

冻结输入为短24568、冷长131060、续接131049 token，0.5 RPS/48秒、长请求第16秒到达；原工具期望不变。短请求在第46秒停止到达，不能外推为持续128K常驻并发。冷长与独立短前缀的物理 KV 并集还可能超过池容量，后续长期负载需区分容量约束与可消除的执行损失。

本次仍是 MorphX-on-SGLang，实际作者编译器/profiler/动态SM已运行，但 CUDA-copy 与作者 GDRCopy、编排和原生库覆盖仍存在差异。当前证据不能代表作者性能上限。后续 EB+ 还需忠实适配和本机校准，源码审计不等于已测效能。

全部构建、缓存和原始结果在 `vlm-server:/home/guokc/mypro/gnn-mount-sda1/InferX/morphx-control/` HDD。每个 GPU campaign 使用既有归属/空闲保护、GPU0停止状态检查和 finally 原256K/MTP7服务及双manifest恢复；最终状态将在本轮全部实验退出后再次核验。
