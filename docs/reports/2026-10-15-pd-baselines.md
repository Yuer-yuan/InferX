# 开源 PD 方案适配与验证（进行中）

目前不能把已发现的长短请求干扰称为现有系统尚未解决的研究空白。llama.cpp 分块调优后三轮独立重复仍有明显尾延迟；MorphX 移植尚未通过正确性门槛；vLLM 已实际启动并通过代码生成检查，但24K工具预热发生GDN临时张量OOM，正在调整容量。以下数据不构成最终跨引擎排名。

## 已得到的结果

同一 Qwen3.8-27B GGUF、单张 4090、C4、MTP关闭、共享 KV 池212992 token、q4_0，短请求含约24K历史。固定48秒短请求波次，第16秒加入128K长请求。所有表内混合组均实际完成长请求，没有截断或拒绝它来降低短请求延迟。

| llama.cpp 配置 | 短到达率 | 纯短请求 P95 | 加冷长请求后的短 P95 | 加续接长请求后的短 P95 |
|---|---:|---:|---:|---:|
| chunk2048，诊断各1轮 | 0.5 RPS | 1.972 s | 132.968 s | 未测 |
| chunk128，诊断各1轮 | 0.5 RPS | 1.974 s | 58.557 s | 42.320 s |
| chunk128，正式重复第1轮 | 0.5 RPS | 1.980 s | 58.451 s | 42.392 s |
| chunk128，正式重复第2轮 | 0.5 RPS | 1.460 s | 58.452 s | 151.347 s |
| chunk128，正式重复第3轮 | 0.5 RPS | 1.995 s | 57.848 s | 42.376 s |

这表明较小分块改善了当前配置的干扰，但尚未达到5秒短请求SLO。正式实验已完成9组，每组独立重启服务并通过完整代码检查，222/222请求工具语义有效，6次完整128K长请求均完成，未发生HTTP错误。单波次内的请求相互关联，独立重复数为3，不将请求总数当作独立样本量计算置信区间。完整摘要及原始文件哈希见 [正式三轮结果](../../results/pd-baselines/llama-main-v2-summary.json)。历史组名resume-high在这次实验中实际到达率为0.5 RPS，以记录的rate字段为准。

第二次正式续接重复的短P95进一步升到151.347秒，最大171.591秒；长请求反而在126.189秒完成。长请求仍命中24608 token，与首轮相同，因此不能用长请求缓存失效解释这个波动。原生日志中首轮长请求处于slot2，第二轮处于slot1；这是准入/分块调度需要进一步检查的相关证据，尚未做固定槽位的因果消融。所有重复均保留，不能只用42秒的一轮代表配置表现。当前明细见 [重复实验中间记录](../../results/pd-baselines/llama-repetition-progress.json)。

诊断轮中冷长请求实际输入131049 token、缓存0，耗时164.762秒；续接实际输入131038 token、命中24608、另算106430，耗时147.809秒。两组均25/25请求协议及工具语义有效。机器摘要见 [llama-ch128-diagnostic.json](../../results/pd-baselines/llama-ch128-diagnostic.json)。

另一个2.5 RPS的llama.cpp纯短控制组已自身过载，P95达102.098秒。因此不把该负载下的混合延迟全部归因于PD干扰。此前NInfer的2.5 RPS结果与这里0.5 RPS结果也不能直接计算调度改善倍数。

## MorphX 的实际状态

已固定作者源码4404aed4c180ad83c31179e1ce08ae2234390853，完成LLVM18/SM89编译器适配、CUDA Runtime真正拦截和独立数值探针。原GDRCopy路径实际调用gdr_open失败，设备/dev/gdrdrv不存在；当前实验使用显式标注的映射主存控制通道。这改变了控制数据所在位置及访问开销，不能冒充论文原始GDRCopy环境。

| 完整 CTest 调用 | 实际执行 | 通过 | 失败 | 缺外部模型而跳过 |
|---|---:|---:|---:|---:|
| 相同Clang构建，关闭hook | 79 | 79 | 0 | 5 |
| 允许默认流的初始移植 | 79 | 50 | 29 | 5 |
| 修正统计表访问、退出生命周期及队列发布 | 79 | 74 | 5 | 5 |

调试器捕获到后台agent访问gCI统计表时崩溃，主线程正在退出并等待agent。修正后原先多项崩溃消失。另一个显式兼容变体在Graph capture期间保留原生kernel节点，避免重放已经释放的单次控制缓冲区；针对原先5项失败的复测中，3项Graph测试通过，仍剩GDN投影launch failure和GQA数值错误。该变体尚未重新完整调用79项，不能将定向测试合并成一次完整通过。

完整27B模型也已经实际尝试：hook路径在预热中命中了改写后的bf16_gdn_gating_proj_gemm_mma_kernel，并发生cudaErrorLaunchFailure。关闭hook的Clang对照能启动，但一次完整代码验证生成了缩进错误，未通过质量门槛。这些失败均保留；没有用失败配置生成“有效MorphX吞吐”。见 [兼容性证据](../../results/pd-baselines/compatibility/progress.json)。

因此当前结论是移植尚未合格，不能回答MorphX在本机能改善多少，也不能据此宣称MorphX原方法无效。尤其仅改写kernel并不会自动改变NInfer的单prefill准入策略；要评价PD机制，还需要配套请求准入及实际P/D重叠，并保留相同准入策略关闭MorphX的消融。

另已实际运行共享内存生命周期整数探针，原生、原MorphX改写、增加虚拟CTA结束屏障三条路径均通过32768项独立整数校验，后两者均观察到改写kernel启动。因此这个探针未复现候选问题，不能证明结束屏障是GDN/GQA故障的修复。屏障变体尚未做完整模型算子验证，不替代原始失败记录。见 [探针原始诊断](../../results/pd-baselines/compatibility/morphx-shared-lifetime.json)。

## vLLM 的实际状态与下一项验证

同源27B INT4模型已从hf-mirror下载，revision为91bd022d5b49442a868bc35008f6c21e1860edfa，9个文件完成校验。vLLM固定0.24.0：默认wheel实际导入时缺libcudart.so.13；换用官方CUDA12.9 wheel后，Torch2.11.0+cu128与vLLM核心stable扩展导入通过，实际CLI解析也通过。wheel哈希及包版本见 [安装阶段环境清单](../../results/pd-baselines/vllm-environment.json)。

初始候选为C4、完整131328 token窗口、FP8 KV、Mamba align缓存、batch预算4096、单请求待计算token上限2048、关闭Graph的容量探针。后续根据实际分配、质量和延迟再调优Graph及分块；212992只是比较容量目标，vLLM实际分配的池大小必须另行观察，不能把目标值写成实测容量。

单请求上限会给同一轮其他请求留出token预算；其作用需要实测。[官方参数说明](https://docs.vllm.ai/en/v0.24.0/configuration/engine_args/)和[该版本V1调度源码](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/v1/core/sched/scheduler.py)是配置依据。本次安装源码中的V1路径直接使用long_prefill_token_threshold，不应仅凭max_num_partial_prefills配置项存在就认定它已经改变实际调度。

实际GPU探针V2中模型占16.84 GiB，显存比例0.98时分配5.49 GiB缓存、166348 token实际池，随后通过完整代码检查；4个24K工具预热请求触发引擎OOM，首个失败位置为FLA的recompute_w_u_fwd申请20 MiB张量。故尚未进入128K请求验证，也没有有效PD性能数据，不能用启动时显示的最大并发量作为实际容量通过。见 [失败摘要与完整引擎回溯](../../results/pd-baselines/vllm-mem98-warmup-failure.json)。下一探针只将显存比例降至0.95，保留相同模型、上下文、分块及质量门槛，检查预填充工作空间与实际KV池的取舍。

V1探针在测量前发现FlashInfer将约17 MB新生成编译缓存写入默认home缓存，已受控停止并恢复服务；随后只迁移本次生成的0.6.12目录至HDD，保留旧版本目录，显式配置FLASHINFER_WORKSPACE_BASE和TORCH_EXTENSIONS_DIR，并验证解析后的路径位于HDD。该中断不算模型性能或容量失败。

## 可复现性与剩余工作

完整原始数据、源码、编译日志和模型均在服务器HDD的InferX/pd-baselines目录。当前正式llama实验的启动时4个关键Python文件已按其启动时SHA256精确重建并留档，明确标为事后重建；见 [重建清单](../../results/pd-baselines/harness-llama-main-v2/provenance.json)。后续controller会在启动时保存完整脚本及inferx模块字节，避免工作目录更新覆盖实验实现。

还需完成vLLM实际128K容量与混合负载、MorphX剩余数值兼容问题、合格配置的同引擎消融及独立重复、确定性导出、独立复审，以及最终原256K/MTP7服务和两份manifest、OpenAI/Anthropic工具接口的恢复检查。当前目标继续执行。
