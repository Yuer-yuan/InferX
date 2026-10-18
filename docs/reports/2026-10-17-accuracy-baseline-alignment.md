# IFBench 精度与赛题参考值的对齐

02:36 UTC update: NInfer **default thinking/default sampling** original first100 finishes with the original report strict prompt **84/100**, strict instruction0.8559 (native rounded), loose prompt91/100 and loose instruction0.9189. All100 generation requests succeed. Root directly reads the complete original parser report and then independently reruns all100 original verifiers from rawHTTP final text: exact counts95/111 strict instruction and102/111 loose instruction match. Root verifies100 natural stops, zero length finishes/empty finals; completion tokens range250..21080, median2916.5. This uses the unchanged9695/e7/model atC2 and does not prove high-batch optimized Verify coverage. [Original report observation](../../results/three-node-optimization/control/NInfer-default-first100-original-report-independent-observation.json).

The vLLM fixed three-arm quality queue is already authorized. Its first client reaches no HTTP inference request because dataset offline cache lookup fails; the initial attempt is preserved and a client-only cache/environment repair is authorized. The ready original FP8 service is reused.

2026-10-17，02:04 UTC：**Root已用原版200次verifier独立复算：同一NInfer/C2/K3贪心协议，仅切换thinking，前100题prompt strict从54%升至82%。** 全部样本保留，ON的6个length、5个空final均计入失败。这个结果直接解释此前低分的重要来源，但不是vLLM FP8对照，也未覆盖高B的Verify新算术。

官方[Qwen3.8-27B模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)明确给出 **IFBench 79.5**。Root已读取原文，并固定revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`及README SHA256。约80%是需要解释和复现的模型能力参考；此前53%–55%的无思考前100题结果不能标为精度正常或已达标。新Verify优化相对OFF的结果只能评估这次变化，不能解释此前就存在的绝对差距。

**速度与精度是两套独立请求。** 赛题性能使用随机8192 token IDs的`/v1/completions`，temperature0、ignore_eos=true，总生成固定512；没有chat thinking模板。Root重读历史vLLM六档全部244条raw（216正式+28预热），均实际8192/512、finish_reason=length，无messages/chat_template_kwargs/enable_thinking参数。服务的`--reasoning-parser qwen3`不等于这些raw请求启用了chat thinking，也不等于全局关闭了模型思考。精度使用IFBench chat前100/batch2，仅覆盖max_tokens32768，自然EOS；thinking与final共用此输出预算，不是32768+额外thinking，也不是512个最终答案token。

Root又直接读取根目录`eval.zip`，其`README.md`、`runner.py`、`run.sh`分别与`submission/organizer/`副本逐字一致；runner SHA256为d80df390…。**ZIP本身就提供两个阶段的编排：性能调用AIPerf0.13.0/completions，精度调用EvalScope1.13.0/chat。** 同一入口可选perf/accuracy/all。我们另外实现的服务owner、实验控制、采集、消融与审计是外层工具；MD复现包装器明确更改MD相对ZIP的请求数/预热/超时/种子/cache_salt，仍调用原版AIPerf helper与统计解析器。C7/pilot等扩展和自定义thinking对照须单独标记，不能都称作ZIP默认全流程。证据：[原ZIP逐文件与实际调用函数独立审查](../../results/three-node-optimization/control/eval-ZIP-original-runner-provenance-independent-review.json)。

[赛题原文](../../Qwen3.8-27B基准测试.md)使用IFBench前100题、batch2、max_tokens32768，报告76%–80%。前100题继续作为快速定位与官方协议对齐的基准；不能以题数较少来淡化当前二十多个百分点的差距。模型卡本身不构成固定前100题能精确估计全300题的统计证明，最终仍保留完整集验证。score的四项指标、实际thinking/采样等复现细节需要记录，但这些信息缺失不能成为接受低分的理由。

模型卡同时明确默认开启thinking、默认reasoning_effort=xhigh，并推荐thinking采样temperature=1、top_p=.95、top_k=20、min_p=0、presence_penalty=0、repetition_penalty=1。**这些是已核验的默认/推荐配置，不是已经取得的79.5精确评测配方。**此前NInfer服务强制使用`--greedy --no-thinking`，其源代码还会用server.greedy覆盖请求的temperature；因此即使只打开thinking，仍未对齐默认采样。配置差异已确认，能解释多少分差仍待完整对照结果。

Root从已保存的逐题评分重新聚合，并核验当前全300计数与独立原版600次verifier复核一致。当前300条prompt哈希和指令数量与先前FP8参考逐条相同。

| 对照 | 前100题 prompt strict | 全300题 prompt strict |
| --- | ---: | ---: |
| 当前NInfer Verify A8 OFF，C7 | 54/100=54% | 109/300=36.33% |
| 当前NInfer Verify A8 ON，C7 | 54/100=54% | 118/300=39.33% |
| 实用FP8参考，关闭thinking及MTP，C1 | 53/100=53% | 116/300=38.67% |
| 早期冻结NInfer原版100题 | 55/100=55% | 未测 |
| 新NInfer thinking OFF，贪心C2/K3 | 54/100=54% | 未测 |
| 新NInfer thinking ON，贪心C2/K3 | 82/100=82% | 未测 |

这些运行在前100题均没有length终止。当前ON前100的四项指标依次为：prompt strict54%、instruction strict64/111=57.66%、prompt loose54%、instruction loose65/111=58.56%。因此，即使换看另一项，也还不能复现76%–80%。全300的ON四项分别39.33%、42.73%、41.67%、45.06%；5个length终止全部保留评分。

39.33%与此前55%的差异包含明确的样本范围变化；对齐前100后，当前为54%。但这不解释54%与赛题76%–80%的差距。相同无思考/贪心协议下FP8也只有53%，说明差距并非只有最新低比特核才出现；框架、MTP、权重、KV及并发仍不同，这不能排除量化损失，也不能代替官方默认设置复现。

用户明确赛题80%参考来自vLLM FP8。**现有SGLang默认参考不能代替vLLM复现。** 历史vLLM实际完成37题（0..34、36、37，缺35及38..99），原review聚合prompt strict35/37=94.59%、instruction strict38/40=95%；这只是未完成子集，存在600秒timeout/retry，不能外推完整100题。37条实际均有reasoning及final，其质量预算是32768。当前优先取得vLLM FP8原版完整100、NInfer默认完整100，另外实际测vLLM thinking ON/OFF的512预算完整100题，后两项明确属于诊断。不能截取32768回答的前512来替代新请求，也不把82%填作512预算精度。

新的C8容量阶段仍仅CPU准备，GPU授权保持false；原300评分与性能成绩继续保留。量化/MTP、高B任务质量及vLLM精确基线差异仍需对照。

实际模板SHA已核验，默认开启thinking/xhigh，每题会额外增加40个输入token；FP8模型默认temperature1/top_k20/top_p.95。Root分别复现全部200条OFF/ON渲染文本，两个worker的渲染也逐条相同。早期SGLang35条/vLLM37条有推理内容、仅最终content计分，仍是未完成子集，不据此外推100题成绩。

ucloud1固定原前100/C2贪心thinking OFF→ON已完成，原二进制e7、模型及993源文件独立预检一致，KV131072/context65536/K3/native8、不采profile。Root02:02:18 UTC原版200次verifier复算与逐题原记录、四指标和10k seed42配对CI全部一致：prompt strict54→82%，delta+28pp、95%CI[+16,+41]pp；instruction strict64→92/111；prompt loose54→90%；instruction loose65→100/111。原生两臂各100 start/done、thinking分别false/true、temperature0、max32768、全部cold；所有HTTP原始content与prediction逐字一致，HTTP错误均0。ON长度终止keys36/50/56/88/90/91全部评分失败，后5条空final；输出median2240.5/max32768。两臂只有请求thinking布尔值不同，贪心分支不使用默认top_p差异。旧owner/children实际终态及原253文件归档由worker完成并记录。此对照不等于默认采样或vLLM参考。

随后ucloud1已按条件批准启动同一源/模型/二进制的原版runner默认thinking+默认采样前100/C2。仅移除服务的`--greedy --no-thinking`。worker报告01:57:02 UTC owner315652/start42342773，服务315672/start42344007、client315720/start42344789，首4条native thinkingtrue/temp1/top20/p.95/penalties0/seed42。尚无完整100成绩；进程当前存活状态以新的实际观测为准。

ucloud2的原版runner/EvalScope1.13.0默认thinking前100/batch2参考已经运行。Root实际观测01:51:24 UTC：003阶段完成19题、21次HTTP attempt，owner59367/start42250348、client59421/start42250569、复用服务58803/start42224741均实际存活。001的数据预检失败、002的health空响应解析失败及其启动health请求全部保留；003沿用原服务，不重载模型。实际参数为原FP8全架构/BF16KV/MTPoff、原生容量2、KV131072/context65536。默认采样不显式覆盖，完整记录原runner重试和每次HTTP attempt，读取timeout7200，数值生成参数仍仅max_tokens32768。它与旧53%同时存在API、采样、thinking和容量差异，不能把差值全归于thinking。

ucloud3已收到vLLM FP8三臂准备任务：512预算thinking ON/OFF各原100，随后原版32768默认thinking100；三者自然EOS、完整保存reasoning/final/usage/finish和每HTTP attempt。现成vLLM0.31环境、FP8模型、EvalScope环境均经Root只读盘点存在，01:58:44 UTC GPU compute apps为空。只批准独占vLLM原配置质量服务启动，客户端待CPU冻结包审查；尚无512精度结果。fusion CPU接入代码仍独立保留，新的Q5 C32仅CPU机制准备，不与质量服务重叠执行GPU。

证据：[thinking200原版verifier独立复算](../../results/three-node-optimization/control/thinking-toggle-first100-independent-verifier-review.json)、[历史vLLM244条速度/长度/模板独立审查](../../results/three-node-optimization/control/vLLM-performance-thinking-length-contract-independent-review.json)、[历史vLLM37原review部分聚合](../../results/three-node-optimization/control/historical-vLLM37-partial-score-aggregation.json)、[官方卡片固定版本与解读](../../results/three-node-optimization/control/qwen38-model-card-reference-20261017/reference.json)、[卡片快照](../../results/three-node-optimization/control/qwen38-model-card-reference-20261017/README.md)、[模板和默认参数](../../results/three-node-optimization/control/default-thinking-metadata-render-independent-review.json)、[NInfer默认采样源审](../../results/three-node-optimization/control/NInfer-registered-sampling-defaults-source-review.json)、[后续NInfer默认生成授权](../../results/three-node-optimization/control/NInfer-default-thinking-sampling-first100-next-GPU-authorization.json)、[FP8默认参考授权](../../results/three-node-optimization/control/FP8-default-thinking-first100-GPU-authorization.json)、[vLLM服务启动范围](../../results/three-node-optimization/control/vLLM-FP8-ucloud3-service-startup-GPU-authorization.json)。

证据：[本次聚合复核](../../results/three-node-optimization/control/official-baseline-accuracy-alignment-review.json)、[原版600次verifier复核](../../results/three-node-optimization/control/verify-Q5-A8-quality300-independent-verifier-review.json)、[FP8原300独立审计](../../results/three-node-optimization/control/fp8-full300-independent-cache-native-bootstrap-review.json)、[早期原版100题](../../submission/BENCHMARK.md)。
