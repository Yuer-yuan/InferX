// Edit this file: both language editions and live previews update together.
#import "@preview/touying:0.8.0": themes, config-page, config-colors, config-common
#import "native-figures.typ": illustration
#import "charts.typ": chart
#let data = json("benchmark-data.json")
#let details = json("result-details.json")
#let campaign = json("selected-campaign.json")
#let quality = json("selected-quality.json")
#let repro = json("clean-reproduction.json")
#let accent = rgb("#7E2635")
#let ink = rgb("#252930")
#let muted = rgb("#657180")
#let light = rgb("#D5DBE1")

#let render(language) = {
  set text(font: ("Liberation Sans", "Noto Sans CJK SC"), size: 20pt,
    fill: ink, lang: language, hyphenate: false)
  set par(leading: 8pt, spacing: 10pt, justify: false)
  set list(indent: 14pt, body-indent: 7pt, spacing: 10pt,
    marker: text(fill: accent)[•])
  let L(zh, en) = if language == "zh" {zh} else {en}
  let p(id) = data.performance.at(id)
  let fmt(v) = str(calc.round(v, digits: 2))
  let fmt3(v) = str(calc.round(v, digits: 3))
  let c14 = campaign.performance.filter(r => r.client_c == 14)
  let bullets(items) = list(tight: false, ..items)
  // Touying Metropolis provides the page frame, progress bar and title slide.
  show: themes.metropolis.metropolis-theme.with(
    aspect-ratio: "16-9", align: top,
    footer: pad(left: 30pt, text(9pt, fill: muted)[InferX · Track A]),
    footer-right: context pad(right: 30pt, text(9pt, fill: muted)[#counter(page).display() / #counter(page).final().first()]),
    footer-progress: true,
    config-page(width: 320mm, height: 180mm,
      margin: (top: 29mm, bottom: 20mm, x: 14mm)),
    config-colors(primary: accent, primary-light: rgb("#EADDE0"),
      secondary: accent, neutral-lightest: white,
      neutral-dark: ink, neutral-darkest: ink),
    config-common(new-section-slide-fn: none),
  )
  let slide(title, chapter, body) = themes.metropolis.slide(
    title: pad(left: 20pt, text(23pt, weight: "medium", title)), align: top,
  )[
    // Section labels remain in the source; temporarily hidden for a cleaner slide.
    #block(width: 292mm, height: 126mm, breakable: false)[#body]
  ]
  let mechanism(title, name, items) = slide(title,
    L("核心实现","Implementation"), [
      #align(center)[#illustration(name,language,width:276)]
      #v(3mm)
      #bullets(items)
    ])
  let charts(left-title,left-name,right-title,right-name)=grid(
    columns:(1fr,1fr),gutter:10mm,
    [#text(16pt,weight:"bold")[#left-title]#v(2mm)
     #chart(left-name,language)],
    [#text(16pt,weight:"bold")[#right-title]#v(2mm)
     #chart(right-name,language)])
  let pipeline(labels) = grid(columns:(1fr,12mm,1.3fr,12mm,1fr),align:center+horizon,
    rect(width:100%,height:25mm,stroke:.8pt+light,inset:3mm)[#align(center+horizon)[#text(15pt)[#labels.at(0)]]],
    text(20pt,fill:muted)[→],
    rect(width:100%,height:32mm,stroke:1pt+accent,fill:accent.lighten(96%),inset:3mm)[#align(center+horizon)[#text(15pt)[#labels.at(1)]]],
    text(20pt,fill:muted)[→],
    align(center)[#text(15pt)[#labels.at(2)]])

  // Native Metropolis title slide: left aligned, with one burgundy rule.
  themes.metropolis.title-slide(
    config: config-page(margin: (x: 12mm, top: 0mm, bottom: 0mm))
      + config-common(freeze-slide-counter: false),
    title: text(36pt, weight: "bold", fill: accent)[InferX],
    subtitle: block(above: 4mm, below: 4mm)[
      #text(24pt)[#L("单卡大模型推理部署与优化",
        "Single-GPU LLM Inference Deployment and Optimization")]
    ],
    author: block(above: 7mm)[
      #text(18pt)[#L(data.team.members.join("、"),
        "Wenxin Chen · Zhixiong Zhang · Kunchang Guo")]
      #v(4mm)
      #text(16pt, fill: muted)[#L("你说啥都队", "Team Ni Shuo Sha Dou Dui")]
    ],
    date: none, institution: none, contact: none,
    extra: block(above: 7mm)[
      #text(13pt, fill: muted)[#L("项目介绍与答辩 · 赛道 A", "Project Defense · Track A") · 2026-10-18]
      #v(3mm)
      #text(13pt, fill: muted)[#link("https://github.com/Yuer-yuan/InferX")[github.com/Yuer-yuan/InferX]]
    ],
  )

  slide(L("项目概述","Project overview"),L("目标与交付","Goal and deliverable"),[
    #text(22pt,weight:"bold")[#L("目标：在SLO约束下，提高单卡并发能力。",
      "Goal: higher single-GPU concurrency within latency SLOs.")]
    #v(7mm)
    #grid(columns:(134mm,1fr),gutter:14mm,
      [#v(1mm)#illustration("overview",language,width:132)],
      [
        #text(18pt,weight:"bold")[#L("交付：基于NInfer的推理服务",
          "Deliverable: a NInfer-based inference service")]
        #v(7mm)
        #grid(columns:(8mm,1fr),column-gutter:3mm,row-gutter:7mm,
          text(19pt,fill:accent)[1], [
            #text(20pt,weight:"bold")[#L("4bit计算","4-bit computation")]
            #v(2mm)
            #text(17pt)[#L("权重与激活均采用INT4矩阵运算。",
              "INT4 matrix kernels for weights and activations.")]
          ],
          text(19pt,fill:accent)[2], [
            #text(20pt,weight:"bold")[#L("MTP并发适配","Concurrent MTP integration")]
            #v(2mm)
            #text(17pt)[#L("扩展批处理能力，各请求独立提交、回滚状态。",
              "Expand batching; commit and roll back each request’s state independently.")]
          ],
          text(19pt,fill:accent)[3], [
            #text(20pt,weight:"bold")[#L("基于请求进度的调度","Progress-aware scheduling")]
            #v(2mm)
            #text(17pt)[#L("长输入分块处理；按进度和紧迫性安排解码。",
              "Chunk long inputs; use progress and urgency to allocate decode service.")]
          ],
        )
      ],
    )
    #v(5mm)
    #text(19pt)[*#L("结果：","Result:")* #L(
      "支持14并发请求，两次测试均满足SLO；总吞吐306.04–309.95 tok/s。",
      "Supports 14 concurrent requests; both tests meet SLOs at 306.04–309.95 tok/s.")]
    #v(2mm)
    #text(18pt)[*#L("精度：","Quality:")* #L("IFBench前100题：严格通过85题，宽松通过88题。",
      "First 100 IFBench prompts: 85 strict passes and 88 loose passes.")]
  ])

  // ASSISTANT SECTION START: problems, design, implementation, evaluation.
  slide(L("问题与目标：影响并发能力的两项瓶颈", "Problems: two bottlenecks in concurrent inference"),
    L("问题与目标", "Problems and goals"),[
    #align(center)[#illustration("problems",language,width:236)]
    #v(5mm)
    #bullets((
      [*I · #L("预填充与解码相互干扰。", "P/D interference.")* #L(
        "长输入的预填充（Prefill）占用GPU，延后解码（Decode），增加尾延迟。",
        "Long prefill work delays decode on the shared GPU; scheduling affects tail latency.")],
      [*II · #L("算子效率。", "Operator efficiency.")* #L(
        "长输入处理与批量验证的矩阵形状不同；低比特量化的收益受算子、数据转换和访存开销影响。",
        "Prefill and batched verification use different matrix shapes; low-bit gains depend on execution and conversion costs.")],
    ))
    #v(3mm)
    #text(18pt,fill:accent)[#L("目标：提高满足TTFT P95 ≤ 10s、TPOT P95 ≤ 40ms的并发能力，并验证任务质量。",
      "Goal: increase concurrency within TTFT P95 ≤ 10s and TPOT P95 ≤ 40ms; check task quality.")]
  ])

  slide(L("整体方案：调度与INT4优化，MTP并发适配", "Design: scheduling, INT4, and concurrent MTP integration"),
    L("整体方案", "System design"),[
    #align(center)[#illustration("architecture",language,width:240)]
    #v(3mm)
    #grid(columns:(112mm,1fr),gutter:10mm,
      [
        #text(17pt,weight:"bold")[#L("技术路线", "Technical route")]
        #v(2mm)
        #text(16pt)[#L("评测模型：Qwen3.8-27B", "Given model: Qwen3.8-27B")]
        #v(2mm)
        #set text(17pt)
        #set par(leading:4pt)
        #set list(spacing:8pt)
        #bullets((
          [*#L("离线：", "Offline: ")* #L("GPTQ校准与权重量化。",
            "GPTQ → 4-bit weights.")],
          [*#L("在线：", "Online: ")* #L("分块调度 → INT4 / MTP；",
            "Chunking → INT4 / MTP;")\
           #L("按请求提交状态、反馈进度。", "Per-request commit and feedback.")],
        ))
      ],
      [
        #text(17pt,weight:"bold")[#L("框架取舍：硬件适配与改造成本", "Framework choice: hardware fit and integration")]
        #v(2mm)
        #set text(15.5pt)
        #set par(leading:3pt)
        #table(columns:(43mm,1fr),inset:(x:5pt,y:5pt),stroke:none,
          table.hline(stroke:.8pt+muted),
          [*#L("框架", "Framework")*], [*#L("依据与取舍", "Rationale and trade-off")*],
          table.hline(stroke:.5pt+light),
          [*#L("NInfer（采用）", "NInfer (chosen)")*],
          [#L("面向4090 / SM89，复用量化与MTP；", "4090 / SM89 path; reuse quantization and MTP;")\
           #L("统一改造算子、状态和调度，需自行维护。", "Joint kernel/state/scheduler edits; own upkeep.")],
          table.hline(stroke:.5pt+light),
          [vLLM / SGLang],
          [#L("支持MTP，适配取决于具体模型；", "Support MTP for compatible models;")\
           #L("本项目的量化与状态管理改造需额外适配。", "Our W4A4 and state changes need porting.")],
          table.hline(stroke:.8pt+muted),
        )
      ],
    )
  ])

  slide(L("选型依据：ninfer-4090的针对性优势", "Why ninfer-4090 fits this project"),
    L("框架选型", "Framework selection"),[
    #text(20pt)[#L(
      "相较vLLM / SGLang，更贴合本项目的固定模型、单张4090与底层改造。",
      "Against vLLM / SGLang, ninfer-4090 offers a focused base for single-GPU optimization.")]
    #v(8mm)
    #set text(18pt)
    #set par(leading:6pt)
    #table(columns:(54mm,118mm,1fr),inset:(x:8pt,y:12pt),stroke:none,
      table.hline(stroke:.8pt+muted),
      [*#L("优势", "Advantage")*],
      [*#L("ninfer-4090的基础", "ninfer-4090 foundation")*],
      [*#L("对本项目的价值", "Value to this project")*],
      table.hline(stroke:.5pt+light),
      [*#L("模型与硬件", "Model–GPU")*\ *#L("专用适配", "specialization")*],
      [#L("面向Qwen3.8-27B与单张RTX 4090；", "Qwen3.8-27B on one RTX 4090;")\
       #L("采用SM89原生CUDA执行路径。", "native SM89 CUDA execution.")],
      [#L("围绕固定模型结构与矩阵形状优化，", "Tune for fixed model and matrix shapes;")\
       #L("聚焦4090上的单卡推理。", "focus on inference on one 4090.")],
      table.hline(stroke:.5pt+light),
      [*#L("低位执行链", "Low-bit path")*\ *#L("直接复用", "reuse")*],
      [#L("Q4/Q5权重、量化KV与MTP已配套；", "Q4/Q5 + quantized KV + MTP;")\
       #L("目标模型的加载与验证路径可复用。", "reuse model loading and verification.")],
      [#L("在已有部署链上接入INT4计算，", "Add INT4 compute to an existing stack;")\
       #L("减少定制路径的迁移与集成工作。", "reduce porting of our custom paths.")],
      table.hline(stroke:.5pt+light),
      [*#L("执行与状态", "Execution /")*\ *#L("统一控制", "state control")*],
      [#L("C++20 / CUDA直接管理执行；", "C++20 / CUDA execution control;")\
       #L("ReplaySSM提供状态提交与回滚。", "ReplaySSM commit / rollback.")],
      [#L("便于联改算子、批验证与调度，", "Joint kernel, MTP, and scheduler edits;")\
       #L("扩展状态、工作区与CUDA Graph。", "extend state, workspace, and graphs.")],
      table.hline(stroke:.8pt+muted),
    )
  ])

  mechanism(L("设计I：分块预填充与请求进度调度", "Design I: chunking and request progress"),"schedule",(
    [*#L("调度时机：", "Scheduling points: ")* #L(
      "8192 Token输入拆为4个2048块；每块结束后重新选择预填充或解码，块内不抢占。",
      "Split 8192 input tokens into four 2048-token chunks; choose prefill or decode between non-preemptible chunks.")],
    [*#L("剩余工作：", "Remaining work: ")* #L(
      "按请求统计已提交Token和解码轮数，估计每轮产出；用批次均值会低估慢请求的剩余轮次。",
      "Track committed tokens and decode rounds per request; batch-average yield can underestimate slow requests.")],
    [*#L("决策依据：", "Decision rule: ")* #L(
      "剩余轮次乘以当前批次轮耗时，得到预计解码耗时；比较两阶段时限余量，保留预填充期限约束。",
      "Multiply remaining rounds by current batch round time; compare deadline slack while preserving the prefill deadline guard.")],
  ))

  slide(L("设计I：从实际提交进度估计剩余轮次", "Design I: estimate work from committed progress"),
    L("核心实现", "Implementation"),[
    #set par(leading:5pt,spacing:6pt)
    #grid(columns:(146mm,1fr),gutter:12mm,
      [
        #text(20pt,weight:"bold")[#L("每个请求独立估计", "Estimate each request separately")]
        #v(2mm)
        #align(center)[#text(22pt)[$ y_i = (g_i - 1) / d_i $]]
        #v(4mm)
        #align(center)[#text(20pt)[$ R_i = max(ceil(r_i / max(1, y_b)), ceil(r_i / max(1, y_i))) $]]
        #v(3mm)
        #text(18pt)[#L("g：已提交Token数；d：解码轮数", "g: committed tokens; d: decode rounds")\
          #L("r：剩余Token预算；y：每轮产出", "r: remaining tokens; y: yield per round")\
          #L("i：单个请求；b：当前批次", "i: individual request; b: current batch")\
          #L("减1排除预填充产生的首个Token。", "Subtract one to exclude the prefill token.")]
      ],
      [
        #set text(18pt)
        #set par(leading:5pt)
        #set list(spacing:10pt)
        #bullets((
          [*#L("提交后更新。", "Update after commit. ")* #L("统计经输出裁决的实际提交量；取消行不计入解码轮数。",
            "Use the output policy’s actual commit count; cancelled rows do not count as rounds.")],
          [*#L("无历史时回退。", "Cold-start fallback. ")* #L("尚无解码记录时，沿用当前批次估计。",
            "Use the existing batch estimate before any decode history is available.")],
          [*#L("保留原估计下限。", "Keep the batch lower bound. ")* #L("慢请求可以提高剩余轮次，快请求不压低原估计。",
            "Slow requests may increase predicted work; fast requests never reduce the original estimate.")],
        ))
      ])
    #v(4mm)
    #text(18pt,weight:"bold")[#L("同机16并发，仅改进度估计：128条正式请求 + 8条预热", "Matched C16; estimator change only; 128 measured + 8 warmup")]
    #v(2mm)
    #set text(18pt)
    #table(columns:(104mm,64mm,64mm,1fr),rows:(11mm,11mm,11mm),inset:(x:7pt,y:6pt),stroke:none,
      table.hline(stroke:.8pt+muted),
      [*#L("估计方法", "Estimator")*],[*TTFT P95 (s)*],[*TPOT P95 (ms)*],[*tok/s*],
      table.hline(stroke:.5pt+light),
      L("批次估计", "Batch estimate"),[9.856],[48.784],[311.77],
      L("加入请求进度", "With request progress"),[9.838],[45.745],[315.73],
      table.hline(stroke:.8pt+muted))
    #v(2mm)
    #text(18pt)[#L("TPOT改善6.23%，仍高于40ms；模型算子与CUDA Graph保持一致。",
      "TPOT improves 6.23%, still above 40 ms; model operators and CUDA Graph are unchanged.")]
  ])

  slide(L("设计I：有界选择紧迫请求，随后恢复全员服务", "Design I: bounded urgent service, then two full rounds"),
    L("核心实现 · 100版本", "Implementation · Build 100"),[
    #align(center)[#illustration("bounded-decode",language,width:276)]
    #v(4mm)
    #set text(18pt)
    #set par(leading:5pt)
    #bullets((
      L("先由原调度器选择解码；仅当10–14个请求就绪、存在负时限余量且暂缓请求余量达标时缩减。",
        "Check only after the original scheduler chooses decode: 10–14 ready requests, a negative slack, and sufficient slack for every omitted row."),
      L("每次最多连续缩减一轮；随后两个实际解码轮服务全部就绪请求。预填充不消耗这两个轮次。",
        "At most one subset round at a time; the next two actual decode rounds serve all ready requests. Prefill does not consume this count."),
      L("仅给实际服务的请求记解码进度；这是服务机会的重分配，总吞吐和SLO效果需用完整服务实测。",
        "Count progress only for served rows. This redistributes service; evaluate aggregate throughput and SLOs on the complete service."),
    ))
  ])

  mechanism(L("设计II：权重与激活的INT4计算", "Design II: INT4 weights and activations"),"int4",(
    [*#L("离线权重：", "Offline weights: ")* #L(
      "每512维做局部Hadamard变换，按512通道分组做GPTQ校准；缓存4bit码与缩放因子。",
      "Apply local 512-dimensional Hadamard transforms and GPTQ in 512-channel groups; cache 4-bit codes and scales.")],
    [*#L("在线激活：", "Online activations: ")* #L(
      "BF16激活做对应变换与逐组动态量化；权重、激活均送入S4整数矩阵指令。",
      "Transform BF16 activations and dynamically quantize each group; feed both operands to S4 matrix instructions.")],
    [*#L("输出恢复：", "Output reconstruction: ")* #L(
      "组内INT32累加，组间结合缩放因子做FP32累加，输出BF16；保留非线性和残差的原有语义。",
      "Accumulate INT32 within groups and scaled FP32 across groups; output BF16 while preserving nonlinear and residual semantics.")],
  ))

  slide(L("设计II：从量化缓存接入真实模型", "Design II: integrate quantized paths into the model"),
    L("核心实现", "Implementation"),[
    #v(2mm)
    #set text(18pt)
    #set par(leading:5pt)
    #table(columns:(52mm,143mm,97mm),inset:(x:7pt,y:9pt),stroke:none,
      table.hline(stroke:.8pt+accent),
      [*#L("改造位置", "Component")*],[*#L("具体实现", "Implementation")*],[*#L("验证重点", "Validation")*],
      table.hline(stroke:.5pt+light),
      L("量化缓存", "Quantized caches"),
      L("先覆盖64层FFN的128个矩阵；校准用4条8192 Token请求，另留2条验证。",
        "Start with 128 FFN matrices across 64 layers; use four 8192-token calibration requests and two held out."),
      L("核对码值与缩放因子，检查留出数据误差。",
        "Check codes, scales, and held-out error."),
      table.hline(stroke:.5pt+light),
      L("算子与路由", "Kernels and routing"),
      L("从FFN扩展至输入/输出投影；接入真实权重、工作区和Graph依赖。",
        "Extend FFN paths to input/output projections; connect real weights, workspace, and Graph dependencies."),
      L("完整计时包括变换、量化、矩阵计算与写回。",
        "Time transforms, quantization, matrix work, and writeback together."),
      table.hline(stroke:.5pt+light),
      L("MTP验证阶段", "MTP verification"),
      L("47层GDN输出投影启用T33–128；T1–32、L0与全注意力输出保留原路由。",
        "Enable T33–128 in 47 GDN output layers; keep prior routes for T1–32, L0, and full-attention outputs."),
      L("T为矩阵Token行数；按层、阶段和形状核验覆盖。",
        "T is the token-row count; verify coverage by layer, phase, and shape."),
      table.hline(stroke:.8pt+muted))
    #v(5mm)
    #set text(19pt)
    #bullets((
      [*#L("本项目实现：", "Our implementation: ")* #L(
        "基于已有GPTQ与Hadamard方法，完成缓存生成、CUDA算子及模型路由适配。",
        "Build on existing GPTQ and Hadamard methods; implement caches, CUDA operators, and model routing.")],
    ))
  ])

  mechanism(L("工程实现：MTP的并发适配", "Implementation: adapting MTP for concurrency"),"state",(
    [*#L("算子与布局：", "Operators and layouts: ")* #L(
      "扩展attention/GDN验证形状、行到槽位映射、输入输出缓冲和状态记录，支持16/32请求。",
      "Extend attention/GDN verification shapes, row-to-slot mapping, I/O buffers, and state records for 16/32 requests.")],
    [*#L("运行时资源：", "Runtime resources: ")* #L(
      "按并发配置分配KV、状态和工作区，同步更新CUDA Graph捕获参数及缓冲区边界。",
      "Size KV, state, and workspace for configured concurrency; update CUDA Graph arguments and buffer bounds.")],
    [*#L("一致性要求：", "State invariant: ")* #L(
      "各请求独立维护KV、GDN递推及卷积状态；接受数量可不同，状态必须对应实际提交序列。",
      "Maintain separate KV, GDN recurrent, and convolution states; unequal acceptance must still match each committed sequence.")],
  ))

  mechanism(L("工程实现：按请求提交与回滚状态", "Implementation: per-request state commit and rollback"),"commit-lifecycle",(
    [*#L("先裁决，再提交：", "Resolve output first: ")* #L(
      "目标验证后，按EOS/stop和剩余预算裁定提交长度；取消请求不提交。",
      "After target verification, EOS/stop and remaining budget determine actual commit lengths; cancelled rows commit zero.")],
    [*#L("按请求更新：", "Update per request: ")* #L(
      "先处理待提交状态，再更新已提交Token、生成预算和解码轮数；最后发布有效输出。",
      "Resolve pending state, then update committed tokens, generation budget, and decode rounds; publish valid output last.")],
    [*#L("验证覆盖：", "Validation coverage: ")* #L(
      "检查高编号槽位、0–4部分提交、取消/重置与Graph重放。状态融合阶段2013条记录一致，86项测试通过、4项跳过。",
      "Check high slots, mixed 0–4 commits, cancel/reset, and Graph replay. The state-fusion stage has 2013 matching records; 86 tests pass, four skip.")],
  ))

  slide(L("结果与证据：并发请求下的延迟与吞吐", "Results: latency and throughput under concurrent load"),
    L("结果与证据 · 六组正式测量", "Evaluation · Six formal measurements"),[
    #chart("selected-campaign",language,width:292mm)
    #v(1mm)
    #text(17pt)[#L("每次输入8192 Token、输出512 Token；每配置128次测试、8次预热。",
      "8192 input / 512 output tokens; 128 measured requests and 8 warmups per setting.")]
    #v(2mm)
    #text(17pt)[#L("每条曲线对应一台节点；14并发测了两次，等待时间计入首字延迟。",
        "Each curve represents one node. Two tests at concurrency 14; TTFT includes queueing.")]
  ])

  slide(L("结果与证据：达标配置与失败边界", "Results: passing configurations and capacity limits"),
    L("结果与证据 · 100版本配置边界", "Evaluation · Build 100 configuration boundary"),[
    #text(20pt,weight:"bold")[#L("14并发两次测试均达标；15、16并发的生成延迟仍超限。",
      "Concurrency 14 meets both SLOs twice; 15 and 16 exceed the TPOT limit.")]
    #v(4mm)
    #let boundary = (campaign.comparison_runs.first(),) + c14 + campaign.comparison_runs.filter(r => r.client_c == 15 or r.native_c == 16)
    #set text(16pt)
    #table(columns:(48mm,49mm,54mm,57mm,44mm,1fr),inset:(x:5pt,y:7pt),stroke:none,
      table.hline(stroke:.8pt+muted),
      [*#L("节点 / 轮次", "Node / run")*],[*#L("客户端 / 引擎", "Client / native")*],[*TTFT P95 (s)*],[*TPOT P95 (ms)*],[*tok/s*],[*SLO*],
      table.hline(stroke:.5pt+light),
      ..boundary.map(r => ([#r.node · r#r.repeat],[#r.client_c / #r.native_c],[#fmt3(r.ttft_s)],
        [#fmt3(r.tpot_ms)],[#fmt(r.tps)],[#text(fill:if r.joint_slo {rgb("#287657")} else {rgb("#B44B45")})[#if r.joint_slo { [PASS] } else { [FAIL] }]])).flatten(),
      table.hline(stroke:.8pt+muted))
    #v(5mm)
    #set text(18pt)
    #set par(leading:5pt)
    #bullets((
      L("达标配置：引擎最多接纳13个活动请求，其余请求等待；等待计入首字延迟。",
        "Passing configuration: admit at most 13 active requests; others wait, with queueing included in TTFT."),
      L("目标：首字延迟P95 ≤ 10s，逐Token生成延迟P95 ≤ 40ms。",
        "Targets: TTFT P95 ≤ 10 s and TPOT P95 ≤ 40 ms."),
    ))
  ])

  slide(L("结果与证据：IFBench前100题的指令遵循","Results: instruction following on the first 100 IFBench prompts"),L("结果与证据 · 精度评测","Evaluation · Quality"),[
    #v(5mm)
    #grid(columns:(138mm,1fr),gutter:14mm,
      [
        #text(18pt,weight:"bold")[#L("前100道题，共111条指令约束", "100 prompts, 111 instruction constraints")]
        #v(5mm)
        #set text(18pt)
        #table(columns:(90mm,40mm),inset:7pt,stroke:none,
          table.hline(stroke:.8pt+muted),
          [*#L("评分对象","Score")*],[*#L("结果","Result")*],
          table.hline(stroke:.5pt+light),
          L("题目严格评分","Prompt strict"),[#int(quality.counts.prompt_strict) / 100],
          L("题目宽松评分","Prompt loose"),[#int(quality.counts.prompt_loose) / 100],
          L("指令严格评分","Instruction strict"),[#int(quality.counts.instruction_strict) / 111],
          L("指令宽松评分","Instruction loose"),[#int(quality.counts.instruction_loose) / 111],
          table.hline(stroke:.8pt+muted))
      ],
      [#set text(18pt)
        #bullets((
          L("评测范围：原评测脚本选取的IFBench前100题，按原评分器自动判定。",
            "Scope: the first 100 IFBench prompts selected by the original evaluation script, scored with its original verifier."),
          L("题目严格评分：一道题的全部指令约束通过，才算该题通过。",
            "Strict prompt scoring: a prompt passes only when every instruction constraint is satisfied."),
          L("测试设置：开启思考模式，最多生成32768 Token；与性能测试使用同一二进制。",
            "Setup: thinking enabled, up to 32768 output tokens; the same binary as the performance tests."),
        ))])
  ])

  slide(L("质量案例：成功与失败的具体回答", "Quality cases: what passes and what fails"),
    L("结果与证据 · 原始精度案例", "Evaluation · Original quality examples"),[
    #let success = quality.cases.find(c => c.key == "80")
    #let failure = quality.cases.find(c => c.key == "88")
    #text(17pt)[#L("IFBench前100题中的实际回答", "Actual responses from the first 100 IFBench prompts")]
    #v(4mm)
    #set par(leading:4pt)
    #grid(columns:(138mm,1fr),gutter:16mm,
      [
        #text(20pt,weight:"bold",fill:rgb("#287657"))[#L("成功 · key80", "PASS · key80")]
        #v(4mm)
        #block(height:28mm)[
          #text(17pt,weight:"bold")[#L("题目要求", "Prompt requirements")]
          #v(2mm)
          #set text(17pt)
          #set par(leading:4pt)
          #L("以Arya将别人伪装成自己为主题，写三句等长、用词不重复的句子；每行逐级缩进。",
            "Imagine Arya disguising others as herself. Write three equal-length sentences using distinct words; indent each line further.")
        ]
        #text(16pt,weight:"bold")[#L("实际回答（完整）", "Actual answer (complete)")]
        #v(3mm)
        #block(height:29mm,fill:rgb("#F0F7F3"),inset:4mm,width:100%)[
          #show raw: set text(font:"Liberation Mono",size:16pt)
          #set par(leading:5pt)
          #raw(success.final,block:true)
        ]
        #v(4mm)
        #set text(17pt)
        #set list(spacing:7pt)
        #bullets((
          L("缩进为0 / 4 / 8个空格；去除缩进后，每句均为28个字符，15个单词无重复。",
            "0 / 4 / 8 leading spaces; 28 characters per sentence after trimming; 15 distinct words."),
          L("两条指令均通过，题目严格评分为1。",
            "Both instructions pass; prompt strict score = 1."),
        ))
      ],
      [
        #text(20pt,weight:"bold",fill:rgb("#B44B45"))[#L("失败 · key88", "FAIL · key88")]
        #v(4mm)
        #block(height:28mm)[
          #text(17pt,weight:"bold")[#L("题目要求", "Prompt requirements")]
          #v(2mm)
          #set text(17pt)
          #set par(leading:4pt)
          #L("回答给定流动问题的n范围；以 !?!? 列表作答，相邻单词的首字母不能相同。",
            "Give the range of n for the stated flow problem. Use !?!? as the list marker; adjacent words must start with different letters.")
        ]
        #text(16pt,weight:"bold")[#L("实际回答（完整）", "Actual answer (complete)")]
        #v(3mm)
        #block(height:29mm,fill:rgb("#FCF1EF"),inset:4mm,width:100%)[
          #show raw: set text(font:"Liberation Mono",size:16pt)
          #set par(leading:5pt)
          #raw(failure.final,block:true)
        ]
        #v(4mm)
        #set text(17pt)
        #set list(spacing:7pt)
        #bullets((
          L("两项都用 !??，要求为 !?!?；标记少一个字符。",
            "Expected !?!?; both items use !??, missing one character from the required marker."),
          L("两条指令仅一条通过，因此这道题未通过。",
            "Only one of two instructions passes, so the prompt fails."),
        ))
      ],
    )
  ])

  // Source/build/portable evaluation from the independent ucloud3 reproduction.
  slide(L("工程化设计：获取与校验", "Engineering: obtain and verify"),
    L("完整源码与固定输入", "Complete source and fixed inputs"), [
    #text(21pt, weight:"bold", fill:accent)[#L(
      "① 取得复验材料，在新目录准备输入", "① Obtain the reproduction bundle and prepare a fresh directory")]
    #v(3mm)
    #show raw: it => { set text(font:"Liberation Mono",size:16pt); set par(leading:3pt); it }
    #raw(repro.commands.obtain, lang:"bash", block:true)
    #v(4mm)
    #set text(17pt)
    #table(columns:(57mm,111mm,124mm), inset:(x:7pt,y:7pt), stroke:none,
      table.hline(stroke:1pt+accent),
      [*#L("输入", "Input")*], [*#L("固定身份", "Fixed identity")*], [*#L("本轮核验", "Checks in this run")*],
      table.hline(stroke:.5pt+light),
      L("完整引擎源码", "Complete engine source"),
      L("1023个文件 · c976692d…", "1023 files · c976692d…"),
      L("归档及逐文件SHA；构建前后复核", "Archive and per-file SHA; checked before/after build"),
      table.hline(stroke:.5pt+light),
      L("模型与量化缓存", "Model and quantized caches"),
      L("固定模型revision；025 / 041 / 047", "Pinned model revision; 025 / 041 / 047"),
      L("模型及478个服务文件重新计算SHA", "Rehashed the model and all 478 serving files"),
      table.hline(stroke:.5pt+light),
    )
    #v(3mm)
    #text(17pt)[#L("在复验材料目录运行；u3源码复验使用原只读缓存；u2量化重建的新产物独立验证。完整流程见 reproduction.md。",
      "Run from the bundle directory. u3 tests reuse existing caches; u2 regenerates and validates separate assets. See the bilingual guide.")]
  ])

  slide(L("工程化设计：量化缓存重建", "Engineering: regenerate quantized caches"),
    L("固定输入，依赖顺序重建", "Fixed inputs, dependency-ordered rebuild"), [
    #text(21pt,weight:"bold",fill:accent)[#L(
      "② 从FP8与固定校准输入重建三类缓存", "② Rebuild three cache families from FP8 and fixed calibration inputs")]
    #v(3mm)
    #show raw: it => { set text(font:"Liberation Mono",size:16pt); set par(leading:3pt); it }
    #raw(repro.commands.quantization,lang:"bash",block:true)
    #v(4mm)
    #set text(17pt)
    #table(columns:(40mm,138mm,114mm),inset:(x:7pt,y:8pt),stroke:none,
      table.hline(stroke:1pt+accent),
      [*#L("阶段", "Stage")*],[*#L("重新采集与拟合", "Fresh capture and fitting")*],[*#L("生成对象", "Outputs")*],
      table.hline(stroke:.5pt+light),
      [025],L("原Q4/Q5前向采集 → FFN拟合", "Base Q4/Q5 forward capture → FFN fit"),
      L("64层gate/down · 256文件", "64 gate/down layers · 256 files"),
      table.hline(stroke:.5pt+light),
      [041],L("加载新025采集 → 输入投影拟合", "Capture with new025 → input projection fit"),
      L("64层输入投影 · 128文件", "64 input projections · 128 files"),
      table.hline(stroke:.5pt+light),
      [047],L("加载新025＋041采集 → 输出投影拟合", "Capture with new025 + 041 → output fit"),
      L("固定47层输出投影 · 94文件", "47 selected output projections · 94 files"),
      table.hline(stroke:.8pt+muted))
    #v(4mm)
    #text(17pt)[#L("固定4条训练、2条留出输入；FP8按逐文件revision/SHA获取，基础.ninfer模型单独获取。",
      "Four fixed training and two held-out inputs; per-file FP8 revisions/hashes and a separately acquired base .ninfer model.")]
    #v(3mm)
    #text(17pt,fill:accent)[#L(repro.quantization_note_zh,repro.quantization_note_en)]
  ])

  slide(L("工程化设计：源码构建", "Engineering: build from source"),
    L("新构建与新身份", "New build, new identity"), [
    #text(21pt, weight:"bold", fill:accent)[#L(
      "③ 正常完整构建，绑定新二进制", "③ Build normally and bind the new executable")]
    #v(3mm)
    #show raw: it => { set text(font:"Liberation Mono",size:18pt); set par(leading:3pt); it }
    #raw(repro.commands.build, lang:"bash", block:true)
    #v(5mm)
    #set text(19pt)
    #set list(spacing:8pt)
    #bullets((
      L("Release / SM89 · CUDA13.2.51 / GCC13.4；新build目录完整编译。",
        "Release / SM89 · CUDA13.2.51 / GCC13.4; a full build in a fresh directory."),
      L("构建前后1023个源文件一致；7项调度CPU测试通过，保存55项动态依赖。",
        "All 1023 source files match before/after build; 7 scheduler CPU tests pass, with 55 runtime dependencies recorded."),
      L("构建产物与运行配置一起归档；后续性能、精度复验直接使用这一构建。",
        "Archive the executable with its runtime configuration; use this build for the following performance and quality reproduction."),
    ))
    #v(5mm)
    #text(17pt,fill:accent)[#L("核对 build-job.json、runtime-identity.json 和 rebuilt-recipe-binding.json。",
      "Inspect build-job.json, runtime-identity.json and rebuilt-recipe-binding.json.")]
  ])

  slide(L("工程化设计：启动服务", "Engineering: start the service"),
    L("复验性能配置", "Reproduction performance profile"), [
    #text(21pt, weight:"bold", fill:accent)[#L(
      "④ 在新部署目录启动性能服务", "④ Start the performance service in the new deployment")]
    #v(3mm)
    #show raw: it => { set text(font:"Liberation Mono",size:17pt); set par(leading:3pt); it }
    #raw(repro.commands.serve_performance, lang:"bash", block:true)
    #v(6mm)
    #set text(19pt)
    #bullets((
      L("客户端14 / 引擎13；普通MTP K3，KV121856，上下文65536，输入块2048。",
        "Client 14 / native 13; ordinary MTP K3, KV121856, context 65536 and prefill chunk 2048."),
      L("本次服务和客户端均用新构建配方；19个环境开关保持冻结100配置。",
        "Server and client both select the rebuilt recipe; the 19 environment flags retain the frozen 100 configuration."),
      L("服务日志和launch.json记录真实进程、二进制、路径与参数。",
        "Service logs and launch.json record the live process, executable, paths and arguments."),
    ))
  ])

  slide(L("工程化设计：性能复验", "Engineering: reproduce performance"),
    L("原128条测量", "Original 128 measurements"), [
    #text(21pt, weight:"bold", fill:accent)[#L(
      "⑤ 同机另一个终端运行原性能评测", "⑤ Run the original benchmark in another local terminal")]
    #v(3mm)
    #show raw: it => { set text(font:"Liberation Mono",size:17pt); set par(leading:3pt); it }
    #raw(repro.commands.performance, lang:"bash", block:true)
    #v(4mm)
    #set text(19pt)
    #bullets((
      L("128正式＋8预热＋1探测；冷8192 / 输出512，保留正式阶段初始请求。",
        "128 measured + 8 warmup + 1 probe; cold 8192 / output 512, retaining initial measured requests."),
      L("入口核对真实服务身份和全部137条请求；原始记录在performance-C14-001/c14。",
        "Verify the live service and all 137 payloads; raw records are in performance-C14-001/c14."),
    ))
  ])

  slide(L("工程化设计：复验性能结果", "Engineering: reproduced performance"),
    L("本次新构建的实际结果", "Measured result of this rebuild"), [
    #text(21pt,weight:"bold",fill:accent)[#L("⑥ 本次复验的原始JSON节选", "⑥ Original JSON excerpt from this reproduction")]
    #v(3mm)
    #text(18pt)[#L("ucloud3 · 新二进制664cfbb9… · 客户端14 / 引擎13 · 延迟单位ms",
      "ucloud3 · new executable 664cfbb9… · client 14 / native 13 · latency in ms")]
    #v(3mm)
    #show raw: it => { set text(font:"Liberation Mono",size:18pt); set par(leading:4pt); it }
    #raw(repro.performance_json,lang:"json",block:true)
    #v(3mm)
    #text(17pt,fill:accent)[#L(repro.performance_note_zh,repro.performance_note_en)]
  ])

  slide(L("工程化设计：质量复验", "Engineering: reproduce quality"),
    L("原100题独立配置", "Separate original 100 profile"), [
    #set par(leading:3pt,spacing:4pt)
    #text(19pt,weight:"bold",fill:accent)[#L(
      "⑦ 正常停止性能服务，再启动质量服务", "⑦ Stop the performance service, then start quality")]
    #v(2mm)
    #show raw: it => { set text(font:"Liberation Mono",size:16pt); set par(leading:2pt); it }
    #raw(repro.commands.serve_quality,lang:"bash",block:true)
    #v(3mm)
    #text(19pt,weight:"bold",fill:accent)[#L("同机另一个终端评测", "Evaluate in another local terminal")]
    #v(2mm)
    #raw(repro.commands.quality,lang:"bash",block:true)
    #v(3mm)
    #text(17pt)[#L("客户端/引擎13 · KV532480 · max32768 · 自然EOS、默认thinking与采样。",
      "Client/native 13 · KV532480 · max32768 · natural EOS, default thinking and sampling.")]
    #v(2mm)
    #text(17pt,fill:accent)[#L("入口先核对进程并等待模型就绪；原始质量输出在quality100-002/evaluation/accuracy。",
      "Verify process identity and wait for model readiness; quality evidence: quality100-002/evaluation/accuracy.")]
  ])

  slide(L("工程化设计：复验质量结果", "Engineering: reproduced quality"),
    L("同一新构建的独立质量评测", "Separate quality evaluation of the same rebuild"), [
    #text(21pt,weight:"bold",fill:accent)[#L("⑧ IFBench前100题的复验评分", "⑧ Reproduction scores on the first 100 IFBench prompts")]
    #v(3mm)
    #text(17pt)[#L("新二进制664cfbb9… · C13 / KV532480 · 分值为比例",
      "New executable 664cfbb9… · C13 / KV532480 · scores are proportions")]
    #v(3mm)
    #show raw: it => { set text(font:"Liberation Mono",size:17pt); set par(leading:3pt); it }
    #raw(repro.quality_json,lang:"json",block:true)
    #v(3mm)
  ])

  slide(L("工程化设计：异常处理","Engineering: exception handling"),
    L("失败有出口，结果有记录","Bound failures and preserve their records"),[
    #v(3mm)
    #set text(18pt)
    #table(columns:(57mm,113mm,122mm),inset:(x:8pt,y:12pt),stroke:none,
      table.hline(stroke:1pt+accent),
      [*#L("异常场景","Failure")*],[*#L("处理方式","Response")*],[*#L("保留记录","Retained evidence")*],
      table.hline(stroke:.5pt+light),
      L("启动或端口故障","Startup failure / port in use"),
      L("启动前检查端口，设置服务就绪超时。","Check before launch; bound the readiness wait."),
      L("启动日志、退出码、实际配置。","Startup log, exit code and actual config."),
      table.hline(stroke:.5pt+light),
      L("显存不足","Out of memory"),
      L("检查KV缓存、并发上限和CUDA Graph占用，调整后复测。","Inspect KV, active capacity and Graph; retest a revised config."),
      L("OOM日志与原参数。","OOM log and original parameters."),
      table.hline(stroke:.5pt+light),
      L("非法请求","Invalid request"),
      L("模型ID或输入Token无效时，返回HTTP 400。","Reject an invalid model ID or token with HTTP 400."),
      L("请求、状态码与错误响应。","Request, status code and error response."),
      table.hline(stroke:.5pt+light),
      L("请求异常结束","Timeout / empty / truncated output"),
      L("保留超时、空输出及截断记录，按原规则统计。","Keep failures and partial output; score by the original rules."),
      L("客户端记录、结束原因、输出长度。","Client record, finish reason and output length."),
      table.hline(stroke:.8pt+muted))
  ])

  slide(L("工程化设计：可靠性","Engineering: reliability"),
    L("从接口契约到状态一致性","From API contracts to state consistency"),[
    #v(3mm)
    #set text(18pt)
    #table(columns:(48mm,118mm,126mm),inset:(x:8pt,y:12pt),stroke:none,
      table.hline(stroke:1pt+accent),
      [*#L("检查对象","Scope")*],[*#L("一致性要求","Invariant")*],[*#L("验证方式","Verification")*],
      table.hline(stroke:.5pt+light),
      L("API输出","API output"),
      L("流式响应的结束状态、Token计数与结束原因一致。","Consistent stream termination, token counts and finish reasons."),
      L("通过真实请求验证流式与非流式响应、用量统计及参数校验。","Probe streamed/non-streamed output, usage and invalid inputs."),
      table.hline(stroke:.5pt+light),
      L("并发状态","Concurrent state"),
      L("各请求独立提交、回滚，复用槽位时保持状态隔离。","Independent commit and rollback; slot reuse preserves isolation."),
      L("覆盖高编号槽位、部分提交、取消/重置及CUDA Graph重放。","High slots, partial commits, cancel/reset and Graph replay."),
      table.hline(stroke:.5pt+light),
      L("服务进程","Service process"),
      L("清理前核对身份，避免误停其他进程。","Verify process identity before cleanup."),
      L("核对PID、启动时间及可执行文件，通过pidfd回收进程。","Check PID, start time and executable; clean up via pidfd."),
      table.hline(stroke:.8pt+muted))
    #v(7mm)
    #text(19pt)[#L("状态融合实验：2013条配对记录一致；86项测试通过，4项跳过。",
      "State-fusion experiment: 2013 matching records; 86 tests passed, 4 skipped.")]
    #v(3mm)
  ])

  slide(L("工程化设计：可观测性","Engineering: observability"),
    L("同时观察用户等待、引擎执行与GPU资源","Observe user latency, engine work and GPU resources"),[
    #text(20pt,weight:"bold")[#L("请求执行时序：开始处理 → 首个Token → 完成",
      "Real request traces: admission → first token → finish")]
    #v(3mm)
    #text(18pt)[#L("085版本、16并发：展示最先开始处理的8个正式请求，以首个请求开始处理为零点",
      "085 / C16 · First 8 measured requests, ordered by engine admission")]
    #v(3mm)
    #align(center)[#chart("request-trace",language,width:276mm)]
  ])

  // Cost/performance/tradeoffs: same-run diagnosis and memory, then formal policy comparisons.
  slide(L("当前瓶颈：预填充干扰解码执行", "Bottleneck: prefill delays decode service"),
    L("成本、性能与取舍 · 执行时序证据", "Cost and tradeoffs · Execution timing evidence"),[
    #v(4mm)
    #grid(columns:(174mm,1fr),gutter:10mm,
      chart("cost-prefill-attribution",language,width:174mm),
      bullets((
        L("TPOT P95为49.40ms，其中其他请求的预填充占31.93ms（64.6%），本请求参与的解码占17.45ms。",
          "Competing prefill contributes 64.6% of TPOT P95: 31.93 of 49.40 ms."),
        L("8192 Token输入的预填充耗时中位数为0.909s，仍是优化重点。",
          "Each 8192-token input still needs a median 0.909 s of prefill execution."),
        L("085版本16并发单轮诊断，128个请求、无前缀缓存复用；按P95对应请求加权分解。",
          "One 085 / C16 diagnostic run, 128 cold-input requests; original P95 contributors."),
        L("依据执行器同步记录的耗时分析请求间干扰，不代表GPU独占执行时间。",
          "Synchronized executor wall time measures service interference, not exclusive GPU time."),
      )))
  ])

  slide(L("资源消耗：推理服务的显存占用", "Resources: GPU memory during actual serving"),
    L("成本、性能与取舍 · 同一085诊断轮", "Cost and tradeoffs · The same 085 diagnostic run"),[
    #v(4mm)
    #grid(columns:(174mm,1fr),gutter:10mm,
      chart("cost-serving-memory",language,width:174mm),
      [
        #text(18pt,weight:"bold")[#L("显存组成（GiB）", "Memory breakdown (GiB)")]
        #v(4mm)
        #set text(16pt)
        #table(columns:(84mm,24mm),inset:(x:3pt,y:8pt),stroke:none,
          table.hline(stroke:.8pt+muted),
          L("权重与量化缓存", "Weights + quant. caches"),[21.53],
          L("KV缓存", "KV cache"),[4.66],
          L("递推与序列状态等", "Recurrent / sequence state"),[4.91],
          L("工作区与CUDA Graph", "Workspace + Graph"),[0.47],
          L("其他（未细分）", "Unclassified residual"),[0.40],
          table.hline(stroke:.8pt+muted))
        #v(5mm)
        #set text(18pt)
        #bullets((
          L("085同轮测试采样173秒、共173次，显存占用均为31.97GiB。",
            "Same 085 run: 173 samples over 173 s, all at 31.97 GiB."),
          L("分项依据启动分配记录，总量依据设备采样；CUDA Graph实际占用138MiB。",
            "Startup allocations explain components; device samples give totals. Graph uses 138 MiB."),
        ))
      ])
  ])

  slide(L("并发取舍：用首字延迟预算缓解生成阶段竞争", "Tradeoff: spend TTFT headroom to limit contention"),
    L("成本、性能与取舍 · 请求准入", "Cost and tradeoffs · Request admission"),[
    #text(20pt,weight:"bold")[#L("同为14并发请求，比较引擎的两种活动请求上限。",
      "14 concurrent client requests; compare two active-request limits.")]
    #v(3mm)
    #chart("admission-c14",language,width:292mm)
    #v(2mm)
    #set text(18pt)
    #set par(leading:4pt)
    #bullets((
      L("机制与代价：最多接纳13个活动请求，其余等待；等待计入TTFT，以此限制生成阶段的资源竞争。",
        "Mechanism and cost: admit at most 13 active requests; others wait. Queueing counts toward TTFT and limits active contention."),
      L("选择依据：14个活动槽的TPOT超限；13个槽两轮同时达标，总吞吐306.04–309.95 tok/s。",
        "Choice: 14 slots miss TPOT; 13 slots meet both SLOs twice at 306.04–309.95 tok/s."),
    ))
  ])

  slide(L("总结与不足","Conclusions and limitations"),L("完成度与下一步","Delivered work and next steps"),[
    #v(4mm)
    #pipeline((L("INT4执行","INT4 execution"),
      L("请求状态隔离与调度","Request state + scheduling"),
      L("延迟约束下的并发能力","Latency-constrained capacity")))
    #v(10mm)
    #bullets((
      L("完成INT4推理、进度调度与MTP并发适配；14并发两次测试均满足SLO。",
        "Delivered INT4 inference, progress-aware scheduling and concurrent MTP; both tests at concurrency 14 meet SLOs."),
      L("IFBench前100题：严格通过85题，宽松通过88题。",
        "First 100 IFBench prompts: 85 strict passes and 88 loose passes."),
      L("当前不足：预填充仍影响解码，15/16并发的生成延迟超限。",
        "Remaining bottleneck: prefill still delays decode; concurrency 15/16 exceeds the TPOT limit."),
      L("下一步：优化预填充算子与调度，增加重复测试和完整数据集评测。",
        "Next: improve prefill kernels and scheduling, repeat load tests, and evaluate the full dataset."),
    ))
  ])
}
