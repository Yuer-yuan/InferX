# InferX 答辩展示 / Academic presentation

| 版本 / Edition | 实时预览 / Live preview | PDF | Typst 入口 / Entry |
|---|---|---|---|
| 中文 | [打开预览](http://127.0.0.1:18081/) | [InferX-ZH.pdf](InferX-ZH.pdf) | [InferX-ZH.typ](InferX-ZH.typ) |
| English | [Open preview](http://127.0.0.1:18082/) | [InferX-EN.pdf](InferX-EN.pdf) | [InferX-EN.typ](InferX-EN.typ) |

两版各 **31 页，16:9**，含封面。使用 [Touying 0.8.0 的 Metropolis 主题](https://touying-typ.github.io/docs/themes/metropolis)：左对齐封面、暗红细线与标题栏、普通无衬线字体、图 + bullet points；按要求依次覆盖八项结构。汇报预算五分钟，现场 Demo 另计。[main.pdf](main.pdf)始终与中文版相同。

**编辑 [deck.typ](deck.typ)** 可同时修改中英文正文与版式。保存后，本地预览自动刷新，两份 PDF 自动编译。预览地址仅在运行预览服务的这台机器上有效。

## 编译与实时预览

在仓库根目录运行：

```sh
# 开启或重新连接预览；服务在后台持续运行
python3 submission/tooling/inferx-submission/scripts/preview_typst.py

# 手动编译两版，并渲染全部页面供检查
python3 submission/tooling/inferx-submission/scripts/build_typst.py --render

# 停止本项目的预览进程
python3 submission/tooling/inferx-submission/scripts/preview_typst.py --stop
```

运行环境：Typst 0.15.1、Touying 0.8.0（首次编译自动下载）、Tinymist 0.15.10、Python 3.11+、Poppler。正文使用 Liberation Sans 与 Noto Sans CJK SC。服务进程和编译日志位于 `submission/records/typst-live/`；编译记录位于 `submission/records/typst-build.json`。

English: edit the shared `deck.typ`; saving it updates both live previews and both PDFs. Preview links are local to the machine running the service. Each edition has 31 slides including the cover.

## 图表与机制图

- 测量图由 [build_charts.py](../../tooling/inferx-submission/scripts/build_charts.py) 读取已核验的 [results.json](../../track-a/data/results.json)，用 **Matplotlib 3.10.7** 导出到 [figures/](figures/) 的矢量 PDF，供 Typst 直接引用。保留坐标、单位与阈值；中文图表用 Droid Sans Fallback，英文用 Liberation Sans，字体均嵌入。
- 重绘测量图：`uv run submission/tooling/inferx-submission/scripts/build_charts.py`。预览服务会在数据或绘图脚本改变时重绘；更新实验记录前先核对版本、条件与来源。
- 机制图由 [deck_figures.py](../../tooling/inferx-submission/scripts/deck_figures.py) 定义独立节点、标签、连接与端口；[native-figures.typ](native-figures.typ) 用原生矢量对象排版。
- 同一套对象导出 [中文 draw.io](InferX-figures-ZH.drawio) / [English draw.io](InferX-figures-EN.drawio)，均为可编辑的原生对象。修改图形请优先改生成器；重新生成会覆盖 draw.io 导出。

图表为真实测量值；机制图的时间块和候选数仅解释机制，不表示实测持续时间或接受率。

主讲围绕P/D干扰与算子效率两项瓶颈，MTP作为并发工程适配。结果部分直接展示当前并发测试、IFBench前100题和成功/失败案例；移除旧9270结果页、内部历史编号和重复的统计限定。完整比较条件、区间估计及历史数据保留在讲稿和技术材料中。

## 章节与讲解预算

| 页 / Slide | 章节 / Chapter | 本页重点 / Focus | 秒 / s |
|---:|---|---|---:|
| 1 | 封面 / Cover | 项目与团队 / Project and team | 5 |
| 2 | 项目概述 / Overview | 单卡服务目标与交付 / Goal and deliverable | 10 |
| 3 | 问题与目标 / Goals | 两项性能瓶颈 / Two performance bottlenecks | 14 |
| 4 | 整体方案 / Design | 设计位置与框架选择 / Architecture and framework | 13 |
| 5 | 整体方案 / Design | ninfer-4090的适配优势 / Why ninfer-4090 fits | 12 |
| 6 | 核心实现 / Implementation | I：分块与调度决策 / Chunking and scheduling | 12 |
| 7 | 核心实现 / Implementation | I：实际提交进度与估计公式 / Commit history and estimator | 12 |
| 8 | 核心实现 / Implementation | 有界9成员与两轮全员 / Bounded nine-row service | 14 |
| 9 | 核心实现 / Implementation | II：H512/G512与INT4计算 / H512/G512 and INT4 | 12 |
| 10 | 核心实现 / Implementation | II：缓存、算子接入及路由验证 / Cache and route integration | 12 |
| 11 | 核心实现 / Implementation | MTP并发工程适配 / Concurrent MTP integration | 12 |
| 12 | 核心实现 / Implementation | 逐请求状态提交 / Per-request state commit | 12 |
| 13 | 结果与证据 / Evaluation | 100版本六组性能曲线 / Build100 campaign | 14 |
| 14 | 结果与证据 / Evaluation | C14两轮通过及C15/C16失败 / Capacity boundaries | 12 |
| 15 | 结果与证据 / Evaluation | IFBench前100题 / First 100 IFBench prompts | 10 |
| 16 | 结果与证据 / Evaluation | 成功与失败原文 / Original pass and fail cases | 14 |
| 17 | 工程化设计 / Engineering | 获取与校验输入 / Obtain and verify inputs | 6 |
| 18 | 工程化设计 / Engineering | 量化缓存重建 / Regenerate quantized caches | 8 |
| 19 | 工程化设计 / Engineering | 从源码构建服务 / Build the service from source | 8 |
| 20 | 工程化设计 / Engineering | 启动性能配置 / Start the performance profile | 6 |
| 21 | 工程化设计 / Engineering | 运行原128性能测试 / Run the original 128 performance test | 4 |
| 22 | 工程化设计 / Engineering | 复验性能结果 / Reproduction performance results | 4 |
| 23 | 工程化设计 / Engineering | 运行原100质量测试 / Run the original 100 quality test | 4 |
| 24 | 工程化设计 / Engineering | 复验质量结果 / Reproduction quality results | 4 |
| 25 | 工程化设计 / Engineering | 异常处理 / Exception handling | 8 |
| 26 | 工程化设计 / Engineering | 可靠性验证 / Reliability checks | 10 |
| 27 | 工程化设计 / Engineering | 真实请求时序 / Measured request traces | 10 |
| 28 | 成本与取舍 / Tradeoffs | 预填充干扰归因 / Prefill interference | 8 |
| 29 | 成本与取舍 / Tradeoffs | 同轮显存组成 / Same-run memory | 5 |
| 30 | 成本与取舍 / Tradeoffs | 14并发的准入取舍 / Admission tradeoff at C14 | 8 |
| 31 | 总结与不足 / Conclusions | 局限与下一步 / Limitations and next steps | 5 |

复现八页依次覆盖固定v1包获取与输入校验、116量化重建、正常源码构建、性能服务启动、原128条性能评测、实际性能JSON、独立原100题质量配置和实际质量JSON。完整命令见[中文复验指南](../../track-a/reproduction.md) / [English guide](../../track-a/reproduction-en.md)；新量化另有[中文](../../track-a/quantization-reproduction.md) / [English](../../track-a/quantization-reproduction-en.md)完整流程。服务和客户端绑定同一新recipe；性能client14/native13/KV121856，质量client13/native13/KV532480，先后运行。

[clean-reproduction.json](clean-reproduction.json)导入ucloud3正常重建的新二进制664cfbb9…及原始结果：性能TTFT P95 9.89641735465s、TPOT P95 38.28584995440313ms、吞吐304.89665937584215 tok/s；质量严格90/100、宽松96/100，指令100/111、106/111，错误0、空final1、长度截断1。空final与截断是同一个key88，保留在分母中。JSON节选保留原字段、嵌套及数值；原质量汇总的指令比例为四舍五入后的0.9009/0.955，不从小数反推整数计数。

这些664实测复用了已固定的旧478文件缓存和宿主环境。116从公开FP8输入重建缓存，单独报告阶段、加载与评测，见[交付入口](../../DELIVERY.md)；不继承664或历史e4ed成绩。首次质量001因就绪等待不足在0题时超时，记录保留；r2等待只读/models就绪后实际完成002，未增加生成探测或重试已评题目。便携包保留实际旧性能wrapper和r2质量的对应关系，新运行另建输出目录。

主结果页仍为历史100版本e4ed的六组性能曲线与105完整质量85/88，不由664结果替换。原105的空回答key50没有长度截断，和664/002的key88情况不同。[selected-reproduction.json](selected-reproduction.json)保留旧100/105节选，当前工程页使用独立clean-reproduction.json。

可观测性页使用真实的`admission`、`first_token`、`request_finish`事件。图中“开始处理”对应引擎的`admission`事件，以此排序展示前8个正式请求。每行表示请求生命周期；开始处理时间不同于HTTP到达时间，连线不表示GPU独占执行。全部128条正式请求先与同轮客户端记录唯一关联，再选择展示范围。

保留的[生成脚本](../../tooling/inferx-submission/scripts/build_request_trace.py)读取已有日志，可在CPU上重新生成中英文矢量图、[完整事件与来源审计](../../track-a/data/request-trace.json)及[Chrome Trace文件](../../track-a/data/request-trace.chrome.json)。Chrome Trace文件可在兼容的trace viewer中打开，按request_id查看各请求的开始处理、首个Token和完成事件。

```sh
uv run submission/tooling/inferx-submission/scripts/build_request_trace.py
```

脚本默认展示8条请求；`--trace-dir`指定原始日志目录，`--requests`指定展示条数，`--output-dir`与`--data-dir`分别指定图表和审计/trace输出目录。常规`build_charts.py`也会生成这张图。

讲稿：[中文](../defense-notes.md) / [English](../defense-notes-en.md)。演示：[现场性能评测命令](../demo.md)。证据与部署：[技术报告](../../track-a/README.md)。

历史 Beamer 源文件保留在 `submission/records/legacy-beamer/`；当前版本以 Typst 源码与 PDF 为准。

成本与取舍的前两页使用同轮085诊断与显存记录，[cost-evidence.json](../../track-a/data/cost-evidence.json)登记27个原始来源及哈希；CPU复算命令为`python3 submission/tooling/inferx-submission/scripts/build_cost_evidence.py`。第30页为100版本的14并发准入对照：ucloud1引擎14与ucloud3引擎13两轮，数据来自[selected-campaign.json](selected-campaign.json)，跨节点差值不作为准入控制净收益。测量图由`cost_charts.py`生成Matplotlib矢量PDF，等待计入TTFT，活动请求上限不等于每轮实际批量。历史085/086/088正式调度对照仍保留于技术报告与图表资产。intro仅写“支持14并发请求”，详细配置保留于对照和复现页。

已从主讲中移除的历史并发曲线，使用9270版本在同一ucloud1节点的C12/C13/C14复测；TTFT与TPOT由原128条正式请求重新计算P95，吞吐沿用原phase分母。每档8次预热、冷8192/512、K3；活动槽与KV容量同步增加，每槽8704 Token。三档均为单轮观察，保留在技术记录与原图资产中。

案例来自100版本105完整原100题评测：key80和key88均展示完整final，缩进与符号保持原样。key80每句28字符、缩进0/4/8、15个不同单词；key88误用!??而非!?!?，两条指令仅一条通过，题目严格/宽松均0。原预测、题目、review、HTTP与native记录交叉核对，583份归档成员逐个SHA验证，汇总在[selected-quality.json](selected-quality.json)。历史6632原始案例仍保存在[result-details.json](result-details.json)，不冒充本版结果。

数据选择入口为[result-details-sources.json](../../track-a/data/result-details-sources.json)，CPU复算使用[build_result_details.py](../../tooling/inferx-submission/scripts/build_result_details.py)。`build_charts.py`与`build_typst.py`都会核验该来源；更新来源后重新生成曲线、编译PDF并检查版面。

当前100版本六组测量（C4/8/12/13、C14两轮）由[selected-campaign-sources.json](../../track-a/data/selected-campaign-sources.json)固定来源，[build_selected_campaign.py](../../tooling/inferx-submission/scripts/build_selected_campaign.py)核验原始P95、profiling时间窗分母、全部137条冷请求以及binary/source与活动槽/KV。u1的client14/native14失败作为独立comparison_runs保留，不画成同机纯因果收益。100是独立重建二进制，不宣称纯host加速。

有界策略图来自`deck_figures.py`中的`bounded-decode`：原调度选择decode后，仅在B10–14且触发余量保护时服务最紧迫9个，随后强制两次实际全员解码。预填充不减少两轮计数；图中箭头表示策略顺序，不表示实测时长。图同时导出原生Typst和双语draw.io对象。

100版本的边界表保留C14/native14、C15/native13和C16/native16的完整失败结果。C16/native13为额外准入诊断，两项延迟均失败，保留于selected-campaign.json和讲稿；不加入六点主曲线，不提高已观察达标容量。吞吐按原profiling阶段起止计算，不用请求时间包络替换。

完整质量导入器：[build_selected_quality.py](../../tooling/inferx-submission/scripts/build_selected_quality.py)，固定输入注册表：[selected-quality-sources.json](../../track-a/data/selected-quality-sources.json)。只做CPU校验与导出，不发起推理。质量与性能二进制/source一致，但质量C13/KV532480、性能C14/native13/KV121856为独立协议。原100题结果不代表全量IFBench；没有配对关闭组，不推断量化无损。

本轮移除原第13、14页；保留同时整合的复验工程内容，合计31页。表内主讲预算288秒，另留12秒用于转场与指图。正文中的原始评分JSON只保留与展示有关的字段，不改变实际样本、分母或得分。
