---
name: inferx-submission
description: Prepare or refresh InferX Track A submission documents and bilingual academic Typst slides, export Matplotlib vector-PDF charts, run live previews and check all rendered pages. Use when updating this team's materials or results; does not run model experiments or publish submissions.
---

# InferX 参赛材料更新

队名“你说啥都队”，项目 InferX，成员陈文欣、张志雄、郭坤昌，依据现有提交入口；真实研发分工仍由团队填写，不编造贡献。展示姓名从 `track-a/data/sources.json` 的 `team.members` 生成，不硬编码 A/B/C。规则依据是仓库 `submission/organizer/AI比赛评分规则-v3-for现场学生.md`，PPT遵循1.3的八个章节结构和顺序，覆盖第2部分10/40/25/10/15分。“约8页”是建议，用户已明确页数可按内容展开，不设固定页数；同时控制项目汇报在规则的5分钟内。先读取 `submission/records/2026-10-18-submission-checklist.md` 及最新执行记录；旧准备记录仅作历史。

## 最新用户要求：Typst 学术答辩与实时预览（2026-10-18）

最新封面要求：使用 Touying 0.8.0 Metropolis 主题；封面所有信息左对齐，移除 `Qwen3.8-27B · 1 × 48GB RTX 4090` 硬件说明行，采用暗红细线、清晰的标题/副标题/作者层级和留白。正文仍用暗红标题栏、图 + bullet points。不要恢复居中封面或在封面加入硬件参数。

当前用户要求覆盖后面的旧 Beamer、Madrid、TeX、白底海报等条款：用 **Typst** 排版，中文与英文两份独立版本，含正式封面（页数按当前内容编译生成）；暗红色（#7E2635）标题栏、普通无衬线字体、图 + 2–4 个 bullet points，规整学术风格。正文共享 `submission/basic/presentation/deck.typ`，入口 `InferX-ZH.typ` / `InferX-EN.typ`。不要恢复被用户否定的海报布局和夸张大数字。

- 测量图用 Matplotlib 3.10.7 读取已核验 results.json，导出矢量 PDF 嵌入 Typst。执行 `uv run submission/tooling/inferx-submission/scripts/build_charts.py`。不得手画测量点或编造新结果。
- 原生矢量机制图和可编辑双语 draw.io 由 `deck_figures.py` 的同一对象定义生成；Typst 用 `native-figures.typ` 渲染。生成器是几何与语义的权威源。
- 编译并渲染全部页：`python3 submission/tooling/inferx-submission/scripts/build_typst.py --render`；实际打开中英全部页面后，才在 `submission/records/typst-build.json` 记录当前 PDF SHA 的视觉 PASS。
- 开启持续预览及保存后 PDF 编译：`python3 submission/tooling/inferx-submission/scripts/preview_typst.py`。中文 http://127.0.0.1:18081/ ，英文 http://127.0.0.1:18082/ 。后台进程及日志在 `submission/records/typst-live/`；只停止该记录内属于本项目的进程。
- 验证：`uv run submission/tooling/inferx-submission/scripts/check_typst.py`。它检查当前源文件、PDF、预览图、视觉记录、证据哈希及矢量文字；不能代替人工审图或干净环境复验。
- 核心设计要解释具体实现：调度统计什么、何时更新、怎样进入决策；INT4的校准、变换、分组、累加与实际路由；MTP扩容涉及的算子、布局、状态和Graph，以及部分提交/取消的处理顺序。用公式、机制图或表格配合要点，必要时拆页，并让每项验证对应其实际版本与范围。
- 中文按专业技术汇报的习惯独立组织表达，不逐词翻译英文。统一正文、图例、坐标轴和表头的术语；优先用“批处理”“并发上限”“待处理的预填充负载”等明确说法，避免内部简称堆叠。保留必要英文术语、原始JSON字段和命令，不为润色改变测量含义。
- 标题下的灰色章节标签和小字条件栏暂时隐藏；不恢复底部小字说明。关键结论的单轮、版本及比较范围用必要的正文说明，完整条件留在技术材料与讲稿。
- 工程化设计按复现、异常处理、可靠性、可观测性展开。复现保留用户指定的080命令，性能汇总与精度分别贴真实JSON字段节选，不改成汇总表；不恢复“先完成性能测试”“等待队列设容量”等底部说明。总页码由Typst实际页数生成。
- 可观测性展示原始日志的实际请求生命周期，用`build_request_trace.py`生成中英文矢量PDF、审计JSON和Chrome Trace。保留脚本与源文件哈希，不用示意时间冒充trace；引擎接纳与HTTP到达分开，请求生命期不称GPU独占执行。
- 当前以 Typst 工作流为准。旧 `refresh.py` 的 TeX 生成功能及 `assets/main.tex.in` 只保留作历史，不能用它覆盖当前 PDF。原始证据变化先人工核对 registry、统计和比较边界，再更新 results.json 及本工作流。
- 仍需保留八章顺序、上游署名、失败结果、单轮范围、质量与性能不同版本身份。旧基线100+8与新版128+8不得包装为统一加速比。

## 学术叙事要求（2026-10-18）

最新用户要求：主线围绕P/D阶段干扰与算子效率两项瓶颈。MTP多活动槽定位为工程适配，讲批处理、缓冲区、Graph与逐请求状态管理，不包装成第三项独立研究挑战。整体方案标出优化与适配的位置；核心实现解释观察、设计理由与状态不变量；结果页直接展示当前并发能力与任务质量，不再主讲旧9270容量页或“至9270”等内部历史编号。

面向听众组织正文：不反复添加“尚不能证明长期稳定”“吞吐不等于SLO容量”“不能作因果归因”等审计口吻，也不在主讲页显示置信区间、重采样方法或failures字段解释。完整比较条件、统计方法与限制放在讲稿和技术材料，正文用清楚的节点/配置标识与实测次数保持准确，不改写原始数值或声称未证明的收益。质量必须直写“IFBench前100题”，不是全量数据集；保留85/100等实际分母、当前失败配置与真实失败案例。JSON继续用原始字段节选，可省略与本页无关的计数。

对应证据分开：进度调度同条件对照、INT4历史整版本进展、并发执行与SLO边界、独立质量评价。缺少GPU物理时间线、逐项消融、MTP开关对照时，不能声称已经量化瓶颈份额或证明全部因果。编辑范围以用户当次要求为准，保留其他部分的并行修改。

中文图表使用本领域自然术语：首字延迟、逐Token生成延迟、并发请求数、调度策略、压测时间、请求准入。横轴标变量和单位，不把实验条件写成轴名；时间零点在正文交代，不用英文长句直译。

## 成本、性能与取舍（2026-10-18）

用户最新将这一节交给助手补充，保留其余并行编辑。用三页讲清Prefill干扰、同轮serving显存组成、调度取舍；常规图+bullets，条件与方法融入必要正文，不恢复小字条件栏或底部说明。

- `scripts/build_cost_evidence.py`独立复算085/C16 traceON的原128请求P95归因、同轮173秒显存，并核对085/086/088的正式traceOFF对照；数据在`track-a/data/cost-evidence.json`，每个来源有SHA。
- executor墙钟时段支持服务干扰归因，不是GPU独占时间或SM效率。诊断49.40ms不可替代正式50.19ms。
- 显存组件来自同轮启动分配，GPU总量来自约1Hz采样；不虚构逐池实时曲线。保留未细分余项，Graph实际分配与预算分开。
- 086/088分别基于085，各一次同机同请求对照；不包装为累计路线或稳定因果结论。图由`cost_charts.py`生成矢量PDF，预览同时监测新数据和绘图脚本。

## 提交要求与易读性

最新提交约定（2026-10-18）：用户明确采用完整源码与固定输入重建量化缓存，不把12 GB缓存附件作为必交项。默认复验入口为quantization-reproduction.md，已有旧缓存的664历史复验单独保留。现场Demo提供完整C14性能评测Markdown。用户随后明确授权先Git提交与push，且附件必须是单个小于50,000,000字节的文件，不分卷。先将当前完整源码、文档和必要原始证据推送到固定提交，再用scripts/package_attachment.py从该提交打包；附件含002完整重建源码，未附大记录的文档链接必须改为固定提交链接，保留真实网络获取范围。package_submission.py生成的全量包仅作本地备份。

- 按“交什么、去哪里找、怎样算完成”说明要求。先讲结论和操作顺序，再给参数和哈希；术语首次出现用普通中文解释。正文避免大段版本简称、机器路径和内部流程口令。
- 区分主办方必交项与团队建议：八项结构不等于八页；基础材料为视频或可运行 Demo，现场以真实运行为主，约三分钟录屏为建议备用；赛道 A 架构图可选。P0/P1 是团队优先级，不能自动当作比赛硬门槛。
- 完整复验材料包必须支持他人在同规格机器上取得材料、构建、启动和复测。提供可访问仓库及固定快照或版本；所选版本全部源码、依赖、构建和测试步骤须齐全。旧基础版源码、增量补丁、文档和校验值不能单独替代完整交付。
- 大型公开权重不必重复上传，但须给准确来源、固定版本及获取方式。新增量化缓存须能取得或按完整输入和脚本重建；原服务器路径不等于可用获取方式。
- 原始结果、测试条件、关键配置及异常须可追溯；最终提交文件不得含真实密钥或未经授权的敏感数据；现场改变代码、模型或配置须主动说明。
- 当前入口、成员姓名、展示文件及复验说明保持一致。历史部署文档和草稿标明范围；真实未完成项保留在内部清单，不把“已写说明”标成“已打包、已复验”。

## 提交正文与内部验收

`submission/basic/`、`submission/track-a/` 和入口README直接面向最终提交，不写编辑提示、尚待团队补齐、录屏/分工占位、结果未冻结、请运行校验等内部工作说明。将这些内容集中到 `submission/records/` 的验收记录，列出缺项、具体交付物、责任人/团队及检查方式；每次更新保留此清单。

测试条件、实际版本、样本量、失败结果和因果范围是实验事实，不得为排版或措辞美化而伪造。不能通过删除版本身份把不同配置成绩暗示成联合达标。若用户对实验事实如何呈现有未明确要求，先确认。规则要求的技术局限和“总结与不足”仍按实测研究结论填写。

## 绘图、展示与比较

每次制作或刷新展示材料前，读取 [presentation-and-comparison.md](references/presentation-and-comparison.md)，落实以下要求：

- 使用 `$systems-paper-figurecraft` 设计和核验架构图、机制图与设计决策图。读取该 skill 及适用参考资料，先确定图的主张、语义和证据；沿用可编辑 TikZ，测量图用数据驱动的 PGFPlots 等工具。依赖缺失时先说明并解决，不能声称已使用。
- Beamer 显式使用 `\usetheme{Madrid}`，保留主题自带标题栏和页脚，不另造主题样式；图表内部可按语义配色。标题只写本页所属章节，严格保留规则1.3的八个章节顺序；同一章节可连续多页，不把一章等同于一页。
- 每页只集中讲清一个问题或主题，一句主旨配少量说明，不把章节的全部内容塞进一页。先为各页写出“所属章节、本页唯一问题/结论、版式、主要视觉证据、讲解时长”，再排版；内容需要时拆页，细节可移到技术文档/讲稿，不靠缩小字号塞满。无需仅因超过8页再次确认。
- 按内容交替使用图片页、概念页、对比页、过程页和实践页，具体结构见参考文件。图片页以一张较大的图为主体，保持原始比例并预留图注、脚注和Madrid页脚；连续讲述同一过程时保持图文位置和大小稳定。实践链按代码→流程图→真实运行截图→结果组织，分步展示而不堆满一页。
- 关键优化按“问题→设计选择→效果”组织：说明瓶颈证据、选择理由和实测收益/代价，明确未验证部分。用图和简要文字展现工作量，不以代码行数或记录数代替优化成果。
- 性能比较先检查 baseline，再按可比性选择对照；可加入之前的优化版本，保留失败路线与真实取舍。跨机器、协议、样本量或版本混合的结果不能包装成受控加速比。

先核对并修改 `assets/main.tex.in` 再 refresh，不能因旧模板可编译而沿用旧样式。仅修改流程时将待执行的模板整改记入 records，不把旧PDF标记为已满足新规范。新规范验收须逐项审阅；脚本check通过不自动证明主题、叙事、因果和比较公正性。

## 更新已有结果

1. 找到 InferX 仓库根目录；读取 `submission/track-a/data/sources.json` 和上次 `results.json`。运行本 skill 下 `scripts/refresh.py inventory --repo ROOT`，查看未注册的新报告。inventory 仅为候选发现，不是自动资格判断。
2. 阅读新完成报告与原始汇总；确认样本量、预热、ISL/OSL、错误、版本/缓存身份和服务终态。不要把 status.json 的 last_result 当最新结论。新来源必须人工加入 registry；未完成或仍在写的任务只进待办。
3. 为新来源沿用 registry 的字段：`id/label/node/summary/binary/source_identity/n/warmup/k/kv/identity/raw/native/report`。summary 为原版解析器的单并发结果；identity 是包含完整 binary SHA 的 JSON。`c`是客户端并发；native容量不同时须另填 `native_capacity`，默认值才是`c`，不能把client16/native12称为C12。K、fusion和Verify等开关仍按实际配置分别追踪。identity可使用原证据中的`binary_sha256/source_sha256`或`binary_SHA256/source_SHA256`字段，须核对registry source身份，不伪造归一化来源。原始 JSONL 缺失时将 raw/native 留空，报告会降级验证范围，不承诺原始核验。quality 是原版100评分汇总，需单独版本和报告。role 指定容量图、质量图和成果卡片；改变主版本也须更新 notes、身份和叙述。
4. 执行 `python3 SCRIPT refresh --repo ROOT`。它拒绝计数/长度/P95不符、身份不符或正在变动的输入，生成results.json、generated-results.md、主TeX和manifest，随后自动编译main.pdf。流程待办和registry notes另写 `records/generated-materials-findings.md`。此前生成版本保存在history；最终冻结状态只在内部记录中准确维护。
5. 阅读生成表格，更新 `track-a/` 六类文档、`basic/defense-notes.md` 和 `basic/demo.md`。数值以生成表为准，减少正文重复。按上节规范和 figurecraft 的图形契约更新主模板 `assets/main.tex.in`，核对Madrid、八个章节的覆盖和顺序、单页一个重点、五类版式的合理交替、设计决策及比较范围；重新 refresh。图用 TikZ/PGFPlots 保持可编辑，不需要外部图片；frame数及PDF页数按内容安排，不硬编码为8。若使用图片，复制到 main.tex 同目录，ASCII重命名并引用相对文件名，调整资源校验后保持Overleaf独立编译。
6. 编译是默认交付步骤，不能只做TeX静态检查。优先用已安装XeLaTeX，否则使用 `~/.local/bin/tectonic`（用户级XeTeX体系）；本机已安装Tectonic0.17.0和Fandol字体，发行及字体SHA见records内安装记录。独立编译可执行 `python3 SCRIPT compile --repo ROOT`。refresh和compile会按PDF实际页数自动渲染全部PNG；也可用render子命令单独渲染。编译失败保存日志并修复，不能把旧PDF作为当前TeX结果。核对实际页数与计划、缺字和overfull警告，并实际打开全部渲染页面检查图表/版面，修复后重编译。块级图表后的说明应另起段落，零overfull并不保证没有孤字或错位。
7. 实际看完所有页面后，在 `records/latex-build.json` 设置 `visual_inspection` 为对象：`status: PASS`、`pages: 实际PDF页数`、`pdf_sha256: 当前PDF SHA`、`reviewer`与具体检查说明。只能在真正完成全页视觉检查后登记，不能由生成器自动宣称看过。check会核对实际PDF页数、预览全页覆盖及视觉登记页数，拒绝pending视觉状态、缺字/溢出警告、旧PDF或改动过的预览图。
8. 执行 `python3 SCRIPT check --repo ROOT` 检查链接、当前TeX/PDF绑定和规则映射；结合参考文件的人工验收项更新验收记录，写明单页重点与内容密度、版式与图片比例/留白、连续页稳定性、图形语义/矢量检查、问题与设计及效果的对应、baseline和可比性结论。交付TeX、实际编译的PDF和Markdown。保持TeX可在Overleaf XeLaTeX独立编译。录屏、分工、未发布缓存/源码、干净重建等团队待办只列验收记录，不反复写进提交正文。

SCRIPT 指本 skill 实际目录的 `scripts/refresh.py`，不要硬编码用户级路径。仓库内可直接运行 `python3 submission/tooling/inferx-submission/scripts/refresh.py ...`。skill说明、参考文件、脚本或模板修改后，将交付副本和用户安装副本同步并核验 SHA。

质量来源须显式填 `client_c` 与可读的 `protocol`，不能把所有结果默认标为C2。有实际runtime汇总时填 `runtime`，核对其binary/source、collector batch、native容量、K、KV和`route_flags`；质量identity里的 `source_aggregate_sha256` 可与registry完整source核对。各质量配置分列；同binary但KV、节点、采样或负载不同，仍不能冒充同配置联合达标。

## 关键判断

- 性能/质量按 binary、source、量化缓存、路由、采样、thinking 分别追踪。禁止把某版本86%与另一个版本的C12通过拼成同一配置联合达标。质量空final、length、错误完整保留；短输入C2不证明高batch新算术覆盖。
- 原ZIP C1/2/4/8、100+8与扩展C12/16/32、128+8分表。TPOT先逐请求算(E2E−TTFT)/(OSL−1)再取线性P95；吞吐保留原phase分母，不能使用看似相近的benchmark_duration重算。
- 同机同原始payload的增量比较与跨版本进程展示分开。工作量的ledger行数仅称“实验/检查记录”，不能称全部独立GPU实验或成功实验；校准矩阵数、测试数须注明所属阶段。
- 六类专项材料和五项评分映射必须保留；失败路线、上游署名、算子与端到端差别、单轮与稳定性差别不可删除。
- 复验必须包括新增源码、量化缓存及配置。旧NInfer源码包与旧启动方式不自动覆盖新W4A4版本。缺缓存、依赖、原始数据或可移植命令时，在验收记录列具体补齐项；提交正文提供准确流程和路径，不编造交付已完成。
- 仅整理本地已有证据。不要为了补材料启动GPU、停止服务、读取秘密、提交Git、上传或联系第三方。用户要求最终发布时，先形成可审阅材料和manifest，再按已有授权处理。

## 最终冻结

只有用户指定最终版本后才将内部状态改为最终。固定registry、原始数据、私有源码、模型/缓存和配置哈希，离线打包必要证据，重跑check并实际验证归档。代码/缓存/录屏/分工/干净重建/回执缺失记入验收记录，不创建虚假完成标记；实际完成PDF编译和逐页视觉检查后才交付展示文件。
