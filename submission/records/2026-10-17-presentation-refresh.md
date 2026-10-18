# 默认 Beamer 展示刷新与证据验收

执行依据：用户调用 inferx-submission；沿用已确认范围及最新四项展示要求。执行计划见 `docs/superpowers/plans/2026-10-17-submission-presentation-refresh.md`。结果为材料更新快照，未指定最终发布版本。

## 来源审查与比较契约

- baseline 搜索涵盖旧ZIP NInfer b604（C1/2/4/8，100+8，K4）与框架参考报告（vLLM0.31.0/SGLang0.5.19，FP8、无MTP、4槽、12–100正式请求）。后者跨机器、计算后端、KV精度、窗口、样本与协议，不放入统一增益图；保留模型选型中的来源及限制。旧ZIP独立表说明问题，不与128+8新版本计算净加速。
- 同节点C32累计版本 c3f482→40b3→9270 为历史进展，K3、128+8、8192/512相同，但量化覆盖、代码和质量覆盖不同；40b3只有本地原版summary，图不加单优化增益标签。
- 新增原始来源：u3 `gqa-highbatch-C32-official128-001`（8b20）、`gdn-fusion-C32-official128-001`及`gdn-fusion-C16-official128-001`（2362）。逐份阅读REPORT、根summary、原runner的分档summary与完整identity。服务终态记录均为实际missing、GPU空；固定128正式+8预热+1probe、seed20261016、450W、K3、PF2048、cold8192/512，零错误。仅注册原版分档summary，client/native由refresh独立重算。
- GQA算子B16/T4未到原1.2x筛选门槛，原SCREEN_FAIL保留；后来工程集成不修改该门槛。8b20与u1/9270跨机器且多个实现不同，不计算GQA净服务收益。
- 8b20/2362的u3 C32为同137实际payload（仅去cache_salt）及相同warm/measured cohort顺序对照；GDN fold+record为实现变量。报告支持共同执行decode均值下降、PF服务成本接近，但TPOT尾部恶化。无新trace，nondecode不能全部归为PF，无同版任务质量或稳定性结论。
- 2362 C16与dc99为组合版本变化（GQA+GDN），只列版本结果，不归因GDN；不继承9270 C12通过或40b3质量。
- 其他inventory候选：CPU/source-only核查、算子局部探针及准备阶段不加入整服务性能表；原负例W2、FP16状态、history-four和compact-conv按真实局限保留。没有启动GPU、停止服务、发布或提交Git。

## 问题、设计选择与效果

| 优化点 | 问题及证据 | 设计与理由 | 效果/代价及边界 |
|---|---|---|---|
| dense量化覆盖 | 旧ZIP C8吞吐增加有限、TPOT恶化；原生计时/Nsight定位dense和串行竞争 | H512/G512+GPTQ；在真实矩阵中接入A4/S4 MMA，保留scale、SiLU及残差 | 025共128个FFN矩阵，041输入、047输出47层；C32历史吞吐305.91→352.41→363.58，非单项因果或同版质量证明 |
| 批扩容/调度 | pending数量不等于原生batch；Prefill占用生成机会 | B16/B32扩展Graph、状态、工作区；chunk与completion预算 | 能运行且有数值/状态门；C16/C32仍超延迟，不能单独声称净收益 |
| GDN fold+record | 状态物化/记录开销 | 合并记录与折叠、保留pending生命周期、bank/slot和回滚语义 | 2013条native配对无差异是补充证据；u3 C32吞吐略升而P95 TPOT恶化，未晋级；额外持久状态103,023,360B |
| down053 | small路径wholeOp可优化 | 独立gate接入，原表示检查与实际payload复验 | 同机C16吞吐+1.41%，TPOT+4.66%，未晋级；未测试其C12 |

## 图形契约（制作前）

权威源：`tooling/inferx-submission/assets/main.tex.in`，生成 `basic/presentation/main.tex`；原生TikZ/PGFPlots与编译PDF。16:9宽约16cm；图表半幅6.4–6.8cm，高2.8–4.4cm，普通标签8pt以上，必要脚注8pt；不缩放整图挤字。Beamer主题/标题/正文保留默认，图内中性灰表示已有基础，蓝色表示实测或状态，橙色表示Prefill/新增覆盖；文字/形状冗余区分，不依赖颜色。

| 图/页 | 主张与证据 | 对象、边及边界 | 禁止暗示 |
|---|---|---|---|
| 竞争示意/2 | PF影响首token及decode机会；optimization.md和旧计时报告 | 三请求时间行，PF/Decode/等待明确标注；时间箭头仅方向，长度为示意 | 实测比例、带宽饱和、所有等待均来自PF |
| 架构/3 | 校准缓存、调度与原生执行共同构成方案 | 请求→准入队列→预算调度→PF或Decode/Verify→响应；真实执行/Graph范围包围两类执行；KV/GDN状态与执行层双向读写、完成状态反馈到调度；缓存加载到执行层，不穿越调度边 | 缓存是在线顺序处理阶段、状态属于评测器、控制边是请求payload |
| 设计覆盖与版本/4 | 工作量是具体覆盖及历史结果 | 三个优化决策的瓶颈→选择→实测边界；独立C32版本柱，从零起，原始数值来自registry | 柱差为某单算子收益、记录数等于实验成功数 |
| SLO与质量/5 | 版本明确的独立测量，只有左下点联合通过 | TTFT秒/TPOT毫秒，10/40阈值；四独立散点不连线，表列binary；质量独立表列错误/空final/length | 同版容量曲线、性能与质量联合达标、跨样本直接排名 |
| 证据记录/6 | 实际运行身份与统计须共同绑定 | source/binary、cache/config、client/native三组输入汇入同一证据记录；记录→原版统计/独立复算；不画虚假运行顺序 | manifest意味着完整复验包已经发布、材料check代表实验成功 |
| GDN取舍/7 | 同payload仍可能吞吐升而尾延迟恶化 | 8b20/2362 C32 TPOT柱零起、40ms线、原phase吞吐并列；顺序单轮 | 误差条、长期稳定收益、PF全部因果归因、继承质量 |

## 编译与人工验收

已完成当前PDF的实际编译与逐页视觉检查，SHA `3f4dc7d4093e73d8d0833dd20f2cd0e4f2a42074b7baca96aaf18dfdcee2f7f7`，八页、零缺字/overfull。全部八张最终预览分别打开；第3页原有越界与交叉通过执行层边界、短连接修复，第5页C12标签移离阈值线，第7页柱图留出横轴标签余量。机制示意、历史版本和单轮受控对照均明确标注，不隐藏数值失败。

结构审计通过figurecraft脚本的audit_pdf函数逐页执行（脚本CLI要求SVG，本项目原生TikZ，不伪造SVG包装）。临时隔离环境安装PyMuPDF1.28.2；八页均有矢量路径、可选文本、零栅格。原始最小字体告警来自默认Beamer透明导航链接句点；alpha=0，与可见内容分开解释，所有可见标签最低8 TeX pt/7.970 PDF bp，原审计告警保留于 `latex/vector-audit.json`。PDF字体均已嵌入。主题保留默认导航图标，未通过自定义模板隐藏它们。

仓库及安装副本check、quick_validate、5项静态门正负例和git diff --check通过；运行文件5份SHA一致。统计为14性能、2完整质量、1552正式及104预热独立重算、1233条native。新增三个性能来源全部具备原始client/native；input-c32仍降级为summary-only。真实投影设备与现场讲稿计时未执行，不声称已彩排。

团队清单继续以 `2026-10-17-final-submission-acceptance.md` 为准，录屏、真实分工、最终版本、完整源码/缓存发布、同版质量、独立seed稳定性、干净重建和提交回执仍需团队完成；不将尚缺项移入提交正文。
