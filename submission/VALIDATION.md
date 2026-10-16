# 材料与部署检查记录

本轮已完成两机NInfer/MTP基线与第一台原版完整评测；正式数字见[BENCHMARK.md](BENCHMARK.md)，原始记录见[official-r1](../results/ninfer-performance/official-r1/)。此处区分请求完整性、GPU采样覆盖和最终提交验收。

- 原版性能C1/C2/C4/C8各100正式+8预热，全部8192/512、零请求错误；原版runner退出0，独立解析复核通过。C1/C2满足两项延迟SLO，C4/C8未满足。
- IFBench100预测、100review、100个prompt评分完整；strict/loose prompt55%，指令65/111与66/111。原生100chat均stop_token，无32768截断，输出4–2362 tokens。保留参数展开、原生报告和全部失败输出。
- 目标1Hz采样2747点：两处8.402/6.682秒间隔使全程及C1/C8覆盖partial，C2/C4通过。保留严格阈值及首次拒绝诊断；分析程序完成不代表GPU覆盖完整。GPU数值只作为已记录点统计。
- 远端冻结归档SHA256为`2639defbd36ae0ddd1fc7c925335bb87d118524533657b8cd5565e1c02df877f`；113个原始文件逐字节哈希验证通过，41个原生文本/数据文件无损gzip，原始/存储SHA分别保留。原版解析器使用的JSON/JSONL保留明文格式。
- 两机模型SHA与二进制SHA一致，精确内嵌tokenizer核验；原生构建、真实raw/chat/流式/ignore_eos/usage/缓存隔离和工具调用已验证。第二台仅做功能与重启检查，没有重复正式压测。当前进程/健康与构建核验见[交接快照](../results/ninfer-performance/handoff/)。
- 通用命名与路径参数化已落实：scripts/configs以NInfer/功能命名，机器名仅用于清单和运行归属。隔离依赖锁保存；第二台复制已核验构建，explicit锁只做了重放dry-run，不宣称完成全新环境二次重建。
- 本轮最终运行代码的CPU测试为**384 passed,44 skipped，124.54秒**，见[日志及核验](../results/ninfer-performance/analysis/verification-pilot.json)；之后只有材料、配置模板及离线分析追加。44个skip不作为原生外部环境通过。API原生契约/GPU实测另有证据，不能用CPU测试替代。
- 19个PILOT档共152正式+38预热，独立原版解析及输入哈希核验通过；PILOT不混入100请求正式结果。计时与prefill归因的解释边界见性能报告。
- 功能回归：46867-token三处精确检索通过；生成merge_intervals通过已有seccomp隔离测试。逻辑题384/2048预算均截断，2048中出现12分钟可行时间表但证明未完成，保留为未完整通过。
- 一次备用检查误用测量机SSH别名，连接复用失效后回退到该机；在发API请求前因目录不匹配中止。核对两机实际hostname/GPU后改用正确别名并加身份断言；正式冻结日志仍为433completions+100chat，没有额外请求，诊断保留。
- 独立审阅：原生API、driver的信号/端口修正、可迁移配置、PILOT/计时解释及正式结果均APPROVE。正式结果审阅复核400性能、100精度、113快照SHA；部分GPU覆盖被明确保留。
- 空白检查保留原生EvalScope HTML模板的尾空格，排除该原始报告；其真实性由113个快照SHA保证。源码/文档/其他新增文件按git diff --check验证，不为风格检查改写原生证据。
- 原始评分MD/eval.zip/提取runner的6个SHA，以及用户PD补测8个SHA保持一致。最新相对链接、8页结构、压缩和输入检查见[材料检查](evidence/material-checks.json)。本轮新增文件另做常见秘密模式扫描，不将模式扫描当作安全性保证。

两机已实际同步main `2d8dc2c7aad0d87bbb352455aae654669723a089`，各3067个跟踪文件完整SHA核验，原版runner哈希一致；[两机同步记录](../results/ninfer-performance/handoff/)已保留。该版本源码归档的6个输入SHA和Git bundle验证通过，忽略的本地skills/状态不在源码包中。最终记录补充也同步到两机，实际最终提交由外部release-manifest记录，避免在版本内自引用。

最终main源码同步和归档以`artifacts/release-manifest.json`的实际版本/校验和为准；正式测量时运行主仓快照为8e974a9，引擎/API补丁未在正式运行中切换。主办方差异口径、持久存储、干净重建、成员分工、投屏/录屏和提交回执仍依[TODO](TODO.md)验收，不能将本轮测量完成当作最终比赛提交完成。

## 此前材料整理记录

以下为部署前、截至2026-10-16T08:10Z的历史检查；当前状态以上文和最新成绩表为准。


本文件记录本次材料整理和代码集成的检查，不是 UCloud 模型评测成绩。

- 原始输入和评测文件：6 个 SHA256/字节数通过，ZIP 内 3 个文件与提取文件逐字节相等。
- 文档相对链接、JSON 和 8 页 HTML 结构：检查通过；详见 [检查记录](evidence/material-checks.json)。HTML 只做结构检查，尚未完成现场投屏彩排。
- 原版 runner 的 builtin / 显式 tokenizer 两种 dry-run：均退出 0，未创建工具环境、未安装、未请求模型；[输出 1](evidence/eval-dry-run-builtin.txt)、[输出 2](evidence/eval-dry-run-explicit.txt)。占位 tokenizer 仅用于 dry-run，不能用于正式评测。
- 项目 CPU 测试：`/home/yuer/.cache/uv/archive-v0/LAYH7qN9Xs5rVIJ5/bin/python -m pytest -q`，**366 passed, 44 skipped，40.65 秒**。不涵盖 UCloud GPU 部署、正式成绩或跳过的原生环境检查。
- 用户原有 PD 补测文件：与既有 `preexisting-worktree.json` 的 8 个 SHA256 全部一致；本轮不修改其测量内容。
- 源码历史：检查 1,793 个可达 blob 的常见密钥/私钥/明文 SSH 密码/带凭据 URL 模式，无候选命中；这是模式扫描，不保证覆盖所有秘密类型。新增材料另行检查。
- 空白检查：两份原始规则文件保留其 CRLF、尾空格和末尾空行，未经规范化；对其余新增/修改文件执行 `git diff --cached --check`。原始输入的真实性由 SHA256 检查保证。
- 独立合入前审阅：无 Critical / Important；已按建议精确说明“API 公布数值上下文上限”才触发精度预检查，未知上限由实际请求暴露超限。
- 开发内容已快进集成到 `main`，首次集成提交 `be5e2535fbe0523577c32f6857362ff252a2e13a`；全部 4 个本地分支的提交均被 main 包含，研究分支保留。已正常推送 `origin/main`（首次同步从 `e1764f9` 到 `be5e253`），未强推；本检查记录的提交也同步至同一分支。
- 合入 main 后复测：同一 CPU 测试命令，**366 passed, 44 skipped，42.01 秒**。
- 固定版本源码快照已导出，Git bundle 验证通过，归档校验和通过。文件在 `artifacts/InferX-source.tar.gz`、`artifacts/InferX-main.bundle`、`artifacts/SHA256SUMS`；最终整理提交后重新导出，实际对应 SHA 以 `artifacts/release-manifest.json` 为准。
- 首次同步前另检查 2,440 个跟踪路径，含 41 个解压检查的归档；未命中常见凭据模式，记录见 [提交文件检查](evidence/pre-push-source-checks.json)。

当时UCloud仅完成只读盘点，未安装、下载、启动或评测；正式提交的剩余条件维护在 TODO.md。

按用户要求清理版本控制：`.agents/` 及本地 agent 状态、缓存、构建/运行目录、导出包、编辑器文件纳入 `.gitignore`；39 份已跟踪的 skill 文件取消跟踪，本地 SHA256 全部保持不变。26 个应忽略路径与 12 个应保留路径的检查通过，官方 eval、评分规则、项目配置与实验数据仍可跟踪。源码包按清理后的 main 重新导出；Git bundle 作为历史备份保留既有提交，不改写历史。
