# 材料整理检查记录

本文件记录本次材料整理和代码集成的检查，不是 UCloud 模型评测成绩。

- 原始输入和评测文件：6 个 SHA256/字节数通过，ZIP 内 3 个文件与提取文件逐字节相等。
- 文档相对链接、JSON 和 8 页 HTML 结构：检查通过；详见 [检查记录](evidence/material-checks.json)。HTML 只做结构检查，尚未完成现场投屏彩排。
- 原版 runner 的 builtin / 显式 tokenizer 两种 dry-run：均退出 0，未创建工具环境、未安装、未请求模型；[输出 1](evidence/eval-dry-run-builtin.txt)、[输出 2](evidence/eval-dry-run-explicit.txt)。占位 tokenizer 仅用于 dry-run，不能用于正式评测。
- 项目 CPU 测试：`/home/yuer/.cache/uv/archive-v0/LAYH7qN9Xs5rVIJ5/bin/python -m pytest -q`，**366 passed, 44 skipped，40.65 秒**。不涵盖 UCloud GPU 部署、正式成绩或跳过的原生环境检查。
- 用户原有 PD 补测文件：与既有 `preexisting-worktree.json` 的 8 个 SHA256 全部一致；本轮不修改其测量内容。
- 源码历史：检查 1,793 个可达 blob 的常见密钥/私钥/明文 SSH 密码/带凭据 URL 模式，无候选命中；这是模式扫描，不保证覆盖所有秘密类型。新增材料另行检查。
- 空白检查：两份原始规则文件保留其 CRLF、尾空格和末尾空行，未经规范化；对其余新增/修改文件执行 `git diff --cached --check`。原始输入的真实性由 SHA256 检查保证。
- 独立合入前审阅：无 Critical / Important；已按建议精确说明“API 公布数值上下文上限”才触发精度预检查，未知上限由实际请求暴露超限。
- 开发内容集成 main 与 origin 同步：待完成。
- 固定版本源码快照和 SHA256：待导出。

UCloud 当前仅完成只读盘点，未安装、下载、启动或评测；正式提交的剩余条件维护在 TODO.md。
