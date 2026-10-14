# 在 vlm-server 用 Qwen：OpenCode、Pi、Claude Code

三个客户端已安装并接入 GPU1 的 NInfer + Qwen3.8-27B，模型名 `qwen38-27b`。后端当前运行，地址为 `http://127.0.0.1:18081`，MTP7、256K 总上下文，关闭 thinking。本轮不继续 KV 管理研究。

## 直接启动

先登录服务器，进入你希望 Agent 修改的仓库，然后任选一个客户端：

```bash
ssh vlm-server
cd /path/to/your/repository
TASK_ROOT=/home/guokc/mypro/gnn-mount-sda1/InferX

# 三选一；退出当前客户端后再换另一个
bash "$TASK_ROOT/code-tuning/scripts/run_local_agent.sh" opencode
bash "$TASK_ROOT/code-tuning/scripts/run_local_agent.sh" pi
bash "$TASK_ROOT/code-tuning/scripts/run_local_agent.sh" claude
```

`cd` 的目录就是客户端的工作目录。启动器设置本次进程的 Node、模型和配置位置，无须改 shell profile 或原有云端账户。当前后端只同时处理一个请求，日常使用一次运行一个客户端。

可以先输入：`先阅读这个仓库，说明运行测试的方法，暂时不要修改文件。` 后续再要求修改代码。日常启动没有启用实验的 `INFERX_AGENT_RESTRICT=1`，也没有增加自动批准工具的参数；保持各客户端普通交互方式。

## 实际配置位置

下表路径都相对于服务器 HDD 根目录 `$TASK_ROOT`：

| 用途 | 实际文件 | 关键设置 |
|---|---|---|
| OpenCode | `context-agents/config/opencode.json` | provider `inferx`；主模型、小模型及压缩模型固定为 `inferx/qwen38-27b` |
| Pi | `agent-state/pi/models.json` | provider `inferx`；模型 `qwen38-27b`；API `openai-completions` |
| Pi 默认与压缩 | `agent-state/pi/settings.json` | 默认本地模型、thinking off；自动压缩，预留 16K token |
| Claude Code | `context-agents/config/claude.json` | `ANTHROPIC_BASE_URL` 和全部主/辅助模型别名固定为本地 Qwen |
| 环境与目录 | `code-tuning/scripts/agent-env.sh` | 独立 Node 22、客户端 PATH、HDD 配置/状态/缓存位置 |
| 三个启动入口 | `code-tuning/scripts/run_local_agent.sh` | 加载上述配置，保留传入的客户端参数 |

OpenCode 的会话位于 `agent-state/xdg-data`、`agent-state/xdg-state`；Pi 会话位于 `agent-state/pi`；Claude Code 会话位于 `agent-state/claude`；缓存位于 `cache`。配置的仓库快照见 [OpenCode](../configs/agents/opencode.json)、[Pi](../configs/agents/pi.json)、[Claude Code](../configs/agents/claude.json)。修改实际生效配置时编辑表中服务器文件，仓库快照不会自动同步过去。

## 接入方式与理由

OpenCode/Pi 使用 OpenAI Chat Completions 兼容接口，base URL 是 **`http://127.0.0.1:18081/v1`**。OpenCode 通过 `@ai-sdk/openai-compatible` 注册自定义 provider；Pi 在 `models.json` 中注册自定义 provider。对应官方说明：[OpenCode 自定义 provider](https://opencode.ai/docs/providers/)、[Pi 自定义模型](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/models.md)。

Claude Code 使用后端的原生 Anthropic Messages 兼容接口，**`ANTHROPIC_BASE_URL=http://127.0.0.1:18081`**，不加 `/v1`，由客户端追加 `/v1/messages`。设置 `ANTHROPIC_MODEL`、Sonnet/Opus/Haiku 等别名为 `qwen38-27b`，避免辅助调用使用其他模型。这里无需协议转换代理。[Claude Code 网关配置](https://code.claude.com/docs/en/llm-gateway)

三份配置中的 `inferx-local` 是本地占位 API key，当前回环后端未启用鉴权；无需登录云端账号。路径集中到 HDD 满足下载/缓存落盘要求，也便于复现和修改。主调用和辅助调用都固定到本地模型；输出上限 8192 token，自动压缩提前留出工具及输出空间。

当前启动参数关闭附加插件/MCP/扩展；Pi 的 `--no-context-files` 和 Claude 的 `--bare` 也关闭自动读取项目指令文件。需要项目 AGENTS.md/CLAUDE.md、skills 或 MCP 时，可调整启动器相应开关，并检查新的配置仍使用本地模型。这套配置以最少依赖的已验证接入为起点。

## 已验证的能力与使用范围

- OpenCode 1.19.0、Pi 1.2.0、Claude Code 2.1.92 都完成了真实的读取文件、修改源码、执行 pytest，外部复验通过。
- OpenCode/Pi 声明 262144 上下文、8192 输出；Claude Code 对这个自定义模型内部仍报告约 200K 上下文，实际请求输出限制为 8192。后端 256K 不代表 Claude CLI 能完整使用 256K。
- 三个仓库任务各跑一次：OpenCode 与 Claude Code 各通过 2/3；Pi 三次均在 600 秒上限停止，包含反复调用工具的情况。Pi 已接通，但当前这组模型/提示/客户端配置的仓库工作流仍有质量问题；可先从 OpenCode 或 Claude Code 使用。小样本不能作为普遍排名。
- 接近满 256K 的冷输入首字约 201 秒；短代码提示的生成约 182–231 token/s。这些是不同提示下的测量，不是 Agent 完成任务的速度保证。

详细结果、原始记录位置与回退说明见 [验证报告](reports/2026-10-14-context-agents.md)。

## 本机访问与后端状态

本轮客户端配置在服务器上。如果在自己的电脑装客户端，先保持 SSH 隧道运行：

```bash
ssh -N -L 18081:127.0.0.1:18081 vlm-server
```

随后使用同样的 API 地址、模型和协议，把配置/状态目录改成本机位置；服务器的绝对路径启动器不能直接在本机执行。本机客户端没有在本轮安装或验证。

在服务器检查健康状态：

```bash
curl -s http://127.0.0.1:18081/health
curl -s http://127.0.0.1:18081/v1/models
```

当前服务已经运行，无需重复启动。尚未配置开机自启；服务器重启后确认 18081 未占用，再运行 `bash "$TASK_ROOT/code-tuning/scripts/start_agent_server.sh"`。运行信息位于 `context-agents/runtime.json`，日志位于 `context-agents/agent-server.log`。GPU0 的 18080 服务保留。
