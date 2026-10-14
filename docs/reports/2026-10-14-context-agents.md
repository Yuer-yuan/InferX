# Qwen3.8-27B：256K 上下文与三个本地代码 Agent

直接使用请看 [三个客户端启动与配置位置](../agent-quickstart.md)。本轮按最新用户要求暂不继续 KV 管理研究。

本轮从已完成的64K NInfer/MTP部署继续，按用户要求先增加上下文，再接入OpenCode、Pi、Claude Code。使用一张RTX4090（GPU1）；GPU0原服务继续在18080。所有依赖、缓存、配置、任务副本和原始记录放在HDD `/home/guokc/mypro/gnn-mount-sda1/InferX`，复用hf-mirror下载并校验的模型。

## 上下文结果

原生模型上限262,144token。当前Agent后端选择 **262,144总上下文、rk4v4-e8 KV、MTP7、prefill2048、一个活动请求、prefix reuse**。保留128K INT8与64K INT8回退。

| 候选 | 冷输入token | 首字延迟 | 前/中/后检索 | 1秒采样最大显存 | 最少空闲显存 |
|---|---:|---:|---|---:|---:|
| 128K、INT8 | 122,879 | 69.312s | 3/3正确 | 22,665MiB | 1,443MiB |
| 256K、4-bit E8 | 253,914 | 200.804s | 3/3正确 | 22,801MiB | 1,307MiB |

每项长提示一次，预留至少8192输出token，输出是三处确定性的随机哈希值。检索答案在输入的前、中、后三处，均完整匹配。它比仅问末尾标记更有区分度，但仍是有限检索/容量检查，不证明所有256K推理任务的准确率。显存从启动到请求完成后10秒按1秒采样，未观测OOM；采样不是连续峰值保证。

两个候选各运行四项独立提示三次。LRU/拓扑排序共12份代码输出通过原有沙箱执行门槛。概率性能样本仍在1024token预算截断，未当成完整质量通过。生成速度和范围见结果JSON；长冷输入的主要代价是预填充，MTP不消除200秒的首次处理成本。

短提示三个重复的生成速度中位数如下，单位 token/s：

| 候选 | LRU | 拓扑排序 | 概率题（截断） | 迁移说明 |
|---|---:|---:|---:|---:|
| 128K INT8 | 232.4 | 179.3 | 211.7 | 95.5 |
| 256K 4-bit E8 | 230.6 | 182.0 | 201.1 | 103.3 |

速度随输出内容变化；这些小样本不是端到端 Agent 任务吞吐，也没有同条件统计检验证明两配置等速。

总上下文同时容纳系统提示、工具schema、历史消息、工具结果和输出。预留8K输出时，256K输入预算最多248K，实际用户文件预算还要减去Agent框架开销。此次工作没有增加模型原生上限，也没有启用YaRN。

## 客户端配置与协议验证

| 客户端 | 固定版本 | 接口 | 窗口与输出 |
|---|---|---|---|
| OpenCode | 1.19.0 | 原生OpenAI Chat Completions | 262,144 / 8,192 |
| Pi | @earendil-works/pi-coding-agent 1.2.0 | 原生OpenAI Chat Completions | 262,144 / 8,192 |
| Claude Code | 已安装2.1.92 | 原生Anthropic Messages | 客户端元数据200K，实际请求输出8,192 |

Node22.23.3在HDD独立前缀，官方SHA256校验通过；系统Node18不改。Pi已更换上游命名空间，避免误装旧包。npm依赖锁文件保存用于复现；依赖生命周期脚本关闭，单独运行核对过的OpenCode二进制初始化脚本。

三个客户端均实际读取两个文件、修复错误实现、执行pytest，并由外部进程再次验证原始测试未修改及测试通过。原生NInfer API足以完成这些流程，没有额外协议桥接。后续正式试验使用的受限模式也分别重测通过。

Claude Code对未知模型仍在输出里显示200K/32K元数据；服务端256K是上限，不能据此宣称CLI完整利用256K。实际四次初始调用均向本地qwen38-27b发送8192输出预算；通过本进程环境固定所有模型别名及辅助调用，配置自动压缩窗口。其内置美元cost字段按云端价格估算，没有作为实际费用统计。客户端日志中的token/cache计数也可能采用不同口径，正式汇总以NInfer原生请求记录为准。

客户端配置和状态分别放在HDD的context-agents/config、agent-state和cache，配置不会写入原用户账号文件。初次版本检查曾创建空的OpenCode默认目录及4字节Pi默认状态；核对创建时间后已将这些新生成项迁至HDD，安装脚本提前设置目录，后续使用隔离路径。

## 仓库任务

复用此前固定版本的ExecRepoBench私有源副本。从calibration-v5按既定顺序取前三个仓库的第一个dev任务，生成前固定：

- ExecRepoBench/604：json_repair。
- ExecRepoBench/1155：xmltodict。
- ExecRepoBench/33：PTable。

这是一组公开仓库的单文件缺口修复任务。客户端获得文件名、缺口附近行号与运行测试的命令，自主读取代码、编辑并根据测试继续；没有要求修改多个文件或自主定位整个issue，也没有宣称SWE-bench覆盖。三个客户端收到相同任务，各用全新副本，串行共享同一后端。失败消融在Agent运行前必须让原始完整测试失败。

正式验收将客户端编辑的目标源码放回另一份原始仓库，在现有Landlock/seccomp验证器里运行原始测试，核对收集数量。客户端改测试、只声称通过、退出0却没有实际工具操作，均不能成为成功结果。测试命令显式优先当前仓库/src路径，避免意外导入已安装的同名库。

每个客户端/任务上限600秒，保存完整JSON事件、stderr、代码diff和原生请求计数。超时仅终止其自有进程组；控制程序stdin与客户端隔离，防止客户端消耗后续控制命令。

九次正式试验均已结束。以下是独立原始测试验收结果，每格括号内为客户端运行秒数：

| 客户端 | json_repair /604 | xmltodict /1155 | PTable /33 |
|---|---|---|---|
| OpenCode | 超时（600.08） | 通过，60 tests（47.19） | 通过，66 tests（21.12） |
| Pi | 超时（600.03） | 超时（600.02） | 超时（600.02） |
| Claude Code | 未改目标源码（35.75） | 通过，60 tests（63.41） | 通过，66 tests（73.04） |

OpenCode与Claude Code各2/3，Pi为0/3。三次Pi均有真实工具执行：/604完成341次工具、/1155完成444次（其中439次工具报错）、/33完成381次；不将它们写成连接失败或成功。OpenCode/604亦反复执行Bash，没有改目标源码。小样本结果只适用于冻结任务、600秒上限、当前关闭thinking/greedy与客户端版本，不能推断一般客户端排名。

原生请求口径：OpenCode最高提示92301token，累计54746输出token；Pi最高130209，累计121857输出；Claude最高26740，累计23778输出。前缀命中token占各客户端累计prompt-token约99.21%、99.71%、84.97%；它反映重复请求历史，不等于显存利用率或模型任务质量。客户端运行总耗时分别668.39、1800.07、172.21秒，包含读取/执行工具等开销，不能用输出token除总耗时替代引擎decode速度。

正式raw事件、stderr和diff均保留；所有未通过任务也保留，初始未受限尝试单独归档。

### 被排除的初始尝试

最初OpenCode在json_repair任务中忽略离线提示，下载上游wheel并写入/tmp/jr9。它由我们核对身份后停止，所有日志与源码副本保留在 `context-agents/attempts/initial-unrestricted/`；该下载也迁到HDD。没有把这次尝试计入正式9次结果。

正式任务因此增加继承式Landlock ABI4限制：只允许私有仓库及HDD状态/缓存/临时目录写入，原始数据与参考实现不能读取；TCP-connect只允许18081、18082（推理后端与OpenCode内部服务）。这是端口限制，不能描述为按目的IP限制的完整网络沙箱；它针对本轮非对抗性客户端，且正常交互启动器不默认强制此实验限制。三种客户端均在同样限制下重测工具往返通过。

## 日常使用与回退

当前服务回环地址 `http://127.0.0.1:18081`，模型 `qwen38-27b`。启动后端：

```bash
# 在vlm-server；当前服务健康时无需重复启动
source /home/guokc/mypro/gnn-mount-sda1/InferX/env.sh
bash "$INFERX_ROOT/code-tuning/scripts/start_agent_server.sh"
```

在自己的仓库中使用三个客户端，状态仍落HDD：

```bash
bash /home/guokc/mypro/gnn-mount-sda1/InferX/code-tuning/scripts/run_local_agent.sh opencode
bash /home/guokc/mypro/gnn-mount-sda1/InferX/code-tuning/scripts/run_local_agent.sh pi
bash /home/guokc/mypro/gnn-mount-sda1/InferX/code-tuning/scripts/run_local_agent.sh claude
```

从本机访问API先建隧道 `ssh -N -L 18081:127.0.0.1:18081 vlm-server`；客户端如果装在本机，按configs/agents的模型/地址/窗口配置，目录参数应改为本机自己的位置。本轮验证的是服务器上的CLI版本和配置。

替换或停止服务时核对runtime.json内PID、start ticks、UID、exe和GPU归属，使用pidfd向匹配进程发送SIGTERM，不能按旧PID文件或pkill批量杀进程。64K回退仍使用start_best_server.sh；128K回退参数为：

```bash
bash "$INFERX_ROOT/code-tuning/scripts/start_tuned_server.sh" ninfer 7 int8 2048 131072 1 warm
```

改变后端后必须同步客户端预算，尤其不能把256K客户端配置接到64K回退而继续保留旧上限。没有配置开机自启。生成仍关闭thinking、greedy；Agent质量结论只适用于本轮配置。

## 复现与证据

最终检查：完整远端测试套件88通过（最终回归后5.82秒）；四个安装/启动脚本bash语法检查通过，三个CLI版本符合固定值。GPU1服务PID2726457的UID、可执行文件和start ticks均匹配runtime；18081与GPU0的18080健康，requests_processing/deferred均为0。GPU0原PID2504785仍在。服务和客户端留给用户直接使用，没有开机自启。

最终审查修复汇总器重复读取自己的导出文件：两个真实 CLI 回归先失败再通过；在实际 HDD 目录连续导出两次，九次结果和三份摘要的 SHA256 均保持不变。审查与本轮决定见 [执行记录](2026-10-14-context-agents-execution.md)。

仓库轻量证据：`results/context-agents/summary.json`、`trials-compact.jsonl`（九次）、`tasks-manifest.json`、`clients.json`、`client-package-lock.json`及`raw-manifest.json`（35份原始文件的路径/大小/SHA256）。`summary.json`含两种上下文的功能探针、短提示速度、客户端汇总和运行配置。原生API请求记录在HDD `context-agents/agent-server-requests.jsonl`，不把仍随服务追加的日志声明为冻结文件。

完整上下文请求、CLI事件、原生服务日志和diff在HDD context-agents。仓库保留版本、冻结任务、摘要、单次指标及原始文件SHA256；完整代码与对话留在HDD。来源与功能参考：[OpenCode](https://opencode.ai/docs/providers/)、[Pi](https://github.com/earendil-works/pi)、[Claude Code网关](https://code.claude.com/docs/en/llm-gateway)。
