# NInfer 基线环境与后续实验

脚本和配置以引擎/功能命名，机器别名只用于连接与结果归属。部署方法见[复现说明](../submission/REPRODUCE.md)，冻结参数见[基线模板](../configs/ninfer/qwen38-27b-baseline.template.json)，测量与局限见[性能报告](reports/2026-10-16-ninfer-performance.md)。

## 本轮节点

| 实验角色 | 节点名 | SSH 别名 | CPU | 服务地址 |
|---|---|---|---|---|
| 完整性能与 IFBench 测量 | ucloud1 | `ucloud` | EPYC 7A23 | `http://127.0.0.1:8088/v1` |
| 相同基线备用，短功能核验 | ucloud2 | `ucloud2` | EPYC 7B13 | `http://127.0.0.1:8088/v1` |
| 新节点，NInfer 基线就绪，另备 vLLM / SGLang 环境 | ucloud3 | `ucloud3` | EPYC 7A23 | `http://127.0.0.1:8088/v1` |

实时状态以各节点`config/active-service.json`为准；本轮结束时的[身份/健康快照](../results/ninfer-performance/handoff/)另存，PID不作为长期常量。

均为单卡49,140MiB RTX4090，工作前缀为`/workspace/InferX`。该路径是可写overlay，实例释放后是否保留尚未确认；关键结果回传仓库。ucloud1/2权重和二进制SHA一致，CPU不同；ucloud3复制校验见下方记录。表中端口都是各节点自身的回环地址。

`ucloud3` 的[部署记录](../results/ninfer-performance/preparation/ucloud3-peer-copy.json)保存复制来源和校验结果：[硬件盘点](../results/ninfer-performance/inventory/ucloud3.json)、[服务/API 就绪快照](../results/ninfer-performance/handoff/ucloud3.json)分别记录新节点身份与真实请求。公共 NInfer、构建和评测环境从 `ucloud2` 经内网 SSH 复制，权重、二进制、tokenizer 和原版 runner 均通过冻结 SHA256 校验；vLLM 0.31.0 来自 `ucloud1`，SGLang 0.5.19 来自 `ucloud2`。平台已挂载的 `/model/ModelScope/Qwen/Qwen3.8-27B-FP8` 可直接访问，无需复制。

新节点当前运行 NInfer；vLLM / SGLang 已通过包导入、依赖一致性、CLI 和 CUDA 张量运算检查，尚未在该节点启动模型服务或执行正式性能评测。登录后可分别加载环境：

```bash
ssh ucloud3
source /workspace/InferX/config/vllm-env.sh
# 或在另一个新 shell 中使用 SGLang：
source /workspace/InferX/config/sglang-env.sh
```

加载环境不会停止服务。启动另一个模型引擎前，应先按下文的身份核验流程停止当前 NInfer，释放显存。

从本机访问其中一台：

```bash
ssh -N -L 8088:127.0.0.1:8088 MACHINE_ALIAS
# 另一个终端访问 http://127.0.0.1:8088/v1
```

连接两台时将第二个本地端口改为8089：`ssh -N -L 8089:127.0.0.1:8088 MACHINE_ALIAS`。

## 节点内目录与控制

| 相对工作前缀的路径 | 用途 |
|---|---|
| `code/` | 固定main快照；来源见`config/code-source.json` |
| `source/`、`build/ninfer-sm89/` | 固定上游源码及本机编译目录 |
| `models/qwen38-ninfer/`、`models/tokenizer/` | 固定v2权重及六份内嵌资源 |
| `envs/build/`、`eval-envs/` | 隔离工具链及两个官方客户端环境 |
| `config/active-service.json` | 当前服务PID、启动tick、UID、argv及启动时版本身份 |
| `runs/`、`logs/` | 分次实验输出；每次使用新目录 |

节点内加载环境：

```bash
export INFERX_ROOT=/workspace/InferX  # 其他机器改为实际数据目录
source "$INFERX_ROOT/config/node-env.sh"
source "$INFERX_ROOT/config/ninfer-env.sh"
cat "$INFERX_ROOT/config/active-service.json"
curl -fsS http://127.0.0.1:8088/v1/models
```

结束本机实验并保存输出后，如需停止当前受管服务，使用身份校验而非固定PID：

```bash
PYTHONPATH="$INFERX_ROOT/code" /usr/bin/python3 - <<'PY'
import json, os
from pathlib import Path
from inferx.service_ownership import stop_matching_process
m = json.loads((Path(os.environ['INFERX_ROOT']) / 'config/active-service.json').read_text())
identity = {'pid': m['pid'], 'proc_start_ticks': m['start_ticks'],
            'uid': m['uid'], 'executable': m['command'][0]}
stop_matching_process(identity, timeout=30)
PY
```

`active-service.json`中的source_main_commit是服务启动时快照，`code-source.json`是当前源码快照；追加报告/结果后两者可能不同，运行二进制身份仍用SHA核验。

身份不匹配或进程已不存在时，保留异常并重新盘点，不向其他PID发信号。本轮使用系统Python提供pidfd；工具链Python没有该接口。新服务的前台启动见复现说明；原生HTTP keepalive可能使旧端口暂留TIME_WAIT。通用sweep只在自身已回收服务后有界等待端口释放，外来监听者会直接报错。

## 后续改动的比较方式

当前仅测量节点运行完整基准；备用节点不重复该任务。后续并行验证机制时，各节点先保存自己的原配置基线，再运行改动。保持权重、tokenizer、请求集/seed、采样、输出预算、预热、网络和统计口径一致；不能把不同CPU节点的绝对差作为优化收益。

PILOT与正式评测分目录，保留失败及退出码。吞吐和联合SLO分别报告；IFBench保留四项原生分数。原版runner始终保持eval.zip字节内容，API适配只在服务侧进行。新的机制研究与尚暂停的SOTA对照另行安排。
