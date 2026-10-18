# 现场性能评测：启动服务并运行C14

本页用于在已经部署好的单张48GB RTX 4090上，重新运行完整性能测试。默认使用已完成源码复验的664部署目录；先确认模型、缓存、构建产物及评测环境已准备好。从零部署请按[源码与环境准备](../track-a/reproduction.md)及[量化缓存重建](../track-a/quantization-reproduction.md)操作，现场不重新下载或拟合权重。

配置固定为**客户端14并发、引擎13个执行槽、KV121856、MTP K3**。共128条正式请求、8条预热、1条探针；输入8192、输出512。历史完整测量阶段约215秒，另加加载、预热和汇总时间，请预留5–8分钟；不把缩短的演示测试当成正式成绩。

## 1. 两个终端设置相同路径

两边均复制执行。`INFERX_RUN`每次换一个新名字，避免覆盖原始结果。下面路径对应已有的源码复验目录；若在新目录部署，改第一行，并选用该次构建生成的recipe，不能用旧recipe冒充新二进制。

```bash
export INFERX_DEMO_ROOT=/workspace/InferX/reproductions/selected100-single-node-20261018
export INFERX_ENTRY="$INFERX_DEMO_ROOT/entrypoint"
export INFERX_RECIPE="$INFERX_ENTRY/candidate_recipes/100-k3-u3-47-rebuilt-20261018.json"
export INFERX_RUN=defense-c14-001
export INFERX_PORT=18540
export INFERX_SERVICE="$INFERX_DEMO_ROOT/$INFERX_RUN-service"
export INFERX_PERF="$INFERX_DEMO_ROOT/$INFERX_RUN-performance"
export CUDA_VISIBLE_DEVICES=0
cd "$INFERX_DEMO_ROOT"
```

## 2. 终端一：确认环境并启动服务

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
test -f "$INFERX_RECIPE"
test ! -e "$INFERX_SERVICE"
test ! -e "$INFERX_PERF"
python3 -B "$INFERX_ENTRY/serve_candidate.py" \
  --recipe "$INFERX_RECIPE" \
  --profile perf-c14 \
  --root "$INFERX_DEMO_ROOT" \
  --host 127.0.0.1 --port "$INFERX_PORT" \
  --log-dir "$INFERX_SERVICE"
```

服务保持前台运行。启动入口会核对二进制SHA、资产集合及配置；不要在同一张卡上同时运行其他推理或量化任务。

## 3. 终端二：等待模型就绪并评测

先重复第1节的变量设置，再执行下面的只读就绪检查。它仅查询模型列表，不生成额外回答。

```bash
python3 - <<'PY'
import json, os, time, urllib.request, urllib.error
url = f"http://127.0.0.1:{os.environ['INFERX_PORT']}/v1/models"
deadline = time.monotonic() + 300
while True:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            models = json.load(response)
        if not any(m.get('id') == 'qwen38-27b' for m in models.get('data', [])):
            raise RuntimeError('模型列表中没有 qwen38-27b，请检查当前服务。')
        print('模型已就绪：qwen38-27b')
        break
    except (urllib.error.URLError, TimeoutError):
        if time.monotonic() >= deadline:
            raise SystemExit('等待模型就绪超时，请检查终端一。')
        time.sleep(2)
PY
```

就绪检查成功后再运行完整压测。客户端还会校验服务PID、启动时刻、可执行文件、recipe和监听端口，防止连到另一份服务。

```bash
OPENAI_API_KEY=EMPTY python3 -B "$INFERX_ENTRY/perf_candidate.py" \
  --recipe "$INFERX_RECIPE" \
  --concurrency 14 \
  --root "$INFERX_DEMO_ROOT" \
  --service-launch "$INFERX_SERVICE/launch.json" \
  --base-url "http://127.0.0.1:$INFERX_PORT/v1" \
  --output-dir "$INFERX_PERF"
```

## 4. 查看结果

```bash
cat "$INFERX_PERF/evaluation-identity.json"
cat "$INFERX_PERF/summary.json"
```

`evaluation-identity.json`的`status`应为`complete`；失败或中断时保留目录，不用已有历史结果代替本轮结果。原始汇总中的两项延迟单位均为毫秒，按下面命令读取并判定：

```bash
python3 - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['INFERX_PERF'])
identity = json.loads((root / 'evaluation-identity.json').read_text())
s = json.loads((root / 'summary.json').read_text())
assert identity['status'] == 'complete', '本轮评测未完整结束'
assert not s.get('issues'), s.get('issues')
ttft = s['metrics']['time_to_first_token']['p95']
tpot = s['metrics']['inter_token_latency']['p95']
print(f'TTFT P95: {ttft / 1000:.3f} s（要求 ≤ 10 s）')
print(f'TPOT P95: {tpot:.3f} ms（要求 ≤ 40 ms）')
print('总输出吞吐:', s['output_token_throughput'], 'token/s')
print('两项延迟SLO:', 'PASS' if ttft <= 10000 and tpot <= 40 else 'FAIL')
PY
```

原解析器可能不提供`meets_slo`字段，不以该字段的`null`或程序退出码代替两项P95判断。发生请求错误时，同时展示原汇总，不能只展示延迟通过。

| 证据 | 本轮路径 |
|---|---|
| 服务版本与实际参数 | `$INFERX_SERVICE/launch.json` |
| 客户端、请求与评测配置 | `$INFERX_PERF/plan.json` |
| 流程终态及完整汇总 | `$INFERX_PERF/evaluation-identity.json`、`summary.json` |
| 实际输入与逐请求输出 | `$INFERX_PERF/c14/inputs.json`、`profile_export.jsonl` |
| 原压测日志 | `$INFERX_PERF/c14/console.log` |

## 5. 停止服务或重新测量

测试完成后，在终端一按`Ctrl+C`停止本次前台服务。重新测量时先确认原服务退出，再修改两个终端的`INFERX_RUN`并从第2节开始；端口占用时先核对所属进程，不终止其他任务。启动、评测或SLO失败均保留原目录。

历史664复验参考值为TTFT 9.896秒、TPOT 38.286毫秒、304.897 token/s；这里只供核对量级，现场结果以本轮原始输出为准。
