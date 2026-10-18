# 冻结100：从完整源码到性能与质量复验

本次提交采用**源码重建缓存**的方式，不随归档重复附大型量化缓存。首次复验先完成本页第1–2节和第3节公共基础模型/tokenizer准备，再执行[量化重建指南](quantization-reproduction.md)。该流程会生成缓存、构建并绑定新服务、运行性能和精度。下文固定旧缓存的步骤用于已有该资产时复查664历史结果，并非本次提交的必要附件。

[English instructions](reproduction-en.md) · [历史100/105结果](selected-results.md) · [实际脚本与命令](../../results/three-node-optimization/ucloud3/selected100-single-node-reproduction-20261018/COMMANDS.md)

本流程在单张48GB RTX 4090上，以冻结100的完整源码重新构建服务，并顺序运行C14性能与原100题质量。性能和质量使用同一个新构建产物，但各自独立启动服务、使用各自的KV容量，并分别保存结果。

本次ucloud3复验目录为 `/workspace/InferX/reproductions/selected100-single-node-20261018`。这是已有Linux/CUDA环境中的新源码、新build和新运行目录；模型、量化缓存、tokenizer及已安装环境由原目录复用。它不等于重新安装操作系统或重新生成量化缓存。历史100二进制 `e4ed1bf4…` 与评测105的85%/88%仍是历史结果；新构建只报告自身身份和实测。

**本次提交默认从源码重新生成缓存，按[116完整量化流程](quantization-reproduction.md)执行。** 本页第1–2节及第3节首个代码块提供环境、公共基础模型和tokenizer准备；准备完成后直接转入116流程。本次不附12 GB旧缓存，下面664旧缓存复验分支仅供已经持有该固定资产时复查。

## 1. 取得并校验正式复验包

从InferX交付仓库根目录开始，使用不存在的新目录。正式包为`reproduction-kit-v1.tar.gz`，40,181,061字节；下列SHA同时固定源码、入口、依赖清单和实际复验记录。包内不含大型模型、缓存、已安装环境或旧二进制。

```bash
export INFERX_REPO="$PWD"
export INFERX_BUNDLE="$INFERX_REPO/results/three-node-optimization/ucloud3"
export INFERX_BUNDLE="$INFERX_BUNDLE/selected100-single-node-reproduction-20261018"
export INFERX_REPRO=/workspace/InferX/reproductions/selected100-rerun01
mkdir -p /workspace/InferX/reproductions
mkdir "$INFERX_REPRO"
printf '%s  %s\n' \
  62fc432a278f2b7f8c60472c173929fc1f537ea2785908207e8b1ce7703ed545 \
  "$INFERX_BUNDLE/reproduction-kit-v1.tar.gz" | sha256sum -c -
tar -xzf "$INFERX_BUNDLE/reproduction-kit-v1.tar.gz" \
  -C "$INFERX_REPRO" --strip-components=1
cd "$INFERX_REPRO"
sha256sum -c SHA256SUMS
export INFERX_KIT="$INFERX_REPRO"
```

包内MANIFEST列出1207个payload文件，全部1023个引擎源码文件齐全。源码聚合SHA为`c976692d9657408308193b50772c700382a6e243d9da353165205dc5252f76a8`，定义为按路径排序的紧凑JSON文件SHA映射的SHA；它和压缩包SHA不同。构建脚本在编译前后分别逐文件核验。

`evidence/`保存ucloud3真正执行的命令和664构建回执。可移植包把构建锁改到本目录、用`asset-sources.json`显式映射输入；入口使用已实际评测的r2版本，等待同一服务的`/models`就绪后再调用原评测器。包内代码与实际入口对应关系见`evidence/packaged-vs-actually-tested-code.json`。一次新构建的SHA可以不同，不能预先写成664或继承旧成绩。

## 2. 新机器首次安装依赖

本轮ucloud3复验复用了已安装环境。下面是第三方首次安装步骤，未冒称本轮重新安装过这些依赖。要求Linux x86_64、48GB RTX 4090、适配CUDA13.2的驱动、CUDA Toolkit13.2.51，以及可执行的micromamba（记录版本2.9.0）。`nvcc`与驱动分别提供编译工具和GPU运行支持；本流程不修改系统驱动。

从刚解压的复验包目录继续，保留上一节的环境变量。以下把依赖安装在`/workspace/InferX`，与本轮命令的基础前缀一致；已有环境应先保留，不要覆盖。Conda使用带包校验值的`build-conda-explicit.lock`，不要把人类可读的`.txt`列表当成安装锁。

```bash
# Continue from the extracted kit root.
export INFERX_KIT="$INFERX_REPRO"
export INFERX_PROVENANCE="$INFERX_KIT/asset-origin"
export INFERX_LOCKS="$INFERX_PROVENANCE/remote-provenance/dependency-locks"
export INFERX_DEPENDENCIES=/workspace/InferX
export INFERX_MAMBA="$(command -v micromamba)"
export INFERX_CUDA_ROOT=/usr/local/cuda-13.2
mkdir -p "$INFERX_DEPENDENCIES"/{envs,eval-envs,config,cache,tools,source}

"$INFERX_MAMBA" --no-rc create --yes \
  --root-prefix "$INFERX_DEPENDENCIES/cache/mamba" \
  --prefix "$INFERX_DEPENDENCIES/envs/build" \
  --file "$INFERX_KIT/evidence/build-conda-explicit.lock"
```

```bash
# Preserve the original freeze. packaging comes from the explicit conda lock.
python3 - "$INFERX_LOCKS/build-pip-freeze.txt" \
  "$INFERX_DEPENDENCIES/config/build-pip-install.txt" <<'PY'
from pathlib import Path
import sys
lines = Path(sys.argv[1]).read_text().splitlines()
local = [line for line in lines if ' @ file:' in line]
assert len(local) == 1 and local[0].startswith('packaging @ file:///')
Path(sys.argv[2]).write_text('\n'.join(line for line in lines if line not in local) + '\n')
PY
export INFERX_BUILD_PY="$INFERX_DEPENDENCIES/envs/build/bin/python"
"$INFERX_BUILD_PY" -m pip install --no-deps \
  --index-url https://pypi.org/simple \
  -r "$INFERX_DEPENDENCIES/config/build-pip-install.txt"
"$INFERX_BUILD_PY" -c 'import packaging; assert packaging.__version__ == "26.3"'
"$INFERX_BUILD_PY" -m pip install --no-deps \
  --index-url https://pypi.org/simple uv==0.11.29
"$INFERX_DEPENDENCIES/envs/build/bin/uv" --version
"$INFERX_BUILD_PY" -m pip check

for inferx_eval_spec in aiperf:0.13.0 evalscope:1.13.0; do
  inferx_eval_name=${inferx_eval_spec%:*}
  inferx_eval_version=${inferx_eval_spec#*:}
  inferx_eval_dir="$INFERX_DEPENDENCIES/eval-envs/$inferx_eval_name-$inferx_eval_version"
  "$INFERX_BUILD_PY" -m venv "$inferx_eval_dir"
  "$inferx_eval_dir/bin/python" -m pip install \
    --index-url https://pypi.org/simple pip==26.2.1
  "$inferx_eval_dir/bin/python" -m pip install --no-deps \
    --index-url https://pypi.org/simple \
    -r "$INFERX_LOCKS/$inferx_eval_name-pip-freeze.txt"
  "$inferx_eval_dir/bin/python" -m pip check
done
```

```bash
# Download the three NLTK resources and verify their original content hashes.
python3 - \
  "$INFERX_PROVENANCE/original-support/results/ninfer-performance/preparation/nltk-resources.json" \
  "$INFERX_DEPENDENCIES/eval-envs/nltk_data" <<'PY'
from pathlib import Path
import hashlib, json, sys, urllib.request, zipfile
manifest, root = Path(sys.argv[1]), Path(sys.argv[2])
archives = root / '.archives'
archives.mkdir(parents=True, exist_ok=True)
for item in json.loads(manifest.read_text()):
    target = archives / item['file']
    if not target.exists():
        with urllib.request.urlopen(item['source'], timeout=120) as response:
            target.write_bytes(response.read())
    data = target.read_bytes()
    assert len(data) == item['bytes']
    assert hashlib.sha256(data).hexdigest() == item['sha256']
    kind = item['source'].split('/packages/', 1)[1].split('/', 1)[0]
    with zipfile.ZipFile(target) as archive:
        archive.extractall(root / kind)
PY
export NLTK_DATA="$INFERX_DEPENDENCIES/eval-envs/nltk_data"
"$INFERX_DEPENDENCIES/eval-envs/evalscope-1.13.0/bin/python" - <<'PY'
import nltk
for resource in ('tokenizers/punkt_tab', 'corpora/stopwords',
                 'taggers/averaged_perceptron_tagger_eng'):
    print(nltk.data.find(resource))
PY
```

```bash
# Obtain the exact grammar dependency; CUDA and its driver are prerequisites.
git clone https://github.com/mlc-ai/xgrammar.git \
  "$INFERX_DEPENDENCIES/source/xgrammar_source-src"
git -C "$INFERX_DEPENDENCIES/source/xgrammar_source-src" \
  checkout 4546988d1cf51670a10eafb8795d37e4a2f0d833
git -C "$INFERX_DEPENDENCIES/source/xgrammar_source-src" \
  submodule update --init --recursive
export PATH="$INFERX_DEPENDENCIES/envs/build/bin:$INFERX_CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$INFERX_DEPENDENCIES/envs/build/lib:$INFERX_CUDA_ROOT/lib64"
"$INFERX_CUDA_ROOT/bin/nvcc" --version
"$INFERX_DEPENDENCIES/envs/build/bin/x86_64-conda-linux-gnu-g++" --version
```

`build-pip-freeze.txt`中的`packaging @ file:///...`是Conda打包机路径，不能在新机器安装。上述步骤保留原freeze，只从pip安装列表移除这一项；`packaging==26.3`由显式Conda锁提供并单独校验。两个评测环境分别按原始freeze安装，并执行`pip check`。NLTK资源按原URL下载后核对完整SHA；任何不符都应停止并保留文件。

量化拟合使用单独的GPU PyTorch环境，其具体依赖随量化重建流程记录；不要用这里的API评测环境代替拟合环境。

### 独立新环境安装的实际验证

另外在新prefix完成了27步安装，857.424秒、各步退出码0；三个环境的pip check、未修改runner的check，以及原v1 prepare_assets组合均通过。[完整回执](../../results/three-node-optimization/ucloud3/selected100-single-node-reproduction-20261018/environment-install-addendum/delivery/FINAL-SUMMARY.json)与[冻结附件](../../results/three-node-optimization/ucloud3/selected100-single-node-reproduction-20261018/environment-install-addendum/environment-data-delivery.tgz)独立于v1包保存，附件SHA为`f82b9fc992e9cfac84c64bf8a1dfa87fe7c5d8c72da20fbbec3161a3bc1b2c36`。原EvalScope离线加载器实际读取固定全部300题并逐内容核对，前100题111指令一致，6份原评分器源文件SHA一致；没有发推理请求或生成新分数。

133项Conda锁一致；原性能138项、质量123项依赖版本均保留，新环境各增加显式pip26.2.1。55项运行库中42项原始SHA相同，另13项仅在安装prefix字符串和NUL填充处不同；严格规范这些位置后其余字节一致。这不等于55项原始SHA全部相同。

664的DT_RPATH仍指向原build环境；上述性能和质量不是在新prefix重跑。该附件只验证新环境安装、原加载器及资产准备，没有在新环境重建服务或做GPU评测。v1完整构建路线使用本指南默认依赖prefix`/workspace/InferX`；其build_owner仍固定这个路径，不能把本次安装测试解释为任意自定义prefix的完整构建已通过。


## 3. 准备固定模型、缓存与原始评测数据

本节复现ucloud3已报告的旧缓存配置。固定公共模型为18,210,531,328字节；tokenizer从其内嵌资源提取。历史478个缓存服务文件的独立归档为`selected100-u3-47-caches.tar`（12,082,360,320字节），本次提交不附该大文件；[附件来源与校验](../../results/three-node-optimization/ucloud1/selected100-reproduction-chain-CPU/README.md)记录了已生成的原始归档。如已持有该归档，才执行本节第二个代码块校验解压；首次复验在下方完成公共模型和tokenizer准备后，直接转[116量化流程](quantization-reproduction.md)，不执行旧缓存后续分支。

如选择从公开FP8输入重新生成缓存，应走独立量化重建流程，并绑定新资产清单。只有内容SHA完全相同时才可使用本节旧缓存配方；新缓存不得继承本节664成绩。

```bash
export INFERX_MODEL_DIR="$INFERX_DEPENDENCIES/models/qwen38-ninfer"
export INFERX_TOKENIZER="$INFERX_DEPENDENCIES/models/tokenizer"
export INFERX_FROZEN_ASSETS="$INFERX_DEPENDENCIES/inputs/selected100-assets"
export INFERX_DATA_CACHE="$INFERX_DEPENDENCIES/cache/modelscope-rerun01"
mkdir -p "$INFERX_DEPENDENCIES/inputs"
"$INFERX_DEPENDENCIES/envs/build/bin/hf" download \
  neroued/Qwen3.8-27B-NInfer qwen3_8_27b.ninfer \
  --revision 3526913004b1cf552cb57b88d6a5c6f5e4a89a70 \
  --local-dir "$INFERX_MODEL_DIR"
printf '%s  %s\n' \
  eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e \
  "$INFERX_MODEL_DIR/qwen3_8_27b.ninfer" | sha256sum -c -
python3 "$INFERX_KIT/asset-origin/extract_tokenizer.py" \
  --model "$INFERX_MODEL_DIR/qwen3_8_27b.ninfer" --out "$INFERX_TOKENIZER"
```

**以下仅供已持有旧缓存的历史复查。默认116路线在完成上面的模型/tokenizer准备后转入量化指南。**

```bash
# Optional historical branch only: use an existing verified old-cache archive.
export INFERX_CACHE_ARCHIVE="$INFERX_DEPENDENCIES/inputs/selected100-u3-47-caches.tar"
printf '%s  %s\n' \
  a8936aa7d39e9f971256a30b415f6f271f4f70418d0705233c0c84bfabfc3fb7 \
  "$INFERX_CACHE_ARCHIVE" | sha256sum -c -
mkdir "$INFERX_FROZEN_ASSETS"
tar -xf "$INFERX_CACHE_ARCHIVE" -C "$INFERX_FROZEN_ASSETS"
(cd "$INFERX_FROZEN_ASSETS" && sha256sum -c SHA256SUMS)
python3 "$INFERX_KIT/asset-origin/verify_assets.py" \
  --root "$INFERX_FROZEN_ASSETS"
```

```bash
export INFERX_IFBENCH_PACKAGE="$INFERX_REPO/results/three-node-optimization/ucloud1"
export INFERX_IFBENCH_PACKAGE="$INFERX_IFBENCH_PACKAGE/ifbench-fixed-dataset-acquisition-CPU"
export INFERX_IFBENCH_PACKAGE="$INFERX_IFBENCH_PACKAGE/ifbench-fixed-dataset-cache-r1.tar.gz"
printf '%s  %s\n' \
  c267da418e8c0a79b402e37237b14fde4a17233f60b8d5f866b85493fda757aa \
  "$INFERX_IFBENCH_PACKAGE" | sha256sum -c -
export INFERX_IFBENCH="$INFERX_REPRO/ifbench-data"
mkdir "$INFERX_IFBENCH"
tar -xzf "$INFERX_IFBENCH_PACKAGE" -C "$INFERX_IFBENCH" --strip-components=1
export MODELSCOPE_CACHE="$INFERX_DATA_CACHE"
python3 "$INFERX_IFBENCH/stage_ifbench.py" \
  --cache-root "$MODELSCOPE_CACHE" \
  --receipt "$INFERX_REPRO/ifbench-stage-receipt.json"
"$INFERX_DEPENDENCIES/eval-envs/evalscope-1.13.0/bin/python" \
  "$INFERX_IFBENCH/check_offline_cache.py" \
  --cache-root "$MODELSCOPE_CACHE" --evalscope \
  --receipt "$INFERX_REPRO/ifbench-original-loader-check.json"
```

```bash
python3 - "$INFERX_REPRO/asset-sources.json" \
  "$INFERX_DEPENDENCIES" "$INFERX_MODEL_DIR" "$INFERX_TOKENIZER" \
  "$INFERX_FROZEN_ASSETS" "$INFERX_DATA_CACHE" <<'PY'
from pathlib import Path
import json, sys
path, prefix, model, tokenizer, caches, data_cache = map(Path, sys.argv[1:])
mapping = {
    'assets/model/qwen3_8_27b.ninfer': str(model/'qwen3_8_27b.ninfer'),
    'assets/tokenizer': str(tokenizer),
    'envs/build': str(prefix/'envs/build'),
    'eval-envs': str(prefix/'eval-envs'),
    'tools': str(prefix/'tools'),
    'cache/modelscope': str(data_cache),
}
mapping.update({f'assets/{name}':str(caches/name)
                for name in ('weights025','weights041','weights047')})
path.write_text(json.dumps(mapping, indent=2)+'\n')
PY
cd "$INFERX_REPRO"
python3 prepare_assets.py
cat asset-verification.json
cat evaluation-dependency-check.json
```

确认`asset-verification.json`中`complete`与`content_sha_rehashed`均为`true`，缓存数为478，评测依赖检查退出码为0。025覆盖64层、256文件；041覆盖64层、128文件；047服务选择47层、94文件。准备阶段完整重算模型和缓存SHA；日常服务入口主要检查文件大小和目录集合，两者范围不同。

IFBench补充包保留原始300题、344条指令；原runner仍选前100题、111条指令。固定ModelScope修订`cf340e885a36d4b7b02692ddb7da242ea2643f4e`与HF修订`2e8a48de45ff3bf41242f927254ca81b59ca3ae2`的Parquet逐字节相同，SHA为`80037e4d99c39a55c1e2e7d5a863d8d9edeb5ebe136ba8c1e849f8c015027c6a`，且100题metadata与105、quality002实测输入相同。这是可准确获取的等内容固定修订；历史加载时的revision未记录，不倒填为上述修订。预置数据不发送模型请求，也不改变原评测器。

本轮ucloud3的实际操作复用了原只读资产和已安装环境，重新计算全部内容SHA。上述新机器准备命令保留这一区别；路径通过`asset-sources.json`显式配置，解压包中的旧节点路径不再是隐藏前提。

## 4. 正常完整构建并保存身份

本轮工具链为Linux x86_64、CUDA13.2.51、GCC13.4、CMake/Ninja，Release，CUDA架构89。构建与评测期间使用单张空闲GPU；不要同时运行另一个推理服务。

```bash
python3 "$INFERX_REPRO/build_owner.py"
cat "$INFERX_REPRO/build-job.json"
cat "$INFERX_REPRO/registered-CPU-CTest.log"
cat "$INFERX_REPRO/runtime-identity.json"
```

`build_owner.py`从 `source/` 配置新的 `build/`，正常编译 `ninfer-serve` 及调度CPU测试，没有复制旧对象或替换设备归档。完整CMake参数和实际执行argv写入 `build-job.json`，编译日志为 `normal-build.log`。要求构建状态为 `complete`、7项注册CPU测试通过、编译前后源码身份相同，并保存新二进制SHA与55项动态库清单。

这些CPU测试只覆盖所选调度契约。新构建的GPU正确性、性能及质量由后续实际运行独立验证；不能仅凭源码相同沿用旧二进制成绩。

## 5. 将便携入口绑定到新二进制

```bash
python3 "$INFERX_REPRO/bind_rebuilt_recipe.py"
cat "$INFERX_REPRO/rebuilt-recipe-binding.json"
cd "$INFERX_REPRO"
python3 -B -m unittest discover -s entrypoint/tests -p "test_*.py" -v
```

该步骤把新产物复制到 `artifacts/100-k3-rebuilt-20261018/ninfer-serve`，新增 `100-k3-u3-47-rebuilt-20261018.json` 配方及index条目，记录实际binary/source SHA。原 `100-k3-u3-47.json` 和历史二进制身份保留。新配方初始状态不继承任何性能或质量结果；运行结论以本次输出为准。

服务与两个评测客户端都必须选择此新配方。原评测runner仍固定为 `d80df390…`，数据、原始请求namespace、采样和统计口径不变。`entrypoint-runtime-freeze.json`记录绑定后的实际入口清单；原配方保留，新配方读取自身资格描述。r2入口的18项CPU测试包含服务身份和就绪等待，但不能替代后续GPU实测。

## 6. 启动服务并运行C14性能测试

以下两个终端均先 `cd "$INFERX_REPRO"`；新终端须先设置相同的 `INFERX_REPRO`。终端一前台启动：

```bash
python3 -B entrypoint/serve_candidate.py \
  --recipe entrypoint/candidate_recipes/100-k3-u3-47-rebuilt-20261018.json \
  --profile perf-c14 --root "$PWD" --port 18540 \
  --log-dir "$PWD/service-perf-c14-001"
```

终端二执行下列命令；r2入口会核对同一服务身份并等待`/models`真正就绪，不把端口监听当作模型加载完成：

```bash
OPENAI_API_KEY=EMPTY python3 -B entrypoint/perf_candidate.py \
  --recipe entrypoint/candidate_recipes/100-k3-u3-47-rebuilt-20261018.json \
  --concurrency 14 --root "$PWD" \
  --service-launch "$PWD/service-perf-c14-001/launch.json" \
  --base-url http://127.0.0.1:18540/v1 \
  --output-dir "$PWD/performance-C14-001"

cat performance-C14-001/evaluation-identity.json
cat performance-C14-001/summary.json
```

配置为客户端14、引擎13、KV121856、上下文65536、2048-token输入块、普通MTP K3。共137次请求：128正式、8预热、1探测；冷输入8192、输出512，temperature0、ignore_eos。保留正式阶段最初的请求；等待进入引擎的时间计入TTFT。

入口先核对同机服务的PID、启动时刻、二进制、配方、argv/env与监听socket，评测后核对全部137条实际请求的canonical SHA。原版解析器的 `summary.json` 没有 `meets_slo` 字段，入口回执中的该值因此为 `null`。应读取 `metrics.time_to_first_token.p95` 和 `metrics.inter_token_latency.p95`（均为毫秒），分别核对≤10000与≤40；两项同时满足才通过。合法SLO失败仍保留完整记录，不能用进程退出码0代替SLO判定。

## 7. 停止性能服务，再运行独立原100题质量评测

性能评测结束后，在终端一用 `Ctrl+C` 正常停止该性能服务，确认原PID与监听端口已经退出，再启动质量配置。不要同时启动两套服务。

终端一：

```bash
python3 -B entrypoint/serve_candidate.py \
  --recipe entrypoint/candidate_recipes/100-k3-u3-47-rebuilt-20261018.json \
  --profile quality100 --root "$PWD" --port 18541 \
  --log-dir "$PWD/service-quality100-002"
```

服务就绪后，终端二：

```bash
OPENAI_API_KEY=EMPTY python3 -B entrypoint/accuracy_candidate.py \
  --recipe entrypoint/candidate_recipes/100-k3-u3-47-rebuilt-20261018.json \
  --root "$PWD" \
  --service-launch "$PWD/service-quality100-002/launch.json" \
  --base-url http://127.0.0.1:18541/v1 \
  --output-dir "$PWD/quality100-002"

cat quality100-002/evaluation-identity.json
cat quality100-002/evaluation/summary.json
```

质量配置为客户端/引擎均13、KV532480、普通K3。原IFBench前100题包含111条指令约束；生成配置只显式覆盖 `max_tokens=32768`，保留默认thinking、模型采样和自然EOS。完整评分包括所有100题，空最终答案、格式失败和其他错误均保留，不筛选成功样本。

`evaluation-identity.json`标识流程是否完成；题目严格/宽松、指令严格/宽松分数来自原版EvalScope报告。完成后正常停止质量服务。GPU上每次只运行一个服务；两种KV配置及两轮评测不得拼成同一次测试。

## 8. 结果在哪里、怎样核对

以下路径均相对于 `INFERX_REPRO`。

| 要核对的内容 | 文件或目录 |
|---|---|
| 实际构建命令、退出码和编译日志 | `build-job.json`、`configure.log`、`normal-build.log` |
| 构建前后源码、新binary与动态库 | `actual-source-before.json`、`actual-source-after.json`、`runtime-identity.json` |
| 模型与缓存全量校验 | `model-verification.json`、`asset-verification.json`、`deployment-layout.json` |
| 服务真实身份及请求日志 | `service-perf-c14-001/`、`service-quality100-002/` |
| 性能汇总和原始记录 | `performance-C14-001/summary.json`、`performance-C14-001/c14/` |
| 质量汇总和配置 | `quality100-002/evaluation/summary.json`、`quality100-002/evaluation/accuracy/{task_config,resolved_config}.json` |
| 逐题预测、review和原评分 | `quality100-002/evaluation/accuracy/native/{predictions,reviews,reports}/` |
| 便携入口实际版本 | `entrypoint-runtime-freeze.json`、`rebuilt-recipe-binding.json` |

性能逐请求计算 `TPOT=(E2E−TTFT)/(输出Token数−1)`，再取线性P95；吞吐沿用原profiling阶段起止时间，不替换为请求时间包络。质量核对100题、111条约束、四项分数及错误、空最终答案和截断数量。

本指南按两个前台终端操作，不依赖原实验的自动owner。包内`evidence/`保存旧性能入口及r2质量入口的实际执行命令和回执；质量首次启动超时生成0题，已封存。r2仅增加同一进程的只读`/models`就绪等待，期间不发送生成探针、不重试已评测题目。

不同二进制、缓存、采样或服务配置的结果分别列出。对照历史100/105时，展示新旧身份和实际差异；新一轮未满足SLO时如实保留，不修改历史通过曲线。

## 9. 本次独立复验的实际结果

以下为ucloud3已经完成的独立新构建，二进制SHA为`664cfbb989da8de5aaf58b755ecb68717a966c6992a3e377a8b04e2b2f1f7518`，源码仍为`c976692d…`。模型及478个旧缓存服务文件已完整重哈希；本表不使用新量化缓存。

| 测试 | 实际结果 | 原始证据 |
|---|---|---|
| C14性能，客户端14 / 引擎13，KV121856 | 原128＋8＋1；错误0；TTFT P95 **9.89641735465 s**、TPOT P95 **38.28584995440313 ms**，两项SLO通过；吞吐 **304.89665937584215 tok/s** | [原JSON](../../results/three-node-optimization/ucloud3/selected100-single-node-reproduction-20261018/performance-delivery/performance-C14-001/summary.json)、[独立复算](../../results/three-node-optimization/ucloud3/selected100-single-node-reproduction-20261018/root-C14-independent-audit.json) |
| 原100题质量，客户端/引擎13，KV532480 | 题目严格 **90/100**、宽松 **96/100**；指令严格 **100/111**、宽松 **106/111**；错误0、空最终回答1、长度截断1 | [原JSON](../../results/three-node-optimization/ucloud3/selected100-single-node-reproduction-20261018/quality-delivery/quality100-002/evaluation/summary.json)、[独立复算](../../results/three-node-optimization/ucloud3/selected100-single-node-reproduction-20261018/root-quality-independent-audit.json) |

吞吐分母是profiling阶段214.944959168秒，分子65536输出Token。原质量JSON将指令分数四舍五入为0.9009和0.955；上表整数分子来自全100题逐项重算，不是由显示小数倒推。唯一空回答和截断属于同一道key88，达到32768输出上限；仍计入全部100题。

质量第一次启动在模型加载完成前超时，生成0题，记录完整保留；r2等待`/models`就绪3次、32.947901秒，期间生成探针0次。性能和第二次质量服务均已正常回收。历史e4ed构建的85/88分与本轮90/96分独立记录；没有配对重复试验证明差异由性能优化或量化造成。

## 10. 从公开输入重新量化

[完整量化重建流程](quantization-reproduction.md)说明冻结002包的获取、FP8逐文件校验、4训练/2留出校准、拟合环境安装、三阶段新缓存生成、新源码服务绑定及原性能/质量评测。该分支不依赖旧量化缓存附件；结果与本页664旧缓存复验分开。实际完成范围见[交付入口](../DELIVERY.md)。

116更新：三类量化已实际重建239矩阵／478文件，与原selected478的SHA及大小全部一致、差异0；正常重建的新二进制为`e919f180…878f5`，使用独立116-fresh47配方。详细表与回执见[量化产物完成情况](quantization-reproduction.md#已完成的116量化产物)。该新二进制已完成独立C14：TTFT P95 9.86221024060s、TPOT P95 39.42860192427ms，两项SLO通过，吞吐305.53286821935 tok/s；128正式请求全部成功、错误0。独立原100题质量已完成：题目strict85/100、loose88/100，指令95/111、99/111，0错误；key50同时为空final和32768截断，全部保留计分。原verifier全100独审通过，服务正常回收；这是e919自己的结果，不继承本页664成绩。原始记录与独立核对见[116性能结果](quantization-reproduction.md#116新二进制的实际c14性能)及[116质量终态](quantization-reproduction.md#116新二进制的原100题质量)。
