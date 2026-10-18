# 从公开FP8重建025 / 041 / 047量化缓存

[English workflow](quantization-reproduction-en.md) · [源码构建、依赖和基础模型准备](reproduction.md) · [交付与实际运行回执](../DELIVERY.md)

这条流程生成新的量化缓存并运行自己的性能、质量评测。先完成主指南第1–2节依赖安装，以及第3节的公共基础模型下载和tokenizer提取；无需取得旧缓存附件，也不需要运行主指南的旧缓存评测。FP8、基础.ninfer和安装环境是明确列出的外部输入。

## 获取完整量化流程包

冻结002包为38,965,052字节，包含完整四套构建源码、采集器、拟合、绑定和原评测入口。根协调已独立核验归档实际4277个文件（4276个payload＋MANIFEST），没有旧服务缓存、旧编译对象或预编译服务。以下在主指南同一终端执行，目标目录必须不存在。

```bash
# Continue with INFERX_REPO and INFERX_DEPENDENCIES from the main guide.
export INFERX_Q="$INFERX_DEPENDENCIES/reproduction-inputs/portable116-002"
export INFERX_Q_ARCHIVE="$INFERX_REPO/results/three-node-optimization/ucloud2"
export INFERX_Q_ARCHIVE="$INFERX_Q_ARCHIVE/full-quantization-reproduction-116-portable-source-002.tar.gz"
mkdir -p "$INFERX_DEPENDENCIES/reproduction-inputs"
mkdir "$INFERX_Q"
printf '%s  %s\n' \
  7300c70a5857221537b878046d8c869ce640c788de4773d91d4e523b6bb41f9e \
  "$INFERX_Q_ARCHIVE" | sha256sum -c -
tar -xzf "$INFERX_Q_ARCHIVE" -C "$INFERX_Q" --strip-components=1
python3 - "$INFERX_Q" <<'PY'
from pathlib import Path
import hashlib, json, sys
root = Path(sys.argv[1])
files = json.loads((root / 'MANIFEST.json').read_text())['files']
assert len(files) == 4276
for name, spec in files.items():
    data = (root / name).read_bytes()
    assert len(data) == spec['bytes'], name
    assert hashlib.sha256(data).hexdigest() == spec['sha256'], name
print('Verified', len(files), 'payload files')
PY
```

## 量化重建的输入与边界

重建对象是本项目新增的025 FFN、041输入投影、047输出投影缓存。固定公共`.ninfer`基础容器作为独立外部输入下载；它来自另一固定BF16模型，本轮没有从FP8重新制造该基础容器。新缓存、新加载服务和新评测分别保存身份，不能沿用旧缓存的性能或精度分数。

### 获取FP8输入

从InferX交付仓库根目录执行，目标目录使用新路径。以下脚本只需Python3标准库，逐文件使用官方固定revision、字节数及SHA；72个内容文件共30,879,892,015字节，不下载旧挂载元数据`.mv/.msc`，也不假定存在一个统一snapshot revision。

```bash
export INFERX_FP8_FETCH="$INFERX_Q/fp8-acquisition"
export INFERX_FP8="$INFERX_DEPENDENCIES/inputs/Qwen3.8-27B-FP8"
mkdir -p "$INFERX_DEPENDENCIES/inputs"
python3 "$INFERX_FP8_FETCH/fetch_fp8.py" \
  --directory "$INFERX_FP8" \
  --receipt "$INFERX_DEPENDENCIES/inputs/fp8-acquisition-complete.json"
```

完整成功回执要求72个文件全部校验。已有正确文件会重新哈希；不匹配文件、符号链接或残留partial会被拒绝并保留。获取脚本的独立联网检查只实际下载了51,350字节的`config.json`；这项小文件验证不等于30.9GB权重已经重新下载。实际重建使用哪个FP8目录，以该轮全量校验回执为准。

公共基础模型单独使用`neroued/Qwen3.8-27B-NInfer@3526913004b1cf552cb57b88d6a5c6f5e4a89a70`，SHA `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e`。原发布说明记录BF16来源`Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`、转换器`Neroued/ninfer@52320554b5e71a9da96bff809ddf67ac5773ed63`及recipe `qwen3_8_27b-v1`。这里保留来源关系；本流程直接取得固定公共容器，不执行55.56GB BF16下载与转换。

### 固定校准数据和拟合算法

使用WikiText-2-raw-v1的4条训练、2条留出输入，每条8192个原始Token ID。训练选择seed为20261017，留出选择seed为20261018，请求seed为42。六条请求整体SHA为`f93b4d2b44c9ca642277f14c9260e64d8c95be2858b6c4e4b1db16874381481a`；每条请求生成1个Token、temperature0、K0。数据修订、偏移、原始Token ID和请求盐均随材料保留，IFBench题目不参与校准。

每条输入分为4个2048-token块，每块保留512行激活。每层16个训练块对应8192行，8个留出块对应4096行；先冻结拟合输出，再计算留出指标。拟合采用H512/G512、固定逐行scale、FP64 GPTQ反馈、damp0.01、block128、无actorder、关闭TF32。FP8逻辑值直接由E4M3乘BF16块scale恢复为FP32，不额外经一次BF16舍入。

| 执行顺序 | 采集时的服务输入 | 新产物 |
|---|---|---|
| 024采集 → 025拟合 | 固定公共base的原Q4/Q5前向，关闭完整S4路径 | 64层gate/down，256个服务文件 |
| 038采集 → 041拟合 | 固定base＋本轮新025 | 64层输入投影，128个服务文件 |
| output采集 → 047拟合 | 固定base＋本轮新025＋新041 | 固定47层输出投影，94个服务文件 |
| 选择并加载 | 保留全部新产物；047按固定47层mask拟合 | 256＋128＋94＝478个服务文件及独立资产清单 |

新目录中的每个阶段只读取本轮前驱产物。便携入口不依赖旧020/021大数组：025的L31检查改为新expanded/scales完整SHA与固定预期比较，并保留常量来源及原零差回执；拟合核心函数和训练主体保持一致。保留每矩阵数值检查和独立CPU256行反馈检查，省略冻结后的重复留出大矩阵诊断。这些检查不代替完整模型任务精度。

### 拟合环境首次安装

GPU拟合环境使用Python 3.12与PyTorch 2.13.0+cu130，和服务C++/CUDA13.2构建环境分开。44个依赖按官方wheel URL及SHA固定；原环境的METADATA已核对，但本轮没有重新下载全部wheel或重新安装该环境。以下供新机器安装，先设置主复验指南中的`INFERX_REPO`和`INFERX_DEPENDENCIES`。

```bash
export INFERX_QUANT="$INFERX_Q"
export INFERX_FIT_ENV="$INFERX_DEPENDENCIES/envs/quant-fit"
"$INFERX_DEPENDENCIES/envs/build/bin/python" -m venv "$INFERX_FIT_ENV"
"$INFERX_FIT_ENV/bin/python" -m pip install --no-deps --no-index \
  --require-hashes -r "$INFERX_QUANT/fit-environment/requirements-wheels.lock"
"$INFERX_FIT_ENV/bin/python" -m pip check
export FIT_PY="$INFERX_FIT_ENV/bin/python"
"$FIT_PY" -c 'import torch; assert torch.__version__ == "2.13.0+cu130"; print(torch.__version__, torch.version.cuda)'
```


## 在新目录执行完整流程

设置幻灯片中使用的路径变量。`INFERX_W`必须不存在，不能指向原116运行目录；依赖和公开模型路径接续主指南。系统Python默认`/usr/bin/python3`，须支持pidfd；默认端口19048、19050、19051用于采集，19052和19053用于最终性能及质量服务。

```bash
export INFERX_W="$INFERX_DEPENDENCIES/reproductions/quantization-rerun01"
export INFERX_M="$INFERX_MODEL_DIR/qwen3_8_27b.ninfer"
export INFERX_T="$INFERX_TOKENIZER"
export INFERX_B="$INFERX_DEPENDENCIES/envs/build"
export INFERX_C=/usr/local/cuda-13.2
export INFERX_X="$INFERX_DEPENDENCIES/source/xgrammar_source-src"
export INFERX_E="$INFERX_DEPENDENCIES/eval-envs"
export INFERX_U="$INFERX_DEPENDENCIES/tools"
```
```bash
python3 "$INFERX_Q/reproduce.py" \
  --work-dir "$INFERX_W" --fp8 "$INFERX_FP8" \
  --model "$INFERX_M" --tokenizer "$INFERX_T" \
  --fit-python "$FIT_PY" --build-env "$INFERX_B" \
  --cuda-root "$INFERX_C" --xgrammar "$INFERX_X" \
  --eval-envs "$INFERX_E" --tools "$INFERX_U"
```

该入口先正常构建采集与服务源码，再执行024→025→038→041→output→047，之后绑定新478文件缓存和新二进制，运行原C14以及独立原100题。`--prepare-only`仅展开、绑定路径和做CPU检查，不编译、不推理；若先运行该选项，正式执行须更换新的work目录。

IFBench固定数据由入口置入work目录下独立缓存，原300题/344指令、前100题/111指令内容保持不变。评测前保留独立的AIPerf0.13.0、EvalScope1.13.0及NLTK环境。所有编译和拟合结束后才进行性能测量。

## 核对产物、失败与真实完成范围

| 产物 | 核对内容 |
|---|---|
| `invocation.json`、`generated-path-bindings.json` | 外部输入和实际路径；没有调用旧历史owner |
| `portable-job.json`、`normal-builds-job.json` | 各阶段真正终态、退出码及新构建身份 |
| 各025/041/047目录 | 新packed/scales、逐矩阵数值检查及冻结清单 |
| `deployment116/` | 新资产manifest、recipe/index、真实服务身份和加载记录 |
| `deployment116/`下性能与质量目录 | 原始请求、128正式测量、100题预测/review和完整失败 |

性能必须同时满足TTFT P95≤10s、TPOT P95≤40ms，吞吐保留原profiling时间窗；质量保留全部100题、四项分数、错误、空final和长度截断。进程启动、目录存在或包校验通过均不等同评测完成。

实际116的首次024采集在追加源码重建要求前用了旧对象；后续另做正常完整源码采集，已对3072个激活文件逐SHA双边重新哈希，差异0，128项拟合训练绑定全部核对通过，因此允许复用本轮已拟合的新025。此先后顺序保留在[实际输入等价回执](../../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/024-normal-source-input-equivalence.json)及[独立核对](../../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/root-024-input-equivalence-audit.json)中；该3072文件对照只证明采集输入一致；后续三类拟合已另有完成回执，见下一节，新二进制评测仍单独报告。交付的002默认直接从新源码开始采集，不要求第三方走这段历史补验。

实际新缓存的阶段、加载、性能和质量以[交付索引](../DELIVERY.md)及116最终附件为准。664复验使用旧缓存所得304.8967 tok/s、90/96分，不是新缓存结果。基础容器来源、量化输入与缓存身份均分别保留。

## 已完成的116量化产物

116已实际完成全部239个服务矩阵的重新拟合：025为128个、041为64个、047为固定47个，共478个packed/scales文件。由新产物的完整清单逐项比对原selected478合同，SHA与字节数全部一致，差异0；见[三类缓存对照](../../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/fresh-vs-original478-SHA.json)和[实际拟合证据](../../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/fresh239-fit-light-evidence-001/full-fresh-fit-evidence.json)。这项结论来自本轮重新生成和校验的产物，没有用复制旧缓存代替拟合。

| 缓存 | 新拟合矩阵 | 服务文件 | 与原缓存SHA/大小一致 |
|---|---:|---:|---:|
| 025 FFN | 128 | 256 | 256 / 256 |
| 041输入投影 | 64 | 128 | 128 / 128 |
| 047输出投影 | 47 | 94 | 94 / 94 |

这些新文件已绑定独立[116-fresh47配方](../../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/deployment116/entrypoint/candidate_recipes/116-fresh47.json)。正常重建的新二进制SHA为`e919f18084f3cc09718158cfae071cdf8c58d5660715648db375643016c878f5`，源码聚合仍为`c976692d9657408308193b50772c700382a6e243d9da353165205dc5252f76a8`。新二进制的C14性能已独立实测通过，结果见下一节；原100题质量亦已完成，严格85/100、宽松88/100，详细终态见末节。不继承664或历史e4ed成绩，内容一致不能替代新构建的端到端评测。

### 116新二进制的实际C14性能

新二进制e919f180…与本轮新478缓存完成原C14评测：客户端14／引擎13，KV121856、普通MTP K3，冷8192输入／512输出；128正式＋8预热＋1探测。128条测量全部成功、错误0。TTFT P95 **9.862210240600s**，TPOT P95 **39.42860192427ms**，同时满足10s／40ms；吞吐 **305.532868219352 tok/s**。吞吐分母为原profiling观察时间窗214.497380861秒，输出65536 Token；保留初始测量和准入等待。

[原始汇总](../../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/fresh-C14-raw-001/deployment116/performance-C14/summary.json)保留原数值。原解析器没有meets_slo字段，入口回执为null；联合PASS按两项原P95分别比较阈值得到。性能服务已正常回收，随后独立完成原100题质量；其独立分数及完整失败见末节。此行为e919自己的实测，不替换主展示664或历史100/105的记录。

独立逐请求复算见[材料侧原raw审计](../records/typst/clean-reproduction/fresh-performance-audit.json)，阈值和服务回收记录见[C14终态判定](../../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/C14-threshold-evaluation.json)。

吞吐分母核对：`phase_manifest.json`中profiling的`end_ns=1792287366484028998`、`start_ns=1792287151986648137`，整数相减后除以10⁹得到214.497380861s；65536除以该值为305.53286821935166 tok/s。原AIPerf另导出的`benchmark_duration=214.49216998400001s`是不同字段，不用于替换上述吞吐分母。

### 116新二进制的原100题质量

同一e919二进制与新478缓存另启质量服务：客户端／活动槽13，KV532480、普通MTP K3，保留默认thinking、采样temperature=1、top_k=20、top_p=0.95及seed42，仅显式覆盖max_tokens=32768，按自然EOS结束。原IFBench前100题、111条指令全部完成，生成错误0；全100题的顺序、prompt、指令ID和kwargs与历史固定集合一致。

| 计分层级 | 严格strict | 宽松loose |
|---|---:|---:|
| 题目prompt |85/100 |88/100 |
| 指令instruction |95/111 |99/111 |

99题自然结束，1题触及长度上限。ordinal50／key50生成32768 Token后截断，保留reasoning但final为空；这是同一题的两种状态，仍完整计入全部分母。原EvalScope 1.13.0 verifier已独立重算100题，与原review逐项一致。原summary指令分数保留其四位小数0.8559／0.8919；上表整数由逐题计分汇总。质量服务正常回收，所属进程均退出、端口关闭、观察时GPU计算进程为空。

来源：[原始质量汇总](../../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/fresh-quality100-raw-and-CPU-001/deployment116/quality100/evaluation/summary.json)、[原verifier独立复核](../../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/fresh-quality100-raw-and-CPU-001/quality-independent-CPU-001/summary.json)、[完整失败及空答案／截断记录](../../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/fresh-quality100-raw-and-CPU-001/quality-independent-CPU-001/failures-empty-length.jsonl)、[实际终态](../../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/fresh-quality100-raw-and-CPU-001/actual-terminal-and-GPU-116.json)、[材料侧47文件及100题核对](../records/typst/clean-reproduction/fresh-quality-audit.json)。题级85/88与历史105相同，但严格指令分数为95/111，历史105为96/111；不称为精度完全一致。这里只覆盖原前100题，不代表全300题或一般任务精度，也不能把不同构建间的单轮差异归因于量化。
