#!/usr/bin/env bash
# Prepare the original judge's exact tool versions, without altering the judge.
set -euo pipefail
export INFERX_ROOT=${1:?Usage: prepare_judge_env.sh /absolute/data/prefix}
source "$INFERX_ROOT/config/ninfer-env.sh"
inferx_index=${INFERX_PYPI_INDEX:-https://pypi.org/simple}
export UV_HTTP_TIMEOUT=${INFERX_DOWNLOAD_TIMEOUT:-120}
export UV_CONCURRENT_DOWNLOADS=${INFERX_PARALLEL_DOWNLOADS:-4}
"$INFERX_ROOT/envs/build/bin/python" -m pip install \
  --index-url "$inferx_index" 'uv==0.11.29'
inferx_uv="$INFERX_ROOT/envs/build/bin/uv"
inferx_envs="$INFERX_ROOT/eval-envs"
mkdir -p "$inferx_envs" "$INFERX_ROOT/config"
for inferx_spec in aiperf:0.13.0 evalscope:1.13.0; do
  inferx_package=${inferx_spec%:*}
  inferx_version=${inferx_spec#*:}
  inferx_python="$inferx_envs/$inferx_package-$inferx_version/bin/python"
  if [[ ! -x "$inferx_python" ]]; then
    "$inferx_uv" venv --python "$INFERX_ROOT/envs/build/bin/python" \
      "$inferx_envs/$inferx_package-$inferx_version"
  fi
  if [[ "$inferx_package" == aiperf ]]; then
    "$inferx_uv" pip install --python "$inferx_python" \
      --index-url "$inferx_index" "aiperf==$inferx_version"
  else
    # API-only evaluation requires no CUDA PyTorch; keep the client off the GPU.
    "$inferx_uv" pip install --python "$inferx_python" --torch-backend cpu \
      --index-url "$inferx_index" \
      "evalscope[ifbench]==$inferx_version" emoji syllapy nltk
    "$inferx_python" - "$inferx_envs/nltk_data" <<'PY'
import sys
import nltk
nltk.data.path.insert(0, sys.argv[1])
for name, kind in (('punkt_tab', 'tokenizers'), ('stopwords', 'corpora'),
                   ('averaged_perceptron_tagger_eng', 'taggers')):
    try:
        nltk.data.find(f'{kind}/{name}/')
    except LookupError:
        nltk.download(name, download_dir=sys.argv[1], raise_on_error=True)
PY
  fi
  "$inferx_uv" pip freeze --python "$inferx_python" \
    > "$INFERX_ROOT/config/$inferx_package-pip-freeze.txt"
done
bash "$INFERX_ROOT/code/submission/organizer/run.sh" --mode check --env-dir "$inferx_envs"
