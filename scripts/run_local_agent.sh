#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/agent-env.sh"
inferx_client=${1:?usage: run_local_agent.sh opencode|pi|claude [args...]}
shift
case "$inferx_client" in
  opencode)
    exec opencode --pure "$@"
    ;;
  pi)
    exec pi --provider inferx --model qwen38-27b --thinking off \
      --no-extensions --no-mcp --no-skills --no-prompt-templates --no-context-files --offline "$@"
    ;;
  claude)
    # Configure only this process; installed CLI and global account remain untouched.
    eval "$(/usr/bin/python3 - <<'PY'
import json,os,shlex
from pathlib import Path
settings=json.loads((Path(os.environ['INFERX_ROOT'])/'context-agents/config/claude.json').read_text())
for key,value in settings['env'].items():print('export '+key+'='+shlex.quote(value))
PY
)"
    exec /home/guokc/.local/bin/claude --bare --model qwen38-27b \
      --settings "$INFERX_ROOT/context-agents/config/claude.json" --setting-sources '' \
      --strict-mcp-config --mcp-config '{"mcpServers":{}}' --disable-slash-commands "$@"
    ;;
  *) echo "Unknown client: $inferx_client" >&2; exit 2 ;;
esac
