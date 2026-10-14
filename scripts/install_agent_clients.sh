#!/usr/bin/env bash
set -euo pipefail
source /home/guokc/mypro/gnn-mount-sda1/InferX/env.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/agent-env.sh"
inferx_node_version=v22.23.3
inferx_archive="node-$inferx_node_version-linux-x64.tar.xz"
inferx_tools="$INFERX_ROOT/toolchains"
mkdir -p "$inferx_tools" "$INFERX_ROOT/agent-clients"
cd "$inferx_tools"
if [[ ! -x "$inferx_tools/node-$inferx_node_version-linux-x64/bin/node" ]]; then
  curl -fL --retry 3 --connect-timeout 30 --max-time 300 \
    "https://nodejs.org/dist/$inferx_node_version/SHASUMS256.txt" -o "node-$inferx_node_version-SHASUMS256.txt"
  curl -fL --retry 3 --connect-timeout 30 --max-time 300 \
    "https://nodejs.org/dist/$inferx_node_version/$inferx_archive" -o "$inferx_archive"
  awk -v filename="$inferx_archive" '$2 == filename' "node-$inferx_node_version-SHASUMS256.txt" | sha256sum -c -
  tar -xJf "$inferx_archive"
fi
export PATH="$inferx_tools/node-$inferx_node_version-linux-x64/bin:$PATH"
export npm_config_cache="$INFERX_ROOT/cache/npm"
npm install --prefix "$INFERX_ROOT/agent-clients" --ignore-scripts --no-audit --no-fund --save-exact \
  opencode-ai@1.19.0 @earendil-works/pi-coding-agent@1.2.0
# OpenCode needs its own binary link/copy step; other dependency scripts stay off.
node "$INFERX_ROOT/agent-clients/node_modules/opencode-ai/postinstall.mjs"
node --version
"$INFERX_ROOT/agent-clients/node_modules/.bin/opencode" --version
"$INFERX_ROOT/agent-clients/node_modules/.bin/pi" --version
/home/guokc/.local/bin/claude --version
