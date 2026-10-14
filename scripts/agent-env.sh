#!/usr/bin/env bash
# Source from a local-client wrapper; never alter the user's global profile.
source /home/guokc/mypro/gnn-mount-sda1/InferX/env.sh
export PATH="$INFERX_ROOT/toolchains/node-v22.23.3-linux-x64/bin:$INFERX_ROOT/repo-venv/bin:$INFERX_ROOT/agent-clients/node_modules/.bin:$PATH"
export npm_config_cache="$INFERX_ROOT/cache/npm"
export XDG_CONFIG_HOME="$INFERX_ROOT/agent-state/xdg-config"
export XDG_DATA_HOME="$INFERX_ROOT/agent-state/xdg-data"
export XDG_STATE_HOME="$INFERX_ROOT/agent-state/xdg-state"
export XDG_CACHE_HOME="$INFERX_ROOT/cache/agents-xdg"
export PI_CODING_AGENT_DIR="$INFERX_ROOT/agent-state/pi"
export CLAUDE_CONFIG_DIR="$INFERX_ROOT/agent-state/claude"
export OPENCODE_CONFIG="$INFERX_ROOT/context-agents/config/opencode.json"
export OPENCODE_CONFIG_DIR="$INFERX_ROOT/agent-state/opencode"
export OPENCODE_DISABLE_DEFAULT_PLUGINS=true
export OPENCODE_DISABLE_LSP_DOWNLOAD=true
export OPENCODE_DISABLE_AUTOUPDATE=true
export PI_OFFLINE=1
mkdir -p "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$XDG_CACHE_HOME" \
  "$PI_CODING_AGENT_DIR" "$CLAUDE_CONFIG_DIR" "$OPENCODE_CONFIG_DIR"
