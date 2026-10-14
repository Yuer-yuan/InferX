#!/usr/bin/env bash
set -euo pipefail
# Expanded agent context, verified with 8K output headroom and functional probes.
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/start_tuned_server.sh" \
  ninfer 7 rk4v4-e8 2048 262144 1 warm
