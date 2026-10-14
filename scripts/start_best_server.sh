#!/usr/bin/env bash
set -euo pipefail
# Selected for single-conversation code generation; see the dated tuning report.
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/start_tuned_server.sh" \
  ninfer 7 int8 2048 65536 1 warm
