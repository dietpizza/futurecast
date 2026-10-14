#!/usr/bin/env bash
# ==============================================================================
# futurecast: start the CoinDCX futures web dashboard
# Safety: READ-ONLY (Zero Trade Execution)
#
#   ./run.sh                    http://127.0.0.1:8000  (PORT=9000 ./run.sh to change)
#   ./run.sh --port 8002        any `futurecast --help` option is passed through
#   ./run.sh --offline          simulated candles, no network
# ==============================================================================

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# --web / --ui were the old way to pick the dashboard; it is now the only thing to run.
ARGS=()
for arg in "$@"; do
    [[ "$arg" == "--web" || "$arg" == "--ui" ]] || ARGS+=("$arg")
done

if command -v uv >/dev/null 2>&1; then
    RUNNER=(uv run futurecast)
elif [ -x ".venv/bin/futurecast" ]; then
    RUNNER=(.venv/bin/futurecast)
else
    echo -e "\033[1;31mError: 'uv' not found and '.venv' not found. Please install uv or setup venv.\033[0m"
    exit 1
fi

exec "${RUNNER[@]}" --port "${PORT:-8000}" "${ARGS[@]+"${ARGS[@]}"}"
