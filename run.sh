#!/usr/bin/env bash
# ==============================================================================
# futurecast: start the CoinDCX futures web dashboard
#
#   ./run.sh                    port 8000 on every interface  (PORT=9000 ./run.sh to change)
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

# The page is the Svelte build in web/dist; make it once if it's missing (`mise run build` rebuilds it)
if [ ! -f web/dist/index.html ]; then
    if command -v npm >/dev/null 2>&1; then
        (cd web && npm install && npm run build)
    else
        echo -e "\033[1;33mWarning: web/dist is missing and npm isn't installed: the API runs, the page doesn't.\033[0m"
    fi
fi

exec "${RUNNER[@]}" --port "${PORT:-8000}" "${ARGS[@]+"${ARGS[@]}"}"
