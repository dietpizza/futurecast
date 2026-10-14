#!/usr/bin/env bash
# ==============================================================================
# futurecast: Real-Time Air-Gapped Alpha Signal Streamer
# Supports CoinDCX (e.g. B-DOGE_USDT, B-SOL_USDT, I-BTC_INR) and Binance (e.g. DOGEUSDT)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

WS_PORT="${PORT:-8765}"
SYMBOL="DOGEUSDT"

# Show help immediately if requested
for arg in "$@"; do
    if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
        exec uv run futurecast --help
    fi
    if [[ "$arg" == "--web" || "$arg" == "--ui" ]]; then
        echo -e "\033[1;36m====================================================================\033[0m"
        echo -e "\033[1;36m  ⚡ futurecast: Real-Time Multi-Coin Web Terminal\033[0m"
        echo -e "\033[1;33m  Safety: AIR-GAPPED (Read-Only / Zero Trade Execution)\033[0m"
        echo -e "\033[1;36m====================================================================\033[0m"
        exec uv run futurecast web --port "${PORT:-8000}"
    fi
done

# Parse CLI arguments to inspect symbol
OTHER_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--symbol)
            SYMBOL="$2"
            shift 2
            ;;
        *)
            OTHER_ARGS+=("$1")
            shift
            ;;
    esac
done

# Sanitize symbol for model path lookup (e.g., B-DOGE_USDT -> b_doge_usdt)
CLEAN_SYMBOL=$(echo "${SYMBOL}" | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]' '_' | sed 's/_$//')
MODEL_PATH="models/lightgbm_${CLEAN_SYMBOL}_residual_alpha.txt"

# Display startup banner
echo -e "\033[1;36m====================================================================\033[0m"
echo -e "\033[1;36m  ⚡ futurecast: Modular Cross-Asset Futures Signal Streamer\033[0m"
echo -e "  Target Symbol: \033[1;32m${SYMBOL}\033[0m"
echo -e "\033[1;33m  Safety: AIR-GAPPED (Read-Only / Zero Trade Execution)\033[0m"
echo -e "\033[1;36m====================================================================\033[0m"

# Ensure environment runner (uv or local virtualenv)
if command -v uv >/dev/null 2>&1; then
    RUNNER="uv run"
elif [ -f ".venv/bin/futurecast" ]; then
    RUNNER=".venv/bin/"
else
    echo -e "\033[1;31mError: 'uv' not found and '.venv' not found. Please install uv or setup venv.\033[0m"
    exit 1
fi

# Ensure trained model exists for the target symbol before streaming
if [ ! -f "${MODEL_PATH}" ]; then
    echo -e "\033[1;33mModel for ${SYMBOL} not found at ${MODEL_PATH}.\033[0m"
    echo -e "\033[1;34mBuilding decision tree model for ${SYMBOL} automatically...\033[0m"
    ${RUNNER} futurecast build --symbol "${SYMBOL}" --days 7
fi

echo -e "\033[1;32mStarting live signal stream for ${SYMBOL} (Local WebSocket on port ${WS_PORT})...\033[0m"
echo -e "\033[1;37mPress Ctrl+C to stop.\033[0m"
echo ""

# Launch continuous live stream
exec ${RUNNER} futurecast stream --symbol "${SYMBOL}" --ws-port "${WS_PORT}" "${OTHER_ARGS[@]}"
