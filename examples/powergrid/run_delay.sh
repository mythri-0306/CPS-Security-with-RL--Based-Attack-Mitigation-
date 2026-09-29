#!/bin/bash
# Power Grid CPS Testbed — run_delay.sh
# Launches the tc/netem link-latency attack scenario.
#
# Usage:
#   ./run_delay.sh [duration_sec] [run_label]
# Example:
#   ./run_delay.sh 300 delay_run1

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

export PYTHONPATH="$(cd ../.. && pwd):${PYTHONPATH:-}"

DURATION=${1:-300}
RUN_LABEL=${2:-delay_run1}

echo "=========================================================="
echo " Power Grid CPS — LINK-DELAY ATTACK SCENARIO"
echo " Duration : ${DURATION}s    Label: ${RUN_LABEL}"
echo " Delay    : 800 ms (tc netem on sensor interface)"
echo " Window   : t=60s → t=180s"
echo "=========================================================="

mkdir -p logs

echo "[Setup] Cleaning any previous Mininet artifacts..."
sudo mn -c > /dev/null 2>&1 || true

# Determine Python interpreter (validating it actually runs in Linux)
if [ -n "$VIRTUAL_ENV" ] && "$VIRTUAL_ENV/bin/python" -c "import sys" &>/dev/null; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif [ -f "/home/harshita_k/minicps/venv/bin/python" ] && "/home/harshita_k/minicps/venv/bin/python" -c "import sys" &>/dev/null; then
    PYTHON="/home/harshita_k/minicps/venv/bin/python"
elif [ -f "../../venv/bin/python" ] && "../../venv/bin/python" -c "import sys" &>/dev/null; then
    PYTHON="../../venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

echo "[Setup] Python: $PYTHON ($($PYTHON --version))"

echo "[Setup] Initialising state database..."
$PYTHON init.py

echo "[Simulation] Running link-delay scenario for ${DURATION}s..."
sudo -E $PYTHON run_delay.py --duration "$DURATION" --run-label "$RUN_LABEL" "${@:3}"

echo ""
echo "=========================================================="
echo " Delay Scenario Complete!"
echo " Log : $(pwd)/logs/${RUN_LABEL}.csv"
echo " Plot: $(pwd)/logs/${RUN_LABEL}.png"
echo "=========================================================="
if [ -f "logs/${RUN_LABEL}.csv" ]; then
    echo "CSV preview (first 3 + last 3 data rows):"
    head -n 4 "logs/${RUN_LABEL}.csv"
    echo "..."
    tail -n 3 "logs/${RUN_LABEL}.csv"
fi
