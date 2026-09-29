#!/bin/bash
# Power Grid Frequency Control CPS Testbed - run_baseline.sh
# Entrypoint script to clean up, initialize DB, execute simulation, and plot results.

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

export PYTHONPATH="$(cd ../.. && pwd):${PYTHONPATH:-}"

DURATION=${1:-300}
RUN_LABEL=${2:-baseline_run1}

echo "=========================================================="
echo " Starting Power Grid CPS Baseline Testbed (${DURATION}s, label: ${RUN_LABEL})"
echo "=========================================================="

# Ensure logs directory exists
mkdir -p logs

# Clean up any leftover Mininet instances
echo "[Setup] Cleaning any previous Mininet artifacts..."
sudo mn -c > /dev/null 2>&1 || true

# Determine Python binary (validating it actually runs in Linux)
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

echo "[Setup] Using Python interpreter: $PYTHON ($($PYTHON --version))"

# Initialize SQLite database
echo "[Setup] Initializing state database..."
$PYTHON init.py

# Run simulation with sudo (required for Mininet Open vSwitch and network namespaces)
echo "[Simulation] Running baseline simulation for ${DURATION}s..."
sudo -E $PYTHON run_baseline.py --duration "$DURATION" --run-label "$RUN_LABEL" "${@:3}"

echo ""
echo "=========================================================="
echo " Simulation & Plotting Complete!"
echo " Log file: $(pwd)/logs/${RUN_LABEL}.csv"
echo " Plot:     $(pwd)/logs/${RUN_LABEL}.png"
echo "=========================================================="
if [ -f "logs/${RUN_LABEL}.csv" ]; then
    echo "Summary of generated CSV log (first and last 5 rows):"
    head -n 6 "logs/${RUN_LABEL}.csv"
    echo "..."
    tail -n 5 "logs/${RUN_LABEL}.csv"
fi
