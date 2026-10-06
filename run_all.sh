#!/usr/bin/env bash
# Run the checks that need no retraining: the self-test, the Fig. 1b
# reachability demonstration and the analysis of the IonQ run. About a minute.
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/3] self-test"
python -m mgqc selftest

echo "[2/3] reachability (Fig. 1b)"
python -m mgqc.reachability > results/reachability.json

echo "[3/3] IonQ run"
( cd hardware && python analyze_qpu.py )

echo "done."
