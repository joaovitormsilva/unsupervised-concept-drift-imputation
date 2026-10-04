#!/usr/bin/env bash
# Roda os cenários em paralelo com a suspensão da máquina bloqueada (systemd-inhibit).
# Checkpoints do run_batch.py permitem relançar sem perder pacientes já concluídos.
#
# Uso:
#   ./run_all_scenarios.sh                 # S1 S2 S3, 2 workers cada
#   WORKERS=3 ./run_all_scenarios.sh S2 S3
#   FRESH=1 ./run_all_scenarios.sh         # ignora checkpoints e recomeça
#   TUNED=1 ./run_all_scenarios.sh         # detectores com parâmetros do sweep
set -euo pipefail
cd "$(dirname "$0")"

if [[ -z "${TCC_INHIBITED:-}" ]]; then
    export TCC_INHIBITED=1
    exec systemd-inhibit --what=sleep:idle --mode=block \
        --who=tcc-batch --why="run_batch.py (TCC) em execução" "$0" "$@"
fi

if [[ $# -gt 0 ]]; then SCENARIOS=("$@"); else SCENARIOS=(S1 S2 S3); fi
WORKERS="${WORKERS:-2}"
EXTRA=()
[[ -n "${FRESH:-}" ]] && EXTRA+=(--fresh)
TAG=""
if [[ -n "${TUNED:-}" ]]; then EXTRA+=(--tuned); TAG="_tuned"; fi

LOG_DIR=Analysis/logs
mkdir -p "$LOG_DIR"

for s in "${SCENARIOS[@]}"; do
    ../.venv/bin/python run_batch.py --scenario "$s" --patients all --workers "$WORKERS" "${EXTRA[@]}" \
        > "$LOG_DIR/run${TAG}_$s.log" 2>&1 &
    echo "$s iniciado (PID $!) -> $LOG_DIR/run${TAG}_$s.log"
done
wait
echo "Todos os cenários terminaram."
