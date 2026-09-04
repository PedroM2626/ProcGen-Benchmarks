#!/usr/bin/env bash
# Convergence-complete training for the VALIDATED packages (real gradient training).
# Runs sequentially on the single RTX 4070 (8GB) to avoid VRAM contention.
# Logs to results/logs/. Each experiment saves its JSON incrementally.
set -u
source /root/jax-gpu/bin/activate
cd /mnt/c/Users/Acer/Downloads/MLE
mkdir -p results/logs

STAMP=$(date +%Y%m%d_%H%M%S)
echo "=== Phase 1 convergence runs started at ${STAMP} ==="

echo ">>> [1/3] HRL (flat/skip4/hrl/hrl_learned) — 8M steps, seeds 0 1 2"
python3 experiments/compare_hrl.py --steps 8000000 --num-envs 256 --seeds 0 1 2 \
  --eval-episodes 128 --eval-horizon 1000 2>&1 | tee "results/logs/hrl_${STAMP}.log"

echo ">>> [2/3] MARL (IPPO/VDN/MAPPO/QMIX/MA-POCA) — 2M steps, seeds 0 1 2"
python3 experiments/compare_marl.py --steps 2000000 --num-envs 128 --seeds 0 1 2 \
  --eval-envs 256 2>&1 | tee "results/logs/marl_${STAMP}.log"

echo ">>> [3/3] Representations Triad (CNN/MLP/GNN) — 8M steps, seeds 0 1 2"
python3 experiments/compare_representations_triad.py --steps 8000000 --num-envs 256 --seeds 0 1 2 \
  --eval-episodes 128 --eval-horizon 1000 2>&1 | tee "results/logs/triad_${STAMP}.log"

echo "=== Phase 1 convergence runs FINISHED at $(date +%Y%m%d_%H%M%S) ==="
