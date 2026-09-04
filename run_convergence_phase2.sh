#!/usr/bin/env bash
# Phase 2: convergence training for ALL remaining REAL benchmarks.
# Sequential (one process at a time) so each gets the full RTX 4070 (8GB).
# Pixel experiments use num_envs=64 (image rollouts are memory-heavy).
set -u
source /root/jax-gpu/bin/activate
cd /mnt/c/Users/Acer/Downloads/MLE
mkdir -p results/logs
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

S=$(date +%Y%m%d_%H%M%S)
echo "=== Phase 2 convergence started ${S} ==="

echo ">>> [1/10] Algos (PPO/A2C/DQN) symbolic 8M x3 seeds"
python3 experiments/compare_algos.py --steps 8000000 --num-envs 256 --seeds 0 1 2 2>&1 | tee results/logs/p2_algos_${S}.log

echo ">>> [2/10] Discrete vs Continuous"
python3 experiments/compare_discrete_vs_continuous.py --steps 1000000 --marl-steps 2000000 --num-envs 128 --seeds 0 1 2>&1 | tee results/logs/p2_disc_cont_${S}.log

echo ">>> [3/10] MARL 4 paradigms (clear vs fog)"
python3 experiments/compare_marl_4_paradigms.py --steps 2000000 --num-envs 128 --seeds 0 1 2>&1 | tee results/logs/p2_marl4_${S}.log

echo ">>> [4/10] 3D: Brax (PPO/SAC) + drones 3D"
python3 experiments/compare_3d_benchmarks.py --brax-steps 2000000 --marl-steps 3000000 --num-envs 64 --seeds 0 1 2>&1 | tee results/logs/p2_3d_${S}.log

echo ">>> [5/10] Boxing Grand Prix (SAC teacher + offline suite)"
python3 experiments/run_boxing_grand_prix.py --teacher-steps 1000000 --offline-steps 40000 --dataset-size 200000 --num-envs 64 --num-rounds 64 --seed 0 2>&1 | tee results/logs/p2_boxing_${S}.log

echo ">>> [6/10] Representations Triad (pixels)"
python3 experiments/compare_representations_triad.py --steps 3000000 --num-envs 64 --seeds 0 1 --eval-episodes 64 2>&1 | tee results/logs/p2_triad_${S}.log

echo ">>> [7/10] Contrastive families (pixels)"
python3 experiments/compare_contrastive_types.py --steps 3000000 --num-envs 64 --seeds 0 1 --eval-episodes 64 2>&1 | tee results/logs/p2_contrastive_${S}.log

echo ">>> [8/10] Advanced paradigms (pixels)"
python3 experiments/compare_advanced.py --steps 2000000 --num-envs 64 --seeds 0 1 --eval-episodes 64 2>&1 | tee results/logs/p2_advanced_${S}.log

echo ">>> [9/10] ProcGen components (pixels + recurrent)"
python3 experiments/compare_procgen_missing_components.py --steps 2000000 --num-envs 64 --seeds 0 1 --eval-episodes 64 2>&1 | tee results/logs/p2_procgen_${S}.log

echo ">>> [10/10] Conv Architectures (pixels)"
python3 experiments/compare_architectures.py --steps 3000000 --num-envs 64 --seeds 0 1 --eval-episodes 64 2>&1 | tee results/logs/p2_arch_${S}.log

echo "=== Phase 2 convergence FINISHED $(date +%Y%m%d_%H%M%S) ==="
