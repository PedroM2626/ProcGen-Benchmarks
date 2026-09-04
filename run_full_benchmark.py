"""Orchestrator for the two Craftax single-agent studies (algorithm families + HRL).

Both use REAL gradient training and REAL episodic evaluation. Results are written
to results/algo_families_results.json, results/hrl_results.json and summarized in
results/summary_metrics.json.
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.compare_algos import run_algo_benchmark
from experiments.compare_hrl import run_hrl_benchmark


def _agg(runs):
    n = len(runs)
    return {
        "avg_fps": round(sum(r["fps"] for r in runs) / n, 1),
        "avg_time": round(sum(r["elapsed_sec"] for r in runs) / n, 2),
        "avg_train": round(sum(r["train_score"] for r in runs) / n, 3),
        "avg_unseen": round(sum(r["unseen_score"] for r in runs) / n, 3),
        "avg_gap": round(sum(r["gen_gap"] for r in runs) / n, 3),
    }


def main(total_steps, num_envs, seeds):
    print("=" * 80, flush=True)
    print("      CRAFTAX + PUREJAXRL HIGH-SPEED BENCHMARK SUITE (100% TREINADO)", flush=True)
    print("=" * 80, flush=True)
    start = time.time()

    print("\n[FASE 1/2] Famílias de Algoritmos (PPO vs A2C vs DQN)...", flush=True)
    algo = run_algo_benchmark(total_steps=total_steps, num_envs=num_envs, seeds=seeds)

    print("\n[FASE 2/2] HRL & Abstração Temporal (flat/skip4/hrl/hrl_learned)...", flush=True)
    hrl = run_hrl_benchmark(total_steps=total_steps, num_envs=num_envs, seeds=seeds)

    total = time.time() - start
    summary = {"algos": {k: _agg(v) for k, v in algo.items()},
               "hrl": {k: _agg(v) for k, v in hrl.items()},
               "total_time_seconds": round(total, 2)}

    print("\n" + "=" * 80, flush=True)
    print("TABELA 1 — Famílias de Algoritmos", flush=True)
    for k, v in summary["algos"].items():
        print(f"  {k:<5} FPS={v['avg_fps']:>9,.0f} Train={v['avg_train']:.2f} Unseen={v['avg_unseen']:.2f}", flush=True)
    print("TABELA 2 — HRL & Abstração Temporal", flush=True)
    for k, v in summary["hrl"].items():
        print(f"  {k:<12} FPS={v['avg_fps']:>9,.0f} Train={v['avg_train']:.2f} Unseen={v['avg_unseen']:.2f}", flush=True)
    print(f"\nBENCHMARK COMPLETO EM {total:.1f}s (~{total/60:.2f} min)", flush=True)

    with open("results/summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=8_000_000)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = p.parse_args()
    main(total_steps=a.steps, num_envs=a.num_envs, seeds=tuple(a.seeds))
