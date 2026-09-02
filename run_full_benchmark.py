import time
import json
from pathlib import Path

from experiments.compare_algos import run_algo_benchmark
from experiments.compare_hrl import run_hrl_benchmark


def main():
    print("=" * 80)
    print("      CRAFTAX + PUREJAXRL HIGH-SPEED BENCHMARK SUITE")
    print("      Replicando o Estudo do ProcGen na Velocidade do JAX")
    print("=" * 80)

    start_total = time.time()
    
    # 1. Benchmark de Famílias (PPO vs A2C vs DQN)
    algo_results_file = Path("results/algo_families_results.json")
    if algo_results_file.exists():
        print("\n[FASE 1/2] Resultados de Famílias já computados. Carregando 'results/algo_families_results.json'...")
        with open(algo_results_file, "r") as f:
            algo_results = json.load(f)
    else:
        print("\n[FASE 1/2] Executando Benchmark de Famílias de Algoritmos...")
        algo_results = run_algo_benchmark(total_steps=50000, num_envs=64, seeds=[42, 123])

    # 2. Benchmark de HRL (flat vs skip4 vs hrl vs hrl_learned)
    print("\n[FASE 2/2] Executando Benchmark de HRL e Abstração Temporal...")
    hrl_results = run_hrl_benchmark(total_steps=50000, num_envs=64, seeds=[42, 123])

    total_elapsed = time.time() - start_total

    # Montar Relatório Final
    print("\n" + "=" * 80)
    print("                    TABELA FINAL DE RESULTADOS")
    print("=" * 80)

    print("\n--- TABELA 1: Famílias de Algoritmos (PPO vs A2C vs DQN) ---")
    print(f"{'Algoritmo':<10} | {'Média FPS':<10} | {'Tempo Médio':<12} | {'Train Score':<12} | {'Unseen Score':<12} | {'Gen Gap':<10}")
    print("-" * 75)
    
    summary_data = {"algos": {}, "hrl": {}, "total_time_seconds": round(total_elapsed, 2)}

    for algo, runs in algo_results.items():
        avg_fps = sum(r['fps'] for r in runs) / len(runs)
        avg_time = sum(r['elapsed_sec'] for r in runs) / len(runs)
        avg_train = sum(r['train_score'] for r in runs) / len(runs)
        avg_unseen = sum(r['unseen_score'] for r in runs) / len(runs)
        avg_gap = sum(r['gen_gap'] for r in runs) / len(runs)
        
        summary_data["algos"][algo] = {
            "avg_fps": round(avg_fps, 1),
            "avg_time": round(avg_time, 2),
            "avg_train": round(avg_train, 3),
            "avg_unseen": round(avg_unseen, 3),
            "avg_gap": round(avg_gap, 3)
        }
        print(f"{algo:<10} | {avg_fps:<10.0f} | {avg_time:<10.1f}s | {avg_train:<12.2f} | {avg_unseen:<12.2f} | {avg_gap:<+10.2f}")

    print("\n--- TABELA 2: Hierarchical RL & Abstração Temporal ---")
    print(f"{'Modo HRL':<12} | {'Média FPS':<10} | {'Tempo Médio':<12} | {'Train Score':<12} | {'Unseen Score':<12} | {'Gen Gap':<10}")
    print("-" * 75)
    
    for mode, runs in hrl_results.items():
        avg_fps = sum(r['fps'] for r in runs) / len(runs)
        avg_time = sum(r['elapsed_sec'] for r in runs) / len(runs)
        avg_train = sum(r['train_score'] for r in runs) / len(runs)
        avg_unseen = sum(r['unseen_score'] for r in runs) / len(runs)
        avg_gap = sum(r['gen_gap'] for r in runs) / len(runs)
        
        summary_data["hrl"][mode] = {
            "avg_fps": round(avg_fps, 1),
            "avg_time": round(avg_time, 2),
            "avg_train": round(avg_train, 3),
            "avg_unseen": round(avg_unseen, 3),
            "avg_gap": round(avg_gap, 3)
        }
        print(f"{mode:<12} | {avg_fps:<10.0f} | {avg_time:<10.1f}s | {avg_train:<12.2f} | {avg_unseen:<12.2f} | {avg_gap:<+10.2f}")

    print("=" * 80)
    print(f"BENCHMARK COMPLETO CONCLUÍDO EM: {total_elapsed:.1f} SEGUNDOS (~{total_elapsed/60:.2f} MINUTOS)!")
    print("Comparado às ~9 horas do ProcGen tradicional, este benchmark rodou em tempo recorde.")
    print("=" * 80)

    # Salva resumo estruturado
    with open("results/summary_metrics.json", "w") as f:
        json.dump(summary_data, f, indent=2)


if __name__ == "__main__":
    main()
