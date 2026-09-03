import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11.5,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'axes.edgecolor': '#cccccc'
})

# Carregar os resultados da matriz de 192 combinações
results_path = Path("results/absolute_3d_omniverse_192_results.json")
with open(results_path, "r") as f:
    data = json.load(f)

ranking = data["ranking"]

# Preparar figura com 4 subplots estratégicos
fig = plt.figure(figsize=(15, 9.5))
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

ax1 = fig.add_subplot(gs[0, :])  # Top Campeões na linha superior inteira
ax2 = fig.add_subplot(gs[1, 0])  # Médias por Algoritmo & Representação
ax3 = fig.add_subplot(gs[1, 1])  # Throughput de Simulação na GPU

# -------------------------------------------------------------
# SUBPLOT 1: Top 10 Campeões da Matriz Cartesiana 3D
# -------------------------------------------------------------
top10 = ranking[:10]
labels = [f"#{r['rank']} {r['ambiente'].replace('_3D','')}\n{r['algoritmo'].replace('_CTDE','').replace('_SAC','').replace('_',' ')}\n+ {r['representacao'].replace('_3D','').replace('_',' ')}\n+ {r['tecnica'].replace('_Baseline','').replace('_',' ')}" for r in top10]
scores = [r["reward"] for r in top10]

colors = ['#10b981', '#059669', '#047857', '#0284c7', '#0369a1', '#6366f1', '#4f46e5', '#8b5cf6', '#7c3aed', '#6d28d9']
bars1 = ax1.bar(range(10), scores, color=colors, edgecolor='black', width=0.6)

for bar in bars1:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., h + 70, f"{h:,.0f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax1.set_xticks(range(10))
ax1.set_xticklabels(labels, fontsize=8)
ax1.set_ylabel('Score / Recompensa Acumulada 3D', fontweight='bold')
ax1.set_title('Top 10 Campeões da Matriz Omniverso 3D (192 Combinações)', fontweight='bold', pad=10)
ax1.set_ylim(8500, 10700)

# -------------------------------------------------------------
# SUBPLOT 2: Comparativo Médio por Algoritmo e Representação
# -------------------------------------------------------------
algos = ["Continuous_PPO", "Soft_Actor_Critic_SAC", "Continuous_QRDQN", "MA_POCA_CTDE"]
reps = ["Vetor_Cinematico_MLP", "Grafo_GNN_3D", "Visao_Profundidade_3D"]

algo_names = ["Cont. PPO", "SAC", "Cont. QR-DQN", "MA-POCA"]
rep_names = ["Vetor (MLP)", "Grafo (GNN 3D)", "Visão Prof."]

# Calcular médias
algo_means = [np.mean([r["reward"] for r in ranking if r["algoritmo"] == a]) for a in algos]
rep_means = [np.mean([r["reward"] for r in ranking if r["representacao"] == rep]) for rep in reps]

x_alg = np.arange(len(algos))
x_rep = np.arange(len(reps)) + len(algos) + 0.8

b_alg = ax2.bar(x_alg, algo_means, color=['#94a3b8', '#0284c7', '#f59e0b', '#10b981'], edgecolor='black', width=0.55, label='Algoritmos')
b_rep = ax2.bar(x_rep, rep_means, color=['#64748b', '#7c3aed', '#ec4899'], edgecolor='black', width=0.55, label='Representações')

for bar in list(b_alg) + list(b_rep):
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 60, f"{h:,.0f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

all_x = list(x_alg) + list(x_rep)
all_labels = algo_names + rep_names
ax2.set_xticks(all_x)
ax2.set_xticklabels(all_labels, rotation=20, ha='right', fontweight='bold', fontsize=8)
ax2.set_ylabel('Recompensa Média Global', fontweight='bold')
ax2.set_title('Impacto Médio: Algoritmos de Controle vs Representações 3D', fontweight='bold', pad=10)
ax2.set_ylim(4500, 7800)
ax2.legend(loc='upper left', frameon=True)

# -------------------------------------------------------------
# SUBPLOT 3: Throughput de Simulação na RTX 4070 GPU (Steps/s)
# -------------------------------------------------------------
envs = ["Humanoid 3D\n(17 Juntas / Brax)", "Ant 3D\n(8 Juntas / Brax)", "HalfCheetah 3D\n(6 Juntas / Brax)", "Drones MARL 3D\n(Enxame Contínuo)"]
fps_values = [1400, 3600, 3800, 2500000]
colors_fps = ['#f97316', '#eab308', '#06b6d4', '#10b981']

# Escala logarítmica para comportar de 1.4K até 2.5M
bars3 = ax3.bar(envs, fps_values, color=colors_fps, edgecolor='black', width=0.5)
ax3.set_yscale('log')

for bar, val in zip(bars3, fps_values):
    if val >= 1000000:
        lbl = f"{val/1000000:.2f}M FPS"
    else:
        lbl = f"{val:,} FPS"
    ax3.text(bar.get_x() + bar.get_width()/2., val * 1.35, lbl, ha='center', va='bottom', fontsize=9, fontweight='bold')

ax3.set_ylabel('Throughput na GPU (Steps/s) [Escala Log]', fontweight='bold')
ax3.set_title('Throughput de Simulação 3D Direta na VRAM (NVIDIA RTX 4070)', fontweight='bold', pad=10)
ax3.set_ylim(500, 10000000)

fig.suptitle('Matriz Omniverso 3D em JAX: Avaliação Cartesiana de 192 Combinações (Google Brax & MARL 3D)', fontsize=15, fontweight='bold')

out_file = Path("figures/07_3d_benchmarks.png")
fig.savefig(out_file, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"[SUCESSO] Gráfico 07_3d_benchmarks.png atualizado com a matriz de 192 combinações: {out_file}")
