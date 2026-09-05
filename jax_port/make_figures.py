"""Figuras JAX-vs-SB3 a partir de analysis_full.json + JSONs pareados.

Sem GPU (matplotlib CPU). Saida: jax_port/figures/*.png
  01_global_ci.png      top-8 global com IC95 (n=5)
  02_speedup.png        SPS pareado A/B/C + eval-full
  03_budget.png         curvas 100k/250k/500k (mlp vs resnet18)
  04_hrl.png            4 bracos x jumper/plunder
  05_algo.png           policy vs value por jogo
  06_top10.png          cluster top-5 com n=10
Uso: python -m jax_port.make_figures  (raiz do repo, qualquer python+mpl)
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRADE = "jax_port/results_grade"
FIG = "jax_port/figures"
A = json.load(open("jax_port/analysis_full.json"))


def errbar(ax, labels, means, los, his, title, ylabel):
    import numpy as np
    y = np.arange(len(labels))
    ax.errorbar(means, y, xerr=[np.array(means) - np.array(los),
                                np.array(his) - np.array(means)],
                fmt="o", capsize=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel(ylabel)
    ax.grid(axis="x", alpha=0.3)


def fig_global():
    r = A["main"]["global"]["ranking"][:8]
    fig, ax = plt.subplots(figsize=(8, 5))
    errbar(ax, [x["cell"] for x in r], [x["mean"] for x in r],
           [x["ci95"][0] for x in r], [x["ci95"][1] for x in r],
           "Global top-8 (JAX, 3 games, 5 seeds, eval 100 eps)", "mean ± IC95")
    fig.tight_layout()
    fig.savefig(f"{FIG}/01_global_ci.png", dpi=120)
    plt.close(fig)


def fig_speedup():
    a = json.load(open("jax_port/paired_sb3_n1.json"))["sps"]
    b = json.load(open("jax_port/paired_sb3_n64.json"))["sps"]
    c = json.load(open("jax_port/pa2_coinrun_100k_rerun.json"))["sps"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["SB3 n=1\n(estudo)", "SB3 n=64", "JAX n=64"], [a, b, c])
    for i, v in enumerate((a, b, c)):
        ax.text(i, v + 80, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_title("SPS de treino pareado (coinrun 100k, mesma RTX 4070)")
    ax.set_ylabel("steps/s")
    fig.tight_layout()
    fig.savefig(f"{FIG}/02_speedup.png", dpi=120)
    plt.close(fig)


def fig_budget():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, game in zip(axes, ("starpilot", "dodgeball")):
        for cfg in ("mlp", "resnet18"):
            pts = A["budget"].get(f"{cfg}/{game}", [])
            if pts:
                ax.plot([t for t, _ in pts], [v for _, v in pts],
                        marker="o", label=cfg)
        ax.set_title(game)
        ax.set_xlabel("steps")
        ax.legend()
    axes[0].set_ylabel("eval unseen")
    fig.suptitle("Budget scaling JAX (100k/250k/500k)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/03_budget.png", dpi=120)
    plt.close(fig)


def fig_hrl():
    games = ("jumper", "plunder")
    arms = ("flat", "skip4", "hrl", "hrl_learned")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, game in zip(axes, games):
        means = []
        for arm in arms:
            r = [x for x in A["hrl"][game]["ranking"] if x["cell"] == arm]
            means.append(r[0]["mean"] if r else 0)
        ax.bar(arms, means)
        ax.set_title(game)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    axes[0].set_ylabel("eval unseen")
    fig.suptitle("HRL JAX (100k frames)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/04_hrl.png", dpi=120)
    plt.close(fig)


def fig_algo():
    games = ("starpilot", "dodgeball", "bossfight")
    algs = ("ppo", "a2c", "dqn", "qrdqn")
    x = range(len(games))
    fig, ax = plt.subplots(figsize=(8, 4))
    for alg in algs:
        vals = []
        for game in games:
            r = [t for t in A["algo"][game]["ranking"] if t["cell"] == alg]
            vals.append(r[0]["mean"] if r else 0)
        ax.bar([i + 0.15 * (algs.index(alg) - 1.5) for i in x], vals,
               width=0.15, label=alg)
    ax.set_xticks(list(x))
    ax.set_xticklabels(games)
    ax.legend()
    ax.set_title("Algo families JAX (policy vs value)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/05_algo.png", dpi=120)
    plt.close(fig)


def fig_top10():
    """Cluster top-5 com n=10 (grade + follow-up top10)."""
    import glob
    import numpy as np
    vals = {}
    pats = [os.path.join(GRADE, "main", "*.json"),
            os.path.join("jax_port", "results_followup", "top10", "main",
                         "*.json")]
    for pat in pats:
        for f in glob.glob(pat):
            parts = os.path.basename(f)[:-5].split("__")
            if len(parts) != 4:
                continue
            cfg, game, seed = parts[0], parts[1], parts[2]
            if cfg not in ("aug_noise", "contrastive", "impoola", "mlp",
                           "spatial"):
                continue
            if game not in ("bossfight", "starpilot", "dodgeball"):
                continue
            vals.setdefault((cfg, game), []).append(
                json.load(open(f))["eval_unseen"]["mean"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    for ax, game in zip(axes, ("bossfight", "starpilot", "dodgeball")):
        labs = sorted(vals, key=lambda k: (k[1] != game, k[0]))
        labs = [c for c, g in labs if g == game]
        ms = [float(np.mean(vals[(c, game)])) for c in labs]
        sds = [float(np.std(vals[(c, game)], ddof=1)) for c in labs]
        ax.bar(labs, ms, yerr=np.array(sds) / np.sqrt(10) * 2.262,
               capsize=3)
        ax.set_title(f"{game} (n=10 ± IC95)")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.suptitle("Top-5 cluster at n=10: nobody separates")
    fig.tight_layout()
    fig.savefig(f"{FIG}/06_top10.png", dpi=120)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    fig_global()
    fig_speedup()
    fig_budget()
    fig_hrl()
    fig_algo()
    fig_top10()
    print("figs 01-06 OK ->", FIG)


if __name__ == "__main__":
    main()
