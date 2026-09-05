"""Analise final da grade: rankings + stats vs conclusoes do estudo (§§3-12).

Le jax_port/results_grade/*/*.json (eval_unseen 100 eps), agrupa por
(cfg, jogo) nas seeds, aplica stats.rank_cells e compara com as
ordenacoes publicadas. Saida: jax_port/analysis_full.json + tabelas.
Uso: python jax_port/analyze_grade.py
"""

import glob
import json
import os

from jax_port.stats import auc_norm, rank_cells

GRADE = "jax_port/results_grade"
OUT = "jax_port/analysis_full.json"


def load_cells(suite):
    cells = []
    for f in sorted(glob.glob(os.path.join(GRADE, suite, "*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        ev = (d.get("eval_unseen") or {}).get("mean")
        if ev is None:
            continue
        # cfg vem do nome do arquivo (gêmeos ae/recon/classic partilham
        # extrator mas têm seeds/draws independentes por célula).
        d["_cfg"] = os.path.basename(f).split("__")[0]
        cells.append(d)
    return cells


def group(cells):
    g = {}
    for d in cells:
        g.setdefault((d["_cfg"], d["game"]), []).append(
            d["eval_unseen"]["mean"])
    return g


def group_seeds(cells):
    g = {}
    for d in cells:
        g.setdefault((d["_cfg"], d["game"]),
                     {})[d["seed"]] = d["eval_unseen"]["mean"]
    return g


def main():
    rep = {}
    # main: por jogo + global (media das medias, como §3.7/3.12)
    g = group(load_cells("main"))
    games = sorted(set(game for _, game in g))
    rep["main"] = {game: rank_cells(
        {cfg: vals for (cfg, gm), vals in g.items() if gm == game})
        for game in games}
    cfgs = sorted(set(cfg for cfg, _ in g))
    gs = group_seeds(load_cells("main"))
    seeds = sorted(set(s for v in gs.values() for s in v))
    rep["main"]["global"] = rank_cells({
        cfg: [sum(gs[(cfg, gm)][s] for gm in games) / len(games) for s in seeds]
        for cfg in cfgs if all(s in gs.get((cfg, gm), {}) for gm in games
                               for s in seeds)})
    for suite in ("exploration", "algo", "hrl", "hard", "pilot", "spr",
                  "gnn", "aux"):
        gg = group(load_cells(suite))
        games = sorted(set(game for _, game in gg))
        rep[suite] = {game: rank_cells(
            {cfg: vals for (cfg, gm), vals in gg.items() if gm == game})
            for game in games}
    # budget: curvas por (cfg, jogo)
    curves = {}
    for d in load_cells("budget"):
        key = (d["_cfg"], d["game"])
        curves.setdefault(key, []).append((d["timesteps"], d["eval_unseen"]["mean"]))
    rep["budget"] = {f"{c}/{g}": sorted(v) for (c, g), v in curves.items()}
    # AUC por celula com curva
    n_auc = 0
    for suite in ("main", "exploration", "algo", "hrl", "budget", "hard",
                  "pilot", "spr", "gnn", "aux"):
        for f in glob.glob(os.path.join(GRADE, suite, "*.json")):
            d = json.load(open(f))
            if d.get("curve"):
                n_auc += 1
    rep["meta"] = {"cells_with_curve": n_auc}
    with open(OUT, "w") as fh:
        json.dump(rep, fh, indent=1)
    # tabelas resumidas
    for suite in ("main", "exploration", "algo", "hrl"):
        print(f"== {suite} ==")
        for game, r in rep[suite].items():
            if game == "global":
                continue
            top = [(x["cell"], round(x["mean"], 2)) for x in r["ranking"][:4]]
            print(f"  {game}: {top}")
    g = rep["main"]["global"]["ranking"]
    print("GLOBAL:", [(x["cell"], round(x["mean"], 2)) for x in g[:6]])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
