"""Estatistica do porte — paridade com ``scorecard_analysis.py`` (§3.8) e AUC (§3.9).

IC 95% via t de Student (n=5 -> t=2.776); Cohen's d top1 vs top2;
AUC_norm = trapezio(retorno, steps)/budget (eixo sample-efficiency).
Sem scipy (nao instalado no venv do porte): tabela t embutida df<=30.
"""

import math

_T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
      7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086,
      30: 2.042}


def _t(df):
    if df in _T:
        return _T[df]
    ks = sorted(_T)
    return _T[ks[-1]] if df > ks[-1] else _T[min(k for k in ks if k >= df)]


def mean_ci(vals):
    """(mean, std, n, (lo, hi)) — std amostral (ddof=1)."""
    import numpy as np
    v = np.asarray(vals, float)
    n = len(v)
    m, s = float(v.mean()), float(v.std(ddof=1)) if n > 1 else 0.0
    h = _t(n - 1) * s / math.sqrt(n) if n > 1 else 0.0
    return m, s, n, (m - h, m + h)


def cohen_d(a, b):
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                   / max(1, na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def auc_norm(curve, budget):
    """curva [(steps, ret)] -> integral/budget (estudo §3.9)."""
    import numpy as np
    if not curve:
        return 0.0
    xs = np.array([p["steps"] for p in curve], float)
    ys = np.array([p.get("ret20", p.get("ret", 0.0)) for p in curve], float)
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapz(ys, xs) / budget)


def rank_cells(cell_means):
    """cell->lista de means por seed  =>  ranking + stats top1 vs top2."""
    rows = []
    for cell, vals in cell_means.items():
        m, s, n, (lo, hi) = mean_ci(vals)
        rows.append({"cell": cell, "mean": m, "std": s, "n": n,
                     "ci95": [lo, hi]})
    rows.sort(key=lambda r: -r["mean"])
    out = {"ranking": rows}
    if len(rows) >= 2:
        a = cell_means[rows[0]["cell"]]
        b = cell_means[rows[1]["cell"]]
        ma, _, _, _ = mean_ci(a)
        mb, _, _, _ = mean_ci(b)
        out["top1_vs_top2"] = {
            "top1": rows[0]["cell"], "top2": rows[1]["cell"],
            "diff": ma - mb, "cohen_d": cohen_d(a, b),
            "overlap": not (rows[0]["ci95"][0] > rows[1]["ci95"][1]
                            or rows[1]["ci95"][0] > rows[0]["ci95"][1])}
    return out
