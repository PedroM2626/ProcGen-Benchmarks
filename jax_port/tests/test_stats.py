"""Sanidade de stats.py com valores conhecidos (sem GPU).

mean_ci([1..5]) = 3 ± t4*std/sqrt5; cohen_d = -3; auc triangulo = 1.0.
Uso: python -m jax_port.tests.test_stats
"""

import math


def test_stats():
    from jax_port.stats import auc_norm, cohen_d, mean_ci, rank_cells
    m, s, n, (lo, hi) = mean_ci([1, 2, 3, 4, 5])
    assert m == 3.0 and n == 5
    assert abs(s - math.sqrt(2.5)) < 1e-9
    assert abs((hi - lo) / 2 - 2.776 * math.sqrt(2.5) / math.sqrt(5)) < 1e-9
    assert abs(cohen_d([1, 2, 3], [4, 5, 6]) + 3.0) < 1e-9
    assert abs(auc_norm([{"steps": 0, "ret20": 0.0},
                         {"steps": 100, "ret20": 2.0}], 100) - 1.0) < 1e-9
    r = rank_cells({"a": [1.0, 1.1, 0.9, 1.0, 1.0],
                    "b": [0.0, 0.1, -0.1, 0.0, 0.05]})
    assert r["ranking"][0]["cell"] == "a"
    assert r["top1_vs_top2"]["cohen_d"] > 5  # separacao obvia
    return True


if __name__ == "__main__":
    test_stats()
    print("STATS_OK")
