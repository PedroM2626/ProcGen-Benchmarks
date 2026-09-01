"""
Sondagem do espaço de ações (15 discretas) de jumper/plunder:
para cada ação, aplica por alguns frames e mede variação de pixels e reward —
identifica quais botões movem/pulam/atiram, base para a biblioteca de skills do HRL.
"""
import numpy as np
from procgen_wrapper import make_procgen_env

def probe(game, seeds=(42, 43, 44, 45, 46), frames=16):
    print(f'\n=== {game} (média de {len(seeds)} níveis, {frames} frames/ação, env NOVO por ação) ===')
    print(f"{'act':>4s} {'pixdiff':>10s} {'reward':>8s} {'done%':>7s} {'picoframe':>10s}")
    agg = {}
    for a in range(15):
        for seed in seeds:
            env = make_procgen_env(game, num_levels=1, distribution_mode='easy', seed=seed, vector=False)
            obs0, _ = env.reset()
            total_r, diff, done = 0.0, 0.0, False
            peak = 0.0
            prev = obs0.astype(np.float32)
            for t in range(frames):
                o, r, term, trunc, _ = env.step(a)
                d = float(np.abs(o.astype(np.float32) - prev).mean())
                diff += d
                peak = max(peak, d)
                prev = o.astype(np.float32)
                total_r += r
                if term or trunc:
                    done = True
                    break
            acc = agg.setdefault(a, [0.0, 0.0, 0.0, 0.0])
            acc[0] += diff; acc[1] += total_r; acc[2] += done; acc[3] += peak
            env.close()
    for a in range(15):
        d, r, dn, pk = agg[a]
        n = len(seeds)
        print(f"{a:4d} {d/n:10.2f} {r/n:8.2f} {100*dn/n:6.0f}% {pk/n:10.2f}")

for g in ['jumper', 'plunder']:
    try:
        probe(g)
    except Exception as e:
        print(f'{g}: ERRO {e}')
