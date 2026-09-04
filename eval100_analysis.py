"""
Análise comparativa 30 vs 100 episódios (protocolo definitivo):
- Médias por config nos dois protocolos (mesmos modelos, mesmo seed de eval)
- Mudanças de ordenação por jogo (rank 30 vs rank 100)
- Ranking global da suite @100 vs @30
- Escreve results/eval100_analysis.json
"""
import json
import numpy as np

ev100 = json.load(open('results/eval100_results.json', encoding='utf-8'))
ev30_a = json.load(open('results/re_eval_results.json', encoding='utf-8'))   # 115 new_archs/maze_heist
ev30_b = json.load(open('results/retrain_results.json', encoding='utf-8'))   # 160 suite retrain
ev30 = {**{k: v for k, v in ev30_a.items() if 'error' not in v},
        **{k: v for k, v in ev30_b.items() if 'error' not in v}}

def aggregate(entries):
    rows = {}
    for k, v in entries.items():
        if 'error' in v:
            continue
        gk = k.rsplit('_seed', 1)[0]
        rows.setdefault(gk, []).append(v['stoch_unseen'])
    return {gk: float(np.mean(x)) for gk, x in rows.items()}

a30, a100 = aggregate(ev30), aggregate(ev100)

games = ['bossfight', 'starpilot', 'dodgeball', 'maze', 'heist']
print('=' * 80)
print('POR CONFIG: stoch@30 vs stoch@100 (delta = 100 - 30)')
print('=' * 80)
deltas = []
for g in games:
    keys = sorted(k for k in a100 if k.startswith(g + '_'))
    print(f'\n{g.upper()}')
    print(f"  {'config':34s} {'@30':>6s} {'@100':>6s} {'delta':>7s}")
    for k in keys:
        v30, v100 = a30.get(k, float('nan')), a100[k]
        deltas.append((k, v30, v100, v100 - v30))
        print(f"  {k:34s} {v30:6.2f} {v100:6.2f} {v100-v30:+7.2f}")

dabs = [abs(d) for _, _, _, d in deltas]
print(f"\n|delta| médio: {np.mean(dabs):.3f} | máximo: {max(dabs):.2f} "
      f"({max(deltas, key=lambda x: abs(x[3]))[0]})")

# ordenação por jogo: muda algo?
print('\n' + '=' * 80)
print('ORDENAÇÃO POR JOGO — rank@30 vs rank@100')
print('=' * 80)
rank_changes = {}
for g in games:
    keys = [k for k in a100 if k.startswith(g + '_')]
    r30 = {k: i for i, k in enumerate(sorted(keys, key=lambda x: -a30.get(x, -9)), 1)}
    r100 = {k: i for i, k in enumerate(sorted(keys, key=lambda x: -a100[x]), 1)}
    moved = {k: (r30[k], r100[k]) for k in keys if r30[k] != r100[k]}
    rank_changes[g] = moved
    print(f"\n{g.upper()}: {'SEM mudanças de posição' if not moved else ''}")
    top3_100 = sorted(keys, key=lambda x: -a100[x])[:3]
    print(f"  top-3 @100: {', '.join(f'{k} {a100[k]:.2f}' for k in top3_100)}")
    for k, (p0, p1) in sorted(moved.items(), key=lambda kv: kv[1][1]):
        print(f"  {k:34s} {p0}º -> {p1}º   ({a30[k]:.2f} -> {a100[k]:.2f})")

# ranking global da suite @100 vs @30 (média 3 jogos, configs cnn_/wm_/aug_)
print('\n' + '=' * 80)
print('RANKING GLOBAL SUITE @100 vs @30')
print('=' * 80)
def suite_global(agg):
    archs = set(k.split('_', 1)[1] for g in games[:3] for k in agg if k.startswith(g + '_'))
    out = {}
    for a in archs:
        vals = [agg.get(f'{g}_{a}') for g in games[:3]]
        if all(v is not None for v in vals):
            out[a] = float(np.mean(vals))
    return out

g30, g100 = suite_global(a30), suite_global(a100)
print(f"{'arquitetura':20s} {'@100':>6s} {'@30':>6s} {'delta':>7s}")
for a in sorted(g100, key=lambda x: -g100[x]):
    print(f"  {a:18s} {g100[a]:6.2f} {g30.get(a, float('nan')):6.2f} {g100[a]-g30.get(a, float('nan')):+7.2f}")

json.dump({'deltas': {k: {'v30': round(v30, 3), 'v100': round(v100, 3)} for k, v30, v100, _ in deltas},
           'rank_changes': rank_changes, 'global100': {k: round(v, 3) for k, v in g100.items()},
           'global30': {k: round(v, 3) for k, v in g30.items()}},
          open('results/eval100_analysis.json', 'w'), indent=2, default=str)
print('\nSalvo: results/eval100_analysis.json')
