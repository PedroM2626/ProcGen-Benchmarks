"""Análise do benchmark Value vs Policy-based (seção 12): por jogo × algoritmo + famílias."""
import json
import numpy as np

d = json.load(open('results/algo_families_results.json', encoding='utf-8'))
rows = {}
for k, v in d.items():
    if 'error' in v:
        continue
    rows.setdefault(k.rsplit('_seed', 1)[0], []).append(v)

games = ['starpilot', 'dodgeball', 'bossfight']
algos = ['ppo', 'a2c', 'dqn', 'qrdqn']
fam = {'ppo': 'policy', 'a2c': 'policy', 'dqn': 'value', 'qrdqn': 'value'}

print(f"{'config':24s} {'stoch':>12s} {'det':>12s} {'train':>12s} {'gap':>7s}  família")
print('-' * 78)
summary = {}
for g in games:
    for a in algos:
        vs = rows[f'{g}_{a}']
        st = np.mean([v['stoch_unseen'] for v in vs]); ssd = np.std([v['stoch_unseen'] for v in vs])
        dt = np.mean([v['det_unseen'] for v in vs]); tr = np.mean([v['stoch_train'] for v in vs])
        gp = np.mean([v['gen_gap'] for v in vs])
        summary[f'{g}_{a}'] = dict(stoch=round(float(st), 2), std=round(float(ssd), 2),
                                   det=round(float(dt), 2), train=round(float(tr), 2),
                                   gap=round(float(gp), 2), n=len(vs))
        print(f"{g}_{a:18s} {st:5.2f}±{ssd:4.2f} {dt:5.2f}±{np.std([v['det_unseen'] for v in vs]):4.2f} "
              f"{tr:5.2f}±{np.std([v['stoch_train'] for v in vs]):4.2f} {gp:+7.2f}  {fam[a]}")
    print()

print('FAMÍLIAS (média dos 3 jogos):')
for f in ['policy', 'value']:
    vals = [summary[f'{g}_{a}']['stoch'] for g in games for a in algos if fam[a] == f]
    print(f"  {f:8s} {np.mean(vals):.2f}")
    for a in [x for x in algos if fam[x] == f]:
        per_game = ' / '.join(f"{summary[f'{g}_{a}']['stoch']:.2f}" for g in games)
        print(f"    {a:7s} (S/D/B): {per_game}")

json.dump(summary, open('results/algo_families_analysis.json', 'w'), indent=2)
print('\nSalvo: results/algo_families_analysis.json')
