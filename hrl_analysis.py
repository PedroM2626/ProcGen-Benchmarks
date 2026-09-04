"""Análise final do benchmark HRL (seção 11): 4 braços × 2 jogos × 5 seeds."""
import json
import numpy as np

d = json.load(open('results/hrl_results.json', encoding='utf-8'))
rows = {}
for k, v in d.items():
    if 'error' in v:
        continue
    rows.setdefault(k.rsplit('_seed', 1)[0], []).append(v)

hdr = f"{'config':24s} {'stoch':>13s} {'det':>13s} {'train':>13s} {'gap':>7s}"
print(hdr); print('-' * len(hdr))
summary = {}
for k in sorted(rows):
    vs = rows[k]
    st = [v['stoch_unseen'] for v in vs]; dt = [v['det_unseen'] for v in vs]
    tr = [v['stoch_train'] for v in vs]; gp = [v['gen_gap'] for v in vs]
    summary[k] = dict(stoch=round(np.mean(st), 2), std=round(np.std(st), 2),
                      det=round(np.mean(dt), 2), train=round(np.mean(tr), 2),
                      gap=round(np.mean(gp), 2), n=len(vs))
    print(f"{k:24s} {np.mean(st):5.2f}+/-{np.std(st):4.2f} "
          f"{np.mean(dt):5.2f}+/-{np.std(dt):4.2f} "
          f"{np.mean(tr):5.2f}+/-{np.std(tr):4.2f} {np.mean(gp):+7.2f}")

# deltas por jogo: hrl_learned vs os demais
print('\nDeltas (stoch, média 5 seeds):')
for game in ['jumper', 'plunder']:
    get = lambda arm: summary[f'{game}_{arm}']['stoch']
    f, s4, h, hl = get('flat'), get('skip4'), get('hrl'), get('hrl_learned')
    print(f"  {game:8s} flat={f:.2f} skip4={s4:.2f} hrl={h:.2f} hrl_learned={hl:.2f} "
          f"| learned-skip4={hl-s4:+.2f} learned-flat={hl-f:+.2f}")

json.dump(summary, open('results/hrl_analysis.json', 'w'), indent=2)
print('\nSalvo: results/hrl_analysis.json')
