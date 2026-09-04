"""
Análise completa do retreino da suite (protocolo novo) + merge com new_archs/maze_heist:
- Tabela por jogo/config (stoch, det, gap, n)
- Ranking global novo (média dos 3 jogos da suite) vs ranking antigo (seção 3.7)
- Escreve results/retrain_analysis.json
"""
import json
import numpy as np

base = ''
retr = json.load(open('results/retrain_results.json', encoding='utf-8'))
reev = json.load(open('results/re_eval_results.json', encoding='utf-8'))

def aggregate(entries, prefix_filter=None):
    rows = {}
    for k, v in entries.items():
        if 'error' in v:
            continue
        if prefix_filter and not k.startswith(prefix_filter):
            continue
        gk = k.rsplit('_seed', 1)[0]
        rows.setdefault(gk, []).append(v)
    out = {}
    for gk, vs in rows.items():
        out[gk] = {
            'stoch': round(float(np.mean([v['stoch_unseen'] for v in vs])), 2),
            'std': round(float(np.std([v['stoch_unseen'] for v in vs])), 2),
            'det': round(float(np.mean([v['det_unseen'] for v in vs])), 2),
            'gap': round(float(np.mean([v['gen_gap'] for v in vs])), 2),
            'n': len(vs),
        }
    return out

games = ['bossfight', 'starpilot', 'dodgeball']
suite = {g: aggregate(retr, g) for g in games}

print('=' * 78)
print('SUITE RETRAIN — por jogo (protocolo novo: 30 eps stoch/det + gap)')
print('=' * 78)
for g in games:
    print(f'\n{g.upper()}')
    for gk in sorted(suite[g], key=lambda x: -suite[g][x]['stoch']):
        r = suite[g][gk]
        print(f"  {gk:32s} stoch={r['stoch']:5.2f}±{r['std']:.2f}  det={r['det']:5.2f}  gap={r['gap']:+.2f}  n={r['n']}")

# ranking global NOVO: arquitetura = média simples dos 3 jogos (mesma regra da seção 3.7)
def arch_of(gk):
    # bossfight_cnn_classic -> cnn_classic ; bossfight_wm_vae -> wm_vae
    return gk.split('_', 1)[1]

glob_new = {}
for a in set(arch_of(gk) for g in games for gk in suite[g]):
    vals = [suite[g][f'{g}_{a}']['stoch'] for g in games if f'{g}_{a}' in suite[g]]
    if len(vals) == 3:
        glob_new[a] = round(float(np.mean(vals)), 2)

old_global = {  # seção 3.7 do README (protocolo antigo)
    'spatial': 1.54, 'resnet18': 1.34, 'classic': 1.33, 'cbam': 1.33,
    'mlp_vector': 1.25, 'lstm_attention': 1.20, 'vit': 1.20, 'aug_crop': 1.16,
    'impoola': 1.12, 'ae': 1.11,
}
print('\n' + '=' * 78)
print('RANKING GLOBAL NOVO (suite retrain, média 3 jogos) vs ANTIGO (seção 3.7)')
print('=' * 78)
print(f"{'arquitetura':20s} {'novo':>6s} {'antigo':>7s} {'delta':>7s}")
for a in sorted(glob_new, key=lambda x: -glob_new[x]):
    old = old_global.get(a, None)
    print(f"  {a:18s} {glob_new[a]:6.2f} {old if old is not None else '—':>7} "
          f"{glob_new[a]-old:+.2f}" if old is not None else f"  {a:18s} {glob_new[a]:6.2f}       —       —")

# merge geral: retrain + new_archs + maze_heist num só dicionário (para scorecard)
merged = {}
merged.update(reev)
merged.update({k: v for k, v in retr.items() if 'error' not in v})
ok_total = len(merged)
print(f"\nMERGE: {ok_total} modelos avaliados no protocolo novo "
      f"({len([v for v in reev.values() if 'error' not in v])} re-eval + "
      f"{len([v for v in retr.values() if 'error' not in v])} retrain)")

# gen gap: quem memoriza (gap alto) vs quem generaliza
print('\nGEN GAP — top 5 maiores (memorização relativa) e top 5 menores')
allg = aggregate(merged)
by_gap = sorted(allg.items(), key=lambda kv: -kv[1]['gap'])
for gk, r in by_gap[:5]:
    print(f"  {gk:34s} gap={r['gap']:+.2f} stoch={r['stoch']:.2f}")
print('  ...')
for gk, r in by_gap[-5:]:
    print(f"  {gk:34s} gap={r['gap']:+.2f} stoch={r['stoch']:.2f}")

json.dump({'suite_by_game': suite, 'global_new': glob_new, 'global_old': old_global},
          open('results/retrain_analysis.json', 'w'), indent=2)
print('\nSalvo: results/retrain_analysis.json')
