"""
Braço 'hrl_learned' do benchmark HRL (seção 11): skills APRENDIDAS, sem macros hand-designed.

Hierarquia 2 níveis treinada em conjunto (PPO-lite custom, mesmo budget de 100k frames):
- Meta-controlador: NatureCNN(obs) -> MLP -> K skills latentes; decide a cada DUR=4 frames
- Low-level: NatureCNN(obs) + one-hot(z) -> MLP -> 15 ações primitivas; age por frame
- Ambos atualizam com PPO clip; especialização das skills é emergente (não há incentivo explícito de diversidade)
- gamma do meta = 0.99**DUR (consistente com o desconto por frame do low-level)

Mesmo protocolo de eval dos outros braços: 100 eps unseen stoch (meta amostra) + 100 det
(meta/low argmax) + 15 eps train. Escreve no MESMO results/hrl_results.json
(chaves {game}_hrl_learned_seed{s}) -> RODAR SEQUENCIALMENTE após compare_hrl.py
(ele carrega o JSON no início e sobrescreve a cada job).
"""
import os, json, argparse
import numpy as np, torch, torch.nn as nn
from torch.distributions import Categorical
from stable_baselines3.common.torch_layers import NatureCNN
from procgen_wrapper import make_procgen_env

DUR = 4
FRAMES = 100_000
ITER_FRAMES = 1024
K = 6
GAMMA = 0.99
LAM = 0.95
CLIP = 0.2
EPOCHS = 3
BATCH = 64
LR = 3e-4
ENT_COEF = 0.01
HORIZON = 256           # teto de frames por episódio (jumper tem episódios de 500+; truncamento com bootstrap, como SB3 TimeLimit)

class MetaNet(nn.Module):
    def __init__(self, obs_space, k):
        super().__init__()
        self.cnn = NatureCNN(obs_space)
        self.pi = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, k))
        self.vf = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))
    def forward(self, obs):
        f = self.cnn(obs)
        return self.pi(f), self.vf(f)

class LowNet(nn.Module):
    def __init__(self, obs_space, k, n_act):
        super().__init__()
        self.cnn = NatureCNN(obs_space)
        self.pi = nn.Sequential(nn.Linear(512 + k, 256), nn.ReLU(), nn.Linear(256, n_act))
        self.vf = nn.Sequential(nn.Linear(512 + k, 256), nn.ReLU(), nn.Linear(256, 1))
    def forward(self, obs, z_onehot):
        f = torch.cat([self.cnn(obs), z_onehot], dim=1)
        return self.pi(f), self.vf(f)

def to_t(obs, device):
    return torch.from_numpy(np.asarray(obs)).float().unsqueeze(0).to(device)

def obs_batch(arr, device):
    return torch.from_numpy(np.stack(arr)).float().to(device)

def collect(env, meta, low, device, n_frames):
    """Coleta ~n_frames primitivos; buffers meta (por macro) e low (por frame).
    Episódios são truncados em HORIZON frames com bootstrap de valor (padrão SB3)."""
    meta_b = {'obs': [], 'z': [], 'logp': [], 'val': [], 'R': [], 'done': [], 'next_val': []}
    low_b = {'obs': [], 'z': [], 'a': [], 'logp': [], 'val': [], 'r': [], 'done': [], 'next_val': []}
    frames = 0
    obs, _ = env.reset()
    ep_len = 0
    while frames < n_frames:
        obs_macro = obs
        with torch.no_grad():
            logits_m, val_m = meta(to_t(obs, device))
            dist_m = Categorical(logits=logits_m)
            z = dist_m.sample().item()
            lp_m = dist_m.log_prob(torch.tensor(z, device=device)).item()
        R, macro_done = 0.0, False
        zo = torch.zeros(1, K, device=device); zo[0, z] = 1.0
        for _ in range(DUR):
            with torch.no_grad():
                logits_l, val_l = low(to_t(obs, device), zo)
            dist_l = Categorical(logits=logits_l)
            a = dist_l.sample().item()
            lp_l = dist_l.log_prob(torch.tensor(a, device=device)).item()
            o2, r, term, trunc, _ = env.step(a)
            frames += 1; ep_len += 1; R += r
            hit_horizon = ep_len >= HORIZON
            if term:                    # episódio terminou de verdade
                next_val_l = 0.0; done_flag = True
                o2, _ = env.reset(); ep_len = 0
            elif hit_horizon:           # truncamento: bootstrap no estado pré-reset
                with torch.no_grad():
                    _, nv = low(to_t(o2, device), zo)
                next_val_l = nv.item(); done_flag = True
                o2, _ = env.reset(); ep_len = 0
            else:
                with torch.no_grad():
                    _, nv = low(to_t(o2, device), zo)
                next_val_l = nv.item(); done_flag = False
            low_b['obs'].append(obs); low_b['z'].append(z); low_b['a'].append(a)
            low_b['logp'].append(lp_l); low_b['val'].append(val_l.item()); low_b['r'].append(r)
            low_b['done'].append(done_flag); low_b['next_val'].append(next_val_l)
            obs = o2
            if done_flag:
                macro_done = True
                break
        with torch.no_grad():
            _, nv_m = meta(to_t(obs, device))
        meta_b['obs'].append(obs_macro)
        meta_b['z'].append(z); meta_b['logp'].append(lp_m); meta_b['val'].append(val_m.item())
        meta_b['R'].append(R); meta_b['done'].append(macro_done)
        meta_b['next_val'].append(0.0 if macro_done else nv_m.item())
    return meta_b, low_b, frames

def gae(rewards, vals, next_vals, dones, gamma):
    advs, last = [], 0.0
    vals_ext = vals + [next_vals[-1]]
    for t in reversed(range(len(rewards))):
        d = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * vals_ext[t + 1] * d - vals_ext[t]
        last = delta + gamma * LAM * d * last
        advs.insert(0, last)
    advs_t = torch.tensor(advs, dtype=torch.float32)
    rets_t = advs_t + torch.tensor(vals, dtype=torch.float32)
    return advs_t, rets_t

def ppo_loss(net, obs_t, extra, acts, old_logp, advs, rets):
    logits, vals = net(obs_t, extra) if extra is not None else net(obs_t)
    dist = Categorical(logits=logits)
    logp = dist.log_prob(acts)
    ratio = torch.exp(logp - old_logp)
    pi_loss = -torch.min(ratio * advs, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * advs).mean()
    vf_loss = ((vals.squeeze(1) - rets) ** 2).mean()
    return pi_loss + 0.5 * vf_loss - ENT_COEF * dist.entropy().mean()

def update(meta, low, opt_m, opt_l, meta_b, low_b, device):
    # meta
    advs_m, rets_m = gae(meta_b['R'], meta_b['val'], meta_b['next_val'], meta_b['done'], GAMMA ** DUR)
    advs_m = (advs_m - advs_m.mean()) / (advs_m.std() + 1e-8)
    om = obs_batch(meta_b['obs'], device).to(device)
    zm = torch.tensor(meta_b['z'], device=device)
    lpm = torch.tensor(meta_b['logp'], dtype=torch.float32, device=device)
    advs_m, rets_m = advs_m.to(device), rets_m.to(device)
    # low
    advs_l, rets_l = gae(low_b['r'], low_b['val'], low_b['next_val'], low_b['done'], GAMMA)
    advs_l = (advs_l - advs_l.mean()) / (advs_l.std() + 1e-8)
    ol = obs_batch(low_b['obs'], device).to(device)
    zl_oh = torch.zeros(len(low_b['z']), K, device=device); zl_oh[torch.arange(len(low_b['z'])), torch.tensor(low_b['z'])] = 1.0
    al = torch.tensor(low_b['a'], device=device)
    lpl = torch.tensor(low_b['logp'], dtype=torch.float32, device=device)
    advs_l, rets_l = advs_l.to(device), rets_l.to(device)
    for _ in range(EPOCHS):
        for st in torch.randperm(len(zm)).split(BATCH):
            opt_m.zero_grad()
            ppo_loss(meta, om[st], None, zm[st], lpm[st], advs_m[st], rets_m[st]).backward()
            nn.utils.clip_grad_norm_(meta.parameters(), 0.5)
            opt_m.step()
        for st in torch.randperm(len(al)).split(BATCH):
            opt_l.zero_grad()
            ppo_loss(low, ol[st], zl_oh[st], al[st], lpl[st], advs_l[st], rets_l[st]).backward()
            nn.utils.clip_grad_norm_(low.parameters(), 0.5)
            opt_l.step()

def eval_hierarchy(meta, low, game, seed, num_levels, n_eps, det):
    env = make_procgen_env(game, num_levels=num_levels, distribution_mode='easy',
                           seed=seed + 1000 if num_levels == 0 else seed, vector=False)
    device = next(meta.parameters()).device
    rews = []
    obs, _ = env.reset()
    while len(rews) < n_eps:
        ep_r = 0.0; done = False
        while not done:
            with torch.no_grad():
                logits_m, _ = meta(to_t(obs, device))
                z = int(logits_m.argmax().item()) if det else Categorical(logits=logits_m).sample().item()
            zo = torch.zeros(1, K, device=device); zo[0, z] = 1.0
            for _ in range(DUR):
                with torch.no_grad():
                    logits_l, _ = low(to_t(obs, device), zo)
                a = int(logits_l.argmax().item()) if det else Categorical(logits=logits_l).sample().item()
                obs, r, term, trunc, _ = env.step(a)
                ep_r += r
                if term or trunc:
                    done = True; rews.append(ep_r)
                    obs, _ = env.reset()
                    break
    env.close()
    return float(np.mean(rews))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', nargs='+', default=['jumper', 'plunder'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument('--device', default='auto')
    parser.add_argument('--out', default='results/hrl_results.json')
    args = parser.parse_args()
    device = 'cuda' if (args.device == 'auto' and torch.cuda.is_available()) else ('cpu' if args.device == 'auto' else args.device)

    base = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(base, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    zip_dir = os.path.join(base, 'logs_hrl', 'hrl_zips')
    os.makedirs(zip_dir, exist_ok=True)
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as f: results = json.load(f)

    jobs = [(g, s) for g in args.games for s in args.seeds]
    print(f"{len(jobs)} jobs ({args.games} × hrl_learned × {args.seeds}), device={device}, {FRAMES//1000}k frames")
    for i, (game, seed) in enumerate(jobs):
        name = f"{game}_hrl_learned_seed{seed}"
        if name in results and 'error' not in results[name]: continue
        print(f"\n[{i+1}/{len(jobs)}] {name}")
        try:
            torch.manual_seed(seed); np.random.seed(seed)
            env = make_procgen_env(game, num_levels=200, distribution_mode='easy', seed=seed, vector=False)
            obs_space = env.observation_space
            meta = MetaNet(obs_space, K).to(device)
            low = LowNet(obs_space, K, 15).to(device)
            opt_m = torch.optim.Adam(meta.parameters(), lr=LR)
            opt_l = torch.optim.Adam(low.parameters(), lr=LR)
            total_frames, it = 0, 0
            while total_frames < FRAMES:
                meta_b, low_b, fr = collect(env, meta, low, device, ITER_FRAMES)
                update(meta, low, opt_m, opt_l, meta_b, low_b, device)
                total_frames += fr; it += 1
                if it % 10 == 0:
                    print(f"  frames {total_frames}/{FRAMES}")
            env.close()
            m_st = eval_hierarchy(meta, low, game, seed, 0, 100, det=False)
            m_dt = eval_hierarchy(meta, low, game, seed, 0, 100, det=True)
            m_tr = eval_hierarchy(meta, low, game, seed, 200, 15, det=False)
            try:
                torch.save({'meta': meta.state_dict(), 'low': low.state_dict()},
                           os.path.join(zip_dir, f"{name}.pt"))
            except Exception as e: print(f"  save falhou: {e}")
            results[name] = {'stoch_unseen': round(m_st, 3), 'det_unseen': round(m_dt, 3),
                             'stoch_train': round(m_tr, 3), 'gen_gap': round(m_tr - m_st, 3),
                             'n_unseen': 100, 'n_train': 15, 'frames': FRAMES}
            print(f"  stoch={m_st:.2f} det={m_dt:.2f} train={m_tr:.2f} gap={m_tr-m_st:+.2f}")
        except Exception as e:
            import traceback; traceback.print_exc()
            results[name] = {'error': str(e)}
        with open(out_path, 'w') as f: json.dump(results, f, indent=2)
    print(f"Concluído: {out_path}")

if __name__ == '__main__':
    main()
