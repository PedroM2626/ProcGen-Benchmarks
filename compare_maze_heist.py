"""
Maze+Heist PPO vs ICM vs RND vs NGU — 2 jogos ×4 configs ×5 seeds ×100k
ICM: forward+inverse, RND: random target, NGU: RND+episodic
"""
import os, json, argparse, collections
from datetime import datetime
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import gymnasium as gymn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from procgen_wrapper import make_procgen_env
from models.sb3_extractors import ClassicCNNExtractor

class ICMWrapper(gymn.Wrapper):
    def __init__(self, env, beta=0.01):
        super().__init__(env)
        self.beta=beta
        # phi: same CNN as Classic
        self.phi = nn.Sequential(nn.Conv2d(3,32,8,stride=4), nn.ReLU(), nn.Conv2d(32,64,4,stride=2), nn.ReLU(), nn.Conv2d(64,64,3,stride=1), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            dummy=torch.zeros(1,3,64,64); n_flat=self.phi(dummy).shape[1]
        self.forward_model = nn.Sequential(nn.Linear(n_flat+15, 512), nn.ReLU(), nn.Linear(512, n_flat))
        self.inverse_model = nn.Sequential(nn.Linear(n_flat*2, 512), nn.ReLU(), nn.Linear(512, 15))
        self.opt = torch.optim.Adam(list(self.phi.parameters())+list(self.forward_model.parameters())+list(self.inverse_model.parameters()), lr=1e-4)
        self.prev_phi=None
    def reset(self, **kw):
        obs,_=self.env.reset(**kw)
        chw=np.transpose(obs,(2,0,1)) if obs.shape==(64,64,3) else obs
        # procgen wrapper already CHW, so handle
        if isinstance(obs, np.ndarray) and obs.shape==(3,64,64):
            chw=obs
        elif isinstance(obs, np.ndarray) and obs.shape==(64,64,3):
            chw=np.transpose(obs,(2,0,1))
        else:
            chw=obs
        with torch.no_grad():
            phi=self.phi(torch.from_numpy(chw).unsqueeze(0).float()/255.0)
        self.prev_phi=phi
        return obs,{}
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # intrinsic via forward error (simplified, train every step)
        try:
            # phi(s)
            # need to handle obs CHW vs HWC
            if isinstance(obs, np.ndarray) and obs.shape==(3,64,64):
                chw=obs
            elif isinstance(obs, np.ndarray) and obs.shape==(64,64,3):
                chw=np.transpose(obs,(2,0,1))
            else:
                chw=obs
            phi_next=self.phi(torch.from_numpy(chw).unsqueeze(0).float()/255.0)
            a_onehot=F.one_hot(torch.tensor([action]), num_classes=15).float()
            pred_phi=self.forward_model(torch.cat([self.prev_phi, a_onehot], dim=1))
            loss_forward=F.mse_loss(pred_phi, phi_next)
            intrinsic=loss_forward.item()
            # update
            self.opt.zero_grad(); loss_forward.backward(); self.opt.step()
            reward = reward + self.beta*intrinsic
            self.prev_phi=phi_next.detach()
        except Exception:
            pass
        return obs, reward, terminated, truncated, info

class RNDWrapper(gymn.Wrapper):
    def __init__(self, env, beta=0.01):
        super().__init__(env); self.beta=beta
        self.target=nn.Sequential(nn.Conv2d(3,32,8,stride=4), nn.ReLU(), nn.Conv2d(32,64,4,stride=2), nn.ReLU(), nn.Flatten(), nn.Linear(1024,512))
        self.predictor=nn.Sequential(nn.Conv2d(3,32,8,stride=4), nn.ReLU(), nn.Conv2d(32,64,4,stride=2), nn.ReLU(), nn.Flatten(), nn.Linear(1024,512))
        for p in self.target.parameters(): p.requires_grad=False
        self.opt=torch.optim.Adam(self.predictor.parameters(), lr=1e-4)
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        try:
            if isinstance(obs, np.ndarray) and obs.shape==(3,64,64):
                chw=obs
            elif isinstance(obs, np.ndarray) and obs.shape==(64,64,3):
                chw=np.transpose(obs,(2,0,1))
            else:
                chw=obs
            x=torch.from_numpy(chw).unsqueeze(0).float()/255.0
            with torch.no_grad(): t=self.target(x)
            p=self.predictor(x)
            loss=F.mse_loss(p,t)
            intrinsic=loss.item()
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            reward = reward + self.beta*intrinsic
        except Exception:
            pass
        return obs, reward, terminated, truncated, info

class NGUWrapper(RNDWrapper):
    def __init__(self, env, beta=0.01):
        super().__init__(env, beta)
        self.memory=collections.deque(maxlen=1000)
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        try:
            if isinstance(obs, np.ndarray) and obs.shape==(3,64,64):
                chw=obs
            elif isinstance(obs, np.ndarray) and obs.shape==(64,64,3):
                chw=np.transpose(obs,(2,0,1))
            else:
                chw=obs
            x=torch.from_numpy(chw).unsqueeze(0).float()/255.0
            with torch.no_grad(): t=self.target(x)
            p=self.predictor(x)
            rnd_loss=F.mse_loss(p,t).item()
            # episodic novelty: distance to kNN in memory (simplified)
            phi=p.detach().cpu().numpy().flatten()
            if len(self.memory)>10:
                dists=[np.linalg.norm(phi - m) for m in list(self.memory)[-100:]]
                episodic=np.mean(sorted(dists)[:5])
            else:
                episodic=1.0
            self.memory.append(phi)
            loss=F.mse_loss(p,t)
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            reward = reward + self.beta*(rnd_loss * episodic)
        except Exception:
            pass
        return obs, reward, terminated, truncated, info

def train_one(game, wrapper, timesteps, seed, log_dir, device):
    def make_env():
        env=make_procgen_env(game, num_levels=200, distribution_mode='easy', seed=seed, vector=False)
        if wrapper=='icm': env=ICMWrapper(env)
        elif wrapper=='rnd': env=RNDWrapper(env)
        elif wrapper=='ngu': env=NGUWrapper(env)
        return Monitor(env)
    vec=DummyVecEnv([make_env])
    eval_env=DummyVecEnv([lambda: Monitor(make_procgen_env(game, num_levels=0, distribution_mode='easy', seed=seed+1000, vector=False))])
    model=PPO("CnnPolicy", vec, verbose=0, learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=3, gamma=0.99, gae_lambda=0.95, clip_range=0.2, seed=seed, device=device, policy_kwargs={"features_extractor_class": ClassicCNNExtractor, "features_extractor_kwargs": dict(features_dim=512)}, tensorboard_log=log_dir)
    model.learn(total_timesteps=timesteps)
    mean,std=evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=False)
    vec.close(); eval_env.close()
    return float(mean), float(std), model

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42,43,44,45,46])
    parser.add_argument('--games', type=str, nargs='+', default=['maze','heist'])
    parser.add_argument('--log_dir', type=str, default='./logs_maze_heist')
    parser.add_argument('--device', type=str, default='auto')
    args=parser.parse_args()
    device='cuda' if (args.device=='auto' and torch.cuda.is_available()) else args.device if args.device!='auto' else 'cpu'
    print(f"Device: {device} torch {torch.__version__} MAZE+HEIST ICM/RND/NGU")
    os.makedirs(args.log_dir, exist_ok=True)
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    comp_dir=os.path.join(args.log_dir, f"maze_heist_{'_'.join(args.games)}_{ts}")
    os.makedirs(comp_dir, exist_ok=True)
    wrappers=['ppo','icm','rnd','ngu']
    results={}
    for game in args.games:
        for w in wrappers:
            gk=f"{game}_{w}"
            results[gk]=[]
            for seed in args.seeds:
                print(f"\n{'='*60}\n{gk} seed {seed} {game}\n{'='*60}")
                try:
                    mean,std,model=train_one(game, w if w!='ppo' else None, args.timesteps, seed, args.log_dir, device)
                    print(f"{gk} seed {seed}: {mean:.2f} +/- {std:.2f}")
                    results[gk].append({'seed': seed, 'mean_reward': mean, 'std_reward': std})
                    try: model.save(os.path.join(comp_dir, f"{gk}_seed{seed}.zip"))
                    except: pass
                except Exception as e:
                    import traceback; traceback.print_exc()
                    results[gk].append({'seed': seed, 'mean_reward': None, 'error': str(e)})
    with open(os.path.join(comp_dir,'comparison_results.json'),'w') as f: json.dump(results,f,indent=2)
    stats={}
    for k,v in results.items():
        rewards=[x['mean_reward'] for x in v if x['mean_reward'] is not None]
        stats[k]={'mean':float(np.mean(rewards)),'std':float(np.std(rewards)),'min':float(np.min(rewards)),'max':float(np.max(rewards)),'n':len(rewards)} if rewards else None
    with open(os.path.join(comp_dir,'statistics.json'),'w') as f: json.dump(stats,f,indent=2)
    try:
        import matplotlib.pyplot as plt
        for game in args.games:
            keys=[k for k in stats if k.startswith(game+'_')]
            means=[stats[k]['mean'] if stats[k] else 0 for k in keys]
            stds=[stats[k]['std'] if stats[k] else 0 for k in keys]
            plt.figure(figsize=(10,6)); plt.bar(keys, means, yerr=stds, capsize=4, alpha=0.8)
            plt.xticks(rotation=20, ha='right'); plt.ylabel('Mean Reward'); plt.title(f"{game} - PPO vs ICM/RND/NGU - {args.timesteps} steps 5 seeds")
            plt.tight_layout(); plt.savefig(os.path.join(comp_dir, f"maze_heist_{game}_plot.png"), dpi=150, bbox_inches='tight'); plt.close()
        print(f"Plots salvos em {comp_dir}")
    except Exception as e: print(f"Plot erro: {e}")
    print(f"\nResultados em {comp_dir}\n"+json.dumps(stats,indent=2))

if __name__=='__main__': main()
