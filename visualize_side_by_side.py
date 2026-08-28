"""
Visualização lado-a-lado tempo real: 2-4 agentes Procgen jogando simultaneamente
- Suporta world_models (bossfight) e procgen (coinrun) com PPO CnnPolicy
- Se model.zip não existir, usa agente aleatório para demo
- Python 3.10 + procgen 0.10.7
"""
import os, argparse, glob, time
import numpy as np
import cv2
from procgen_wrapper import make_procgen_env
from stable_baselines3 import PPO

def load_model(path, device='cpu'):
    try:
        if os.path.exists(path):
            return PPO.load(path, device=device)
    except Exception as e:
        print(f"Falha load {path}: {e}")
    return None

def run_side_by_side(benchmark='world_models', game='bossfight', log_dir='./logs_world_models', mode='human', steps=1000, fps=15, out='side_by_side.mp4', seed=0, device='cpu'):
    # descobrir último comparison_*
    comps = sorted(glob.glob(os.path.join(log_dir, 'comparison_*')))
    if not comps:
        print(f"Sem {log_dir}/comparison_*, usando demo aleatório")
        comp_dir = None
    else:
        comp_dir = comps[-1]
        print(f"Usando {comp_dir}")

    if benchmark == 'world_models':
        configs = ['vae','ae','recon','contrastive']
        extractor_map = None
    else:  # procgen
        configs = ['classic_pixels','attention_cbam_pixels','attention_spatial_pixels','mlp_vector']
        game = game or 'coinrun'

    envs = []
    models = []
    for key in configs:
        # vector config não tem imagem -> pular para visual
        if 'vector' in key:
            print(f"Pulando {key} (vector sem imagem)")
            continue
        env = make_procgen_env(game, num_levels=0, distribution_mode='easy', seed=seed, vector=False)
        # gym old -> gymnasium wrapper já converte CHW
        envs.append((key, env))
        # tentar carregar melhor seed
        model = None
        if comp_dir:
            candidates = sorted(glob.glob(os.path.join(comp_dir, f"{key}_seed*.zip")))
            if candidates:
                # pegar maior reward se tiver results.json
                candidates = sorted(candidates) # pega último
                model = load_model(candidates[-1], device=device)
                print(f"{key}: carregado {candidates[-1]}" if model else f"{key}: aleatório (falha load)")
            else:
                print(f"{key}: sem .zip, aleatório")
        models.append(model)

    if not envs:
        print("Nenhum env visual")
        return

    # reset todos
    obs_list = []
    for _, env in envs:
        o,_ = env.reset()
        obs_list.append(o)

    # vídeo writer se mp4
    writer = None
    if mode == 'mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # cada painel é resize para 128x128; canvas = 2 linhas (real em cima, dream embaixo)
        h, w = 128, 128
        total_w = w * len(envs)
        writer = cv2.VideoWriter(out, fourcc, fps, (total_w, h*2))  # *2 para linha de sonho

    for step in range(steps):
        frames = []
        for idx, (key, env) in enumerate(envs):
            obs = obs_list[idx]
            model = models[idx]
            # obs CHW -> para model precisa CHW mesmo, PPO lida
            if model is not None:
                # DummyVecEnv não usado, model.predict espera obs com batch
                action, _ = model.predict(obs, deterministic=False)
                action = int(action)
            else:
                action = env.action_space.sample()
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # obs CHW -> HWC para display, *3 se precisar resize
            if obs_next.shape[0] in [3,12]:  # CHW
                disp = np.transpose(obs_next, (1,2,0))
            else:
                disp = obs_next
            # garantir 64x64x3 uint8
            if disp.dtype != np.uint8:
                disp = (np.clip(disp,0,1)*255).astype(np.uint8) if disp.max()<=1 else disp.astype(np.uint8)
            # resize 2x para ver melhor
            disp = cv2.resize(disp, (128,128), interpolation=cv2.INTER_NEAREST)
            # legenda
            cv2.rectangle(disp, (0,0), (128,14), (0,0,0), -1)
            cv2.putText(disp, key[:12], (2,11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
            frames.append(disp)
            if done:
                o,_ = env.reset()
                obs_list[idx]=o
            else:
                obs_list[idx]=obs_next

        # sonhos (dream) para VAE/AE/Recon
        dream_frames = []
        for idx, (key, env) in enumerate(envs):
            model = models[idx]
            obs = obs_list[idx]
            if model is not None and hasattr(model.policy.features_extractor, 'dream'):
                try:
                    import torch
                    with torch.no_grad():
                        obs_t = torch.from_numpy(obs).unsqueeze(0)
                        dream = model.policy.features_extractor.dream(obs_t)  # 1x3x64x64 0-1
                        dream = dream.squeeze(0).permute(1,2,0).cpu().numpy()
                        dream = (dream*255).astype(np.uint8)
                        dream = cv2.resize(dream, (128,128), interpolation=cv2.INTER_NEAREST)
                        cv2.rectangle(dream, (0,0), (128,14), (0,0,0), -1)
                        cv2.putText(dream, "dream", (2,11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,200,0), 1)
                        dream_frames.append(dream)
                except Exception:
                    dream_frames.append(np.zeros((128,128,3), dtype=np.uint8))
            else:
                # contrastivo sem dream -> mostra obs com noise
                if obs.shape[0] in [3,12]:
                    d = np.transpose(obs, (1,2,0))
                else:
                    d = obs
                d = cv2.resize(d, (128,128), interpolation=cv2.INTER_NEAREST)
                cv2.rectangle(d, (0,0), (128,14), (0,0,0), -1)
                cv2.putText(d, "no dream" if model else "random", (2,11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180,180,180), 1)
                dream_frames.append(d)

        # canvas 2 linhas: real em cima, sonho embaixo
        top = np.hstack(frames)
        bottom = np.hstack(dream_frames)
        canvas = np.vstack([top, bottom])
        cv2.putText(canvas, f"step {step} {game}  top=real bottom=dream", (4, canvas.shape[0]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        if mode == 'human':
            cv2.imshow(f"Side-by-side {benchmark} {game}", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(int(1000/fps)) & 0xFF == 27:
                break
        else:
            # canvas já BGR? writer espera BGR
            writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        if step % 100 == 0:
            print(f"step {step}/{steps}")

    if mode == 'human':
        cv2.destroyAllWindows()
    else:
        writer.release()
        print(f"Vídeo salvo em {out} ({steps} frames)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, default='world_models', choices=['world_models','procgen'])
    parser.add_argument('--game', type=str, default='bossfight', help='bossfight, coinrun, etc.')
    parser.add_argument('--log_dir', type=str, default='./logs_world_models')
    parser.add_argument('--mode', type=str, default='human', choices=['human','mp4'])
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--out', type=str, default='side_by_side.mp4')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    run_side_by_side(benchmark=args.benchmark, game=args.game, log_dir=args.log_dir, mode=args.mode, steps=args.steps, out=args.out, device=args.device)
