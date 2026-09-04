"""Grade runner do porte — Suites do estudo, sequencial, resume-safe.

Suites (--suite, repetiveis e combinaveis):
  main        : 16 configs x jogos (bossfight/starpilot/dodgeball)
  exploration : ppo/icm/rnd/ngu x maze/heist
  algo        : ppo/a2c/dqn/qrdqn x starpilot/dodgeball/bossfight (+--lr-sens)
  hrl         : flat/skip4/hrl/hrl_learned x jumper/plunder (budget frames)
  budget      : resnet18+mlp x starpilot/dodgeball (honra --timesteps dado)
Uso:
    .../train.py ...  # nao; este script orquestra tudo em UM processo
    wsl -e env PYTHONPATH=... /root/procgen-jax/bin/python \
      jax_port/run_grade.py --suite main --games bossfight --seeds 42 \
      --timesteps 100000 --eval-full --out-dir jax_port/results_grade
Celula: {cfg}__{game}__seed{s}__{t}k.json; pula celula pronta (resume).
--eval-full => 100 stoch + 100 det + 15 train (protocolo definitivo).
"""

import argparse
import json
import os
import time
import types

MAIN_CONFIGS = ["classic", "cbam", "spatial", "mlp", "aug_crop", "aug_color",
                "aug_noise", "impala", "impoola", "lstm_attention", "vit",
                "resnet18", "vae", "ae", "recon", "contrastive"]
MAIN_GAMES = ["bossfight", "starpilot", "dodgeball"]
EXPLORE_CONFIGS = ["ppo", "icm", "rnd", "ngu"]
EXPLORE_GAMES = ["maze", "heist"]
ALGO_CONFIGS = ["ppo", "a2c", "dqn", "qrdqn"]
ALGO_GAMES = ["starpilot", "dodgeball", "bossfight"]
HRL_ARMS = ["flat", "skip4", "hrl", "hrl_learned"]
HRL_GAMES = ["jumper", "plunder"]
BUDGET_CONFIGS = ["resnet18", "mlp"]
BUDGET_GAMES = ["starpilot", "dodgeball"]
HARD_CONFIGS = ["classic", "cbam", "spatial", "mlp", "vae", "ae", "recon",
                "contrastive", "aug_crop", "aug_color", "aug_noise"]
PILOT_CONFIGS = ["classic", "cbam", "spatial", "mlp"]

AUG_OF = {"aug_crop": "crop", "aug_color": "color", "aug_noise": "noise"}


def cells(args):
    out = []
    for suite in args.suite:
        if suite == "main":
            games = args.games or MAIN_GAMES
            for cfg in MAIN_CONFIGS:
                ext = {"aug_crop": "classic", "aug_color": "classic",
                       "aug_noise": "classic",
                       "contrastive": "contrastive"}.get(cfg, cfg)
                if cfg == "contrastive":
                    aug, exp = "noise", "none"
                else:
                    aug, exp = AUG_OF.get(cfg, "none"), "none"
                for game in games:
                    for s in args.seeds:
                        for t in args.timesteps:
                            out.append({"suite": suite, "cfg": cfg,
                                        "kind": "ppo", "game": game, "seed": s,
                                        "timesteps": t, "extractor": ext,
                                        "augment": aug, "explore": exp})
        elif suite == "exploration":
            for cfg in EXPLORE_CONFIGS:
                for game in (args.games or EXPLORE_GAMES):
                    for s in args.seeds:
                        for t in args.timesteps:
                            out.append({"suite": suite, "cfg": cfg,
                                        "kind": "ppo", "game": game, "seed": s,
                                        "timesteps": t, "extractor": "classic",
                                        "augment": "none",
                                        "explore": "none" if cfg == "ppo" else cfg})
        elif suite == "algo":
            cfgs = list(ALGO_CONFIGS)
            for cfg in cfgs:
                for game in (args.games or ALGO_GAMES):
                    for s in args.seeds:
                        for t in args.timesteps:
                            out.append({"suite": suite, "cfg": cfg,
                                        "kind": cfg, "game": game, "seed": s,
                                        "timesteps": t, "extractor": "classic",
                                        "augment": "none", "explore": "none"})
            if args.lr_sens:
                for cfg in ("dqn", "qrdqn"):
                    for s in args.seeds:
                        for t in args.timesteps:
                            c = dict(suite=suite, cfg=cfg + "_lr3e-4",
                                     kind=cfg, game="starpilot", seed=s,
                                     timesteps=t, extractor="classic",
                                     augment="none", explore="none", lr=3e-4)
                            out.append(c)
        elif suite == "hrl":
            for arm in HRL_ARMS:
                for game in (args.games or HRL_GAMES):
                    for s in args.seeds:
                        for t in args.timesteps:
                            out.append({"suite": suite, "cfg": arm,
                                        "kind": "hrl", "game": game, "seed": s,
                                        "timesteps": t, "arm": arm})
        elif suite == "budget":
            bsteps = args.budget_steps or args.timesteps
            for cfg in BUDGET_CONFIGS:
                for game in (args.games or BUDGET_GAMES):
                    for s in args.seeds:
                        for t in bsteps:
                            out.append({"suite": suite, "cfg": cfg,
                                        "kind": "ppo", "game": game, "seed": s,
                                        "timesteps": t, "extractor": cfg,
                                        "augment": "none", "explore": "none"})
        elif suite == "hard":
            # Stress test §3.4: mesmas 11 configs da suite, modo hard.
            for cfg in HARD_CONFIGS:
                ext = {"aug_crop": "classic", "aug_color": "classic",
                       "aug_noise": "classic",
                       "contrastive": "contrastive"}.get(cfg, cfg)
                aug = (AUG_OF.get(cfg, "none") if cfg != "contrastive"
                       else "noise")
                for s in args.seeds:
                    for t in args.timesteps:
                        out.append({"suite": suite, "cfg": cfg, "kind": "ppo",
                                    "game": "bossfight", "seed": s,
                                    "timesteps": t, "extractor": ext,
                                    "augment": aug, "explore": "none",
                                    "distribution": "hard"})
        elif suite == "pilot":
            # Piloto coinrun 50k §3.1 (sempre 50k, como no estudo).
            for cfg in PILOT_CONFIGS:
                for s in args.seeds:
                    out.append({"suite": suite, "cfg": cfg, "kind": "ppo",
                                "game": "coinrun", "seed": s, "timesteps": 50000,
                                "extractor": cfg, "augment": "none",
                                "explore": "none"})
        elif suite == "spr":
            # EXTENSAO alem do estudo (sem paridade §1-12): SPR aux.
            for cfg in ("spr", "spr_aug"):
                for game in (args.games or MAIN_GAMES):
                    for s in args.seeds:
                        for t in args.timesteps:
                            out.append({"suite": suite, "cfg": cfg,
                                        "kind": "ppo", "game": game, "seed": s,
                                        "timesteps": t, "extractor": "classic",
                                        "augment": "crop" if cfg == "spr_aug"
                                        else "none",
                                        "explore": "none", "aux": "spr"})
        elif suite == "gnn":
            # EXTENSAO alem do estudo (sem paridade §1-12): GAT patches.
            for game in (args.games or MAIN_GAMES):
                for s in args.seeds:
                    for t in args.timesteps:
                        out.append({"suite": suite, "cfg": "gat",
                                    "kind": "ppo", "game": game, "seed": s,
                                    "timesteps": t, "extractor": "gat",
                                    "augment": "none", "explore": "none"})
    return out


def run_cell(cell, args):
    from jax_port import train as T
    from jax_port import train_dqn as D
    from jax_port import train_hrl as H
    tag = f"{cell['cfg']}__{cell['game']}__seed{cell['seed']}__{cell['timesteps']//1000}k"
    path = os.path.join(args.out_dir, cell["suite"], tag + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and not args.overwrite:
        return {"skipped": path}
    ee = (100, 100, 15) if args.eval_full else (
        args.eval_eps, args.eval_det_eps, args.eval_train_eps)
    if cell["kind"] == "hrl":
        ns = types.SimpleNamespace(
            game=cell["game"], arm=cell["arm"], frames=cell["timesteps"],
            seed=cell["seed"], num_envs=args.num_envs, rollout=128,
            minibatch=1024, eval_eps=ee[0], eval_det_eps=ee[1],
            eval_train_eps=ee[2], eval_envs=8, out=path)
        return H.train(ns)
    if cell["kind"] in ("dqn", "qrdqn"):
        ns = types.SimpleNamespace(
            game=cell["game"], algo=cell["kind"], extractor="classic",
            timesteps=cell["timesteps"], seed=cell["seed"],
            lr=cell.get("lr", 1e-4), num_envs=32, eval_eps=ee[0],
            eval_det_eps=ee[1], eval_train_eps=ee[2], eval_envs=8, out=path)
        return D.train(ns)
    algo = cell["kind"]  # ppo | a2c
    ns = types.SimpleNamespace(
        game=cell["game"], algo=algo, extractor=cell["extractor"],
        distribution=cell.get("distribution", "easy"),
        obs=None, augment=cell.get("augment", "none"),
        explore=cell.get("explore", "none"), aux=cell.get("aux", "none"),
        timesteps=cell["timesteps"],
        seed=cell["seed"], num_envs=args.num_envs, rollout=128,
        minibatch=1024, eval_eps=ee[0], eval_det_eps=ee[1],
        eval_train_eps=ee[2], eval_envs=8, out=path)
    return T.train(ns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", nargs="+",
                    default=["main"],
                    choices=["main", "exploration", "algo", "hrl", "budget",
                             "hard", "pilot", "spr", "gnn"])
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--timesteps", type=int, nargs="+", default=[100000])
    ap.add_argument("--budget-steps", type=int, nargs="+", default=None,
                    help="timesteps so p/ suite budget (default: --timesteps)")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--eval-eps", type=int, default=10)
    ap.add_argument("--eval-det-eps", type=int, default=0)
    ap.add_argument("--eval-train-eps", type=int, default=0)
    ap.add_argument("--eval-full", action="store_true")
    ap.add_argument("--configs", nargs="*", default=None,
                    help="filtra cfgs (ex. classic mlp mlp_vector ppo icm flat)")
    ap.add_argument("--lr-sens", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out-dir", default="jax_port/results_grade")
    ap.add_argument("--master", default="jax_port/results_grade/master.json")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    master = {}
    if os.path.exists(args.master) and not args.overwrite:
        with open(args.master) as fh:
            master = json.load(fh)
    t0 = time.perf_counter()
    all_cells = cells(args)
    if args.configs:
        all_cells = [c for c in all_cells if c["cfg"] in args.configs]
    for i, cell in enumerate(all_cells):
        key = (f"{cell['suite']}/{cell['cfg']}__{cell['game']}__"
               f"seed{cell['seed']}__{cell['timesteps']//1000}k")
        if key in master and not args.overwrite:
            print(f"[{i}] skip {key}", flush=True)
            continue
        print(f"[{i}] {key}", flush=True)
        try:
            r = run_cell(cell, args)
            master[key] = {"ok": True,
                           "sps": r.get("sps"),
                           "eval_unseen": r.get("eval_unseen"),
                           "out": r.get("skipped", "")}
        except Exception as e:  # noqa: BLE001 (grade nao pode morrer)
            import traceback
            traceback.print_exc()
            master[key] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        with open(args.master, "w") as fh:
            json.dump(master, fh, indent=2)
    dt = time.perf_counter() - t0
    ok = sum(1 for v in master.values() if v.get("ok"))
    print(f"grade: {ok}/{len(master)} ok em {dt:.0f}s -> {args.master}")


if __name__ == "__main__":
    main()
