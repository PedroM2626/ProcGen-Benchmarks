"""Sweep de tuning do Dreamer (sequencial, um processo por celula).

Celulas B/C/D sobre o baseline A (raw/ent 3e-4, ja medido ret 0.0):
  B: symlog + ent 3e-4   (isola escala de reward)
  C: symlog + ent 1e-3   (escala + exploracao)
  D: raw    + ent 1e-3   (isola entropia)
Uso (GPU livre!):
    wsl -e env PYTHONPATH=... /root/procgen-jax/bin/python \
      jax_port/run_dreamer_sweep.py --frames 1000000 --seed 42
Resumo em jax_port/dreams/sweep.json.
"""

import argparse
import gc
import json
import subprocess
import sys

CELLS = [
    ("B", ["--reward-mode", "symlog", "--ent-coef", "3e-4"]),
    ("C", ["--reward-mode", "symlog", "--ent-coef", "1e-3"]),
    ("D", ["--reward-mode", "raw", "--ent-coef", "1e-3"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="coinrun")
    ap.add_argument("--frames", type=int, default=1000000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-envs", type=int, default=16)
    args = ap.parse_args()
    summ = {}
    for tag, extra in CELLS:
        out = f"jax_port/dreams/dreamer_sweep_{tag}.json"
        cmd = [sys.executable, "jax_port/train_dreamer.py", "--game", args.game,
               "--frames", str(args.frames), "--seed", str(args.seed),
               "--num-envs", str(args.num_envs), "--out", out,
               "--out-dir", "jax_port/dreams"] + extra
        print(f"[{tag}] {' '.join(extra)}", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout[-800:] if r.stdout else "")
        if r.returncode != 0:
            print(r.stderr[-2000:])
            summ[tag] = {"ok": False}
            continue
        d = json.load(open(out))
        summ[tag] = {"ok": True, "ret": d["train_ret_mean20"],
                     "sps": d["sps"], "wall_s": d["wall_s"]}
        gc.collect()
    json.dump(summ, open("jax_port/dreams/sweep.json", "w"), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
