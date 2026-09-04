"""Smoke de treino ponta-a-ponta (MLP+coinrun, ~1k steps, rapido, sem conv).

Exercita o caminho da grade (train.train em processo). Requer venv do
porte + GPU. Uso: python -m jax_port.tests.test_smoke
"""

import math
import os
import tempfile
import types


def test_smoke_train():
    from jax_port import train as T
    out = os.path.join(tempfile.mkdtemp(), "smoke.json")
    args = types.SimpleNamespace(
        game="coinrun", algo="ppo", extractor="mlp", obs=None,
        augment="none", explore="none", timesteps=1024, seed=3,
        num_envs=8, rollout=32, minibatch=256, eval_eps=2,
        eval_det_eps=0, eval_train_eps=0, eval_envs=2, out=out)
    r = T.train(args)
    assert r["timesteps"] >= 1024 and r["sps"] > 100, r
    assert r["curve"], "curva AUC ausente"
    assert math.isfinite(r["train_ret_mean20"])
    assert r["eval_unseen"]["eps"] == 2
    return {"sps": r["sps"], "ret": r["train_ret_mean20"]}


if __name__ == "__main__":
    print(test_smoke_train())
    print("SMOKE_OK")
