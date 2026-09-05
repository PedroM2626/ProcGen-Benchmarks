# Rebuilding the JAX-port venv (`/root/procgen-jax`)

I run the port in WSL2 Ubuntu 24.04 with a dedicated Python 3.10 venv,
because `procgen 0.10.7` ships no `cp312` wheel and requires
`numpy<2` (see README §14.3). Exact pins live in
`requirements-jax.txt` (a `pip freeze` of my working venv).

```bash
# 1. Python 3.10 (not in Ubuntu 24.04 repos)
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update && sudo apt install -y python3.10 python3.10-venv

# 2. Venv (any path; mine is /root/procgen-jax)
/usr/bin/python3.10 -m venv /root/procgen-jax
/root/procgen-jax/bin/pip install -U pip

# 3. Pinned stack (order matters: numpy<2 BEFORE jax pulls 2.x)
#
# 3a. procgen wheel for cp310 (no compilation needed)
/root/procgen-jax/bin/pip install "procgen==0.10.7" "numpy==1.26.4" \
  "gym==0.26.2" "gymnasium==1.3.0" "gym3==0.3.3"
# 3b. JAX with CUDA 12 (+ deps)
/root/procgen-jax/bin/pip install "jax[cuda12]==0.6.2" \
  "flax==0.10.7" "optax==0.2.8"
# 3c. or everything at once, verbatim:
/root/procgen-jax/bin/pip install -r jax_port/requirements-jax.txt
```

Verify (headless, no display needed):

```bash
/root/procgen-jax/bin/python -c \
  "import jax, procgen, gym; print(jax.devices()); \
   e = gym.make('procgen:procgen-coinrun-v0', num_levels=200); \
   print(e.reset().shape, e.action_space)"
# expect: [CudaDevice(id=0)]  (64, 64, 3)  Discrete(15)
```

Notes I learned the hard way:
- `jax[cuda12]` pulls `numpy 2.x`; pin `numpy==1.26.4` or procgen breaks.
- The `ml-dtypes` version warning is harmless (the GPU op still runs).
- First-ever run pays ~3 min of cuDNN/XLA autotune; it persists in
  `/tmp/jax_port_cache` (`JAX_PORT_CACHE` overrides it).
- The study venv (Windows, `torch cu121`, `requirements.txt`) and this
  venv must stay separate — their `numpy`/`gym` requirements conflict.
