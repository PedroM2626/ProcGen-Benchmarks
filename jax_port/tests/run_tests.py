"""Runner da suite (sem pytest): python -m jax_port.tests.run_tests.

test_stats sempre; test_parity/test_zoo/test_smoke pulam (SKIP) se o
venv nao tiver as deps — nunca falham por falta de ambiente.
"""

import traceback

CASES = [
    ("stats", "jax_port.tests.test_stats", "test_stats", False),
    ("parity", "jax_port.tests.test_parity", "test_env_parity", True),
    ("zoo", "jax_port.tests.test_zoo", "test_zoo", True),
    ("smoke", "jax_port.tests.test_smoke", "test_smoke_train", True),
]


def main():
    import sys
    fails, skips = [], []
    for name, mod, fn, needs_env in CASES:
        try:
            f = getattr(__import__(mod, fromlist=[fn]), fn)
            print(f"[{name}] {f()}", flush=True)
            print(f"[{name}] PASS", flush=True)
        except ImportError as e:
            print(f"[{name}] SKIP ({e})", flush=True)
            skips.append(name)
        except Exception:
            traceback.print_exc()
            fails.append(name)
    print(f"PASS={len(CASES)-len(fails)-len(skips)} "
          f"SKIP={len(skips)} FAIL={fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
