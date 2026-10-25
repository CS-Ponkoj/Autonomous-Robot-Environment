"""DAgger for the CfC policy: the CfC drives training layouts (its memory carried through each
episode) while the built-in driver labels every state; each round's episodes are kept whole (the
CfC trains on sequences), added to the data, and the policy is fine-tuned, benchmarked and logged
(run <run>_r<k>).

    .venv\Scripts\python -m AD.models.cfc.dagger --base cfc_bc --run dagger_cfc --rounds 3 --episodes 40"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ...common import dataset
from ..cnn.dagger import rollout
from . import train
from .train import RUNS


def collect(ckpt: str, seeds: list[int], beta: float, workers: int, out) -> dict:
    with ProcessPoolExecutor(workers, max_tasks_per_child=1) as pool:  # a fresh process per episode
        eps = list(pool.map(rollout, [(ckpt, s, beta, "cfc") for s in seeds]))
    np.savez_compressed(out, x=np.concatenate([e["x"] for e in eps]), cmd=np.concatenate([e["cmd"] for e in eps]),
                        lens=np.array([len(e["x"]) for e in eps]))
    return {"states": int(sum(len(e["x"]) for e in eps)),
            "outcomes": {k: sum(e["outcome"] == k for e in eps) for k in ("success", "collision", "timeout")}}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base", required=True, help="run name of the starting CfC (AD/models/cfc/runs/<base>/model.pt)")
    p.add_argument("--run", required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--beta", type=float, default=0.5, help="expert share in round 1, halved each round")
    p.add_argument("--steps", type=int, default=2000, help="fine-tune steps per round")
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args(argv)
    pool = list(dataset.TRAIN)
    files, ckpt, beta = [], str(RUNS / a.base / "model.pt"), a.beta
    for k in range(1, a.rounds + 1):
        seeds = [pool[((k - 1) * a.episodes + i + 250) % len(pool)] for i in range(a.episodes)]  # other layouts than the CNN's DAgger
        out = RUNS / a.run / f"episodes_r{k}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = out.with_suffix(".json")
        if out.exists() and meta.exists():
            info = json.loads(meta.read_text(encoding="utf-8"))
        else:
            t0 = time.time()
            info = collect(ckpt, seeds, beta, a.workers, out)
            meta.write_text(json.dumps(info), encoding="utf-8")
            print(f"round {k}: collected {info} in {time.time() - t0:.0f} s (beta {beta:.2f})", flush=True)
        files.append(str(out))
        run = f"{a.run}_r{k}"
        train.main(["--run", run, "--init", ckpt, "--steps", str(a.steps), "--lr", "3e-4", "--workers", str(a.workers),
                    "--extra", *files, "--notes", f"CfC DAgger round {k} from {a.base}: +{info['states']} states "
                                                   f"(policy episodes {info['outcomes']}, beta {beta:.2f})"])
        ckpt, beta = str(RUNS / run / "model.pt"), beta / 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
