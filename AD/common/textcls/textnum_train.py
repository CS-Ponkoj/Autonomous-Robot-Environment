"""Train a text-plus-numbers model (textnum.py), check it on the validation data, benchmark it
driving, and log it; or run DAgger rounds with it.

    .venv\\Scripts\\python -m AD.common.textcls.textnum_train train --model minilm --run tn1
    .venv\\Scripts\\python -m AD.common.textcls.textnum_train train --model modernbert_large --run tn1 --lora 16 --grad-ckpt --batch 32
    .venv\\Scripts\\python -m AD.common.textcls.textnum_train dagger --model minilm --base-run tn1 --run tnd --rounds 2

Training data:
- the recordings (AD/data, training seeds), every --stride-th row, with the route source's
  no-route reading (they hold no route values), loss weight --rec-weight so they do not teach the
  model to ignore the route;
- the hybrid's DAgger states (AD/models/hybrid/runs/hybrid_dagger/states_r*.npz, real route values
  from the online source: by default only with --route online, see --hybrid-states);
- --extra state files (this model's own DAgger rounds).
Validation: the recordings of the validation seeds and every 10th row of the state files.

Memory on a 6 GB GPU: the base models fit as they are; the large ones need --lora R (peft) or
--freeze N, and --grad-ckpt. The weights stay fp32; autocast is bf16 where the GPU supports it,
else fp16 with loss scaling. --max-minutes caps one invocation (continue with --resume)."""

from __future__ import annotations

import argparse
import functools
import glob
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .. import benchmark, dataset, tracker
from ..features import ACTION_NAMES, labels_of
from . import textnum
from .textnum import INPUTS, V_CAP, W_CAP, TextNumDriver

MODELS = tracker.AD / "models"
HYBRID_STATES = MODELS / "hybrid" / "runs" / "hybrid_dagger"
LARGE_HIDDEN = 1024  # encoders this wide get one rollout worker by default (GPU memory)


def seed_range(text, default):
    if text is None:
        return default
    a, b = (int(v) for v in text.split("-")) if "-" in text else (int(text), int(text))
    return range(a, b + 1)


def recordings(seeds, stride: int, limit: int = 0) -> dict:
    d = dataset.load(seeds)
    idx = np.arange(len(d["cmd"]))[::max(stride, 1)]
    if limit:
        idx = idx[:limit]
    return {"x": textnum.no_route(d["x"][idx]), "cmd": d["cmd"][idx].astype(np.float32)}


def states(files, valid: bool, limit: int = 0) -> dict:
    xs, cs = [], []
    for f in files:
        e = np.load(f)
        assert e["x"].shape[1] == INPUTS, f"{f}: {e['x'].shape[1]} inputs, not {INPUTS}"
        keep = (np.arange(len(e["x"])) % 10 == 0) == valid  # every 10th row is validation
        xs.append(e["x"][keep]), cs.append(e["cmd"][keep])
    if not xs:
        return {"x": np.zeros((0, INPUTS), np.float32), "cmd": np.zeros((0, 2), np.float32)}
    out = {"x": np.concatenate(xs).astype(np.float32), "cmd": np.concatenate(cs).astype(np.float32)}
    if limit:
        out = {k: v[:limit] for k, v in out.items()}
    return out


def join(rec: dict, st: dict, rec_weight: float) -> dict:
    x, cmd = np.concatenate([rec["x"], st["x"]]), np.concatenate([rec["cmd"], st["cmd"]])
    w = np.concatenate([np.full(len(rec["x"]), rec_weight, np.float32), np.ones(len(st["x"]), np.float32)])
    return {"x": x, "cmd": cmd, "action": np.asarray(labels_of(cmd)), "w": w, "n_rec": len(rec["x"]), "n_states": len(st["x"])}


def amp_dtype(precision: str, dev):
    if dev.type != "cuda" or precision == "fp32":
        return None
    if precision == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]


def evaluate(tok, net, d: dict, a, dev, dtype, len_scale: float) -> dict:
    net.eval()
    with torch.autocast(dev.type, dtype=dtype or torch.float32, enabled=dtype is not None):
        y = textnum.predict(tok, net, d["x"], a.max_len, dev, len_scale)
    if net.head_kind == "discrete":
        return dataset.accuracy(y.argmax(1), d["action"])
    r = dataset.accuracy(np.asarray(labels_of(y)), d["action"])
    err = np.abs(y - d["cmd"])
    r["mae_v"], r["mae_omega"] = round(float(err[:, 0].mean()), 4), round(float(err[:, 1].mean()), 4)
    return r


def key_of(r: dict, head: str) -> float:
    return r["accuracy"] if head == "discrete" else -(r["mae_v"] / V_CAP + r["mae_omega"] / W_CAP)


def add_train_args(p):
    p.add_argument("--model", required=True, help="model folder in AD/models (encoder: textnum.base_of)")
    p.add_argument("--run", required=True)
    p.add_argument("--base", default=None, help="encoder to start from (default: the model's pretrained one)")
    p.add_argument("--init", default=None, help="a finished run of this model to continue from (its encoder and head)")
    p.add_argument("--head", choices=("continuous", "discrete"), default="continuous")
    p.add_argument("--prefix", type=int, default=4, help="numeric prefix embeddings")
    p.add_argument("--width", type=int, default=32, help="lidar conv width")
    p.add_argument("--no-skip", action="store_true", help="the head sees only the encoder (ablation)")
    p.add_argument("--route", default="online", help="route source name (AD/models/hybrid/route.py)")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=5e-5, help="encoder learning rate")
    p.add_argument("--head-lr", type=float, default=1e-3, help="numeric branch and head learning rate")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--warmup", type=float, default=0.06)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--max-len", type=int, default=48)
    p.add_argument("--train", default=None, help="recording seed range a-b (default: dataset.TRAIN)")
    p.add_argument("--valid", default=None, help="recording seed range a-b (default: dataset.VALID)")
    p.add_argument("--stride", type=int, default=2, help="keep every k-th recorded row")
    p.add_argument("--valid-stride", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="at most this many rows per source (smoke tests)")
    p.add_argument("--rec-weight", type=float, default=0.3, help="loss weight of the recorded rows (no route values)")
    p.add_argument("--no-recordings", action="store_true")
    p.add_argument("--hybrid-states", choices=("auto", "on", "off"), default="auto",
                   help="the hybrid's DAgger states (online route values): auto uses them only with --route online")
    p.add_argument("--infer-fp32", action="store_true", help="run the saved model's encoder in fp32 when driving "
                   "(if fp16 inference overflows on a GPU without bf16)")
    p.add_argument("--extra", nargs="*", default=[], help="more state files (x: 82 inputs, cmd)")
    p.add_argument("--lora", type=int, default=0, help="LoRA rank on the encoder's linear layers (peft); 0: off")
    p.add_argument("--freeze", type=int, default=0, help="freeze the embeddings and the lowest N encoder layers")
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    p.add_argument("--max-minutes", type=float, default=40.0)
    p.add_argument("--ckpt-minutes", type=float, default=10.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-bench", action="store_true")
    p.add_argument("--bench-seeds", default=None, help="benchmark only these practice seeds a-b (smoke tests)")
    p.add_argument("--final", action="store_true", help="also the held-out test layouts (finished models only)")
    p.add_argument("--no-log", action="store_true", help="do not log the run (smoke tests)")
    p.add_argument("--notes", default="")


def make_model(a, dev):
    from transformers import AutoTokenizer
    if a.init:
        init = Path(a.init)
        cfg = json.loads((init / "textnum.json").read_text(encoding="utf-8"))
        tok = AutoTokenizer.from_pretrained(init / "encoder")
        net = textnum.build(str(init / "encoder"), head=cfg["head"], prefix=cfg["prefix"], width=cfg["width"], skip=cfg["skip"])
        net.load_state_dict(torch.load(init / "head.pt", map_location="cpu"), strict=False)
    else:
        base = a.base or textnum.base_of(a.model)
        tok = AutoTokenizer.from_pretrained(base)
        net = textnum.build(base, head=a.head, prefix=a.prefix, width=a.width, skip=not a.no_skip)
    if a.freeze:
        from .train import freeze
        freeze(net.encoder, a.freeze)
    if a.lora:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise SystemExit("--lora needs the peft package (pip install peft); or use --freeze N")
        net.encoder = get_peft_model(net.encoder, LoraConfig(r=a.lora, lora_alpha=2 * a.lora, lora_dropout=0.05,
                                                             target_modules="all-linear"))
    if a.grad_ckpt:
        net.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return tok, net.to(dev)


def train(argv=None) -> int:
    p = argparse.ArgumentParser(description="train a text-plus-numbers model")
    add_train_args(p)
    a = p.parse_args(argv)
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = amp_dtype(a.precision, dev)
    _, spec = textnum.route_source(a.route)
    use_hybrid = a.hybrid_states == "on" or (a.hybrid_states == "auto" and spec["name"] == "online")
    if use_hybrid and spec["name"] != "online":
        print(f"warning: the hybrid states hold online route values, mixed here with route source {spec['name']!r}", flush=True)
    files = (sorted(glob.glob(str(HYBRID_STATES / "states_r*.npz"))) if use_hybrid else []) + list(a.extra)
    sources = {"recordings": not a.no_recordings, "hybrid_states": use_hybrid,
               "state_files": [str(Path(f).relative_to(tracker.AD)) if Path(f).is_relative_to(tracker.AD) else str(f)
                               for f in files]}
    empty = {"x": np.zeros((0, INPUTS), np.float32), "cmd": np.zeros((0, 2), np.float32)}
    tr = join(empty if a.no_recordings else recordings(seed_range(a.train, dataset.TRAIN), a.stride, a.limit),
              states(files, False, a.limit), a.rec_weight)
    va = join(recordings(seed_range(a.valid, dataset.VALID), a.valid_stride, a.limit), states(files, True, a.limit), 1.0)
    print(f"{tr['n_rec']} recorded + {tr['n_states']} state rows to train on, {len(va['x'])} to validate; "
          f"{len(files)} state files", flush=True)
    tok, net = make_model(a, dev)
    enc_params = [q for q in net.encoder.parameters() if q.requires_grad]
    ids = {id(q) for q in net.encoder.parameters()}
    rest = [q for q in net.parameters() if q.requires_grad and id(q) not in ids]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr}, {"params": rest, "lr": a.head_lr}], weight_decay=a.wd)
    per_step = a.batch * a.accum
    steps_per_epoch = max(1, -(-len(tr["x"]) // per_step))
    total = max(1, int(steps_per_epoch * a.epochs))
    from transformers import get_linear_schedule_with_warmup
    sched = get_linear_schedule_with_warmup(opt, int(a.warmup * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    if net.head_kind == "discrete":
        counts = np.bincount(tr["action"], minlength=len(ACTION_NAMES)).astype(np.float32) + 1.0
        cls_w = torch.tensor((counts.sum() / counts) ** 0.5, device=dev)
        cls_w = cls_w / cls_w.mean()
    else:
        cls_w = None
    scale = torch.tensor([V_CAP, W_CAP], device=dev)
    X, Y, A, W = (torch.from_numpy(tr[k]) for k in ("x", "cmd", "action", "w"))
    run_dir = MODELS / a.model / "runs" / a.run
    ckpt = run_dir / "checkpoint.pt"
    trainable = lambda: {n: q.detach().cpu().clone() for n, q in net.named_parameters() if q.requires_grad}  # noqa: E731
    step, train_s_before, best, history = 0, 0.0, None, []
    if a.resume and ckpt.exists():
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        net.load_state_dict(st["trainable"], strict=False)
        opt.load_state_dict(st["opt"]), sched.load_state_dict(st["sched"]), scaler.load_state_dict(st["scaler"])
        step, train_s_before, best, history = st["step"], st["train_s"], st["best"], st["history"]
        print(f"resumed at step {step}/{total}", flush=True)

    def save_checkpoint():
        run_dir.mkdir(parents=True, exist_ok=True)
        tmp = ckpt.with_suffix(".tmp")
        torch.save({"trainable": trainable(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "step": step, "train_s": train_s_before + time.time() - start,
                    "best": best, "history": history}, tmp)
        tmp.replace(ckpt)

    start = last_ckpt = time.time()
    stopped = False
    while step < total and not stopped:
        epoch, pos = divmod(step, steps_per_epoch)
        net.train()
        order = torch.randperm(len(X), generator=torch.Generator().manual_seed(a.seed * 1000 + epoch))
        for b0 in range(pos * per_step, len(order), per_step):
            chunk = order[b0:b0 + per_step]
            opt.zero_grad(set_to_none=True)
            for m0 in range(0, len(chunk), a.batch):
                i = chunk[m0:m0 + a.batch]
                ids_, mask = textnum.tokenize(tok, tr["x"][i.numpy()], a.max_len, dev, spec["len_scale"])
                with torch.autocast(dev.type, dtype=dtype or torch.float32, enabled=dtype is not None):
                    out = net(ids_, mask, X[i].to(dev))
                w = W[i].to(dev)
                if net.head_kind == "continuous":
                    per_row = (((out.float() - Y[i].to(dev)) / scale) ** 2).mean(1)
                else:
                    per_row = torch.nn.functional.cross_entropy(out.float(), A[i].to(dev), weight=cls_w, reduction="none")
                loss = (per_row * w).sum() / w.sum().clamp_min(1e-6) * len(i) / len(chunk)
                if not torch.isfinite(loss):
                    print(f"non-finite loss at step {step}; try --precision fp32 or a lower --lr", flush=True)
                    return 3
                scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([q for q in net.parameters() if q.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 200 == 0:
                print(f"step {step}/{total} loss {loss.item() * len(chunk) / len(i):.4f} {time.time() - start:.0f} s", flush=True)
            if step >= total or step % steps_per_epoch == 0:
                break
            if time.time() - start > a.max_minutes * 60:
                stopped = True
                break
            if time.time() - last_ckpt > a.ckpt_minutes * 60:
                save_checkpoint()
                last_ckpt = time.time()
        if not stopped:  # an epoch ended (or the last step): check it, keep the best
            r = evaluate(tok, net, va, a, dev, dtype, spec["len_scale"])
            history.append({"step": step, **{k: r[k] for k in r if k != "per_action"}})
            print(f"epoch {step / steps_per_epoch:.2f}: valid {r['accuracy']:.3f}"
                  + (f" mae v {r['mae_v']:.3f} w {r['mae_omega']:.3f}" if "mae_v" in r else ""), flush=True)
            if best is None or key_of(r, net.head_kind) > best["key"]:
                best = {"key": key_of(r, net.head_kind), "result": r, "step": step}
                run_dir.mkdir(parents=True, exist_ok=True)
                torch.save(trainable(), run_dir / "best.pt")
        save_checkpoint()
        last_ckpt = time.time()
    train_s = train_s_before + time.time() - start
    if dev.type == "cuda":
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    if stopped:
        print(f"stopped at the time cap: step {step}/{total}; continue with --resume", flush=True)
        return 2
    net.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"), strict=False)
    net.eval()
    infer_fp32 = a.infer_fp32 or (bool(a.init) and json.loads((Path(a.init) / "textnum.json").read_text(
        encoding="utf-8")).get("infer_fp32", False))  # a DAgger round keeps its start's setting
    textnum.save(net, tok, run_dir, {"model": a.model, "route": spec, "max_len": a.max_len, "infer_fp32": infer_fp32,
                                     "sources": sources, "base": a.base or textnum.base_of(a.model)})
    n_rec, n_states = tr["n_rec"], tr["n_states"]
    net = opt = sched = X = Y = A = W = tr = va = None  # free memory before the benchmark loads the model again
    import gc
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    settings = {k: v for k, v in vars(a).items() if k not in ("run", "notes", "no_bench", "no_log", "extra")}
    entry = {"model": a.model, "run": a.run, "kind": "finetune", "base": a.init or a.base or textnum.base_of(a.model),
             "data": f"{n_rec} recorded rows (stride {a.stride}, weight {a.rec_weight}) + {n_states} DAgger state rows "
                     f"from {len(files)} files",
             "params": {**settings, "route": spec, "sources": sources, "infer_fp32": infer_fp32, "best_step": best["step"],
                        "history": history},
             "train_s": round(train_s, 1), "offline": best["result"],
             "notes": "; ".join(x for x in ("text plus numbers: short scene text, lidar bins and route as prefix embeddings",
                                            a.notes) if x)}
    if not a.no_bench:
        seeds = tuple(seed_range(a.bench_seeds, None)) if a.bench_seeds else None
        summary = benchmark.run(functools.partial(TextNumDriver, str(run_dir), a.model), workers=1, seeds=seeds)
        entry.update(bench=summary, score=benchmark.score(summary))
        print({k: summary[k] for k in ("no_cats", "4_cats", "latency_ms")}, "score", entry["score"], flush=True)
    if a.final:
        final = benchmark.run(functools.partial(TextNumDriver, str(run_dir), a.model), workers=1, final=True)
        entry.update(final=final, final_score=benchmark.score(final))
    if not a.no_log:
        tracker.log(entry)
    ckpt.unlink(missing_ok=True)
    (run_dir / "best.pt").unlink(missing_ok=True)
    print(a.run, "offline", best["result"]["accuracy"], "score", entry.get("score"), flush=True)
    return 0


def rollout(args) -> dict:
    """One episode with the text model driving and the built-in driver labelling every state."""
    run_dir, model, seed, beta, device = args
    import random

    from robot_env import config as C
    from robot_env.baseline import BaselineDriver
    from robot_env.layout import RoomMap
    from robot_env.system import RobotSystem

    from ..features import LABEL_LEVEL

    policy = TextNumDriver(run_dir, model, device=device)
    expert = BaselineDriver(C.SPEED_LEVELS[LABEL_LEVEL])
    system = RobotSystem(cats=seed % 5, cat_seed=seed)
    task = RoomMap(system.sim.model).sample_task(seed)
    system.reset(*task.start, task.goal)
    policy.reset(initial_pose=task.start)
    coin = random.Random(seed * 7919 + int(beta * 1000))
    X, V, outcome = [], [], "timeout"
    while system.time < C.EPISODE_TIME_LIMIT - 1e-9:
        obs = system.observe()
        if obs.goal_distance <= C.GOAL_RADIUS:
            outcome = "success"
            break
        label = expert.decide(obs)
        mine = policy.decide(obs)
        cmd = label.command if label else None
        X.append(policy.last_x)
        V.append((cmd.v, cmd.omega) if cmd else (0.0, 0.0))
        system.apply(label if (label is not None and coin.random() < beta) else mine)
        system.advance(C.DECISION_PERIOD)
        if system.collisions:
            outcome = "collision"
            break
    system.close()
    return {"seed": seed, "outcome": outcome, "x": np.array(X, np.float32).reshape(-1, INPUTS),
            "cmd": np.array(V, np.float32).reshape(-1, 2)}


def dagger(argv=None) -> int:
    """DAgger rounds: the model drives training layouts (its own states), the built-in driver labels
    them, and the model is fine-tuned on everything so far; each round is benchmarked and logged as
    <run>_r<k>. Options not listed here go to every round's training (for example --lora 16)."""
    p = argparse.ArgumentParser(description=dagger.__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="model folder in AD/models (encoder: textnum.base_of)")
    p.add_argument("--base-run", required=True, help="the finished run of this model to start from")
    p.add_argument("--run", required=True)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--beta", type=float, default=0.5, help="expert share in round 1, halved each round")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--workers", type=int, default=None, help="rollout processes (default 2; 1 for the large models)")
    p.add_argument("--device", default=None, help="rollout device (default: the GPU if any)")
    p.add_argument("--first-seed", type=int, default=240, help="offset into the training seeds (the hybrid used 120-239)")
    p.add_argument("--start-round", type=int, default=1, help="resume after the earlier rounds")
    a, passthrough = p.parse_known_args(argv)
    runs = MODELS / a.model / "runs"
    hidden = json.loads((runs / a.base_run / "encoder" / "config.json").read_text(encoding="utf-8"))["hidden_size"]
    workers = a.workers or (1 if hidden >= LARGE_HIDDEN else 2)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pool = list(dataset.TRAIN)
    files, ckpt, beta = [], runs / a.base_run, a.beta
    for k in range(1, a.start_round):
        files.append(str(runs / a.run / f"states_r{k}.npz"))
        ckpt, beta = runs / f"{a.run}_r{k}", beta / 2
        assert Path(files[-1]).exists() and (ckpt / "textnum.json").exists(), f"round {k} is not complete"
    for k in range(a.start_round, a.rounds + 1):
        seeds = [pool[(a.first_seed + (k - 1) * a.episodes + i) % len(pool)] for i in range(a.episodes)]
        out = runs / a.run / f"states_r{k}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = out.with_suffix(".json")
        if out.exists() and meta.exists():
            info = json.loads(meta.read_text(encoding="utf-8"))
        else:
            t0 = time.time()
            with ProcessPoolExecutor(workers, max_tasks_per_child=1) as ex:  # a fresh process per episode (sim memory)
                eps = list(ex.map(rollout, [(str(ckpt), a.model, s, beta, device) for s in seeds]))
            np.savez_compressed(out, x=np.concatenate([e["x"] for e in eps]), cmd=np.concatenate([e["cmd"] for e in eps]))
            info = {"states": int(sum(len(e["x"]) for e in eps)),
                    "outcomes": {o: sum(e["outcome"] == o for e in eps) for o in ("success", "collision", "timeout")}}
            meta.write_text(json.dumps(info), encoding="utf-8")
            print(f"round {k}: collected {info} in {time.time() - t0:.0f} s (beta {beta:.2f})", flush=True)
        files.append(str(out))
        run = f"{a.run}_r{k}"
        rc = train(["--model", a.model, "--run", run, "--init", str(ckpt), "--epochs", str(a.epochs), "--lr", str(a.lr),
                    "--resume", *passthrough, "--extra", *files,  # --resume: a rerun continues a capped round
                    "--notes", f"DAgger round {k} from {a.base_run}: +{info['states']} states (policy episodes "
                               f"{info['outcomes']}, beta {beta:.2f})"])
        if rc:
            print(f"round {k} training stopped with exit {rc}; rerun with --start-round {k} (its states are kept)", flush=True)
            return rc
        ckpt, beta = runs / run, beta / 2
    return 0


def main(argv=None) -> int:
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("train", "dagger"):
        raise SystemExit("usage: textnum_train train|dagger [options]  (--help after the command for options)")
    return (train if argv[0] == "train" else dagger)(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
