"""Fine-tune a text model as an 8-way action classifier on the recorded scene texts, check its
accuracy on the validation layouts, benchmark it driving (GPU, one process), and log the run.

Called by each model's own train.py (AD/models/modernbert, AD/models/deberta, ...), for example

    .venv\\Scripts\\python -m AD.models.modernbert.train --run ft1 --epochs 2 --lr 5e-5 --max-len 128 --batch 32

Memory on an 8 GB GPU: a base model fits as is; a large one (a ModernBERT-large model) needs
--grad-ckpt and --freeze N (the embeddings and the lowest N encoder layers stay fixed), or
--bf16-weights. --max-minutes caps the training time (the run is still checked and logged)."""

from __future__ import annotations

import argparse
import functools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ...common import benchmark, dataset, tracker
from ...common.features import ACTION_NAMES
from . import formats
from .model import TextDriver, device, predict

MODELS = Path(__file__).resolve().parents[2] / "models"  # runs go in AD/models/<model>/runs/<run>


def parser(default_base: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--base", default=default_base, help="Hugging Face id or local path to start from")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    p.add_argument("--warmup", type=float, default=0.06, help="share of the steps")
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--weighted", action="store_true", help="weight classes by inverse frequency")
    p.add_argument("--format", choices=sorted(formats.FORMATS), default="short_nospeed",
                   help="scene-text variant; the *_nospeed ones leave out the copycat shortcut (formats.py)")
    p.add_argument("--stride", type=int, default=1, help="keep every k-th training example (neighbours are near duplicates)")
    p.add_argument("--limit", type=int, default=0, help="at most this many training examples (0: all)")
    p.add_argument("--freeze", type=int, default=0, help="freeze the embeddings and the lowest N encoder layers")
    p.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (less memory, slower)")
    p.add_argument("--bf16-weights", action="store_true", help="keep the weights in bf16 (less memory)")
    p.add_argument("--precision", choices=("bf16", "fp32"), default="bf16",
                   help="compute precision on the GPU (bf16 autocast or plain fp32)")
    p.add_argument("--max-minutes", type=float, default=40.0, help="training time cap for this invocation")
    p.add_argument("--ckpt-minutes", type=float, default=10.0, help="also checkpoint this often (and at each epoch end)")
    p.add_argument("--resume", action="store_true", help="continue the run from its last checkpoint")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-bench", action="store_true")
    p.add_argument("--final", action="store_true", help="also run the held-out test layouts (finished models only)")
    p.add_argument("--notes", default="")
    return p


def encoder_layers(net):
    """The model's list of encoder layers (ModernBERT: model.layers, DeBERTa and BERT: encoder.layer)."""
    base = net.base_model
    for path in ("layers", "encoder.layer"):
        obj = base
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return list(obj)
        except AttributeError:
            continue
    raise AttributeError("no encoder layer list found")


def freeze(net, n: int) -> None:
    base = net.base_model
    emb = getattr(base, "embeddings", None)
    if emb is not None:
        for q in emb.parameters():
            q.requires_grad = False
    for layer in encoder_layers(net)[:n]:
        for q in layer.parameters():
            q.requires_grad = False


def data(seeds, fmt: str, stride: int = 1, limit: int = 0, seed: int = 0):
    d = dataset.load(seeds, text=True)
    idx = np.arange(len(d["action"]))[::max(stride, 1)]
    if limit and len(idx) > limit:
        idx = np.sort(np.random.default_rng(seed).choice(idx, limit, replace=False))
    return [formats.apply(fmt, d["text"][i]) for i in idx], d["action"][idx]


class ActionTask:
    """What a run learns and how it is checked. The default: the 8 driving actions, from the
    recorded scene texts, benchmarked as the robot's driver. A behavior task (behavior.py)
    supplies its own labels, data, accuracy and benchmark through the same methods."""

    names = ACTION_NAMES
    kind_note = ""

    def load(self, seeds, fmt, stride=1, limit=0, seed=0):
        return data(seeds, fmt, stride, limit, seed)

    def accuracy(self, pred, y):
        return dataset.accuracy(pred, y)

    def bench(self, run_dir, model_id, final=False):
        return benchmark.run(functools.partial(TextDriver, str(run_dir), model_id), workers=1, final=final)

    def text_cfg(self, a) -> dict:
        return {"format": a.format, "max_len": a.max_len, "precision": a.precision}


def main(model_id: str, default_base: str, family_note: str, argv=None, task=None) -> int:
    task = task or ActionTask()
    ACTION_NAMES = task.names  # noqa: N806 (the labels this run learns)
    a = parser(default_base).parse_args(argv)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    tr_text, tr_y = task.load(dataset.TRAIN, a.format, a.stride, a.limit, a.seed)
    va_text, va_y = task.load(dataset.VALID, a.format)
    tok = AutoTokenizer.from_pretrained(a.base)
    lens = [len(t) for t in tok(tr_text[:2000])["input_ids"]]
    print(f"{len(tr_y)} training, {len(va_y)} validation examples; tokens per text: median "
          f"{int(np.median(lens))}, max {max(lens)} (max_len {a.max_len})", flush=True)
    labels = {i: n for i, n in enumerate(ACTION_NAMES)}
    net = AutoModelForSequenceClassification.from_pretrained(
        a.base, num_labels=len(ACTION_NAMES), id2label=labels, label2id={n: i for i, n in labels.items()},
        ignore_mismatched_sizes=True, dtype=torch.float32)  # fp32 master weights: transformers 5 otherwise keeps
        # the checkpoint's dtype, and AdamW on fp16 weights (DeBERTa-v3) turns the first step into NaN
    if a.bf16_weights:
        net.to(torch.bfloat16)
    if a.freeze:
        freeze(net, a.freeze)
    if a.grad_ckpt:
        net.gradient_checkpointing_enable()
    dev = device()
    net.to(dev)
    amp = dev.type == "cuda" and a.precision == "bf16"
    counts = np.bincount(tr_y, minlength=len(ACTION_NAMES)).astype(np.float32)
    weight = torch.tensor(counts.sum() / np.maximum(counts, 1) / len(counts), device=dev) if a.weighted else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    params = [q for q in net.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    per_step = a.batch * a.accum
    steps_per_epoch = math.ceil(len(tr_y) / per_step)
    total = max(1, int(steps_per_epoch * a.epochs))
    sched = get_linear_schedule_with_warmup(opt, int(a.warmup * total), total)
    Y = torch.from_numpy(tr_y)
    run_dir = MODELS / model_id / "runs" / a.run
    ckpt = run_dir / "checkpoint.pt"
    step, train_s_before = 0, 0.0
    if a.resume and ckpt.exists():  # the weights, optimizer, schedule and position in the data
        state = torch.load(ckpt, map_location=dev, weights_only=False)
        net.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        step, train_s_before = state["step"], state["train_s"]
        print(f"resumed at step {step}/{total}", flush=True)

    def save_checkpoint():
        run_dir.mkdir(parents=True, exist_ok=True)
        tmp = ckpt.with_suffix(".tmp")
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                    "train_s": train_s_before + time.time() - start}, tmp)
        tmp.replace(ckpt)  # never a half-written checkpoint

    start = last_ckpt = time.time()
    stopped = False
    net.train()
    while step < total and not stopped:
        epoch, pos = divmod(step, steps_per_epoch)  # the order of each epoch is fixed by its seed, so a resume continues it
        order = torch.randperm(len(Y), generator=torch.Generator().manual_seed(a.seed * 1000 + epoch)).tolist()
        for b0 in range(pos * per_step, len(order), per_step):
            opt.zero_grad(set_to_none=True)
            chunk = order[b0:b0 + per_step]
            for m0 in range(0, len(chunk), a.batch):
                ids = chunk[m0:m0 + a.batch]
                enc = tok([tr_text[i] for i in ids], truncation=True, max_length=a.max_len, padding=True,
                          return_tensors="pt").to(dev)
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                    logits = net(**enc).logits
                loss = loss_fn(logits.float(), Y[ids].to(dev)) * len(ids) / len(chunk)
                if not torch.isfinite(loss):  # a NaN loss never recovers: stop before the run wastes its slot
                    print(f"non-finite loss at step {step}; try --precision fp32 or a lower --lr", flush=True)
                    return 3
                loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                print(f"step {step}/{total} loss {loss.item() * len(chunk) / len(ids):.3f} "
                      f"{time.time() - start:.0f} s", flush=True)
            if step >= total:
                break
            if time.time() - start > a.max_minutes * 60:
                stopped = True
                break
            if time.time() - last_ckpt > a.ckpt_minutes * 60:
                save_checkpoint()
                last_ckpt = time.time()
        save_checkpoint()  # each epoch end, the time cap, and the last step
        last_ckpt = time.time()
    done = step / steps_per_epoch
    train_s = train_s_before + time.time() - start
    if stopped:  # resume later with --resume; nothing is benchmarked or logged until the run is complete
        print(f"stopped at the time cap: step {step}/{total} ({done:.2f} epochs); continue with --resume", flush=True)
        return 2
    net.eval()
    (run_dir / "model").mkdir(parents=True, exist_ok=True)
    net.save_pretrained(run_dir / "model")
    tok.save_pretrained(run_dir / "model")
    (run_dir / "text_cfg.json").write_text(json.dumps(task.text_cfg(a)), encoding="utf-8")
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
        pred = predict(tok, net, va_text, a.max_len).numpy()
    offline = task.accuracy(pred, va_y)
    n_train = len(tr_y)
    # Free everything the benchmark does not need before it loads the saved model again (in this
    # process or a worker): the model, optimizer and schedule, the data and its tokenized
    # tensors, the validation texts. A large model otherwise holds about 8 GB of host memory here.
    net = opt = sched = params = loss_fn = weight = state = None
    tr_text = tr_y = va_text = va_y = Y = enc = logits = loss = order = chunk = ids = pred = lens = None
    tok = None
    import gc
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    settings = {k: v for k, v in vars(a).items() if k not in ("run", "notes", "no_bench")}
    settings["epochs_done"] = round(done, 2)
    entry = {"model": model_id, "run": a.run, "kind": "finetune", "base": a.base,
             "data": f"train seeds {dataset.TRAIN.start}-{dataset.TRAIN.stop - 1}, {n_train} examples "
                     f"(format {a.format}, stride {a.stride})",
             "params": settings, "train_s": round(train_s, 1), "offline": offline,
             "notes": "; ".join(x for x in (family_note, task.kind_note, a.notes) if x)}
    if not a.no_bench:
        summary = task.bench(run_dir, model_id)
        if summary is not None:
            entry.update(bench=summary, score=benchmark.score(summary))
    if a.final:
        final = task.bench(run_dir, model_id, final=True)
        if final is not None:
            entry.update(final=final, final_score=benchmark.score(final))
    tracker.log(entry)
    ckpt.unlink(missing_ok=True)  # the run is complete: the saved model is what remains (the optimizer state is large)
    print(a.run, "accuracy", offline["accuracy"], "score", entry.get("score"), flush=True)
    return 0
