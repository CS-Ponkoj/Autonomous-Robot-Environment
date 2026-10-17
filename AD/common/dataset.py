"""The recorded examples, split by goal seed so validation layouts are never trained on."""

from __future__ import annotations

from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
TRAIN = range(10000, 10500)
VALID = range(10500, 10600)


def load(seeds, text: bool = False) -> dict:
    """Arrays for the given seeds (those recorded): x (N, FEATURES), cmd (N, 2), action (N,)
    relabelled from cmd with the current action set (features.ACTIONS), seed (N,), and with
    text=True the scene texts (list of N strings)."""
    from .features import FEATURE_VERSION, labels_of
    xs, cmds, acts, sd, texts = [], [], [], [], []
    for s in seeds:
        f = DATA / f"seed{s}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        if not len(d["cmd"]):
            continue
        version = int(d["feature_version"]) if "feature_version" in d.files else 1
        if version != FEATURE_VERSION:
            raise ValueError(f"{f.name} was recorded with feature layout {version}, not {FEATURE_VERSION}; record it again")
        xs.append(d["x"]), cmds.append(d["cmd"]), acts.append(labels_of(d["cmd"]))
        sd.append(np.full(len(d["cmd"]), s))
        if text:
            lines = (DATA / f"seed{s}.txt").read_text(encoding="utf-8").split("\n")
            if len(lines) != len(d["cmd"]):
                raise ValueError(f"seed{s}.txt has {len(lines)} lines for {len(d['cmd'])} examples")
            texts += lines
    if not xs:
        raise FileNotFoundError(f"no recordings for these seeds in {DATA}")
    out = {"x": np.concatenate(xs), "cmd": np.concatenate(cmds), "action": np.concatenate(acts),
           "seed": np.concatenate(sd)}
    if text:
        out["text"] = texts
    return out


def accuracy(pred: np.ndarray, action: np.ndarray) -> dict:
    """Overall and per-action accuracy (recall), plus the share of each action in the labels."""
    from .features import ACTION_NAMES
    out = {"accuracy": round(float((pred == action).mean()), 4), "n": int(len(action)), "per_action": {}}
    for i, name in enumerate(ACTION_NAMES):
        sel = action == i
        if sel.any():
            out["per_action"][name] = {"recall": round(float((pred[sel] == i).mean()), 3), "share": round(float(sel.mean()), 3)}
    return out
