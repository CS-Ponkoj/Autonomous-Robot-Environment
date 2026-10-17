"""Track B: behavior selection above the built-in driver. A text model reads the scene text and
the last behaviors, and picks one of 5 behaviors about twice a second; the driver does the
steering (the behavior wrapper in AD/common/behavior.py).

The interface this file expects from AD/common/behavior.py (the counterfactual labels and the
wrapper are built there):

    BEHAVIORS: tuple of the 5 names, in label order
    load(seeds) -> dict with "text" (list of scene texts), "label" (int array), "history"
        (int array, N x HISTORY: the previous behaviors, oldest first, -1 for none)
    run_selector(make_selector, final=False) -> a common.benchmark summary, where
        make_selector() returns an object with select(obs, history: list[str]) -> str

The model input is the formatted scene text, then " | last: " and the previous behaviors (the
same at training and at run time)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import formats
from .model import load as load_model, predict

HISTORY = 3
DEFAULT_BEHAVIORS = ("FOLLOW", "SLOW", "WAIT", "REPLAN", "RECOVER")


def _labels_module():
    from .. import behavior  # AD/common/behavior.py (ARE-1)
    return behavior


def behaviors() -> tuple:
    try:
        return tuple(_labels_module().BEHAVIORS)
    except (ImportError, AttributeError):
        return DEFAULT_BEHAVIORS


def with_history(text: str, history, names) -> str:
    """The model input: the scene text and the last HISTORY behaviors (oldest first)."""
    last = [names[h] if isinstance(h, (int, np.integer)) and h >= 0 else (h if isinstance(h, str) else "none")
            for h in list(history)[-HISTORY:]]
    last = ["none"] * (HISTORY - len(last)) + last
    return f"{text} | last: {' '.join(last)}"


class BehaviorTask:
    """The textcls.train task for behavior selection (see train.ActionTask)."""

    kind_note = "behavior selection (track B): scene text plus the last 3 behaviors"

    def __init__(self):
        self.names = behaviors()

    def load(self, seeds, fmt, stride=1, limit=0, seed=0):
        d = _labels_module().load(seeds)
        idx = np.arange(len(d["label"]))[::max(stride, 1)]
        if limit and len(idx) > limit:
            idx = np.sort(np.random.default_rng(seed).choice(idx, limit, replace=False))
        texts = [with_history(formats.apply(fmt, d["text"][i]), d["history"][i], self.names) for i in idx]
        return texts, np.asarray(d["label"])[idx].astype(np.int64)

    def accuracy(self, pred, y):
        out = {"accuracy": round(float((pred == y).mean()), 4), "n": int(len(y)), "per_behavior": {}}
        for i, name in enumerate(self.names):
            sel = y == i
            if sel.any():
                out["per_behavior"][name] = {"recall": round(float((pred[sel] == i).mean()), 3),
                                             "share": round(float(sel.mean()), 3)}
        return out

    def bench(self, run_dir, model_id, final=False):
        import functools
        mod = _labels_module()
        if not hasattr(mod, "run_selector"):
            print("no behavior benchmark yet (AD/common/behavior.py run_selector): offline only", flush=True)
            return None
        return mod.run_selector(functools.partial(BehaviorSelector, str(run_dir)), final=final)

    def text_cfg(self, a) -> dict:
        return {"format": a.format, "max_len": a.max_len, "precision": a.precision, "task": "behavior", "behaviors": list(self.names),
                "history": HISTORY}


class BehaviorSelector:
    """Picks a behavior for the wrapper: select(obs, history) -> name (GPU, one text per call)."""

    def __init__(self, run_dir: str):
        self.tok, self.net, self.cfg = load_model(run_dir)
        self.names = tuple(self.cfg["behaviors"])

    def select(self, obs, history) -> str:
        from ..features import scene_text
        text = with_history(formats.apply(self.cfg["format"], scene_text(obs)), history, self.names)
        return self.names[int(predict(self.tok, self.net, [text], self.cfg["max_len"])[0])]


def read_cfg(run_dir: str) -> dict:
    return json.loads((Path(run_dir) / "text_cfg.json").read_text(encoding="utf-8"))
