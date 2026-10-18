"""The text-model driver: scene text (in the run's format) -> the most likely of the 8 actions."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from robot_env.types import Decision

from ...common.features import scene_text, to_command
from . import formats

_CACHE: dict[str, tuple] = {}  # one model per checkpoint and process (the benchmark makes a driver per episode)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(run_dir: str):
    """(tokenizer, model, settings) of a fine-tuned run: AD/models/<model>/runs/<run>/model plus text_cfg.json."""
    key = str(Path(run_dir).resolve())
    if key not in _CACHE:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        cfg = json.loads((Path(run_dir) / "text_cfg.json").read_text(encoding="utf-8"))
        tok = AutoTokenizer.from_pretrained(Path(run_dir) / "model")
        bf16 = device().type == "cuda" and cfg.get("precision", "bf16") == "bf16"
        dtype = torch.bfloat16 if bf16 else torch.float32
        net = AutoModelForSequenceClassification.from_pretrained(Path(run_dir) / "model", dtype=dtype)
        net.to(device()).eval()
        _CACHE[key] = (tok, net, cfg)
    return _CACHE[key]


@torch.no_grad()
def predict(tok, net, texts: list[str], max_len: int, batch: int = 256) -> torch.Tensor:
    """Action indices for already formatted texts."""
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc = {k: v.to(device()) for k, v in enc.items()}
        out.append(net(**enc).logits.float().argmax(-1).cpu())
    return torch.cat(out)


class TextDriver:
    """Formats the scene text as its run was trained, and picks the most likely action (GPU)."""

    def __init__(self, run_dir: str, name: str = "text"):
        self.name = name
        self.tok, self.net, self.cfg = load(run_dir)

    def decide(self, obs):
        text = formats.apply(self.cfg["format"], scene_text(obs))
        action = int(predict(self.tok, self.net, [text], self.cfg["max_len"])[0])
        return Decision(to_command(action), obs.seq)
