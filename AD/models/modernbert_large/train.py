"""Fine-tune answerdotai/ModernBERT-large on the recorded scene texts, check it offline,
benchmark it, log it: the control for the decision-head fine-tunes (the same encoder, plainly fine-tuned),
so compare it at the same settings as the matching decision-head run.

    .venv\Scripts\python -m AD.models.modernbert_large.train --run ft1 --epochs 1 --lr 2e-5 --max-len 96 --batch 16 --stride 3 --grad-ckpt --freeze 14

The 395M encoder needs --grad-ckpt and --freeze N on an 8 GB GPU (or --bf16-weights).
All options: AD/common/textcls/train.py."""

from __future__ import annotations

from ...common.textcls.train import main as _main

BASE = "answerdotai/ModernBERT-large"


def main(argv=None) -> int:
    return _main("modernbert_large", BASE, "ModernBERT-large fine-tuned on scene text (control for the decision-head model)", argv)


if __name__ == "__main__":
    raise SystemExit(main())
