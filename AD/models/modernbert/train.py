"""Fine-tune answerdotai/ModernBERT-base on the recorded scene texts, check it offline, benchmark it, log it.

    .venv\Scripts\python -m AD.models.modernbert.train --run ft1 --epochs 2 --lr 5e-5 --max-len 128 --batch 32

All options: AD/common/textcls/train.py (format variants, class weights, stride, freezing, --resume)."""

from __future__ import annotations

from ...common.textcls.train import main as _main

BASE = "answerdotai/ModernBERT-base"


def main(argv=None) -> int:
    return _main("modernbert", BASE, "ModernBERT-base fine-tuned on scene text", argv)


if __name__ == "__main__":
    raise SystemExit(main())
