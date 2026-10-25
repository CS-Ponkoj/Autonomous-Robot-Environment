"""Fine-tune microsoft/deberta-v3-small on the recorded scene texts, check it offline, benchmark it, log it.

    .venv\Scripts\python -m AD.models.deberta.train --run ft1 --epochs 2 --lr 5e-5 --max-len 128 --batch 32

All options: AD/common/textcls/train.py (format variants, class weights, stride, freezing, --resume)."""

from __future__ import annotations

from ...common.textcls.train import main as _main

BASE = "microsoft/deberta-v3-small"


def main(argv=None) -> int:
    return _main("deberta", BASE, "DeBERTa-v3-small fine-tuned on scene text", argv)


if __name__ == "__main__":
    raise SystemExit(main())
