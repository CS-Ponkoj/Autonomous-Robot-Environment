"""Track B for deberta: fine-tune it to pick one of the 5 behaviors above the built-in driver, from
the scene text and the last 3 behaviors (AD/common/textcls/behavior.py), check it offline,
benchmark it through the behavior wrapper, log it.

    .venv\Scripts\python -m AD.models.deberta.behavior --run bh1 --epochs 2 --lr 5e-5 --max-len 64 --batch 32

The same options as train.py (AD/common/textcls/train.py)."""

from __future__ import annotations

from ...common.textcls.behavior import BehaviorTask
from ...common.textcls.train import main as _main
from .train import BASE


def main(argv=None) -> int:
    return _main("deberta", BASE, "behavior selection head on the deberta encoder", argv, task=BehaviorTask())


if __name__ == "__main__":
    raise SystemExit(main())
