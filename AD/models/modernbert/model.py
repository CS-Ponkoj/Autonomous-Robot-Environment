"""The modernbert driver: a fine-tuned run (AD/models/modernbert/runs/<run>) picks one of the 8 actions."""

from __future__ import annotations

from ...common.textcls.model import TextDriver


class Driver(TextDriver):
    def __init__(self, run_dir: str):
        super().__init__(run_dir, name="modernbert")
