"""Every training run and benchmark, logged, and the leaderboard table rebuilt from the log.

Each entry is one row of AD/private/experiments.jsonl (local only) and a copy in the model's
own folder (AD/<model>/runs/<run>/result.json). AD/LEADERBOARD.md (local only) is rebuilt from
the whole log after every entry, best first.

Several computers share the project folder through a synced drive, so each computer appends only
to its own log: the main computer (MAIN_HOST) to experiments.jsonl, any other to
experiments_<host>.jsonl. Reading merges them all. A held-out result for an earlier entry is
appended as a "final_of" record (applied to that entry when reading), so no computer rewrites a
file another one writes. Only the main computer rebuilds the board (one writer per file)."""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path

AD = Path(__file__).resolve().parent.parent
MAIN_HOST = "Ponkoj-PC"  # the computer that owns experiments.jsonl and the board
HOST = socket.gethostname()
LOG = AD / "private" / ("experiments.jsonl" if HOST == MAIN_HOST else f"experiments_{HOST}.jsonl")
BOARD = AD / "LEADERBOARD.md"


def log(entry: dict) -> dict:
    """entry: model, run, kind ("benchmark", "finetune"), and any of base, data, params,
    offline (accuracy on held-out recordings), bench (common.benchmark summary), score, notes."""
    entry = {"time": time.strftime("%Y-%m-%d %H:%M"), **entry, "host": HOST}
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    run_dir = AD / "models" / entry["model"] / "runs" / entry["run"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(json.dumps(entry, indent=1), encoding="utf-8")
    write_board()
    return entry


def entries() -> list[dict]:
    """Every computer's log merged (main first, then the others by name), with the held-out
    "final_of" records applied to their entries."""
    files = sorted(LOG.parent.glob("experiments*.jsonl"), key=lambda f: (f.name != "experiments.jsonl", f.name))
    rows = [json.loads(line) for f in files for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    out = [e for e in rows if e.get("kind") != "final_of"]
    for r in rows:
        if r.get("kind") == "final_of":
            for e in out:
                if e["model"] == r["model"] and e["run"] == r["run"]:
                    e["final"], e["final_score"] = r["final"], r["final_score"]
    return out


def cards() -> dict:
    """Each model folder's model.json: what the model is (family, size, inputs, license...)."""
    out = {}
    for f in sorted(AD.glob("models/*/model.json")):
        card = json.loads(f.read_text(encoding="utf-8"))
        out[card.get("id", f.parent.name)] = card
    return out


def _params(n) -> str:
    if not n:
        return "-"
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"


def _cell(v, fmt="{}"):
    return "-" if v is None else fmt.format(v)


def _goals(b, key):
    return f"{b[key]['success']}/{b[key]['of']}" if b and key in b else "-"


def write_board() -> None:
    if HOST != MAIN_HOST:
        return  # one writer per file: the main computer rebuilds the board (from every log)
    rows = entries()
    info = cards()
    benched = [e for e in rows if e.get("bench") or e.get("final")]

    def rank_key(e):
        f, d = e.get("final"), e.get("bench")
        return (f is not None, e.get("final_score", -9), e.get("score", -9))

    benched.sort(key=rank_key, reverse=True)
    lines = [
        "# Decision model leaderboard",
        "",
        f"Rebuilt {time.strftime('%Y-%m-%d %H:%M')} from `private/experiments.jsonl` ({len(rows)} entries). "
        "Best first: models with a final test score rank above those with only a practice score.",
        "",
        "- **Practice score:** practice layouts (goal seeds 2000-2009), without cats and with 4 cats "
        "(20 drives). Every training run gets one; tuning uses only this.",
        "- **Test score:** the held-out test layouts (goal seeds 1000-1009, 20 drives), run once per "
        "finished model.",
        "- **Score** = (goals reached - 2 x collisions) / drives, from -2 to 1 (1 = every goal, no crash). "
        "The benchmark is latency-aware: the world keeps moving while a model thinks.",
        "- **Action accuracy:** how often the model picks the built-in driver's action (8 actions) on "
        "recordings of layouts it never trained on (seeds 10500-10599).",
        "",
        "| Rank | Model | Size | Run | Test score | Test goals (no cats / 4 cats) | Practice score | Practice goals | "
        "Collisions | Timeouts | Action accuracy | Latency median / p95 (ms) | Notes |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, e in enumerate(benched, 1):
        f, d = e.get("final"), e.get("bench")
        main = f or d
        acc = (e.get("offline") or {}).get("accuracy")
        lines.append(
            f"| {i} | {e['model']} | {_params(info.get(e['model'], {}).get('params'))} | {e['run']} | "
            f"{_cell(e.get('final_score'), '{:.2f}')} | {_goals(f, 'no_cats')} / {_goals(f, '4_cats')} | "
            f"{_cell(e.get('score'), '{:.2f}')} | {_cell(d and d['success_total'], '{}')}{'/' + str(d['of_total']) if d else ''} | "
            f"{main['collisions_total']} | {main['timeouts_total']} | {_cell(acc, '{:.1%}')} | "
            f"{main['latency_ms']['median']} / {main['latency_ms']['p95']} | {e.get('notes', '')} |")
    if info:
        lines += ["", "## Models", "",
                  "| Model | Family | Parameters | Disk | Inputs | Outputs | Pretrained from | License | Source |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for mid, c in info.items():
            lines.append(f"| {mid} | {c.get('family', '-')} | {_params(c.get('params'))} | "
                         f"{_cell(c.get('disk_mb'), '{:.0f} MB')} | {c.get('inputs', '-')} | {c.get('outputs', '-')} | "
                         f"{c.get('pretrained', '-')} | {c.get('license', '-')} | {c.get('source', '-')} |")
    tuned = [e for e in rows if e.get("kind") == "finetune"]
    if tuned:
        lines += ["", "## Every fine-tune", "",
                  "| When | Model | Run | Base | Training data | Settings | Train time | Action accuracy | Practice score | Notes |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for e in tuned:
            o = e.get("offline") or {}
            lines.append(f"| {e['time']} | {e['model']} | {e['run']} | {e.get('base', '-')} | {e.get('data', '-')} | "
                         f"{json.dumps(e.get('params', {}))} | {_cell(e.get('train_s'), '{:.0f} s')} | "
                         f"{_cell(o.get('accuracy'), '{:.1%}')} | {_cell(e.get('score'), '{:.2f}')} | {e.get('notes', '')} |")
    BOARD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_final(model: str, run: str, final: dict, final_score: float) -> None:
    """Attach a held-out test result to an earlier entry (model, run): appended to this computer's
    log as a "final_of" record (entries() applies it), then the board is rebuilt."""
    rec = {"time": time.strftime("%Y-%m-%d %H:%M"), "kind": "final_of", "model": model, "run": run,
           "final": final, "final_score": final_score, "host": HOST}
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    write_board()
