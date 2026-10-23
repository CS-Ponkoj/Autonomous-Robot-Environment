"""Text plus numbers: a pretrained text encoder reads a short scene text while the 72 lidar bins
and the route values enter as numbers, through a small numeric branch whose output becomes a few
extra input embeddings (a prefix) of the encoder; a head gives (v, omega) or one of the 8 actions.

Every input comes from one 82-value vector: AD.common.features.vector(obs) (78) and a route
source's 4 values (route_sin, route_cos, route_known, route_len; AD/models/hybrid/route.py). The
recordings, the DAgger states and the live driver all go through the same functions here.

    encoder input:  [CLS] + PREFIX numeric embeddings + the text tokens (inputs_embeds)
    numeric branch: circular 1D conv over the 72 bins (as AD/models/cnn) + an MLP over the goal,
                    contact and route values
    head:           mean of the encoder's outputs (and, with skip, the numeric features)

The measured speed and turn rate are never used: a cloned policy that sees them learns to keep
doing what the wheels do."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from robot_env import config as C
from robot_env.types import Command, Decision, Observation

from ..features import ACTION_NAMES, FEATURES, LABEL_LEVEL, LIDAR_BINS, to_command, vector

ROUTE = 4
INPUTS = FEATURES + ROUTE  # 82
GOAL_D, GOAL_SIN, GOAL_COS, CONTACT = LIDAR_BINS, LIDAR_BINS + 1, LIDAR_BINS + 2, LIDAR_BINS + 5
R_SIN, R_COS, R_KNOWN, R_LEN = FEATURES, FEATURES + 1, FEATURES + 2, FEATURES + 3
SCALAR_IDX = [GOAL_D, GOAL_SIN, GOAL_COS, CONTACT, R_SIN, R_COS, R_KNOWN, R_LEN]  # no v, omega
V_CAP, W_CAP = C.SPEED_LEVELS[LABEL_LEVEL]  # 0.30 m/s, 1.0 rad/s
MODELS = Path(__file__).resolve().parents[2] / "models"
BASES = {  # model folder -> the pretrained encoder it starts from
    "modernbert": "answerdotai/ModernBERT-base",
    "deberta": "microsoft/deberta-v3-small",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "modernbert_large": "answerdotai/ModernBERT-large",
}


def base_of(model: str) -> str:
    """The encoder a model folder starts from: BASES, else a converted checkpoint in
    AD/models/<model>/base."""
    if model in BASES:
        return BASES[model]
    local = MODELS / model / "base"
    if (local / "config.json").exists():
        return str(local)
    raise SystemExit(f"no encoder for {model!r}: not in textnum.BASES and no AD/models/{model}/base")


def text_of(x: np.ndarray, len_scale: float = 10.0) -> str:
    """The short scene text of one 82-value input: the goal, contact and route, in words
    (len_scale: the route source's route_len scale, metres per unit)."""
    deg = math.degrees(math.atan2(float(x[GOAL_SIN]), float(x[GOAL_COS])))
    route = f"route known {float(x[R_LEN]) * len_scale:.1f} m" if x[R_KNOWN] > 0.5 else "route unknown"
    return (f"goal {float(x[GOAL_D]) * 10:.1f} m at {deg:.0f} deg; "
            f"contact {'yes' if x[CONTACT] > 0.5 else 'none'}; {route}")


def no_route(x78: np.ndarray) -> np.ndarray:
    """82-value inputs for recordings without route values: the route source's no-route reading
    (the goal direction, route unknown, length 0)."""
    r = np.zeros((len(x78), ROUTE), np.float32)
    r[:, 0], r[:, 1] = x78[:, GOAL_SIN], x78[:, GOAL_COS]
    return np.concatenate([x78, r], 1)


def route_source(name: str):
    """(source, spec) by name (AD/models/hybrid/route.py); spec {"name", "len_scale"} goes into
    the checkpoint, so a model is always run with the source it was trained with."""
    from ...models.hybrid import route
    if hasattr(route, "make_route_source"):
        src = route.make_route_source(name)
        return src, route.route_spec(src)
    if name != "online":
        raise ValueError(f"route source {name!r} needs route.make_route_source")
    return route.RouteSensor(), {"name": "online", "len_scale": 10.0}


class TextNum(nn.Module):
    def __init__(self, encoder, head: str = "continuous", prefix: int = 4, width: int = 32, skip: bool = True):
        super().__init__()
        assert head in ("continuous", "discrete")
        self.encoder, self.head_kind, self.n_prefix, self.width, self.skip = encoder, head, prefix, width, skip
        hidden = encoder.config.hidden_size
        w = width

        def conv(cin, cout, k, stride=1):  # the scan is a ring, and so is the padding
            return nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2, padding_mode="circular")
        self.lidar = nn.Sequential(conv(1, w, 5), nn.ReLU(), conv(w, 2 * w, 5, 2), nn.ReLU(),
                                   conv(2 * w, 2 * w, 3, 2), nn.ReLU(),
                                   nn.Flatten(), nn.Linear(2 * w * (LIDAR_BINS // 4), 128), nn.ReLU())
        self.scalars = nn.Sequential(nn.Linear(len(SCALAR_IDX), 64), nn.ReLU())
        self.prefix = nn.Linear(192, prefix * hidden)
        self.out = nn.Sequential(nn.Linear(hidden + (192 if skip else 0), 128), nn.ReLU(),
                                 nn.Linear(128, 2 if head == "continuous" else len(ACTION_NAMES)))
        self.register_buffer("caps", torch.tensor([V_CAP, W_CAP]))
        self.register_buffer("scalar_idx", torch.tensor(SCALAR_IDX), persistent=False)

    def numeric(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.lidar(x[:, None, :LIDAR_BINS]), self.scalars(x[:, self.scalar_idx])], 1)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """ids, mask: the tokenized texts (B, L), [CLS] first; x: (B, 82). Continuous: (v, omega)
        inside the caps; discrete: logits over the 8 actions."""
        z = self.numeric(x.float())
        emb = self.encoder.get_input_embeddings()(ids)
        pre = self.prefix(z).view(len(z), self.n_prefix, -1).to(emb.dtype)
        e = torch.cat([emb[:, :1], pre, emb[:, 1:]], 1)
        m = torch.cat([mask[:, :1], torch.ones(len(z), self.n_prefix, dtype=mask.dtype, device=mask.device), mask[:, 1:]], 1)
        h = self.encoder(inputs_embeds=e, attention_mask=m).last_hidden_state
        mf = m[..., None].to(h.dtype)
        pooled = (h * mf).sum(1) / mf.sum(1)
        y = self.out(torch.cat([pooled.float(), z], 1) if self.skip else pooled.float())
        return torch.tanh(y) * self.caps if self.head_kind == "continuous" else y

    def cfg(self) -> dict:
        return {"head": self.head_kind, "prefix": self.n_prefix, "width": self.width, "skip": self.skip}


def build(base: str, **kw) -> TextNum:
    from transformers import AutoModel
    return TextNum(AutoModel.from_pretrained(base, dtype=torch.float32), **kw)  # fp32 master weights


def save(net: TextNum, tok, run_dir: Path, extra: dict) -> None:
    """The encoder (any adapters merged in) and tokenizer in run_dir/encoder, the rest in head.pt,
    the settings in textnum.json."""
    run_dir.mkdir(parents=True, exist_ok=True)
    enc = net.encoder.merge_and_unload() if hasattr(net.encoder, "merge_and_unload") else net.encoder
    enc.save_pretrained(run_dir / "encoder")
    tok.save_pretrained(run_dir / "encoder")
    torch.save({k: v for k, v in net.state_dict().items() if not k.startswith("encoder.")}, run_dir / "head.pt")
    (run_dir / "textnum.json").write_text(json.dumps({**net.cfg(), **extra}), encoding="utf-8")


def autocast_dtype(device, precision: str = "auto"):
    """The autocast dtype for training and driving alike: None (fp32) off the GPU or with fp32;
    with auto, bf16 where the GPU has it natively (not emulated), else fp16."""
    if torch.device(device).type != "cuda" or precision == "fp32":
        return None
    if precision == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]


def load_head(net: TextNum, path) -> None:
    """head.pt (everything but the encoder) into net; any other missing or extra key is an error."""
    res = net.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    missing = [k for k in res.missing_keys if not k.startswith("encoder.")]
    assert not missing and not res.unexpected_keys, f"{path}: missing {missing}, unexpected {res.unexpected_keys}"


_CACHE: dict[str, tuple] = {}  # one model per run and process (the benchmark makes a driver per episode)


def load(run_dir, device: str):
    """(tokenizer, model, settings) of a finished run. The weights stay fp32 and the driver runs
    them under autocast_dtype, as in training (infer_fp32 in textnum.json: plain fp32)."""
    key = (str(Path(run_dir).resolve()), device)
    if key not in _CACHE:
        from transformers import AutoModel, AutoTokenizer
        run_dir = Path(run_dir)
        cfg = json.loads((run_dir / "textnum.json").read_text(encoding="utf-8"))
        enc = AutoModel.from_pretrained(run_dir / "encoder", dtype=torch.float32)
        net = TextNum(enc, cfg["head"], cfg["prefix"], cfg["width"], cfg["skip"])
        load_head(net, run_dir / "head.pt")
        _CACHE[key] = (AutoTokenizer.from_pretrained(run_dir / "encoder"), net.to(device).eval(), cfg)
    return _CACHE[key]


def tokenize(tok, x: np.ndarray, max_len: int, device, len_scale: float = 10.0):
    enc = tok([text_of(r, len_scale) for r in x], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


@torch.no_grad()
def predict(tok, net: TextNum, x: np.ndarray, max_len: int, device, len_scale: float = 10.0,
            batch: int = 256, dtype=None) -> np.ndarray:
    """(N, 2) commands (continuous) or (N, 8) logits (discrete) for 82-value inputs, under
    autocast in dtype (None: as the caller has set it, fp32 by default)."""
    out = []
    dev_type = torch.device(device).type
    for i in range(0, len(x), batch):
        ids, mask = tokenize(tok, x[i:i + batch], max_len, device, len_scale)
        with torch.autocast(dev_type, dtype=dtype or torch.float32, enabled=dtype is not None):
            out.append(net(ids, mask, torch.from_numpy(x[i:i + batch]).to(device)).float().cpu())
    return torch.cat(out).numpy()


def command_of(net: TextNum, y: np.ndarray) -> Command:
    if net.head_kind == "discrete":
        return to_command(int(np.argmax(y)))
    return Command(float(y[0]), float(y[1]))


class TextNumDriver:
    """A trained text-plus-numbers model as a robot_env driver: the route source reads every
    observation, the model decides (GPU if any). last_x is the latest 82-value input (DAgger)."""

    def __init__(self, run_dir: str, name: str = "textnum", device: str | None = None, route: str | None = None):
        """route: drive with another route source than the run's (DAgger moving a model to a new
        source); the text then states that source's route lengths."""
        torch.set_num_threads(1)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok, self.net, self.cfg = load(run_dir, self.device)
        self.route, self.spec = route_source(route or self.cfg.get("route", {"name": "online"})["name"])
        self.len_scale = float(self.spec["len_scale"])
        self.dtype = None if self.cfg.get("infer_fp32", False) else autocast_dtype(self.device)
        self.name = f"{name}:{Path(run_dir).name}"
        self.last_x = None

    def reset(self, initial_pose=None, **kw) -> None:
        """initial_pose: the task's start pose (x, y, yaw), for route sources that need it."""
        import inspect
        if "initial_pose" in inspect.signature(self.route.reset).parameters:
            self.route.reset(initial_pose=initial_pose, **kw)
        else:  # the older RouteSensor (online only: it needs no pose)
            self.route.reset()

    def decide(self, obs: Observation) -> Decision:
        x = np.concatenate([vector(obs), np.asarray(self.route.read(obs), np.float32)])
        self.last_x = x
        y = predict(self.tok, self.net, x[None], self.cfg["max_len"], self.device, self.len_scale, dtype=self.dtype)[0]
        return Decision(command_of(self.net, y), obs.seq)
