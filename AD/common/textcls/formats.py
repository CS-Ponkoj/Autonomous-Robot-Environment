"""Scene-text variants for the format experiments. Every variant is computed from the recorded
scene text (common.features.scene_text, "key: value; key: value"), so training data and the
driver at run time always see the same format."""

from __future__ import annotations

SHORT_KEYS = {  # shorter labels: fewer tokens, same information
    "front_min_m": "F", "front_left_min_m": "FL", "left_min_m": "L", "back_left_min_m": "BL",
    "rear_min_m": "B", "back_right_min_m": "BR", "right_min_m": "R", "front_right_min_m": "FR",
    "unknown_rays": "unk", "forward_speed_m_s": "v", "turn_rate_rad_s": "w",
    "goal_distance_m": "goal_d", "goal_bearing_deg": "goal_deg", "contact": "hit",
}
GOAL_KEYS = ("goal_distance_m", "goal_bearing_deg")


def pairs(text: str) -> list[tuple[str, str]]:
    out = []
    for part in text.split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            out.append((k.strip(), v.strip()))
    return out


def full(text: str) -> str:
    """The scene text exactly as recorded."""
    return text


def short(text: str) -> str:
    """Short keys, no units in the names."""
    return " ".join(f"{SHORT_KEYS.get(k, k)}={v}" for k, v in pairs(text))


def goal_first(text: str) -> str:
    """The goal first (the most decisive fields), then the ranges, then the rest."""
    p = pairs(text)
    goal = [kv for kv in p if kv[0] in GOAL_KEYS]
    rest = [kv for kv in p if kv[0] not in GOAL_KEYS]
    return "; ".join(f"{k}: {v}" for k, v in goal + rest)


def sentence(text: str) -> str:
    """Plain English, for models pretrained on prose and instruction-style input."""
    d = dict(pairs(text))
    ranges = ", ".join(f"{k.replace('_min_m', '').replace('_', ' ')} {d[k]}" for k in (
        "front_min_m", "front_left_min_m", "left_min_m", "back_left_min_m", "rear_min_m",
        "back_right_min_m", "right_min_m", "front_right_min_m") if k in d)
    return (f"The goal is {d.get('goal_distance_m', '?')} m away at {d.get('goal_bearing_deg', '?')} degrees "
            f"(positive is to the left). Nearest obstacle in metres by direction: {ranges}. "
            f"Speed {d.get('forward_speed_m_s', '?')} m/s, turning {d.get('turn_rate_rad_s', '?')} rad/s, "
            f"unknown rays {d.get('unknown_rays', '?')}, contact {d.get('contact', '?')}.")


# The robot's own motion (its measured speed and turn rate): the reference driver's next command
# almost always equals its current motion, so a model that sees these can learn "keep doing what
# the wheels do" and score well offline while it cannot drive (it spins in place). The *_nospeed
# variants leave both fields out.
MOTION_KEYS = ("forward_speed_m_s", "turn_rate_rad_s")


def _drop_motion(text: str) -> str:
    return "; ".join(f"{k}: {v}" for k, v in pairs(text) if k not in MOTION_KEYS)


def nospeed(text: str) -> str:
    """The recorded scene text without the robot's speed and turn rate."""
    return _drop_motion(text)


def short_nospeed(text: str) -> str:
    return short(_drop_motion(text))


def goal_first_nospeed(text: str) -> str:
    return goal_first(_drop_motion(text))


def sentence_nospeed(text: str) -> str:
    """Plain English without the robot's speed and turn rate."""
    return sentence(text).split(" Speed ", 1)[0] + (
        " Unknown rays {}, contact {}.".format(*[dict(pairs(text)).get(k, "?") for k in ("unknown_rays", "contact")]))


FORMATS = {"full": full, "short": short, "goal_first": goal_first, "sentence": sentence,
           "nospeed": nospeed, "short_nospeed": short_nospeed, "goal_first_nospeed": goal_first_nospeed,
           "sentence_nospeed": sentence_nospeed}


def apply(name: str, text: str) -> str:
    return FORMATS[name](text)
