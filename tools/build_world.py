"""Generate robot_env/world.xml: an indoor office floor and the robot.

    .venv\\Scripts\\python tools\\build_world.py

The floor layout lives here (one source of truth for geometry); room regions are
mirrored in robot_env/config.py (ROOMS), which a test checks against this file.
Everything that collides is a box or cylinder with the same visible shape.

Geom groups: 0 solid (seen by lidar, collides), 2 visual only, 3 ceiling (visual
only, shown in the robot camera, hidden in overview cameras), 4 hidden colliders (the
robot's, and the solid members of furniture shown as a detailed mesh).

Furniture shown as a detailed mesh (robot_env/assets/furniture: CC0 models from Poly Haven
converted by tools/fetch_models.py, and procedural meshes) is built with item(): the mesh is
visual only, and simple boxes and cylinders fitted to it (its members, hidden) are what
collides, what the lidar sees, and what the maps use. tools/furniture_check.py checks that the
members and the mesh agree.
"""

import json
import math
from contextlib import contextmanager
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "robot_env" / "world.xml"
FURNITURE = Path(__file__).resolve().parent.parent / "robot_env" / "assets" / "furniture"

H = 5.0  # inner half size of the floor (walls' inner faces at +/-5 m)
T = 0.1  # wall thickness
WH = 2.4  # wall height
DOOR_GAP = 1.0  # opening in the wall
JAMB = 0.05  # door frame on each side, inside the gap: clear width 0.9 m
DOOR_HEAD = 2.1  # top of the doorway
CORRIDOR = 0.7  # corridor half width (clear width 1.4 m)

geoms: list[str] = []
furniture_assets: list[str] = []  # <asset> entries (meshes, textures, materials) of the furniture meshes
furniture_items: list[dict] = []  # every item(): its members and meshes (for tools/furniture_check.py)
placed_models: list[dict] = []  # every model() placed, inside an item or not (owner: the item's name or None)
_assets_done: set = set()
_item: dict | None = None  # the item being built


@contextmanager
def item(name, support=0.0):
    """A piece of furniture shown by detailed meshes (model()). The named boxes and cylinders made
    inside are its solid members: hidden (group 4) but solid, seen by the lidar and the maps.
    `support` is the height the meshes stand on (the floor, or a desk top)."""
    global _item
    _item = {"name": name, "support": support, "members": [], "meshes": []}
    try:
        yield _item
    finally:
        furniture_items.append(_item)
        _item = None


def model(model_id, pos, yaw=0.0):
    """A detailed furniture mesh (visual only) at its real size: the model's base centre at pos,
    turned yaw degrees about z."""
    rec = json.loads((FURNITURE / f"{model_id}.json").read_text(encoding="utf-8"))
    if model_id not in _assets_done:
        _assets_done.add(model_id)
        for k, part in enumerate(rec["parts"]):
            if part["texture"]:
                furniture_assets.append(f'    <texture name="{model_id}_t{k}" type="2d" file="{part["texture"]}"/>')
                look = f'texture="{model_id}_t{k}"'
            else:
                look = 'rgba="' + " ".join(f"{c:.3f}" for c in part["rgba"]) + '"'
            furniture_assets.append(f'    <material name="{model_id}_m{k}" {look} specular="{part["specular"]}" '
                                    f'shininess="{part["shininess"]}" reflectance="{part["reflectance"]}"/>')
            furniture_assets.append(f'    <mesh name="{model_id}_{k}" file="{part["mesh"]}" inertia="shell"/>')
    for k in range(len(rec["parts"])):
        name = f'name="{_item["name"]}_{model_id}_{k}" ' if _item is not None else ""
        geoms.append(f'    <geom {name}class="visual" type="mesh" mesh="{model_id}_{k}" material="{model_id}_m{k}" '
                     f'pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" euler="0 0 {yaw:.1f}"/>')
    placed = {"model": model_id, "pos": [round(float(c), 4) for c in pos], "yaw": float(yaw)}
    placed_models.append({**placed, "item": _item["name"] if _item is not None else None})
    if _item is not None:
        _item["meshes"].append(placed)


def members(name, pos, yaw, parts, material="metal"):
    """Solid members given in a model's own frame (as model() places it at pos, turned yaw):
    ("box", suffix, (cx, cy, cz), (hx, hy, hz)[, own yaw degrees]) or
    ("cyl", suffix, (cx, cy, cz), r, half_h)."""
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    for part in parts:
        kind, suffix, (mx, my, mz) = part[0], part[1], part[2]
        wx, wy = pos[0] + c * mx - s * my, pos[1] + s * mx + c * my
        if kind == "box":
            turn = (yaw + (part[4] if len(part) > 4 else 0.0)) % 360
            box(f"{name}_{suffix}", (wx, wy, pos[2] + mz), part[3], material, euler=round(turn, 4) if turn else None)
        else:
            cyl(f"{name}_{suffix}", (wx, wy, pos[2] + mz), part[3], part[4], material)


def box(name, pos, half, material=None, rgba=None, cls=None, euler=None, group=None):
    if _item is not None and cls is None:
        group = 4  # a member: solid, hidden behind the item's mesh
        _item["members"].append(name)
    attrs = [f'name="{name}"' if name else "", 'type="box"',
             f'pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"',
             f'size="{half[0]:.4f} {half[1]:.4f} {half[2]:.4f}"']
    if euler:
        attrs.append(f'euler="0 0 {euler}"')
    if material:
        attrs.append(f'material="{material}"')
    if rgba:
        attrs.append(f'rgba="{rgba}"')
    if cls:
        attrs.append(f'class="{cls}"')
    if group is not None:
        attrs.append(f'group="{group}"')
    geoms.append("    <geom " + " ".join(a for a in attrs if a) + "/>")


def cyl(name, pos, radius, half_h, material=None, rgba=None, cls=None):
    attrs = [f'name="{name}"' if name else "", 'type="cylinder"',
             f'pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"', f'size="{radius:.4f} {half_h:.4f}"']
    if _item is not None and cls is None:
        attrs.append('group="4"')  # a member: solid, hidden behind the item's mesh
        _item["members"].append(name)
    if material:
        attrs.append(f'material="{material}"')
    if rgba:
        attrs.append(f'rgba="{rgba}"')
    if cls:
        attrs.append(f'class="{cls}"')
    geoms.append("    <geom " + " ".join(a for a in attrs if a) + "/>")


def shelf_faces(center, half, levels=4, seed=0):
    """Visual shelf boards and cartons on both long faces of a solid shelving unit.
    Details sit on the solid's surface (1 mm proud), inside the visible envelope."""
    import random
    rnd = random.Random(seed)
    cx, cy, cz = center
    hx, hy, hz = half
    long_x = hx > hy
    for side in (-1, 1):
        face = (cy + side * (hy + 0.001)) if long_x else (cx + side * (hx + 0.001))
        for k in range(1, levels + 1):
            z = cz - hz + k * (2 * hz) / (levels + 1)
            if long_x:
                box(None, (cx, face, z), (hx, 0.001, 0.012), "shelf_board", cls="visual")
            else:
                box(None, (face, cy, z), (0.001, hy, 0.012), "shelf_board", cls="visual")
            span = hx if long_x else hy
            pos = -span + 0.08
            while pos < span - 0.15:
                w = rnd.uniform(0.08, 0.16)
                h = rnd.uniform(0.08, 0.16)
                mat = rnd.choice(["cardboard", "cardboard", "bin_blue", "bin_grey"])
                c = (cx + pos + w, face, z + 0.012 + h) if long_x else (face, cy + pos + w, z + 0.012 + h)
                hw = (w, 0.001, h) if long_x else (0.001, w, h)
                box(None, c, hw, mat, cls="visual")
                pos += 2 * w + rnd.uniform(0.03, 0.1)


def cabinet_seams(center, half, side_axis, side, doors):
    """Visual door seams and handles on one face of a solid cabinet."""
    cx, cy, cz = center
    hx, hy, hz = half
    span = hx if side_axis == "y" else hy
    for i in range(1, doors):
        off = -span + i * 2 * span / doors
        if side_axis == "y":
            box(None, (cx + off, cy + side * (hy + 0.001), cz), (0.003, 0.001, hz * 0.85), "seam", cls="visual")
        else:
            box(None, (cx + side * (hx + 0.001), cy + off, cz), (0.001, 0.003, hz * 0.85), "seam", cls="visual")
    for i in range(doors):
        off = -span + (i + 0.5) * 2 * span / doors + 0.06
        if side_axis == "y":
            box(None, (cx + off, cy + side * (hy + 0.002), cz + hz * 0.55), (0.03, 0.002, 0.006), "handle", cls="visual")
        else:
            box(None, (cx + side * (hx + 0.002), cy + off, cz + hz * 0.55), (0.002, 0.03, 0.006), "handle", cls="visual")


def shelving_unit(name, center, half, height, levels, seed, items="cartons"):
    """Open shelving with honest collision: a solid plinth (0 to 0.16 m, seen by the
    lidar), solid corner posts and shelf boards, and visual items on the shelves."""
    import random
    rnd = random.Random(seed)
    cx, cy = center
    hx, hy = half
    along_y = hy >= hx
    box(f"{name}_plinth", (cx, cy, 0.08), (hx, hy, 0.08), "metal_dark")
    for sx in (-1, 1):
        for sy in (-1, 1):
            box(f"{name}_post_{'n' if sy > 0 else 's'}{'e' if sx > 0 else 'w'}",
                (cx + sx * (hx - 0.015), cy + sy * (hy - 0.015), height / 2), (0.015, 0.015, height / 2), "metal")
    tops = [0.16] + [0.16 + k * (height - 0.2) / levels for k in range(1, levels + 1)]
    for k, z in enumerate(tops[1:], 1):
        box(f"{name}_board_{k}", (cx, cy, z - 0.01), (hx, hy, 0.01), "metal")
    span = (hy if along_y else hx) - 0.04
    depth = (hx if along_y else hy) - 0.03
    for k in range(len(tops) - 1):
        z0, z1 = tops[k], tops[k + 1] - 0.02
        pos = -span
        while True:
            if items == "books":
                w, d, h = rnd.uniform(0.008, 0.02), depth * rnd.uniform(0.7, 0.95), (z1 - z0) * rnd.uniform(0.6, 0.85) / 2
                mat = rnd.choice(["book_red", "book_blue", "book_green", "book_cream", "book_dark"])
            else:
                w, d, h = rnd.uniform(0.06, 0.15), depth * rnd.uniform(0.6, 0.95), min((z1 - z0) / 2 - 0.01, rnd.uniform(0.06, 0.16))
                mat = rnd.choice(["cardboard", "cardboard", "cardboard_dark", "bin_blue", "bin_grey"])
            if pos + 2 * w > span:
                break
            off = pos + w
            c = (cx + (0 if along_y else off), cy + (off if along_y else 0), z0 + h)
            box(None, c, (d, w, h) if along_y else (w, d, h), mat, cls="visual")
            pos += 2 * w + (rnd.uniform(0.0, 0.01) if items == "books" else rnd.uniform(0.02, 0.08))


OFFICE_DESK_TOP = 0.788  # metal_office_desk: top surface height


def office_desk(name, pos, yaw=0.0):
    """Steel office desk (metal_office_desk, 2.0 x 0.95 m): two drawer pedestals on short legs,
    a modesty panel at the back, and a 0.96 m kneehole open at the front (-y in its frame)."""
    with item(name):
        model("metal_office_desk", pos, yaw)
        legs = [("box", f"leg_{i}", (sx * 0.725, sy, 0.093), (0.025, 0.025, 0.093))
                for i, (sx, sy) in enumerate(((-1, -0.344), (1, -0.344), (-1, 0.351), (1, 0.351)))]
        members(name, pos, yaw, legs + [
            ("box", "ped_l", (-0.7325, 0.005, 0.468), (0.2525, 0.44, 0.282)),
            ("box", "ped_r", (0.7325, 0.005, 0.468), (0.2525, 0.44, 0.282)),
            ("box", "panel", (0.0, 0.435, 0.468), (0.48, 0.01, 0.282)),
            ("box", "top", (0.0, 0.015, 0.769), (1.0, 0.46, 0.019)),
        ] + [("box", f"handle_{i}", (sx * 0.7175, -0.4555, z), (0.0575, 0.0185, 0.005))  # drawer pulls
             for i, (sx, z) in enumerate(((-1, 0.2935), (-1, 0.4555), (-1, 0.6185),
                                          (1, 0.2935), (1, 0.4585), (1, 0.6215)))], "desk_body")


def _turned(pos, yaw, local):
    """A point given in a model's frame, in world coordinates (as model() places the model)."""
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    return (pos[0] + c * local[0] - s * local[1], pos[1] + s * local[0] + c * local[1], pos[2] + local[2])


SHELF_FEET = 0.033  # levelling feet: the bottom frame then spans 0.123 to 0.148 m, across the lidar plane


def steel_shelf(name, pos, yaw=0.0, books_seed=None):
    """Open steel-frame shelving with wooden boards (steel_frame_shelves_01, 1.10 x 0.50 x 2.14 m) on
    four levelling feet (proc_shelf_feet): four corner posts; at each level a steel frame the full
    width with a wooden board on it, set between the posts (the top level is the frame only)."""
    with item(name):
        model("proc_shelf_feet", pos, yaw)
        members(name, pos, yaw, [("cyl", f"foot_{i}", (sx * 0.534, sy * 0.236, SHELF_FEET / 2), 0.017, SHELF_FEET / 2)
                                 for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1)))], "metal")
        pos = (pos[0], pos[1], pos[2] + SHELF_FEET)
        model("steel_frame_shelves_01", pos, yaw)
        posts = [("box", f"post_{i}", (sx * 0.534, sy * 0.236, 1.07), (0.0165, 0.0165, 1.07))
                 for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1)))]
        levels = ((0.0898, 0.1145, 0.1338), (0.5953, 0.6195, 0.6396), (1.1019, 1.127, 1.146),
                  (1.608, 1.6345, 1.6522), (2.1118, 2.1308, None))  # measured: frame, board top
        boards = []
        for k, (z0, z1, z2) in enumerate(levels):
            boards.append(("box", f"frame_{k}", (-0.0015, -0.001, (z0 + z1) / 2), (0.5475, 0.25, (z1 - z0) / 2)))
            if z2 is not None:
                boards.append(("box", f"board_{k}", (-0.0015, -0.001, (z1 + z2) / 2), (0.5175, 0.25, (z2 - z1) / 2)))
        members(name, pos, yaw, posts + boards, "metal")
        if books_seed is not None:
            model(f"proc_shelf_books_{books_seed}", pos, yaw)  # tools/proc_furniture.py shelf_books()


PLANT_STAND_TOP = 0.22  # proc_plant_stand: its top board's surface

# potted_plant_01: the planter's centre in the model's frame, and stacked cylinders fitted to it by
# tools/fit_round.py (potted_plant_01 0.53 0.0145): each band's largest radius, and within a band the
# member is at most 1.45 cm from the planter's nearest surface (furniture_check: 2 cm). The leaves
# start at 0.55 m, above any actor, and are visual only.
PLANTER_CENTRE = (0.0448, -0.0482)
PLANTER_BANDS = ((0.0, 0.025, 0.1991), (0.025, 0.03, 0.1935), (0.03, 0.0375, 0.1861), (0.0375,
                 0.045, 0.1775), (0.045, 0.0525, 0.169), (0.0525, 0.0575, 0.1617), (0.0575, 0.0625,
                 0.1557), (0.0625, 0.0675, 0.1506), (0.0675, 0.0725, 0.1456), (0.0725, 0.0775,
                 0.1395), (0.0775, 0.0825, 0.135), (0.0825, 0.0875, 0.1289), (0.0875, 0.09, 0.1256),
                 (0.09, 0.1, 0.1233), (0.1, 0.1075, 0.116), (0.1075, 0.1275, 0.1156), (0.1275,
                 0.1375, 0.1188), (0.1375, 0.1425, 0.129), (0.1425, 0.155, 0.1309), (0.155, 0.1625,
                 0.1358), (0.1625, 0.17, 0.1409), (0.17, 0.1775, 0.1456), (0.1775, 0.18, 0.1495),
                 (0.18, 0.2075, 0.1586), (0.2075, 0.2225, 0.165), (0.2225, 0.235, 0.1699), (0.235,
                 0.25, 0.1759), (0.25, 0.265, 0.1813), (0.265, 0.285, 0.1869), (0.285, 0.3, 0.1915),
                 (0.3, 0.3175, 0.1962), (0.3175, 0.335, 0.2015), (0.335, 0.3575, 0.2067), (0.3575,
                 0.38, 0.2115), (0.38, 0.4125, 0.2172), (0.4125, 0.44, 0.2218), (0.44, 0.4625,
                 0.227), (0.4625, 0.4675, 0.2326), (0.4675, 0.485, 0.2384), (0.485, 0.495, 0.2345),
                 (0.495, 0.51, 0.2359), (0.51, 0.53, 0.2393))


def planter(name, xy, yaw=0.0):
    """A tall terracotta planter with a leafy plant (potted_plant_01) on a solid oak plant stand
    (proc_plant_stand), both centred on xy. The stand crosses the lidar plane and covers the
    planter's flared foot, which is then above the robot."""
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    px, py = PLANTER_CENTRE
    z = PLANT_STAND_TOP
    pos = (xy[0] - (c * px - s * py), xy[1] - (s * px + c * py), z)
    with item(name):
        model("proc_plant_stand", (xy[0], xy[1], 0.0), yaw)
        members(name, (xy[0], xy[1], 0.0), yaw, [("box", "stand", (0, 0, 0.125), (0.22, 0.22, 0.095)),
                                                 ("box", "stand_top", (0, 0, 0.21), (0.23, 0.23, 0.01)),
                                                 ("box", "stand_plinth", (0, 0, 0.015), (0.205, 0.205, 0.015))],
                "desk_wood")
        model("potted_plant_01", pos, yaw)
        members(name, pos, yaw, [("cyl", f"pot_{k}", (px, py, (z0 + z1) / 2), r, (z1 - z0) / 2)
                                 for k, (z0, z1, r) in enumerate(PLANTER_BANDS)], "pot")


def _seating(hw, seats):
    """Members of tools/proc_furniture.py _box_seating (model frame): plinth, base, back, arms, seat
    cushions (2 cm into the base), back cushions (above 0.43 m)."""
    parts = [("box", "plinth", (0, 0, 0.03), (hw - 0.02, 0.40, 0.03)),
             ("box", "base", (0, 0, 0.18), (hw, 0.42, 0.12)),
             ("box", "back", (0, 0.3415, 0.43), (hw + 0.003, 0.0815, 0.37)),
             ("box", "arm_l", (-(hw - 0.0785), -0.0815, 0.34), (0.0815, 0.3415, 0.28)),
             ("box", "arm_r", (hw - 0.0785, -0.0815, 0.34), (0.0815, 0.3415, 0.28))]
    inner = hw - 0.16
    for k in range(seats):
        x, w = -inner + (2 * k + 1) * inner / seats, inner / seats - 0.005
        parts += [("box", f"seat_{k}", (x, -0.075, 0.355), (w, 0.335, 0.075)),
                  ("box", f"cushion_{k}", (x, 0.19, 0.59), (w, 0.087, 0.165))]
    return parts


def box_sofa(name, pos, yaw=0.0):
    """Three-seat box-arm sofa (procedural, 2.10 x 0.84 x 0.80 m) on a recessed plinth: its
    upholstered base crosses the lidar plane, nothing under it. Front -y at yaw 0."""
    with item(name):
        model("proc_sofa", pos, yaw)
        members(name, pos, yaw, _seating(1.05, 3), "desk_wood")


def cube_ottoman(name, pos, yaw=0.0):
    """Upholstered cube ottoman (procedural, 0.60 x 0.60 x 0.42 m) on a recessed plinth."""
    with item(name):
        model("proc_ottoman", pos, yaw)
        members(name, pos, yaw, [("box", "plinth", (0, 0, 0.03), (0.28, 0.28, 0.03)),
                                 ("box", "base", (0, 0, 0.18), (0.30, 0.30, 0.12)),
                                 ("box", "cushion", (0, 0, 0.35), (0.30, 0.30, 0.07))], "desk_wood")


def lounge_armchair(name, pos, yaw=0.0):
    """Box-arm lounge chair (procedural, 0.82 x 0.84 x 0.80 m) on a recessed plinth: its
    upholstered base crosses the lidar plane, nothing under it. Front -y at yaw 0."""
    with item(name):
        model("proc_armchair", pos, yaw)
        members(name, pos, yaw, _seating(0.41, 1), "desk_wood")


def coffee_table(name, pos, yaw=0.0):
    """Oak coffee table (procedural, 1.10 x 0.55 x 0.42 m): four 5 cm legs from the floor (they
    cross the lidar plane; the robot fits under the top between them), and the apron, a frame of
    four rails, as one solid: nothing can get inside it."""
    with item(name):
        model("proc_coffee_table", pos, yaw)
        parts = [("box", "top", (0, 0, 0.405), (0.55, 0.275, 0.015)),
                 ("box", "apron", (0, 0, 0.36), (0.515, 0.24, 0.03))]  # the apron's frame closes it
        parts += [("box", f"leg_{i}", (sx * 0.5, sy * 0.225, 0.195), (0.025, 0.025, 0.195))
                  for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1)))]
        members(name, pos, yaw, parts, "desk_wood")


def end_table(name, pos, yaw=0.0):
    """Closed oak end table (procedural, 0.45 x 0.45 x 0.55 m) with a drawer, on a recessed plinth:
    nothing under it. Front -y at yaw 0."""
    with item(name):
        model("proc_end_table", pos, yaw)
        members(name, pos, yaw, [("box", "plinth", (0, 0, 0.015), (0.205, 0.205, 0.015)),
                                 ("box", "body", (0, 0, 0.28), (0.22, 0.22, 0.25)),
                                 ("box", "top", (0, 0, 0.54), (0.23, 0.23, 0.01))], "desk_wood")


def office_chair(name, pos, yaw=0.0):
    """Four-legged office chair (procedural, tools/proc_furniture.py): legs from the floor to the
    seat (they cross the lidar plane), back posts, seat pan and seat, the reclined back (two boxes,
    each fitted to its half of the pad), armrests. yaw 0 faces -y."""
    with item(name):
        model("proc_office_chair", pos, yaw)
        parts = []
        for sd, side in ((-1, "l"), (1, "r")):
            x = sd * 0.205
            parts += [("box", f"leg_front_{side}", (x, -0.195, 0.1975), (0.0125, 0.0125, 0.1975)),
                      ("box", f"leg_back_{side}", (x, 0.195, 0.475), (0.0125, 0.0125, 0.475)),
                      ("cyl", f"glide_front_{side}", (x, -0.195, 0.004), 0.0145, 0.004),
                      ("cyl", f"glide_back_{side}", (x, 0.195, 0.004), 0.0145, 0.004),
                      ("box", f"arm_rail_{side}", (x, 0.0, 0.62), (0.01, 0.195, 0.01)),
                      ("box", f"arm_post_{side}", (x, -0.175, 0.51), (0.01, 0.01, 0.11)),
                      ("box", f"arm_pad_{side}", (x, -0.02, 0.645), (0.028, 0.13, 0.013))]
        parts += [("box", "pan", (0, 0, 0.385), (0.23, 0.22, 0.005)),
                  ("box", "seat", (0, 0, 0.4365), (0.238, 0.228, 0.0435)),
                  ("box", "back_low", (0, 0.1828, 0.664), (0.191, 0.0448, 0.086)),
                  ("box", "back_high", (0, 0.2056, 0.836), (0.191, 0.0453, 0.086))]
        members(name, pos, yaw, parts, "chair_base")


def filing_cabinet(name, pos, yaw=0.0):
    """Four-drawer steel filing cabinet (procedural), 0.47 x 0.63 x 1.32 m, front -y at yaw 0."""
    with item(name):
        model("proc_filing_cabinet", pos, yaw)
        parts = [("box", "body", (0, -0.003, 0.66), (0.235, 0.307, 0.66))]
        dh = (1.29 - 0.05) / 4
        for k in range(4):
            zc = 0.05 + (k + 0.5) * dh + 0.06
            parts.append(("box", f"handle_{k}", (0, -0.319, zc), (0.06, 0.009, 0.008)))
        members(name, pos, yaw, parts, "metal")


def waste_bin(name, pos):
    """Round steel waste bin (procedural), 0.29 m at the rim, 0.34 m tall."""
    with item(name):
        model("proc_waste_bin", pos)
        members(name, pos, 0.0, [("cyl", "low", (0, 0, 0.055), 0.132, 0.055),
                                 ("cyl", "mid", (0, 0, 0.165), 0.139, 0.055),
                                 ("cyl", "top", (0, 0, 0.286), 0.151, 0.066)], "bin_dark")


def standing_lamp(name, pos):
    """Floor lamp (procedural): cast concrete base 0.16 m tall (it crosses the lidar plane),
    brass collar, pole, drum shade at 1.45 to 1.65 m."""
    with item(name):
        model("proc_floor_lamp", pos)
        members(name, pos, 0.0, [("cyl", "base", (0, 0, 0.08), 0.13, 0.08),
                                 ("cyl", "collar", (0, 0, 0.176), 0.024, 0.016),
                                 ("cyl", "pole", (0, 0, 0.82), 0.012, 0.635),
                                 ("cyl", "shade", (0, 0, 1.55), 0.2, 0.1)], "bin_dark")


def sofa(name, center, width, facing_y, seats=2):
    """Upholstered sofa: solid base, armrests and back (all reach the floor), visual cushions."""
    x, y = center
    box(f"{name}_base", (x, y, 0.21), (width / 2, 0.4, 0.21), "fabric")
    for sx in (-1, 1):
        box(f"{name}_arm_{'e' if sx > 0 else 'w'}", (x + sx * (width / 2 - 0.08), y, 0.32), (0.08, 0.4, 0.32), "fabric")
    back_y = y - facing_y * 0.31
    box(f"{name}_back", (x, back_y, 0.4), (width / 2, 0.09, 0.4), "fabric")
    inner = width / 2 - 0.16
    cw = inner / seats
    for k in range(seats):
        cx_ = x - inner + cw * (2 * k + 1)
        box(None, (cx_, y + facing_y * 0.05, 0.47), (cw - 0.012, 0.3, 0.055), "fabric_light", cls="visual")
        box(None, (cx_, back_y + facing_y * 0.13, 0.66), (cw - 0.012, 0.05, 0.16), "fabric_light", cls="visual")


def plant(name, x, y, seed):
    """Potted plant: solid pot (seen by the lidar), soil, stems and layered leaves."""
    import random
    rnd = random.Random(seed)
    cyl(f"{name}_pot", (x, y, 0.2), 0.18, 0.2, "pot")
    cyl(None, (x, y, 0.402), 0.165, 0.004, "soil", cls="visual")
    for k in range(5):
        a = 2 * math.pi * k / 5 + rnd.uniform(-0.3, 0.3)
        cyl(None, (x + 0.03 * math.cos(a), y + 0.03 * math.sin(a), 0.5), 0.006, 0.1, "stem", cls="visual")
    for k in range(30):  # leaves in three layers, varied size, tilt, and shade
        a = 2 * math.pi * k / 10 + rnd.uniform(-0.25, 0.25) + (k // 10) * 0.31
        r = rnd.uniform(0.06, 0.16)
        z = 0.5 + 0.11 * (k // 10) + rnd.uniform(-0.03, 0.05)
        tilt = rnd.uniform(10, 45) - 8 * (k // 10)
        length = rnd.uniform(0.09, 0.14)
        mat = ("leaves", "leaves", "leaves_light")[k % 3]
        geoms.append(f'    <geom class="visual" type="ellipsoid" pos="{x + r * math.cos(a):.3f} {y + r * math.sin(a):.3f} {z:.3f}" '
                     f'size="{length:.3f} {length * 0.32:.3f} 0.005" euler="0 {-tilt:.1f} {math.degrees(a):.1f}" material="{mat}"/>')


def comment(text):
    geoms.append(f"\n    <!-- {text} -->")


def _pieces(a, b, openings):
    """Rectangles (along from, along to, z from, z to) covering a wall from a to b, floor to
    WH, less the openings (along centre, half width, bottom, top) that fall within it."""
    inside = sorted(o for o in openings if a < o[0] < b)
    out, start = [], a
    for c, hw, z0, z1 in inside:
        if c - hw > start:
            out.append((start, c - hw, 0.0, WH))
        out += [r for r in ((c - hw, c + hw, 0.0, z0), (c - hw, c + hw, z1, WH)) if r[3] - r[2] > 1e-6]
        start = c + hw
    if b > start:
        out.append((start, b, 0.0, WH))
    return out


def skin(center, normal, half_along, half_up, material, out=0.001):
    """A visual-only face over a wall or door face whose normal is the world `normal` axis ('x'
    or 'y', pointing to the side the face is seen from, as a signed unit: +1 or -1). The classic
    renderer maps a 2D texture with texuniform only onto a box's local x-y faces, so the
    panel's local x runs along the face and its local y up it; the solid geom behind keeps its
    simple upright box (planners and the lidar read those)."""
    axis, sign = normal
    x, y, z = center
    if axis == "x":
        x += sign * out
        xy = "0 1 0 0 0 1" if sign > 0 else "0 -1 0 0 0 1"
    else:
        y += sign * out
        xy = "-1 0 0 0 0 1" if sign > 0 else "1 0 0 0 0 1"
    geoms.append(f'    <geom class="visual" type="box" pos="{x:.4f} {y:.4f} {z:.4f}" '
                 f'size="{half_along:.4f} {half_up:.4f} 0.0005" xyaxes="{xy}" material="{material}"/>')


def wall(name, axis, fixed, a, b, doors=(), skirt_sides=(-1, 1), openings=()):
    """A wall along `axis` ('x' or 'y') at coordinate `fixed`, from a to b, with door
    openings centered at `doors`. Each door gets jambs, a header, and skirting gaps. With window
    `openings` ((along centre, half width, bottom, top), an outer wall only), the solid stays
    whole but hidden (group 4: planners, the lidar, and physics still meet a wall, as at glass)
    and the wall is drawn as plaster pieces around the openings."""
    cuts = sorted(doors)
    edges = [a] + [e for c in cuts for e in (c - DOOR_GAP / 2, c + DOOR_GAP / 2)] + [b]
    for i in range(0, len(edges), 2):
        s, e = edges[i], edges[i + 1]
        if e - s < 1e-6:
            continue
        mid, half = (s + e) / 2, (e - s) / 2
        group = 4 if openings else None
        if axis == "x":
            box(f"{name}_{i // 2}", (mid, fixed, WH / 2), (half, T / 2, WH / 2), "plaster", group=group)
        else:
            box(f"{name}_{i // 2}", (fixed, mid, WH / 2), (T / 2, half, WH / 2), "plaster", group=group)
        for lo, hi, z0, z1 in _pieces(s, e, openings):  # drawn: the wall less its openings
            pm, ph, zm, zh = (lo + hi) / 2, (hi - lo) / 2, (z0 + z1) / 2, (z1 - z0) / 2
            if openings:
                pos = (pm, fixed, zm) if axis == "x" else (fixed, pm, zm)
                box(None, pos, _half(axis, ph, T / 2, zh), "plaster", cls="visual")
            for side in skirt_sides:  # the painted face on each side seen from a room
                face = fixed + side * T / 2
                centre = (pm, face, zm) if axis == "x" else (face, pm, zm)
                skin(centre, ("y" if axis == "x" else "x", side), ph, zh, "plaster")
        for side in skirt_sides:  # skirting board, solid, 1.2 cm proud of the wall
            off = fixed + side * (T / 2 + 0.006)
            if axis == "x":
                box(None, (mid, off, 0.04), (half, 0.006, 0.04), "skirting")
            else:
                box(None, (off, mid, 0.04), (0.006, half, 0.04), "skirting")
            # visual: its rounded top up to 10 cm, and the thin shadow line along the wall above it
            top = fixed + side * (T / 2 + 0.005)
            line = fixed + side * (T / 2 + 0.0018)
            if axis == "x":
                box(None, (mid, top, 0.09), (half, 0.005, 0.01), "skirting_bead", cls="visual")
                box(None, (mid, line, 0.1015), (half, 0.0015, 0.0015), "shadow_line", cls="visual")
            else:
                box(None, (top, mid, 0.09), (0.005, half, 0.01), "skirting_bead", cls="visual")
                box(None, (line, mid, 0.1015), (0.0015, half, 0.0015), "shadow_line", cls="visual")
    for k, c in enumerate(cuts):
        for sgn in (-1, 1):  # jambs: solid frame inside the gap, slightly proud of the wall
            jc = c + sgn * (DOOR_GAP / 2 - JAMB / 2)
            if axis == "x":
                box(f"{name}_door{k}_jamb{'LR'[sgn > 0]}", (jc, fixed, DOOR_HEAD / 2), (JAMB / 2, T / 2 + 0.02, DOOR_HEAD / 2), "frame")
            else:
                box(f"{name}_door{k}_jamb{'LR'[sgn > 0]}", (fixed, jc, DOOR_HEAD / 2), (T / 2 + 0.02, JAMB / 2, DOOR_HEAD / 2), "frame")
        for side in (-1, 1):  # architrave trim on both wall faces (solid, 1.2 cm proud)
            face = fixed + side * (T / 2 + 0.006)
            for sgn in (-1, 1):
                a_c = c + sgn * (DOOR_GAP / 2 + 0.035)
                if axis == "x":
                    box(None, (a_c, face, (DOOR_HEAD + 0.07) / 2), (0.035, 0.006, (DOOR_HEAD + 0.07) / 2), "frame")
                else:
                    box(None, (face, a_c, (DOOR_HEAD + 0.07) / 2), (0.006, 0.035, (DOOR_HEAD + 0.07) / 2), "frame")
            if axis == "x":
                box(None, (c, face, DOOR_HEAD + 0.035), (DOOR_GAP / 2 + 0.07, 0.006, 0.035), "frame")
            else:
                box(None, (face, c, DOOR_HEAD + 0.035), (0.006, DOOR_GAP / 2 + 0.07, 0.035), "frame")
            # visual: the architrave stands 2.5 cm out (in front of the 2 cm jamb lining), with a
            # raised bead on its inner edge, so the frame reads as moulded trim, not a flat strip
            fa = fixed + side * (T / 2 + 0.0125)
            fb = fixed + side * (T / 2 + 0.028)
            for sgn in (-1, 1):
                a_c = c + sgn * (DOOR_GAP / 2 + 0.035)
                b_c = c + sgn * (DOOR_GAP / 2 + 0.006)
                hz = (DOOR_HEAD + 0.07) / 2
                if axis == "x":
                    box(None, (a_c, fa, hz), (0.035, 0.0125, hz), "frame", cls="visual")
                    box(None, (b_c, fb, hz), (0.006, 0.003, hz), "frame", cls="visual")
                else:
                    box(None, (fa, a_c, hz), (0.0125, 0.035, hz), "frame", cls="visual")
                    box(None, (fb, b_c, hz), (0.003, 0.006, hz), "frame", cls="visual")
            if axis == "x":
                box(None, (c, fa, DOOR_HEAD + 0.035), (DOOR_GAP / 2 + 0.07, 0.0125, 0.035), "frame", cls="visual")
                box(None, (c, fb, DOOR_HEAD + 0.006), (DOOR_GAP / 2, 0.003, 0.006), "frame", cls="visual")
            else:
                box(None, (fa, c, DOOR_HEAD + 0.035), (0.0125, DOOR_GAP / 2 + 0.07, 0.035), "frame", cls="visual")
                box(None, (fb, c, DOOR_HEAD + 0.006), (0.003, DOOR_GAP / 2, 0.006), "frame", cls="visual")
        # visual: the door stop round the opening (a 12 mm bead on the jamb linings and head, mid
        # wall) and the strike plate on the latch jamb (the leaves hinge on the low side)
        for sgn in (-1, 1):
            sc = c + sgn * (DOOR_GAP / 2 - JAMB - 0.006)
            if axis == "x":
                box(None, (sc, fixed, DOOR_HEAD / 2), (0.006, 0.0125, DOOR_HEAD / 2), "frame", cls="visual")
            else:
                box(None, (fixed, sc, DOOR_HEAD / 2), (0.0125, 0.006, DOOR_HEAD / 2), "frame", cls="visual")
        if axis == "x":
            box(None, (c, fixed, DOOR_HEAD - 0.006), (DOOR_GAP / 2 - JAMB, 0.0125, 0.006), "frame", cls="visual")
            box(None, (c + DOOR_GAP / 2 - JAMB - 0.0015, fixed + 0.03, 1.0), (0.0015, 0.012, 0.06), "strike_plate", cls="visual")
        else:
            box(None, (fixed, c, DOOR_HEAD - 0.006), (0.0125, DOOR_GAP / 2 - JAMB, 0.006), "frame", cls="visual")
            box(None, (fixed + 0.03, c + DOOR_GAP / 2 - JAMB - 0.0015, 1.0), (0.012, 0.0015, 0.06), "strike_plate", cls="visual")
        # visual: an aluminium threshold strip across the doorway floor (2 mm, flush enough to drive over)
        if axis == "x":
            box(None, (c, fixed, 0.0022), (DOOR_GAP / 2 - JAMB, T / 2 + 0.01, 0.001), "threshold", cls="visual")
        else:
            box(None, (fixed, c, 0.0022), (T / 2 + 0.01, DOOR_GAP / 2 - JAMB, 0.001), "threshold", cls="visual")
        head_h = (WH - DOOR_HEAD) / 2
        if axis == "x":
            box(f"{name}_door{k}_head", (c, fixed, DOOR_HEAD + head_h), (DOOR_GAP / 2, T / 2, head_h), "plaster")
        else:
            box(f"{name}_door{k}_head", (fixed, c, DOOR_HEAD + head_h), (T / 2, DOOR_GAP / 2, head_h), "plaster")
        for side in (-1, 1):
            face = fixed + side * T / 2
            centre = (c, face, DOOR_HEAD + head_h) if axis == "x" else (face, c, DOOR_HEAD + head_h)
            skin(centre, ("y" if axis == "x" else "x", side), DOOR_GAP / 2, head_h, "plaster")


def door_leaf(name, axis, fixed, center, into):
    """A door fixed open at 90 degrees, hinged at one jamb, swung into the room on the
    `into` side (+1 or -1 along the wall normal). Solid; 0.88 m wide, 4 cm thick."""
    hinge = center - DOOR_GAP / 2 + JAMB  # inner face of the first jamb
    w, t = 0.88, 0.02
    start = fixed + into * (T / 2 + 0.02)
    mid = start + into * w / 2
    zc = 1.035  # the leaf stands 1 cm clear of the floor
    if axis == "x":
        box(name, (hinge + t, mid, zc), (t, w / 2, 1.025), "door")
    else:
        box(name, (mid, hinge + t, zc), (w / 2, t, 1.025), "door")
    veneer = "door_b" if name in ("door_lab", "door_reception") else "door"  # not every leaf alike
    for side in (-1, 1):  # the veneer on both faces of the leaf
        if axis == "x":
            skin((hinge + t + side * t, mid, zc), ("x", side), w / 2, 1.025, veneer)
        else:
            skin((mid, hinge + t + side * t, zc), ("y", side), w / 2, 1.025, veneer)
    # Hinges on the hinge edge, lever handles on both faces near the free edge
    for z in (0.25, 1.0, 1.8):
        h_pos = (hinge + t, start + into * 0.012, z) if axis == "x" else (start + into * 0.012, hinge + t, z)
        cyl(None, h_pos, 0.008, 0.045, "handle", cls="visual")
    for side in (-1, 1):  # kick plates (stainless, the bottom 25 cm) on both faces
        if axis == "x":
            skin((hinge + t + side * t, mid, 0.09), ("x", side), w / 2 - 0.02, 0.075, "kick_plate", out=0.0015)
            skin((hinge + t + side * t, mid, 0.215), ("x", side), w / 2 - 0.02, 0.05, "kick_band", out=0.0015)
        else:
            skin((mid, hinge + t + side * t, 0.09), ("y", side), w / 2 - 0.02, 0.075, "kick_plate", out=0.0015)
            skin((mid, hinge + t + side * t, 0.215), ("y", side), w / 2 - 0.02, 0.05, "kick_band", out=0.0015)
    # an overhead closer on the push side near the hinge, its arm reaching the frame head
    cside = -into
    if axis == "x":
        box(None, (hinge + t + cside * (t + 0.025), start + into * 0.18, 1.98), (0.025, 0.13, 0.03), "closer", cls="visual")
    else:
        box(None, (start + into * 0.18, hinge + t + cside * (t + 0.025), 1.98), (0.13, 0.025, 0.03), "closer", cls="visual")
    free = start + into * (w - 0.07)
    for side in (-1, 1):
        face = hinge + t + side * (t + 0.004)
        if axis == "x":
            geoms.append(f'    <geom class="visual" type="cylinder" pos="{face:.4f} {free:.4f} 1.0" size="0.025 0.004" zaxis="1 0 0" material="handle"/>')
            box(None, (face + side * 0.02, free - into * 0.05, 1.0), (0.008, 0.06, 0.008), "handle", cls="visual")
        else:
            geoms.append(f'    <geom class="visual" type="cylinder" pos="{free:.4f} {face:.4f} 1.0" size="0.025 0.004" zaxis="0 1 0" material="handle"/>')
            box(None, (free - into * 0.05, face + side * 0.02, 1.0), (0.06, 0.008, 0.008), "handle", cls="visual")


def label(name, pos, axis, texture_material, facing=-1):
    """A room sign. A box face maps its texture mirrored on one side, so signs facing +y
    are turned 180 degrees to read correctly."""
    half = (0.25, 0.006, 0.0625) if axis == "x" else (0.006, 0.25, 0.0625)
    box(name, pos, half, texture_material, cls="visual", euler=180 if facing > 0 else None)


def _on_wall(center, axis, facing, along, out, z):
    """World position of a point on a wall's inner face: `along` the wall, `out` into the room."""
    cx, cy = center
    if axis == "x":
        return (cx + along, cy + facing * out, z)
    return (cx + facing * out, cy + along, z)


def _half(axis, along, out, up):
    return (along, out, up) if axis == "x" else (out, along, up)


# Windows on the outer walls: (centre on the wall's inner face, wall axis, facing into the room,
# width, sill height, height). Visual only: the wall behind stays solid, like glass to a robot.
WINDOWS = [((-3.8, H), "x", -1, 1.4, 0.9, 1.2), ((-1.6, H), "x", -1, 1.4, 0.9, 1.2),
           ((1.6, H), "x", -1, 1.2, 1.0, 1.2), ((3.6, H), "x", -1, 1.2, 1.0, 1.2),
           ((H, 3.6), "y", -1, 1.2, 0.9, 1.2), ((1.3, -H), "x", +1, 1.4, 0.9, 1.2),
           ((3.9, -H), "x", +1, 1.0, 0.9, 1.2), ((H, -3.4), "y", -1, 1.2, 0.9, 1.2),
           ((-H, -4.55), "y", +1, 0.6, 1.05, 1.2), ((H, 0.0), "y", -1, 1.0, 0.9, 1.2)]


def _view(center, axis, facing, width, sill, height):
    """(outward heading in degrees clockwise from north, position along the wall, width, height)."""
    out = (0.0, -facing) if axis == "x" else (-facing, 0.0)
    heading = math_deg(out)
    return heading, center[0] if axis == "x" else center[1], width, height


def math_deg(v):
    import math
    return math.degrees(math.atan2(v[0], v[1])) % 360.0


WINDOW_VIEWS = [_view(*w) for w in WINDOWS]  # the view crops are made for these (tools/fetch_view.py)


FRAME_W = 0.07  # m: a window's outer frame width
GLASS_IN = 0.07  # m the glass is set back from the wall's inner face (the wall is T thick)


def window_opening(center, axis, width, sill_z, height=1.2):
    """(along centre, half width, bottom, top) of the hole a window needs in its wall."""
    along = center[0] if axis == "x" else center[1]
    return (along, width / 2 + FRAME_W, sill_z - FRAME_W - 0.01, sill_z + height + FRAME_W - 0.01)


def window(k, center, axis, facing, width, sill_z, height=1.2):
    """Window k set into an outer wall: the glass, 7 cm back from the inner face, showing its own
    crop of the outdoor view (bright: the sky outshines the room); plaster reveals lining the
    opening; a 7 cm frame inside the opening with a sash and mullion a step behind it; a sill
    board from the frame into the room with a rounded front edge; a roller blind a quarter down
    from the head; and soft daylight into the room. Visual only: the wall's hidden solid stays
    whole, like glass to a robot."""
    zc = sill_z + height / 2
    normal = ("y" if axis == "x" else "x", facing)
    _, ow, zb, zt = window_opening(center, axis, width, sill_z, height)
    fw = FRAME_W
    skin(_on_wall(center, axis, facing, 0, -GLASS_IN, zc), normal, width / 2, height / 2, f"view_{k}", out=0.0)
    # reveals: the opening's sides and head, plaster, from the glass to the room face (3 mm
    # inside the opening, so no face lies on the wall pieces' own)
    rd = GLASS_IN / 2
    for a in (-1, 1):
        box(None, _on_wall(center, axis, facing, a * (ow - 0.003), -rd, (zb + zt) / 2), _half(axis, 0.003, rd, (zt - zb) / 2),
            "plaster_reveal", cls="visual")
    box(None, _on_wall(center, axis, facing, 0, -rd, zt - 0.003), _half(axis, ow, rd, 0.003), "plaster_reveal", cls="visual")
    # outer frame, just inside the glass
    fd = 0.0125
    fo = -GLASS_IN + fd
    for zz in (sill_z - fw / 2 + 0.01, sill_z + height + fw / 2 - 0.01):  # head and bottom rail
        box(None, _on_wall(center, axis, facing, 0, fo, zz), _half(axis, width / 2 + fw, fd, fw / 2), "window_frame", cls="visual")
    for a in (-1, 1):  # jambs
        box(None, _on_wall(center, axis, facing, a * (width / 2 + fw / 2), fo, zc), _half(axis, fw / 2, fd, height / 2), "window_frame", cls="visual")
    sw, sd = 0.035, 0.012  # sash rim and mullion, a step into the room from the frame
    so = fo + fd + sd
    box(None, _on_wall(center, axis, facing, 0, so, zc), _half(axis, sw / 2, sd, height / 2), "window_frame", cls="visual")
    for zz in (sill_z + sw / 2, sill_z + height - sw / 2):
        box(None, _on_wall(center, axis, facing, 0, so, zz), _half(axis, width / 2, sd, sw / 2), "window_frame", cls="visual")
    for a in (-1, 1):
        box(None, _on_wall(center, axis, facing, a * (width / 2 - sw / 2), so, zc), _half(axis, sw / 2, sd, height / 2), "window_frame", cls="visual")
    # sill board: 4 cm thick, from the frame to 5 cm proud of the wall, its front edge rounded
    zs = zb + 0.02
    s0, s1 = -GLASS_IN + 2 * fd, 0.05
    box(None, _on_wall(center, axis, facing, 0, (s0 + s1) / 2, zs), _half(axis, ow + 0.04, (s1 - s0) / 2, 0.02), "sill", cls="visual")
    fx, fy, fz = _on_wall(center, axis, facing, 0, s1, zs)
    za = "1 0 0" if axis == "x" else "0 1 0"
    geoms.append(f'    <geom class="visual" type="cylinder" pos="{fx:.4f} {fy:.4f} {fz:.4f}" '
                 f'size="0.02 {ow + 0.04:.4f}" zaxis="{za}" material="sill"/>')
    # roller blind under the head of the opening: the roller, the fabric a quarter down, a rail
    bo = -0.012
    rx, ry, rz = _on_wall(center, axis, facing, 0, bo, zt - 0.02)
    geoms.append(f'    <geom class="visual" type="cylinder" pos="{rx:.4f} {ry:.4f} {rz:.4f}" '
                 f'size="0.018 {ow - 0.01:.4f}" zaxis="{za}" material="blind"/>')
    drop = 0.28 * height
    top = zt - 0.02
    skin(_on_wall(center, axis, facing, 0, bo, top - drop / 2), normal, ow - 0.01, drop / 2, "blind", out=0.0)
    box(None, _on_wall(center, axis, facing, 0, bo, top - drop - 0.01), _half(axis, ow - 0.01, 0.008, 0.012), "blind_rail", cls="visual")
    # daylight: a cool spotlight just inside the glass, aimed into the room and down
    lx, ly, lz = _on_wall(center, axis, facing, 0, 0.1, zc + 0.3)
    dx, dy = (0.0, float(facing)) if axis == "x" else (float(facing), 0.0)
    geoms.append(f'    <light pos="{lx:.3f} {ly:.3f} {lz:.3f}" dir="{dx:.2f} {dy:.2f} -0.7" directional="false" '
                 'cutoff="70" exponent="2" diffuse="0.15 0.17 0.21" specular="0 0 0" attenuation="1 0.2 0.05" '
                 'castshadow="false"/>')


def wall_picture(center, axis, facing, half_size, z, material, frame="window_frame"):
    """A framed panel on a wall face (painting, whiteboard, TV); visual only."""
    hw, hh = half_size
    box(None, _on_wall(center, axis, facing, 0, 0.012, z), _half(axis, hw + 0.03, 0.012, hh + 0.03), frame, cls="visual")
    box(None, _on_wall(center, axis, facing, 0, 0.026, z), _half(axis, hw, 0.002, hh), material, cls="visual")


def floor_lamp(name, xy):
    """Solid base the lidar sees; the pole and shade are visual and stay above the base."""
    x, y = xy
    cyl(name, (x, y, 0.1), 0.16, 0.1, "bin_dark")
    cyl(None, (x, y, 0.85), 0.015, 0.65, "metal", cls="visual")
    cyl(None, (x, y, 1.55), 0.16, 0.1, "lamp_shade", cls="visual")


def build_floor():
    comment("Base floor (physics). Room floors on top are visual only.")
    geoms.append('    <geom name="floor" type="plane" size="5.6 5.6 0.1" material="tile"/>')
    zones = [("office", (-H, -T / 2, CORRIDOR + T, H), "carpet"),
             ("storage", (-H, -T / 2, -H, -CORRIDOR - T), "tile_dark"),
             ("reception", (T / 2, H, -H, -CORRIDOR - T), "wood_floor"),
]
    # (the lab and corridor show the base plane itself: porcelain, the model's first and so its
    # only reflective geom, the classic renderer reflecting in just one)
    for name, (x0, x1, y0, y1), mat in zones:
        box(f"floor_{name}", ((x0 + x1) / 2, (y0 + y1) / 2, 0.0006), ((x1 - x0) / 2, (y1 - y0) / 2, 0.0005), mat, cls="visual")

    comment("Outer walls (inner faces at +/-5 m)")
    def openings(axis, at):
        return [window_opening(c, ax, w, sz, ht) for c, ax, _, w, sz, ht in WINDOWS
                if ax == axis and abs((c[1] if ax == "x" else c[0]) - at) < 1e-9]
    wall("wall_north", "x", H + T / 2, -H - T, H + T, skirt_sides=(-1,), openings=openings("x", H))
    wall("wall_south", "x", -H - T / 2, -H - T, H + T, skirt_sides=(1,), openings=openings("x", -H))
    wall("wall_east", "y", H + T / 2, -H, H, skirt_sides=(-1,), openings=openings("y", H))
    wall("wall_west", "y", -H - T / 2, -H, H, skirt_sides=(1,), openings=openings("y", -H))

    geoms.append("    <!-- OBSTACLES: interior walls, doors, and furniture (removed for the empty test track) -->")
    comment("Corridor walls with doorways (clear width 0.9 m)")
    wall("corridor_north", "x", CORRIDOR + T / 2, -H, H, doors=(-2.5, 2.5))
    wall("corridor_south", "x", -CORRIDOR - T / 2, -H, H, doors=(-3.0, 2.0))
    comment("Dividers between rooms (office and lab connect through a door)")
    wall("divider_north", "y", 0.0, CORRIDOR + T, H, doors=(3.5,))
    wall("divider_south", "y", 0.0, -H, -CORRIDOR - T)

    comment("Doors fixed open (static)")
    door_leaf("door_office", "x", CORRIDOR + T / 2, -2.5, +1)
    door_leaf("door_lab", "x", CORRIDOR + T / 2, 2.5, +1)
    door_leaf("door_storage", "x", -CORRIDOR - T / 2, -3.0, -1)
    door_leaf("door_reception", "x", -CORRIDOR - T / 2, 2.0, -1)
    door_leaf("door_office_lab", "y", 0.0, 3.5, +1)

    comment("Room signs (visual only)")
    label("sign_office", (-1.7, CORRIDOR + 0.004, 1.6), "x", "sign_office")
    label("sign_lab", (3.3, CORRIDOR + 0.004, 1.6), "x", "sign_lab")
    label("sign_storage", (-2.2, -CORRIDOR - 0.004, 1.6), "x", "sign_storage", facing=+1)
    label("sign_reception", (2.8, -CORRIDOR - 0.004, 1.6), "x", "sign_reception", facing=+1)

    comment("Office furniture")
    office_desk("office_desk", (-3.8, 4.505, 0.0))
    top = OFFICE_DESK_TOP
    box(None, (-3.8, 4.68, top + 0.005), (0.09, 0.07, 0.005), "robot_dark", cls="visual")  # monitor stand
    box(None, (-3.8, 4.70, top + 0.10), (0.02, 0.015, 0.09), "robot_dark", cls="visual")
    box(None, (-3.8, 4.68, top + 0.26), (0.3, 0.012, 0.17), "monitor", cls="visual")
    box(None, (-3.8, 4.668, top + 0.26), (0.285, 0.001, 0.155), "screen", cls="visual")
    box(None, (-3.8, 4.38, top + 0.006), (0.21, 0.07, 0.006), "robot_dark", cls="visual")  # keyboard
    box(None, (-3.42, 4.38, top + 0.008), (0.035, 0.05, 0.008), "robot_dark", cls="visual")  # mouse
    office_chair("office_chair", (-3.8, 3.62, 0.0), 180.0)  # tucked in at the kneehole (seat 0.21 m from the desk)
    steel_shelf("office_bookshelf", (-4.74, 2.4, 0.0), 90.0, books_seed=11)
    filing_cabinet("office_cabinet", (-0.45, 4.674, 0.0))
    planter("office_plant", (-0.45, 1.25))
    model("wall_clock", (-2.7, H - 0.024, 1.7))  # on the north wall between the windows (visual, high up)

    comment("Lab furniture")
    box("lab_bench_north", (2.6, 4.55, 0.45), (1.2, 0.4, 0.45), "lab_white")
    cabinet_seams((2.6, 4.55, 0.45), (1.2, 0.4, 0.45), "y", -1, 4)
    box(None, (2.6, 4.55, 0.915), (1.22, 0.42, 0.015), "lab_top", cls="visual")
    box(None, (2.0, 4.65, 0.96), (0.09, 0.12, 0.03), "lab_white", cls="visual")  # microscope base
    box(None, (2.0, 4.72, 1.08), (0.025, 0.03, 0.12), "lab_white", cls="visual")
    box(None, (2.0, 4.66, 1.17), (0.03, 0.07, 0.03), "robot_dark", cls="visual")
    cyl(None, (2.0, 4.6, 1.11), 0.012, 0.05, "robot_dark", cls="visual")
    for k, bx in enumerate((2.6, 2.7, 2.78)):  # glassware
        cyl(None, (bx, 4.5, 0.98 + 0.02 * k), 0.03 - 0.005 * k, 0.05 + 0.02 * k, "glass_ware", cls="visual")
    box(None, (3.3, 4.6, 1.02), (0.2, 0.17, 0.09), "lab_white", cls="visual")  # analyzer
    box(None, (3.3, 4.43, 1.04), (0.12, 0.001, 0.05), "screen", cls="visual")
    box("lab_bench_island", (2.9, 2.5, 0.45), (0.9, 0.4, 0.45), "lab_white")
    cabinet_seams((2.9, 2.5, 0.45), (0.9, 0.4, 0.45), "y", -1, 3)
    cabinet_seams((2.9, 2.5, 0.45), (0.9, 0.4, 0.45), "y", 1, 3)
    box(None, (2.9, 2.5, 0.915), (0.92, 0.42, 0.015), "lab_top", cls="visual")
    shelving_unit("lab_shelving", (4.7, 1.9), (0.25, 0.7), 2.0, 4, seed=1)
    box("lab_cart", (4.3, 4.0, 0.45), (0.3, 0.25, 0.45), "metal")

    comment("Storage furniture: shelving rows with 1.2 m aisles")
    shelving_unit("storage_shelf_west", (-4.75, -3.0), (0.25, 1.2), 2.0, 4, seed=2)
    shelving_unit("storage_shelf_mid", (-2.9, -3.5), (0.25, 1.2), 2.0, 4, seed=3)
    shelving_unit("storage_shelf_east", (-1.15, -3.5), (0.25, 1.2), 2.0, 4, seed=4)
    box("storage_boxes", (-0.45, -4.55, 0.3), (0.3, 0.3, 0.3), "cardboard")

    comment("Reception furniture")
    box("reception_counter", (3.5, -2.2, 0.55), (0.9, 0.3, 0.55), "desk_wood")
    cabinet_seams((3.5, -2.2, 0.55), (0.9, 0.3, 0.55), "y", -1, 3)
    box(None, (3.5, -2.2, 1.115), (0.95, 0.35, 0.015), "lab_top", cls="visual")
    box_sofa("reception_sofa", (1.124, -4.56, 0.0), 180.0)  # against the south wall, facing north
    coffee_table("reception_table", (1.124, -3.415, 0.0))  # 0.45 m in front of the sofa
    end_table("reception_side_table", (2.429, -4.755, 0.0), 180.0)  # at the sofa's east end, drawer to the room
    lounge_armchair("reception_armchair", (3.094, -4.56, 0.0), 180.0)  # beside it, facing north
    cube_ottoman("reception_ottoman", (3.834, -4.68, 0.0))  # the row ends 0.52 m from the lamp
    # (the seating keeps 0.5 m from the frozen held-out starts and goals at (2.555, -3.523) and
    # (3.881, -3.186): robot_env/config.py HELDOUT_TASKS)
    planter("reception_plant", (4.6, -1.2))

    comment("Corridor")
    box("corridor_water_cooler", (4.78, 0.4, 0.55), (0.17, 0.17, 0.55), "lab_white")
    plant("corridor_plant", -4.75, -0.42, seed=23)
    comment("Everyday clutter (solid, at least 0.2 m tall so the lidar sees it)")
    waste_bin("office_trash_bin", (-2.6, 4.75, 0.0))
    standing_lamp("office_lamp", (-4.78, 3.85, 0.0))  # in the corner by the desk: 8 cm to the wall, 8 cm to the desk, closed to cats
    cyl("lab_trash_bin", (0.4, 4.62, 0.17), 0.13, 0.17, "bin_dark")
    cyl("lab_stool_1", (3.32, 1.83, 0.25), 0.17, 0.25, "bin_dark")
    cyl("lab_stool_2", (3.72, 1.83, 0.25), 0.17, 0.25, "bin_dark")
    standing_lamp("reception_lamp", (4.78, -4.78, 0.0))  # in the south-east corner, 9 cm to both walls
    waste_bin("reception_trash_bin", (4.75, -3.0, 0.0))  # 10 cm to the east wall
    box("storage_pallet", (-0.5, -1.6, 0.35), (0.4, 0.35, 0.35), "cardboard")
    box(None, (-0.5, -1.6, 0.06), (0.402, 0.352, 0.06), "pallet_wood", cls="visual")
    for i, (dx, dy) in enumerate(((-0.2, -0.17), (0.2, -0.17), (-0.2, 0.17), (0.2, 0.17))):
        box(None, (-0.5 + dx, -1.6 + dy, 0.44), (0.19, 0.16, 0.002), "seam", cls="visual")
    comment("Decor on the furnished floor (visual only): rugs, wall art, whiteboard, TV")
    # rugs: a 5 mm pile (its bound edge darker) with the woven design on top
    for (cx, cy), (hx, hy), name in (((1.3, -3.55), (0.85, 0.6), "rug_red"), ((-3.8, 3.95), (0.95, 0.65), "rug_blue")):
        box(None, (cx, cy, 0.0026), (hx, hy, 0.0024), f"{name}_edge", cls="visual")
        box(None, (cx, cy, 0.00515), (hx - 0.004, hy - 0.004, 0.00015), name, cls="visual")
    wall_picture((-0.5, -CORRIDOR), "x", +1, (0.4, 0.27), 1.5, "painting_1")
    wall_picture((0.9, CORRIDOR), "x", -1, (0.4, 0.27), 1.5, "painting_2")
    wall_picture((T / 2, -2.6), "y", +1, (0.35, 0.24), 1.5, "painting_2")
    wall_picture((-T / 2, 1.9), "y", -1, (0.6, 0.3), 1.4, "whiteboard")
    wall_picture((T / 2, -3.9), "y", +1, (0.5, 0.28), 1.25, "tv_screen", frame="robot_dark")
    geoms.append("    <!-- /OBSTACLES -->")

    comment("Windows set into the outer walls (visual: the wall's hidden solid stays whole, like glass to a robot)")
    for k, w in enumerate(WINDOWS):
        window(k, *w)
    comment("Fire extinguisher on the corridor's west end wall (visual, mounted above robot height)")
    cyl(None, (-H + 0.055, 0.45, 0.55), 0.055, 0.17, "extinguisher", cls="visual")

    comment("Ceiling and light fittings: group 3, robot camera only (hidden in overview views)")
    box("ceiling", (0, 0, WH + 0.025), (H + T, H + T, 0.025), "ceiling", cls="ceiling")
    for x, y, _, _ in FITTINGS:
        light_fitting(x, y)
    room_lights()


# Ceiling light fittings (x, y, light name, room tint): two per room, four along the corridor;
# each has a spotlight just below it. The first of each room keeps the room's light name.
FITTINGS = [(-3.75, 2.9, "light_office", "1 0.97 0.92"), (-1.25, 2.9, "light_office_2", "1 0.97 0.92"),
            (1.25, 2.9, "light_lab", "0.98 0.99 1"), (3.75, 2.9, "light_lab_2", "0.98 0.99 1"),
            (-3.75, -2.9, "light_storage", "0.98 0.98 0.98"), (-1.25, -2.9, "light_storage_2", "0.98 0.98 0.98"),
            (1.25, -2.9, "light_reception", "1 0.96 0.9"), (3.75, -2.9, "light_reception_2", "1 0.96 0.9"),
            (-3.75, 0.0, "light_corridor_w", "1 0.98 0.95"), (-1.25, 0.0, "light_corridor_w2", "1 0.98 0.95"),
            (1.25, 0.0, "light_corridor_e2", "1 0.98 0.95"), (3.75, 0.0, "light_corridor_e", "1 0.98 0.95")]
FIT_HALF = (0.3, 0.3)  # a 600 x 600 mm surface-mounted panel (corridor: rotated, same size)


def light_fitting(x, y):
    """A surface-mounted LED panel: a 2 cm deep white frame around a diffuser recessed 1 cm."""
    hx, hy = FIT_HALF
    rim, depth = 0.02, 0.02
    for sx in (-1, 1):
        box(None, (x + sx * (hx - rim / 2), y, WH - depth / 2), (rim / 2, hy, depth / 2), "fitting_frame", cls="ceiling")
    for sy in (-1, 1):
        box(None, (x, y + sy * (hy - rim / 2), WH - depth / 2), (hx - rim, rim / 2, depth / 2), "fitting_frame", cls="ceiling")
    box(None, (x, y, WH - 0.005), (hx - rim, hy - rim, 0.005), "light_panel", cls="ceiling")


def room_lights():
    """A spotlight under every fitting (soft, wide, casting shadows) with a little ambient light
    each; the headlight is kept low so the rooms are lit by their fittings and windows."""
    # light bounced off the floor, walls, and ceiling (the renderer has none): two soft fills
    # from different directions, so walls facing different ways differ, as in a real room
    geoms.append('    <light name="fill_1" directional="true" dir="0.45 0.3 -0.84" diffuse="0.34 0.34 0.33" '
                 'specular="0 0 0" castshadow="false"/>')
    geoms.append('    <light name="fill_2" directional="true" dir="-0.3 -0.45 -0.84" diffuse="0.1 0.1 0.1" '
                 'specular="0 0 0" castshadow="false"/>')
    for x, y, name, tint in FITTINGS:
        r, g, b = (float(v) for v in tint.split())
        k = 0.64 if not name.startswith("light_corridor") else 0.48
        geoms.append(f'    <light name="{name}" pos="{x} {y} {WH - 0.06:.2f}" dir="0 0 -1" directional="false" '
                     f'cutoff="70" exponent="2" diffuse="{k * r:.3f} {k * g:.3f} {k * b:.3f}" '
                     f'ambient="0 0 0" specular="0.3 0.3 0.3" attenuation="1 0.25 0.08" castshadow="true"/>')


def _wheel_visuals(side: int) -> str:
    """Tire sidewall, tread blocks around the rim, a metal hub with five bolts.
    All massless and within the tire's own envelope (radius 0.05 m, half width 0.0185 m)."""
    out = [
        '        <geom class="visual" type="cylinder" size="0.0465 0.0151" zaxis="0 1 0" material="tire_wall"/>',
        '        <geom class="visual" type="cylinder" size="0.031 0.0168" zaxis="0 1 0" material="hub"/>',
        '        <geom class="visual" type="cylinder" size="0.011 0.0182" zaxis="0 1 0" material="robot_dark"/>',
    ]
    for k in range(5):  # hub bolts on the outer face
        ang = 2 * math.pi * k / 5
        out.append(f'        <geom class="visual" type="cylinder" size="0.0024 0.0018" '
                   f'pos="{0.019 * math.cos(ang):.4f} {side * 0.0168:.4f} {0.019 * math.sin(ang):.4f}" '
                   f'zaxis="0 1 0" material="robot_dark"/>')
    for k in range(20):  # tread blocks, 0.5 mm proud of the tire surface
        ang = 2 * math.pi * k / 20
        out.append(f'        <geom class="visual" type="box" size="0.0035 0.0135 0.0006" '
                   f'pos="{0.0499 * math.cos(ang):.4f} 0 {0.0499 * math.sin(ang):.4f}" '
                   f'euler="0 {90 - math.degrees(ang):.2f} 0" material="tread"/>')
    return "\n".join(out)


def robot_xml() -> str:
    vents = "\n".join(
        f'      <geom class="visual" type="box" pos="-0.1495 {y:.3f} 0.03" size="0.0008 0.009 0.0016" material="vent"/>'
        for y in (-0.06, -0.03, 0.0, 0.03, 0.06))
    side_vents = "\n".join(
        f'      <geom class="visual" type="box" pos="{x:.3f} {sgn * 0.0995:.4f} 0.036" size="0.006 0.0008 0.0016" material="vent"/>'
        for x in (-0.11, -0.095, -0.08) for sgn in (-1, 1))
    screws = "\n".join(
        f'      <geom class="visual" type="cylinder" pos="{x:.3f} {y:.3f} 0.0663" size="0.003 0.0006" material="hub"/>'
        for x in (-0.112, 0.112) for y in (-0.07, 0.07))
    return ROBOT_TEMPLATE.format(screws=screws, vents=vents, side_vents=side_vents,
                                 left=_wheel_visuals(+1), right=_wheel_visuals(-1))


ROBOT_TEMPLATE = """
    <!-- The robot. Body origin = midpoint between the wheel axles, 5 cm above the floor.
         Colliders (group 4, hidden) are unchanged from the calibrated design; everything
         visible is massless detail inside the physical footprint (tested). -->
    <body name="robot" pos="0 0 0.05">
      <freejoint name="robot_root"/>
      <geom name="chassis" type="box" pos="0 0 0.03" size="0.15 0.10 0.035" mass="2.0" group="4" rgba="0.2 0.2 0.2 1"/>
      <!-- Frictionless ball casters keep the robot level. They sit 1 mm above the
           floor so the wheels carry the weight; if the casters touch the floor,
           they take the load and the wheels slip (found in the step 5 check).
           A soft, well-damped contact (like a sprung caster) so a body nod onto a
           caster does not bounce the tires off the floor. -->
      <geom name="front_caster" type="sphere" pos="0.11 0 -0.024" size="0.025" mass="0.05" condim="1" priority="1" solref="0.1 3" material="robot_dark"/>
      <geom name="rear_caster" type="sphere" pos="-0.11 0 -0.024" size="0.025" mass="0.05" condim="1" priority="1" solref="0.1 3" material="robot_dark"/>
      <geom name="lidar_housing" type="cylinder" pos="0 0 0.075" size="0.035 0.01" mass="0.1" group="4"/>
      <site name="lidar_site" pos="0 0 0.085" size="0.005" group="4"/>

      <!-- Body shell: graphite base with rounded vertical corners -->
      <geom class="visual" type="box" pos="0 0 0.027" size="0.13 0.10 0.03" material="robot_body"/>
      <geom class="visual" type="box" pos="0 0 0.027" size="0.15 0.08 0.03" material="robot_body"/>
      <geom class="visual" type="cylinder" pos="0.13 0.08 0.027" size="0.02 0.03" material="robot_body"/>
      <geom class="visual" type="cylinder" pos="0.13 -0.08 0.027" size="0.02 0.03" material="robot_body"/>
      <geom class="visual" type="cylinder" pos="-0.13 0.08 0.027" size="0.02 0.03" material="robot_body"/>
      <geom class="visual" type="cylinder" pos="-0.13 -0.08 0.027" size="0.02 0.03" material="robot_body"/>
      <!-- Beveled top: a dark trim step, then the inset light cover with screws -->
      <geom class="visual" type="box" pos="0 0 0.0585" size="0.142 0.092 0.0015" material="robot_trim"/>
      <geom class="visual" type="box" pos="0 0 0.0625" size="0.126 0.083 0.0025" material="robot_cover"/>
      <geom class="visual" type="box" pos="0.02 0 0.0652" size="0.06 0.05 0.0003" material="robot_panel"/>
{screws}
      <!-- Side seams, front rubber bumper, rear and side vents -->
      <geom class="visual" type="box" pos="0 0.0998 0.012" size="0.11 0.0006 0.0012" material="robot_dark"/>
      <geom class="visual" type="box" pos="0 -0.0998 0.012" size="0.11 0.0006 0.0012" material="robot_dark"/>
      <geom class="visual" type="box" pos="0.1515 0 0.017" size="0.0035 0.07 0.012" material="rubber"/>
      <geom class="visual" type="cylinder" pos="0.148 0.07 0.017" size="0.007 0.012" material="rubber"/>
      <geom class="visual" type="cylinder" pos="0.148 -0.07 0.017" size="0.007 0.012" material="rubber"/>
{vents}
{side_vents}
      <!-- Status lights: two small amber LEDs at the front show which way is forward -->
      <geom name="front_marker" class="visual" type="cylinder" pos="0.135 0.045 0.0592" size="0.004 0.0012" material="status_light"/>
      <geom class="visual" type="cylinder" pos="0.135 -0.045 0.0592" size="0.004 0.0012" material="status_light"/>
      <!-- Lidar: mount plate, base, scanning window, cap -->
      <geom class="visual" type="cylinder" pos="0 0 0.0665" size="0.03 0.0012" material="robot_dark"/>
      <geom class="visual" type="cylinder" pos="0 0 0.0735" size="0.035 0.006" material="lidar_body"/>
      <geom class="visual" type="cylinder" pos="0 0 0.0825" size="0.0332 0.003" material="lidar_window"/>
      <geom class="visual" type="cylinder" pos="0 0 0.0875" size="0.033 0.002" material="lidar_body"/>
      <geom class="visual" type="cylinder" pos="0 0 0.0898" size="0.022 0.0006" material="robot_trim"/>
      <!-- Depth-camera style head on a short bracket; the camera sits just in front of it -->
      <geom class="visual" type="box" pos="0.134 0 0.07" size="0.006 0.012 0.005" material="robot_dark"/>
      <geom class="visual" type="box" pos="0.146 0 0.083" size="0.009 0.036 0.0095" material="camera_body"/>
      <geom class="visual" type="cylinder" pos="0.1552 0.022 0.083" size="0.0055 0.0004" zaxis="1 0 0" material="lens"/>
      <geom class="visual" type="cylinder" pos="0.1552 -0.022 0.083" size="0.0055 0.0004" zaxis="1 0 0" material="lens"/>
      <geom class="visual" type="cylinder" pos="0.1552 0 0.083" size="0.0045 0.0004" zaxis="1 0 0" material="lens"/>
      <camera name="robot_cam" pos="0.158 0 0.083" xyaxes="0 -1 0 0 0 1" fovy="75"/>

      <body name="left_wheel" pos="0 0.12 0">
        <joint name="left_wheel_joint" type="hinge" axis="0 1 0" damping="0.001"/>
        <!-- Mass and inertia pinned to the calibrated 3 cm wide wheel (0.2 kg, r 0.05 m). -->
        <inertial pos="0 0 0" mass="0.2" diaginertia="0.00014 0.00025 0.00014"/>
        <!-- Collision: a narrow ellipsoid (5 cm rolling radius, 8 mm half width) that touches
             the floor at one point, like a crowned tire; a flat cylinder touches at two points
             and scrubs when pivoting. Stiff contact (solref 0.002 s = 4 physics steps, priority
             over the floor) so the body does not wind up on the tire and rock after a turn.
             No sidewall contact (documented limitation). Hidden; the visual tire below is
             the full 3 cm. -->
        <geom name="left_tire" type="ellipsoid" size="0.05 0.05 0.008" zaxis="0 1 0" friction="1.2 0.005 0.0001" priority="1" solref="0.002 1" group="4" rgba="0.1 0.1 0.1 1"/>
        <geom class="visual" type="cylinder" size="0.05 0.015" zaxis="0 1 0" material="rubber"/>
{left}
      </body>
      <body name="right_wheel" pos="0 -0.12 0">
        <joint name="right_wheel_joint" type="hinge" axis="0 1 0" damping="0.001"/>
        <!-- Mass and inertia pinned to the calibrated 3 cm wide wheel (0.2 kg, r 0.05 m). -->
        <inertial pos="0 0 0" mass="0.2" diaginertia="0.00014 0.00025 0.00014"/>
        <!-- Collision: a narrow ellipsoid (5 cm rolling radius, 8 mm half width) that touches
             the floor at one point, like a crowned tire; a flat cylinder touches at two points
             and scrubs when pivoting. Stiff contact (solref 0.002 s = 4 physics steps, priority
             over the floor) so the body does not wind up on the tire and rock after a turn.
             No sidewall contact (documented limitation). Hidden; the visual tire below is
             the full 3 cm. -->
        <geom name="right_tire" type="ellipsoid" size="0.05 0.05 0.008" zaxis="0 1 0" friction="1.2 0.005 0.0001" priority="1" solref="0.002 1" group="4" rgba="0.1 0.1 0.1 1"/>
        <geom class="visual" type="cylinder" size="0.05 0.015" zaxis="0 1 0" material="rubber"/>
{right}
      </body>
    </body>
"""


HEADER = """<!--
  GENERATED by tools/build_world.py - edit that script, not this file.

  World: a 10 m x 10 m indoor office floor (corridor, office, lab, storage,
  reception, doorways with frames and doors fixed open, furniture) and a two-wheeled
  differential-drive robot. Units: meters, kilograms, seconds.

  Geom groups: 0 solid (lidar sees it, collides), 2 visual only (furniture meshes too),
  3 ceiling (robot camera only), 4 hidden colliders: the robot's, and the solid members
  of furniture drawn by a mesh (the lidar sees them and they collide).
-->
<mujoco model="office_floor">
  <compiler angle="degree"/>
  <!-- 2 kHz physics with an elliptic (direction-independent) friction cone. The default
       pyramidal cone made start/stop grip depend on heading (yaw twitch up to 1.1 rad/s),
       and 500 Hz let the tire contact creep forward during acceleration. Measured and
       agreed in review; see tests/test_motion.py. -->
  <option timestep="0.0005" integrator="implicitfast" cone="elliptic" impratio="1"/>

  <visual>
    <global offwidth="2560" offheight="1440"/>
    <quality shadowsize="2048" offsamples="4"/>
    <headlight ambient="0.28 0.28 0.28" diffuse="0.03 0.03 0.03" specular="0.1 0.1 0.1"/>
    <!-- Clip planes are fractions of the model extent; znear keeps cameras from seeing
         through walls they are close to. -->
    <map znear="0.0005" zfar="3" shadowclip="1" shadowscale="1"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="flat" rgb1="0.86 0.87 0.88" rgb2="0.86 0.87 0.88" width="16" height="16"/>
    <texture name="wood_floor" type="2d" file="floor_wood.png"/>
    <texture name="tile" type="2d" file="floor_tile.png"/>
    <texture name="tile_dark" type="2d" file="floor_tile_dark.png"/>
    <texture name="carpet" type="2d" file="carpet.png"/>
    <texture name="plaster" type="2d" file="plaster.png"/>
    <texture name="door_wood" type="2d" file="door_wood.png"/>
    <!-- CC0 photo textures (tools/fetch_textures.py; sources and licences in assets/textures/*.json) -->
    <texture name="tx_plaster" type="2d" file="paint_roller.png"/>
    <texture name="tx_blind" type="2d" file="hessian_230.png"/>
    <texture name="tx_laminate" type="2d" file="laminate_floor_02.png"/>
    <texture name="tx_veneer" type="2d" file="white_oak_veneer.png"/>
    <texture name="tx_ceiling" type="2d" file="OfficeCeiling001.png"/>
    <texture name="tx_carpet" type="2d" file="Carpet012.png"/>
    <texture name="tx_porcelain" type="2d" file="Tiles040.png"/>
    <texture name="tx_concrete" type="2d" file="Concrete031.png"/>
    <texture name="sign_office" type="2d" file="label_office.png"/>
    <texture name="sign_lab" type="2d" file="label_lab.png"/>
    <texture name="sign_storage" type="2d" file="label_storage.png"/>
    <texture name="sign_reception" type="2d" file="label_reception.png"/>
    <texture name="outdoor_view" type="2d" file="outdoor_view.png"/>
    <texture name="rug_red" type="2d" file="rug_red.png"/>
    <texture name="rug_blue" type="2d" file="rug_blue.png"/>
    <texture name="painting_1" type="2d" file="painting_1.png"/>
    <texture name="painting_2" type="2d" file="painting_2.png"/>
    <texture name="whiteboard" type="2d" file="whiteboard.png"/>
    <texture name="tv_screen" type="2d" file="tv_screen.png"/>
    <!-- floors and walls at their real-world scales (texrepeat: repeats per metre) -->
    <!-- reflectance renders only on the first reflective geom in the model, the base plane (the lab and
         corridor floor); on every other material it does nothing under the classic renderer -->
    <material name="wood_floor" texture="tx_laminate" texrepeat="0.588 0.588" texuniform="true" emission="0" specular="0.3" shininess="0.3" reflectance="0.08" rgba="0.6 0.6 0.6 1"/>
    <material name="tile" texture="tx_porcelain" texrepeat="0.5 0.5" texuniform="true" specular="0.5" shininess="0.4" reflectance="0.22" rgba="0.52 0.52 0.52 1" emission="0"/>
    <material name="tile_dark" texture="tx_concrete" texrepeat="0.25 0.25" texuniform="true" emission="0" specular="0.2" shininess="0.2" reflectance="0.05" rgba="0.7 0.7 0.7 1"/>
    <material name="carpet" texture="tx_carpet" texrepeat="1 1" texuniform="true" emission="0" specular="0" shininess="0" rgba="0.6 0.6 0.6 1"/>
    <material name="plaster" texture="tx_plaster" texrepeat="1 1" texuniform="true" rgba="0.97 0.95 0.91 1" emission="0.33" specular="0.1" shininess="0.1"/>
    <material name="plaster_reveal" rgba="0.87 0.85 0.81 1" emission="0.33" specular="0.1" shininess="0.1"/>
    <material name="ceiling" texture="tx_ceiling" texrepeat="0.278 0.278" texuniform="true" rgba="0.97 0.96 0.94 1" emission="0.72"/>
    <material name="light_panel" rgba="1 0.98 0.92 1" emission="0.75"/>
    <material name="fitting_frame" rgba="0.95 0.95 0.94 1" emission="0.35" specular="0.3"/>
    <material name="skirting" rgba="1 1 0.98 1" emission="0.33" specular="0.25" shininess="0.5"/>
    <material name="skirting_bead" rgba="0.8 0.8 0.78 1" emission="0.3" specular="0.12" shininess="0.5"/>
    <material name="shadow_line" rgba="0.32 0.31 0.3 1" specular="0"/>
    <material name="threshold" rgba="0.72 0.73 0.74 1" emission="0.26" specular="0.8" shininess="0.7"/>
    <material name="kick_plate" rgba="0.62 0.63 0.65 1" emission="0.32" specular="0.9" shininess="0.8"/>
    <material name="kick_band" rgba="0.76 0.77 0.79 1" emission="0.36" specular="0.9" shininess="0.8"/>
    <material name="closer" rgba="0.62 0.63 0.65 1" emission="0.26" specular="0.6" shininess="0.6"/>
    <material name="frame" rgba="1 1 0.99 1" emission="0.3" specular="0.12" shininess="0.4"/>
    <material name="door" texture="tx_veneer" texrepeat="1 1" texuniform="false" specular="0.25" shininess="0.3"/>
    <material name="door_b" texture="tx_veneer" texrepeat="1.6 1" texuniform="false" specular="0.25" shininess="0.3"/>
    <material name="strike_plate" rgba="0.7 0.71 0.73 1" emission="0.26" specular="0.9" shininess="0.8"/>
    <material name="desk_wood" texture="door_wood" texrepeat="1 1" texuniform="true" specular="0.25"/>
    <material name="desk_body" rgba="0.55 0.56 0.58 1" specular="0.2"/>
    <material name="metal" rgba="0.62 0.64 0.67 1" specular="0.5" shininess="0.6"/>
    <material name="lab_white" rgba="0.9 0.91 0.92 1" specular="0.3"/>
    <material name="lab_top" rgba="0.24 0.26 0.28 1" specular="0.4"/>
    <material name="fabric" rgba="0.36 0.42 0.5 1"/>
    <material name="pot" rgba="0.82 0.8 0.76 1"/>
    <material name="leaves" rgba="0.2 0.42 0.2 1" specular="0.2"/>
    <material name="leaves_light" rgba="0.33 0.55 0.27 1" specular="0.2"/>
    <material name="stem" rgba="0.3 0.38 0.2 1"/>
    <material name="soil" rgba="0.25 0.18 0.12 1"/>
    <material name="metal_dark" rgba="0.36 0.38 0.41 1" specular="0.4"/>
    <material name="chair_base" rgba="0.12 0.12 0.13 1" specular="0.5" shininess="0.6"/>
    <material name="fabric_light" rgba="0.44 0.5 0.58 1"/>
    <material name="monitor" rgba="0.08 0.08 0.09 1" specular="0.4"/>
    <material name="screen" rgba="0.12 0.2 0.3 1" emission="0.35" specular="0.8"/>
    <material name="glass_ware" rgba="0.75 0.85 0.9 0.45" specular="0.9" shininess="0.9"/>
    <material name="cardboard_dark" rgba="0.6 0.47 0.32 1"/>
    <material name="book_red" rgba="0.55 0.16 0.14 1"/>
    <material name="book_blue" rgba="0.16 0.26 0.46 1"/>
    <material name="book_green" rgba="0.16 0.36 0.24 1"/>
    <material name="book_cream" rgba="0.86 0.82 0.7 1"/>
    <material name="book_dark" rgba="0.14 0.14 0.16 1"/>
    <material name="cardboard" rgba="0.72 0.58 0.4 1"/>
    <material name="shelf_board" rgba="0.8 0.81 0.83 1" specular="0.4"/>
    <material name="bin_blue" rgba="0.25 0.4 0.62 1"/>
    <material name="bin_grey" rgba="0.5 0.52 0.55 1"/>
    <material name="seam" rgba="0.35 0.36 0.38 1"/>
    <material name="handle" rgba="0.7 0.71 0.73 1" emission="0.26" specular="0.8" shininess="0.7"/>
    <material name="glass" texture="outdoor_view" emission="0.95" specular="0.6" shininess="0.9"/>
    <material name="window_frame" rgba="0.98 0.98 0.97 1" emission="0.27" specular="0.12" shininess="0.4"/>
    <material name="sill" rgba="0.8 0.8 0.79 1" emission="0.2" specular="0.08" shininess="0.45"/>
    <material name="blind" texture="tx_blind" texrepeat="14 14" texuniform="true" rgba="0.97 0.96 0.93 1" emission="0.56" specular="0"/>
    <material name="blind_rail" rgba="0.82 0.82 0.8 1" specular="0.5"/>
    <texture name="view_0" type="2d" file="view_0.png"/>
    <material name="view_0" texture="view_0" texuniform="false" emission="0.59" specular="0.3" shininess="0.9"/>
    <texture name="view_1" type="2d" file="view_1.png"/>
    <material name="view_1" texture="view_1" texuniform="false" emission="0.59" specular="0.3" shininess="0.9"/>
    <texture name="view_2" type="2d" file="view_2.png"/>
    <material name="view_2" texture="view_2" texuniform="false" emission="0.59" specular="0.3" shininess="0.9"/>
    <texture name="view_3" type="2d" file="view_3.png"/>
    <material name="view_3" texture="view_3" texuniform="false" emission="0.59" specular="0.3" shininess="0.9"/>
    <texture name="view_4" type="2d" file="view_4.png"/>
    <material name="view_4" texture="view_4" texuniform="false" emission="0.59" specular="0.3" shininess="0.9"/>
    <texture name="view_5" type="2d" file="view_5.png"/>
    <material name="view_5" texture="view_5" texuniform="false" emission="0.64" specular="0.3" shininess="0.9"/>
    <texture name="view_6" type="2d" file="view_6.png"/>
    <material name="view_6" texture="view_6" texuniform="false" emission="0.64" specular="0.3" shininess="0.9"/>
    <texture name="view_7" type="2d" file="view_7.png"/>
    <material name="view_7" texture="view_7" texuniform="false" emission="0.64" specular="0.3" shininess="0.9"/>
    <texture name="view_8" type="2d" file="view_8.png"/>
    <material name="view_8" texture="view_8" texuniform="false" emission="0.59" specular="0.3" shininess="0.9"/>
    <texture name="view_9" type="2d" file="view_9.png"/>
    <material name="view_9" texture="view_9" texuniform="false" emission="0.59" specular="0.3" shininess="0.9"/>
    <texture name="tx_rug_red" type="2d" file="rug_red_woven.png"/>
    <texture name="tx_rug_blue" type="2d" file="rug_blue_woven.png"/>
    <material name="rug_red" texture="tx_rug_red" emission="0" specular="0" shininess="0" rgba="0.7 0.7 0.7 1"/>
    <material name="rug_blue" texture="tx_rug_blue" emission="0" specular="0" shininess="0" rgba="0.7 0.7 0.7 1"/>
    <material name="rug_red_edge" rgba="0.119 0.028 0.025 1" specular="0"/>
    <material name="rug_blue_edge" rgba="0.028 0.042 0.091 1" specular="0"/>
    <material name="painting_1" texture="painting_1" emission="0.1"/>
    <material name="painting_2" texture="painting_2" emission="0.1"/>
    <material name="whiteboard" texture="whiteboard" emission="0.15" specular="0.5"/>
    <material name="tv_screen" texture="tv_screen" specular="0.9" shininess="0.9"/>
    <material name="bin_dark" rgba="0.22 0.23 0.25 1" specular="0.3"/>
    <material name="pallet_wood" rgba="0.62 0.5 0.34 1"/>
    <material name="extinguisher" rgba="0.75 0.08 0.06 1" specular="0.6"/>
    <material name="lamp_shade" rgba="0.96 0.92 0.82 1" emission="0.5"/>
    <material name="sign_office" texture="sign_office"/>
    <material name="sign_lab" texture="sign_lab"/>
    <material name="sign_storage" texture="sign_storage"/>
    <material name="sign_reception" texture="sign_reception"/>
    <material name="robot_body" rgba="0.2 0.21 0.23 1" specular="0.35" shininess="0.5"/>
    <material name="robot_cover" rgba="0.8 0.81 0.83 1" specular="0.4" shininess="0.6"/>
    <material name="robot_dark" rgba="0.08 0.08 0.09 1" specular="0.3"/>
    <material name="rubber" rgba="0.06 0.06 0.06 1" specular="0.05"/>
    <material name="hub" rgba="0.6 0.62 0.64 1" specular="0.7" shininess="0.8"/>
    <material name="lidar_window" rgba="0.1 0.25 0.3 1" specular="0.9" shininess="0.9"/>
    <material name="lens" rgba="0.05 0.1 0.2 1" specular="1" shininess="1"/>
    <material name="status_light" rgba="1 0.62 0.12 1" emission="0.9"/>
    <material name="robot_trim" rgba="0.32 0.33 0.36 1" specular="0.4" shininess="0.5"/>
    <material name="robot_panel" rgba="0.72 0.73 0.75 1" specular="0.3"/>
    <material name="vent" rgba="0.03 0.03 0.035 1"/>
    <material name="tire_wall" rgba="0.11 0.11 0.115 1" specular="0.08"/>
    <material name="tread" rgba="0.09 0.09 0.09 1" specular="0.02"/>
    <material name="lidar_body" rgba="0.1 0.1 0.11 1" specular="0.5" shininess="0.7"/>
    <material name="camera_body" rgba="0.14 0.14 0.15 1" specular="0.4"/>
  </asset>

  <default>
    <default class="visual">
      <geom contype="0" conaffinity="0" group="2" mass="0"/>
    </default>
    <default class="ceiling">
      <geom contype="0" conaffinity="0" group="3" mass="0"/>
    </default>
  </default>

  <worldbody>
"""

FOOTER = """
    <!-- Goal marker: moved at every reset; visual only -->
    <body name="goal" mocap="true" pos="2 2 0">
      <geom name="goal_disc" class="visual" type="cylinder" size="0.3 0.002" pos="0 0 0.003" rgba="0.1 0.75 0.35 0.45"/>
      <geom name="goal_pole" class="visual" type="cylinder" size="0.012 0.3" pos="0 0 0.3" rgba="0.1 0.75 0.35 1"/>
      <geom name="goal_flag" class="visual" type="box" size="0.07 0.004 0.045" pos="0.07 0 0.55" rgba="0.1 0.75 0.35 1"/>
    </body>
""" + robot_xml() + """  </worldbody>

  <!-- Wheel motors: each drives its wheel toward a target speed (rad/s).
       Gains are tuned in the step 5 motion check. -->
  <actuator>
    <!-- Wheel motor torque is limited to 0.3 N m, typical of a small gear motor. A much
         stronger motor locked the wheels in a hard stop and slammed the robot onto a caster. -->
    <velocity name="left_motor" joint="left_wheel_joint" kv="2.0" ctrlrange="-25 25" forcerange="-0.3 0.3"/>
    <velocity name="right_motor" joint="right_wheel_joint" kv="2.0" ctrlrange="-25 25" forcerange="-0.3 0.3"/>
  </actuator>
</mujoco>
"""


def generate() -> str:
    """The full world.xml text (deterministic)."""
    geoms.clear()
    furniture_assets.clear()
    furniture_items.clear()
    placed_models.clear()
    _assets_done.clear()
    build_floor()
    header = HEADER.replace("  </asset>", "\n".join(furniture_assets + ["  </asset>"]), 1)
    return header + "\n".join(geoms) + "\n" + FOOTER


def main() -> None:
    OUT.write_text(generate(), encoding="utf-8", newline="\n")
    print("wrote", OUT, f"({len(geoms)} lines of geometry)")


if __name__ == "__main__":
    main()
