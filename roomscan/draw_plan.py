"""Step 7b: draw the floor plan (walls with lengths and ranges, doors orange, windows blue) to SVG/PNG."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["svg.fonttype"] = "none"      # keep labels as searchable text in the SVG
DAMAGE_COLOR = "#d7191c"


def _draw_damage(ax, plan, centres):
    rooms = {r["id"]: r for r in plan["rooms"]}
    for d in plan.get("damage", []):
        room = rooms.get(d["room_id"])
        if room is None:
            continue
        wall = next((w for w in room["walls"] if w["id"] == d["surface_id"]), None)
        if wall is not None and "offset_along_wall" in d:
            a, b = np.array(wall["start"], float), np.array(wall["end"], float)
            p = a + (b - a) / max(np.linalg.norm(b - a), 1e-9) * d["offset_along_wall"]
        else:                                   # floor / ceiling damage: at the room centre
            p = centres[room["id"]] + np.array([0.0, 0.35])
        ax.plot(*p, marker="X", ms=11, color=DAMAGE_COLOR, zorder=5)
        ax.text(p[0], p[1] - 0.15, f"{d['id']} {d['class']} {d['area']['value']:.2f} m²", color=DAMAGE_COLOR,
                fontsize=7, ha="center", zorder=5)
    flags = plan.get("concealed_damage_flags", [])
    if flags:
        lines = [f"{f['rule_id']} ({f['severity']}): {f['description']}  [{', '.join(f['damage_ids'])}]"
                 for f in flags]
        ax.figure.text(0.02, 0.01, "Concealed-damage flags:\n" + "\n".join(lines), fontsize=8,
                       color=DAMAGE_COLOR, va="bottom")


def _fmt(m):
    return f"{m['value']:.2f} m (±{(m['hi'] - m['lo']) / 2 * 100:.0f} cm)"


def setup_axes(ax):
    """Top-down view: looking down from +Y, plan v (world z, rotated) points DOWN the page."""
    ax.set_aspect("equal")
    ax.axis("off")
    ax.invert_yaxis()


def render(plan, out_stem):
    fig, ax = plt.subplots(figsize=(12, 12))
    centres = {}
    for room in plan["rooms"]:
        poly = np.array(room["polygon"] + [room["polygon"][0]])
        ax.fill(poly[:, 0], poly[:, 1], color="#f3efe6", zorder=0)
        ax.plot(poly[:, 0], poly[:, 1], color="#333", lw=3, zorder=1)
        walls = {w["id"]: w for w in room["walls"]}
        for w in room["walls"]:
            a, b = np.array(w["start"]), np.array(w["end"])
            mid = (a + b) / 2
            d = b - a
            nrm = np.array([d[1], -d[0]]) / (np.linalg.norm(d) + 1e-9)
            ang = np.degrees(np.arctan2(d[1], d[0]))
            ang = ang - 180 if ang > 90 else ang + 180 if ang < -90 else ang   # keep text upright
            ax.text(*(mid + nrm * 0.25), _fmt(w["length"]), ha="center", va="center", fontsize=8, rotation=ang)
        for o in room["openings"]:
            w = walls[o["wall_id"]]
            a, b = np.array(w["start"]), np.array(w["end"])
            u = (b - a) / np.linalg.norm(b - a)
            p0 = a + u * o["offset_along_wall"]
            p1 = p0 + u * o["width"]["value"]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#2b8cbe" if o["type"] == "window" else "#e6550d",
                    lw=6, zorder=2)
        c = poly[:-1].mean(0)
        centres[room["id"]] = c
        ax.text(*c, f"{room['id']}\n{room['floor_area']['value']:.2f} m²\n"
                    f"ceiling {room['ceiling_height']['value']:.2f} m", ha="center", fontsize=10)
    for link in plan.get("adjacency", []):
        a, b = centres.get(link["room_a"]), centres.get(link["room_b"])
        if a is not None and b is not None:
            ax.plot([a[0], b[0]], [a[1], b[1]], ls="--", color="#999", lw=1, zorder=3)
    _draw_damage(ax, plan, centres)
    setup_axes(ax)
    ax.set_title(f"{plan['capture']['id']} · tier: {plan['capture']['tier']}")
    for ext in ("svg", "png"):
        fig.savefig(f"{out_stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
