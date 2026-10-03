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


MIN_LABELLED_WALL = 0.5     # m: shorter wall pieces are drawn but not labelled (keeps the plan readable)
DOOR_COLOR, WINDOW_COLOR = "#e6550d", "#2b8cbe"


def _half(m):
    return (m["hi"] - m["lo"]) / 2


def wall_label(w):
    L = w["length"]
    if L["value"] < MIN_LABELLED_WALL:
        return None
    return f"{L['value']:.2f} m ±{_half(L) * 100:.0f} cm"


def room_label(room):
    a = room["floor_area"]
    ceiling = (f"ceiling {room['ceiling_height']['value']:.2f} m" if room.get("ceiling_observed")
               else "ceiling not seen")
    return f"{room.get('name') or room.get('id', '')}\n{a['value']:.1f} m² ±{_half(a):.1f}\n{ceiling}"


def setup_axes(ax):
    """Top-down view: looking down from +Y, plan v (world z, rotated) points DOWN the page."""
    ax.set_aspect("equal")
    ax.axis("off")
    ax.invert_yaxis()


def render(plan, out_stem):
    from matplotlib.lines import Line2D
    from shapely.geometry import Polygon
    fig, ax = plt.subplots(figsize=(12, 12))
    centres = {}
    for room in plan["rooms"]:
        poly = np.array(room["polygon"] + [room["polygon"][0]])
        ax.fill(poly[:, 0], poly[:, 1], color="#f3efe6", zorder=0)
        ax.plot(poly[:, 0], poly[:, 1], color="#333", lw=2.5, zorder=1)
        walls = {w["id"]: w for w in room["walls"]}
        for w in room["walls"]:
            text = wall_label(w)
            if text is None:
                continue
            a, b = np.array(w["start"]), np.array(w["end"])
            d = b - a
            nrm = np.array([d[1], -d[0]]) / (np.linalg.norm(d) + 1e-9)     # outward for a CCW outline
            ang = np.degrees(np.arctan2(d[1], d[0]))
            ang = ang - 180 if ang > 90 else ang + 180 if ang < -90 else ang   # keep text upright
            ax.text(*((a + b) / 2 + nrm * 0.25), text, ha="center", va="center", fontsize=7, color="#444",
                    rotation=-ang, rotation_mode="anchor")
        for o in room["openings"]:
            w = walls[o["wall_id"]]
            a, b = np.array(w["start"]), np.array(w["end"])
            u = (b - a) / np.linalg.norm(b - a)
            p0 = a + u * o["offset_along_wall"]
            p1 = p0 + u * o["width"]["value"]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=WINDOW_COLOR if o["type"] == "window" else DOOR_COLOR,
                    lw=6, zorder=2, solid_capstyle="butt")
        rp = Polygon(room["polygon"]).representative_point()          # always inside the room
        centres[room["id"]] = np.array([rp.x, rp.y])
        ax.text(rp.x, rp.y, room_label(room), ha="center", va="center", fontsize=9, zorder=6,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", alpha=0.85))
    for link in plan.get("adjacency", []):
        a, b = centres.get(link["room_a"]), centres.get(link["room_b"])
        if a is not None and b is not None:
            ax.plot([a[0], b[0]], [a[1], b[1]], ls="--", color="#999", lw=1, zorder=3)
    _draw_damage(ax, plan, centres)
    setup_axes(ax)
    allp = np.concatenate([np.array(r["polygon"]) for r in plan["rooms"]]) if plan["rooms"] else np.zeros((1, 2))
    x0, y1 = allp[:, 0].min(), allp[:, 1].max()
    ax.plot([x0, x0 + 1.0], [y1 + 0.5, y1 + 0.5], color="k", lw=3)               # 1 m scale bar
    ax.text(x0 + 0.5, y1 + 0.75, "1 m", ha="center", fontsize=8)
    handles = [Line2D([], [], color=DOOR_COLOR, lw=6, label="door"),
               Line2D([], [], color=WINDOW_COLOR, lw=6, label="window"),
               Line2D([], [], color="#999", ls="--", label="rooms connected"),
               Line2D([], [], color=DAMAGE_COLOR, marker="X", ls="", ms=9, label="damage")]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    total = sum(r["floor_area"]["value"] for r in plan["rooms"])
    ax.set_title(f"{plan['capture']['id']}  ·  {plan['capture']['tier']} tier  ·  {len(plan['rooms'])} rooms  ·  "
                 f"{total:.1f} m²   (ranges are 95 %)", fontsize=11)
    for ext in ("svg", "png"):
        fig.savefig(f"{out_stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
