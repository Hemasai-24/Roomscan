"""Step 7b: draw the floor plan (walls with lengths and ranges, doors orange, windows blue) to SVG/PNG."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _fmt(m):
    return f"{m['value']:.2f} m (±{(m['hi'] - m['lo']) / 2 * 100:.0f} cm)"


def setup_axes(ax):
    """Top-down view: looking down from +Y, plan v (world z, rotated) points DOWN the page."""
    ax.set_aspect("equal")
    ax.axis("off")
    ax.invert_yaxis()


def render(plan, out_stem):
    fig, ax = plt.subplots(figsize=(10, 10))
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
        ax.text(*c, f"{room['name']}\n{room['floor_area']['value']:.2f} m²\n"
                    f"ceiling {room['ceiling_height']['value']:.2f} m", ha="center", fontsize=10)
    setup_axes(ax)
    ax.set_title(f"{plan['capture']['id']} · tier: {plan['capture']['tier']}")
    for ext in ("svg", "png"):
        fig.savefig(f"{out_stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
