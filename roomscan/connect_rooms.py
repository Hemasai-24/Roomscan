"""Step 4c: which rooms connect to which (through a door), for the whole-property plan."""
import numpy as np

NEAR = 0.4   # a door this close to another room leads into it


def _door_mid(room, op):
    w = next(x for x in room["walls"] if x["id"] == op["wall_id"])
    a, b = np.array(w["start"]), np.array(w["end"])
    u = (b - a) / np.linalg.norm(b - a)
    return a + u * (op["offset_along_wall"] + op["width"]["value"] / 2)


def _dist_to_mask(pt, mask, grid):
    ys, xs = np.nonzero(mask)
    cells = grid.lo + (np.c_[xs, ys] + 0.5) * grid.cell
    return float(np.min(np.linalg.norm(cells - pt, axis=1)))


def adjacency(rooms, masks, grid):
    out = []
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            via = None
            for r, other in ((i, j), (j, i)):
                for op in rooms[r]["openings"]:
                    if op["type"] == "door" and _dist_to_mask(_door_mid(rooms[r], op), masks[other], grid) < NEAR:
                        via = op["id"]
                        break
                if via:
                    break
            if via:
                out.append({"room_a": rooms[i]["id"], "room_b": rooms[j]["id"], "via": via})
    return out
