"""
Batch-of-3 planning — no ROS, no hardware, no camera.

One photo of the belt shows 3 blocks in a line: the mode's required items plus one reject (anything
that is not on the mode's checklist). From the photo we work out the order of the blocks along the belt,
leading block first (the one closest to the pickup point), and then how far the belt has to advance
before each block sits on the pickup point. Rejects are never picked: the belt keeps running and they
roll off the end.

Geometry (mm, from the build): blocks are BLOCK_MM long along the belt, GAP_MM apart edge to edge, and
the leading block's near edge starts FIRST_EDGE_MM before the pickup point.
"""
from typing import Dict, List, Optional

import numpy as np

BLOCK_MM = 25.0
GAP_MM = 24.0
FIRST_EDGE_MM = 25.0
BATCH_SIZE = 3

# direction the belt carries the blocks in the image, as an image-space unit vector (x right, y down)
_TOWARD = {"right": (1.0, 0.0), "left": (-1.0, 0.0), "down": (0.0, 1.0), "up": (0.0, -1.0)}


class BatchError(RuntimeError):
    """The photo does not show a usable batch. Nothing should move."""


class OrderError(BatchError):
    """The right blocks are there but not in the order the mode needs (e.g. the stack base must come first)."""


def pitch_mm() -> float:
    return BLOCK_MM + GAP_MM


def advance_distances(widths: Optional[List[float]] = None, n: int = BATCH_SIZE) -> List[float]:
    """Belt travel (mm) from one stop to the next, leading block first. `widths` = each slot's length along the belt
    (default BLOCK_MM). First stop brings block 0's centre onto the pickup point (its near edge starts FIRST_EDGE_MM
    before it); each later stop moves half the previous block + the gap + half the next block."""
    w = list(widths) if widths else [BLOCK_MM] * n
    out = [FIRST_EDGE_MM + w[0] / 2.0]
    for i in range(1, len(w)):
        out.append(w[i - 1] / 2.0 + GAP_MM + w[i] / 2.0)
    return out


MAX_SIZE_RATIO = 1.7   # largest/smallest block size within a batch (perspective makes near blocks bigger)
MAX_SPAN = 5.5    # farthest two blocks of one batch may be apart, in block sizes (3 blocks in a row ~ 4)


def select_batch(dets: List[Dict], required: List[str]) -> List[Dict]:
    """Keep only the blocks that form the batch: the tightest group made of one detection per required item,
    plus any other (non-required) detection lying close to that group. Stray blocks elsewhere in the picture
    (a spare on the table etc.) are dropped. If a required item is missing, returns `dets` unchanged so
    plan_slots reports it."""
    import itertools
    import math
    pools = [[d for d in dets if d["name"] == item] for item in required]
    if not all(pools):
        return dets

    def size(d):
        return math.sqrt(d["w"] * d["h"])

    def span(group):
        if max(size(d) for d in group) > MAX_SIZE_RATIO * min(size(d) for d in group):
            return float("inf")                               # blocks of one batch are about the same size
        sz = sum(size(d) for d in group) / len(group)
        return max(math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]) for a, b in itertools.combinations(group, 2)) / sz

    best = min(itertools.product(*pools), key=span) if len(required) > 1 else (pools[0][0],)
    if len(best) > 1 and span(best) > MAX_SPAN:
        return dets                                           # no plausible batch: let plan_slots refuse
    chosen = list(best)
    cx = sum(d["cx"] for d in chosen) / len(chosen)
    cy = sum(d["cy"] for d in chosen) / len(chosen)
    sz = sum(size(d) for d in chosen) / len(chosen)
    for d in dets:
        if d["name"] not in required and math.hypot(d["cx"] - cx, d["cy"] - cy) <= MAX_SPAN * sz / 2 + sz:
            chosen.append(d)
    return chosen


def plan_slots(dets: List[Dict], required: List[str], toward: str = "right",
               must_order: Optional[List[str]] = None) -> List[Optional[str]]:
    """Slot list, leading block first, length BATCH_SIZE. Each entry is a required item name, or None (reject).

    dets: [{"name", "conf", "cx", "cy", "w", "h"}], already confidence-filtered and including the reject
    (name "reject" or any class not in `required`). The order is taken along the line the blocks lie on, pointed
    the way `toward` says the belt moves in the image, so a tilted or rotated camera view still works.
    Raises BatchError unless the photo shows exactly one of each required item plus rejects filling the batch."""
    if toward not in _TOWARD:
        raise BatchError(f"toward must be one of {sorted(_TOWARD)}")
    for item in required:
        n = sum(1 for d in dets if d["name"] == item)
        if n != 1:
            raise BatchError(f"expected exactly one {item}, saw {n}")
    if len(dets) != BATCH_SIZE:
        raise BatchError(f"saw {len(dets)} blocks, expected {BATCH_SIZE} ({len(required)} required + the reject)")

    from kitting_logic.blob_logic import belt_axis
    axis = belt_axis(dets)
    if axis @ np.array(_TOWARD[toward]) < 0:
        axis = -axis
    ordered = sorted(dets, key=lambda d: d["cx"] * axis[0] + d["cy"] * axis[1], reverse=True)
    slots = [d["name"] if d["name"] in required else None for d in ordered]
    check_order(slots, must_order)
    return slots


def check_order(slots: List[Optional[str]], must_order: Optional[List[str]]) -> None:
    """Raise OrderError unless the items of `must_order` appear in the slots in that order (leading block first)."""
    if must_order:
        got = [x for x in slots if x in must_order]
        if got != list(must_order):
            raise OrderError(f"not in order of height: {' must come before '.join(must_order)} "
                             f"(leading block first, the photo shows {' then '.join(got)})")


def batch_steps(slots: List[Optional[str]], belt_mm_per_s: float, sizes: Optional[Dict[str, float]] = None,
                reject_mm: float = BLOCK_MM) -> List[Dict]:
    """Ordered actions: ("belt", seconds) to advance, ("pick", item) for a required block, ("skip", None) for a reject.
    The belt is stopped for picks and keeps running past rejects. A trailing flush clears the last block."""
    steps: List[Dict] = []
    sizes = sizes or {}
    widths = [sizes.get(slot, BLOCK_MM) if slot else reject_mm for slot in slots]
    for slot, dist in zip(slots, advance_distances(widths)):
        steps.append({"action": "belt", "seconds": dist / belt_mm_per_s})
        steps.append({"action": "pick", "item": slot} if slot else {"action": "skip"})
    return steps
