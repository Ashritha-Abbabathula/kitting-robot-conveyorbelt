"""
One-item-at-a-time kitting flow — no ROS, no hardware, no camera.

The operator puts items on the shared pickup spot one after another; the
camera confirms the *next expected* item is there (and stays there), then the
arm moves just that item to its place target. Used by kitting_run.py.
"""
import math
from typing import Dict, List, Optional

from kitting_logic.dobot_logic import build_sequence


class StableTracker:
    """True only once something has been present continuously for
    `hold_seconds`. One missed frame resets it, which stops flicker and a
    hand passing through the view from triggering the arm."""

    def __init__(self, hold_seconds: float):
        self.hold_seconds = hold_seconds
        self._since: Optional[float] = None

    def update(self, present: bool, now: float) -> bool:
        if not present:
            self._since = None
            return False
        if self._since is None:
            self._since = now
        return (now - self._since) >= self.hold_seconds

    def progress(self, now: float) -> float:
        """0.0-1.0 for an on-screen progress hint."""
        if self._since is None:
            return 0.0
        return min(1.0, (now - self._since) / self.hold_seconds) if self.hold_seconds > 0 else 1.0

    def reset(self):
        self._since = None


def ordered_items(coords: dict, required: List[str], mode: str) -> List[str]:
    """Items in the order they must be moved: the order of coords['place'][mode]
    (e.g. cube before the red block that goes on top of it), restricted to the
    mode's required items. Falls back to the checklist order."""
    placements = coords.get("place", {}).get(mode)
    if not placements:
        return list(required)
    order = [p["item"] for p in placements if p["item"] in required]
    return order + [i for i in required if i not in order]


def item_steps(item: str, coords: dict, mode: str) -> List[Dict]:
    """Move/suck steps for ONE item, without the home actions build_sequence
    brackets a run with (we don't re-home the arm between items)."""
    return [s for s in build_sequence([item], coords, mode) if s["action"] != "home"]


def max_radius(steps: List[Dict]) -> float:
    """Largest horizontal distance (mm) from the arm base over all move steps —
    for a reach warning before anything moves."""
    radii = [math.hypot(s["x"], s["y"]) for s in steps if s["action"] == "move"]
    return max(radii) if radii else 0.0
