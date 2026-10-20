"""
Checklist lookup logic — pure data, no ROS, no hardware.

Loads config/checklists.yaml, which maps a mode name to the list of items
required for that mode. Fill in your real item names tonight.
"""
from typing import Dict, List
import yaml


def load_checklists(path: str) -> Dict[str, List[str]]:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    return data.get("modes", {})


def get_required_items(checklists: Dict[str, List[str]], mode: str) -> List[str]:
    return list(checklists.get(mode, []))
