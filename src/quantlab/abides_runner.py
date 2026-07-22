"""Helpers for running / importing the vendored ABIDES tree."""

from __future__ import annotations

import sys
from pathlib import Path


def abides_root() -> Path:
    """Path to ``third_party/abides`` inside this repository."""
    return Path(__file__).resolve().parents[2] / "third_party" / "abides"


def ensure_abides_on_path() -> Path:
    """Insert the ABIDES root on ``sys.path`` if present; return the path."""
    root = abides_root()
    if not root.is_dir():
        raise FileNotFoundError(
            f"ABIDES not found at {root}. Expected vendored copy under third_party/abides."
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def abides_available() -> bool:
    return abides_root().is_dir() and (abides_root() / "Kernel.py").is_file()


def agent_module_path() -> Path:
    return (
        abides_root()
        / "contributed_traders"
        / "kchebs_myagent"
        / "kchebs_myagent.py"
    )


def agent_description() -> str:
    """Short description of the custom ABIDES trading agent."""
    return (
        "EMA mid-price momentum (fast vs slow window) combined with "
        "inventory-aware bid/ask skew and end-of-day flat; cancels resting "
        "orders on each wake, pauses after fills to avoid over-trading."
    )
