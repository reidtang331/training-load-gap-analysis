"""Configuration loading. One source of truth: config/config.yaml."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG) -> "Config":
        with open(path) as fh:
            return cls(yaml.safe_load(fh))

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def enabled_dimensions(self) -> dict[str, dict]:
        """Dimensions currently switched on. `density` ships disabled -- see config."""
        return {k: v for k, v in self.raw["dimensions"].items() if v.get("enabled", True)}

    @property
    def dimension_metrics(self) -> list[str]:
        out: list[str] = []
        for spec in self.enabled_dimensions.values():
            out.extend(spec["metrics"])
        return sorted(set(out))


def paths() -> dict[str, Path]:
    return {
        "root": ROOT,
        "raw": ROOT / "data" / "raw",
        "interim": ROOT / "data" / "interim",
        "processed": ROOT / "data" / "processed",
        "outputs": ROOT / "outputs",
    }
