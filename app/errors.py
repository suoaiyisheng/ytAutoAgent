from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Stage1Error(Exception):
    code: str
    message: str
    http_status: int = 400

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"
