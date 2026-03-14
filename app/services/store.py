from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.models import TaskRecord


class TaskStore:
    CONTRACT_FILES = {
        "physical_manifest": "01_physical_manifest.json",
        "raw_scene_descriptions": "02_raw_scene_descriptions.json",
        "character_bank": "03_character_bank.json",
        "aligned_storyboard": "04_aligned_storyboard.json",
        "final_production_table": "05_final_production_table.json",
    }

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def task_dir(self, task_id: str) -> Path:
        path = self.root_dir / task_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def manifest_path(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "manifest.json"

    def log_path(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "run.log"

    def contract_path(self, task_id: str, contract_name: str) -> Path:
        if contract_name not in self.CONTRACT_FILES:
            raise ValueError(f"unknown contract: {contract_name}")
        return self.task_dir(task_id) / self.CONTRACT_FILES[contract_name]

    def index_path(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "index.json"

    def markdown_path(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "final_prompts.md"

    def save_manifest(self, record: TaskRecord) -> None:
        path = self.manifest_path(record.task_id)
        path.write_text(
            json.dumps(record.to_manifest(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_all(self) -> list[TaskRecord]:
        records: list[TaskRecord] = []
        for manifest in sorted(self.root_dir.glob("*/manifest.json")):
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                records.append(TaskRecord.from_manifest(payload))
            except Exception:  # noqa: BLE001
                continue
        return records

    def write_contract(self, task_id: str, contract_name: str, payload: dict[str, Any]) -> Path:
        path = self.contract_path(task_id, contract_name)
        self.write_json(path, payload)
        return path

    def write_index(self, task_id: str, payload: dict[str, Any]) -> Path:
        path = self.index_path(task_id)
        self.write_json(path, payload)
        return path

    def write_markdown(self, task_id: str, content: str) -> Path:
        path = self.markdown_path(task_id)
        path.write_text(content, encoding="utf-8")
        return path

    def write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def append_log(self, task_id: str, message: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self.log_path(task_id).open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] {message}\n")
