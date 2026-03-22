from __future__ import annotations

"""
模块作用：
- 负责 Stage 2 原始分镜描述生成，把视觉结果收敛为文档约定的 `subjects + desc` 契约。

当前文件工具函数：
- `_normalize_raw_descriptions`：统一 Stage 2 输出结构，补齐主体编号并生成标准场景描述。
- `_normalize_subject_id`：把主体编号统一收敛为 `subject_n`。
- `_build_scene_desc`：基于主体、地点与动作拼装 `subject_x 在什么地方正在做什么` 句式。
"""

import math
import re
from typing import Any, Callable

from app.models import TaskRecord


class PipelineStage2Mixin:
    def _run_stage2(
        self,
        task: TaskRecord,
        params: dict[str, Any],
        physical_manifest: dict[str, Any],
        progress_cb: Callable[[float, str], None],
        started_at: float,
    ) -> dict[str, Any]:
        scene_inputs = [
            {
                "scene_id": item["scene_id"],
                "keyframe_path": item["keyframe_path"],
                "start_time": item["start_time"],
                "end_time": item["end_time"],
            }
            for item in physical_manifest["scenes"]
        ]

        batch_size = int(params.get("batch_size") or self.default_batch_size)
        retry_max = int(params.get("retry_max") if params.get("retry_max") is not None else self.default_retry_max)
        stage2_provider = getattr(self, "stage2_vlm_provider", self.vlm_provider)
        stage2_default_model = getattr(self, "default_stage2_vlm_model", self.default_vlm_model)
        vlm_model = str(params.get("stage2_vlm_model") or params.get("vlm_model") or stage2_default_model)

        all_rows: list[dict[str, Any]] = []
        total_batches = max(1, math.ceil(len(scene_inputs) / batch_size))

        for batch_idx, start in enumerate(range(0, len(scene_inputs), batch_size), start=1):
            batch = scene_inputs[start : start + batch_size]
            rows = stage2_provider.describe_scenes(
                scene_inputs=batch,
                model=vlm_model,
                retry_max=retry_max,
            )
            all_rows.extend(rows)
            pct = 30.0 + (batch_idx / total_batches) * 18.0
            progress_cb(pct, f"原始分镜描述批次 {batch_idx}/{total_batches}")
            self._assert_not_timeout(started_at)

        normalized = self._normalize_raw_descriptions(scene_inputs, all_rows)
        contract = {
            "project_id": task.task_id,
            "scenes": normalized,
        }
        self.store.write_contract(task.task_id, "raw_scene_descriptions", contract)
        return contract

    def _normalize_raw_descriptions(
        self,
        scene_inputs: list[dict[str, Any]],
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        by_scene: dict[int, dict[str, Any]] = {}
        for row in rows:
            scene_id = int(row.get("scene_id", 0))
            if scene_id > 0:
                by_scene[scene_id] = row

        normalized: list[dict[str, Any]] = []
        for item in scene_inputs:
            scene_id = int(item["scene_id"])
            row = by_scene.get(scene_id) or {"scene_id": scene_id, "subjects": [], "desc": ""}

            raw_subjects = row.get("subjects")
            if not isinstance(raw_subjects, list):
                raw_subjects = []

            subjects: list[dict[str, str]] = []
            subject_id_map: dict[str, str] = {}
            for idx, subject in enumerate(raw_subjects, start=1):
                normalized_subject_id = self._normalize_subject_id(
                    raw=str((subject or {}).get("subject_id") or (subject or {}).get("temp_id") or ""),
                    idx=idx,
                )
                original_subject_id = str((subject or {}).get("subject_id") or (subject or {}).get("temp_id") or "").strip()
                if original_subject_id:
                    subject_id_map[original_subject_id] = normalized_subject_id
                subject_id_map[normalized_subject_id] = normalized_subject_id
                subjects.append(
                    {
                        "subject_id": normalized_subject_id,
                        "appearance": str((subject or {}).get("appearance", "")).strip(),
                    }
                )

            desc = str(row.get("desc", "")).strip()
            if desc:
                desc = self._normalize_desc_subject_ids(desc, subject_id_map, len(subjects))
            else:
                desc = self._build_scene_desc(
                    subjects=subjects,
                    raw_subjects=raw_subjects,
                    row=row,
                )

            normalized.append(
                {
                    "scene_id": scene_id,
                    "subjects": subjects,
                    "desc": desc,
                }
            )
        return normalized

    def _normalize_subject_id(self, raw: str, idx: int) -> str:
        text = raw.strip()
        match = re.search(r"(\d+)", text)
        if match:
            return f"subject_{int(match.group(1))}"
        return f"subject_{idx}"

    def _normalize_desc_subject_ids(
        self,
        desc: str,
        subject_id_map: dict[str, str],
        subject_count: int,
    ) -> str:
        normalized = desc.strip()
        for original, target in sorted(subject_id_map.items(), key=lambda item: len(item[0]), reverse=True):
            if not original or original == target:
                continue
            normalized = re.sub(rf"\b{re.escape(original)}\b", target, normalized)

        normalized = re.sub(r"[，,]\s*(?=subject_\d+\b)", "；", normalized)
        normalized = self._split_joint_subject_clauses(normalized)
        if "subject_" not in normalized and subject_count == 1:
            normalized = f"subject_1 {normalized}".strip()
        return normalized

    def _split_joint_subject_clauses(self, desc: str) -> str:
        pattern = re.compile(r"(subject_\d+)\s*和\s*(subject_\d+)([^；。]+)")

        def _replace(match: re.Match[str]) -> str:
            subject_a = match.group(1)
            subject_b = match.group(2)
            tail = match.group(3)
            return f"{subject_a}{tail}；{subject_b}{tail}"

        previous = desc
        while True:
            current = pattern.sub(_replace, previous)
            if current == previous:
                return current
            previous = current

    def _build_scene_desc(
        self,
        *,
        subjects: list[dict[str, str]],
        raw_subjects: list[dict[str, Any]],
        row: dict[str, Any],
    ) -> str:
        clauses: list[str] = []
        env_location = ""
        visual_analysis = row.get("visual_analysis")
        if isinstance(visual_analysis, dict):
            env_location = str((visual_analysis.get("environment") or {}).get("location", "")).strip()

        for idx, subject in enumerate(subjects, start=1):
            raw_subject = raw_subjects[idx - 1] if idx - 1 < len(raw_subjects) and isinstance(raw_subjects[idx - 1], dict) else {}
            subject_id = str(subject.get("subject_id", "")).strip()
            location = str(raw_subject.get("location", "")).strip() or env_location or "画面中"
            action = str(raw_subject.get("action", "")).strip() or "停留"
            clauses.append(f"{subject_id} 在{location}正在{action}")

        if not clauses:
            return ""
        return "；".join(clauses) + "。"
