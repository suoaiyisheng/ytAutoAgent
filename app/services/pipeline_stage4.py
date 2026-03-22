from __future__ import annotations

"""
模块作用：
- 负责 Stage 4 归一化，把 Stage 2 的原始描述与 Stage 3 的最小角色库融合成标准化描述。

当前文件工具函数：
- `_build_mapping_lookup`：通过 `characters[].scene_presence` 反查 `scene_id + subject_id -> ref_id`。
- `_build_appearance_lookup`：构建 `scene_id + subject_id -> appearance` 索引。
- `_normalize_scene_desc`：按子句重写 `subject_id` 为 `Ref_x` 并注入当前分镜外貌描述。
- `_inject_subject_clause`：保持原始地点与动作语义不变，只替换主语部分。
"""

import re
from typing import Any, Callable

from app.errors import Stage1Error
from app.models import TaskRecord


class PipelineStage4Mixin:
    def _run_stage4(
        self,
        task: TaskRecord,
        raw_scene_descriptions: dict[str, Any],
        character_bank: dict[str, Any],
        progress_cb: Callable[[float, str], None],
        started_at: float,
    ) -> dict[str, Any]:
        mapping_lookup = self._build_mapping_lookup(character_bank)
        appearance_lookup = self._build_appearance_lookup(raw_scene_descriptions)

        scenes: list[dict[str, Any]] = []
        raw_scenes = sorted(raw_scene_descriptions.get("scenes", []), key=lambda item: int(item.get("scene_id", 0)))
        total = max(1, len(raw_scenes))
        for idx, scene in enumerate(raw_scenes, start=1):
            scene_id = int(scene.get("scene_id", 0))
            desc = str(scene.get("desc", "")).strip()
            normalized_desc = self._normalize_scene_desc(
                scene_id=scene_id,
                desc=desc,
                mapping_lookup=mapping_lookup,
                appearance_lookup=appearance_lookup,
            )
            scenes.append(
                {
                    "scene_id": scene_id,
                    "desc": normalized_desc,
                }
            )
            pct = 72.0 + (idx / total) * 10.0
            progress_cb(pct, f"归一化进度 {idx}/{total}")
            self._assert_not_timeout(started_at)

        contract = {
            "project_id": task.task_id,
            "scenes": scenes,
        }
        self.store.write_contract(task.task_id, "normalized_scene_descriptions", contract)
        return contract

    def _build_mapping_lookup(self, character_bank: dict[str, Any]) -> dict[tuple[int, str], str]:
        lookup: dict[tuple[int, str], str] = {}
        for character in character_bank.get("characters", []):
            ref_id = str(character.get("ref_id", "")).strip()
            if not ref_id:
                continue
            for item in character.get("scene_presence", []):
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                key = (int(item[0]), str(item[1]).strip())
                if key in lookup and lookup[key] != ref_id:
                    raise Stage1Error("subject_mapping_duplicate", f"主体映射重复: {key}", 500)
                lookup[key] = ref_id
        return lookup

    def _build_appearance_lookup(self, raw_scene_descriptions: dict[str, Any]) -> dict[tuple[int, str], str]:
        lookup: dict[tuple[int, str], str] = {}
        for scene in raw_scene_descriptions.get("scenes", []):
            scene_id = int(scene.get("scene_id", 0))
            for subject in scene.get("subjects", []):
                subject_id = str(subject.get("subject_id", "")).strip()
                if not subject_id:
                    continue
                key = (scene_id, subject_id)
                if key in lookup:
                    raise Stage1Error("scene_subject_duplicate", f"分镜主体重复: {key}", 500)
                lookup[key] = str(subject.get("appearance", "")).strip()
        return lookup

    def _normalize_scene_desc(
        self,
        *,
        scene_id: int,
        desc: str,
        mapping_lookup: dict[tuple[int, str], str],
        appearance_lookup: dict[tuple[int, str], str],
    ) -> str:
        clauses = [item.strip() for item in re.split(r"[；;。]", desc) if item.strip()]
        normalized_clauses: list[str] = []
        for clause in clauses:
            subject_ids = re.findall(r"\bsubject_\d+\b", clause)
            if not subject_ids:
                raise Stage1Error("desc_subject_missing", f"分镜 {scene_id} 的子句缺少 subject_id: {clause}", 500)
            if len(subject_ids) != 1:
                raise Stage1Error("desc_subject_ambiguous", f"分镜 {scene_id} 的子句主体不唯一: {clause}", 500)

            subject_id = subject_ids[0]
            ref_id = mapping_lookup.get((scene_id, subject_id))
            if not ref_id:
                raise Stage1Error("subject_mapping_missing", f"分镜 {scene_id} 缺少主体映射: {subject_id}", 500)

            appearance = appearance_lookup.get((scene_id, subject_id))
            if appearance is None:
                raise Stage1Error("subject_appearance_missing", f"分镜 {scene_id} 缺少主体外貌: {subject_id}", 500)

            normalized_clauses.append(
                self._inject_subject_clause(
                    clause=clause,
                    subject_id=subject_id,
                    ref_id=ref_id,
                    appearance=appearance,
                )
            )

        if not normalized_clauses:
            return ""
        return "；".join(normalized_clauses) + "。"

    def _inject_subject_clause(
        self,
        *,
        clause: str,
        subject_id: str,
        ref_id: str,
        appearance: str,
    ) -> str:
        appearance_text = str(appearance).strip()
        replacement = f"（{appearance_text}）的{ref_id}" if appearance_text else ref_id
        rewritten = re.sub(rf"\b{re.escape(subject_id)}\b", replacement, clause, count=1)
        if rewritten == clause:
            raise Stage1Error("desc_rewrite_failed", f"主体注入失败: {clause}", 500)
        return rewritten
