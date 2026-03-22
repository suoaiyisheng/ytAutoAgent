from __future__ import annotations

"""
模块作用：
- 负责 Stage 4 归一化，把 Stage 2 的原始描述与 Stage 3 的最小角色库融合成标准化描述。

当前文件工具函数：
- `_build_mapping_lookup`：通过 `characters[].scene_presence` 反查 `scene_id + subject_id -> ref_name`。
- `_build_appearance_lookup`：构建 `scene_id + subject_id -> appearance` 索引。
- `_normalize_scene_desc`：按子句重写 `subject_id` 为 `参考图x` 体系并注入当前分镜外貌描述。
- `_inject_subject_clause`：保持原始地点与动作语义不变，替换子句里该主体的全部 `subject_id` 引用。
"""

import re
from typing import Any, Callable

from app.errors import Stage1Error
from app.models import TaskRecord

SUBJECT_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9_])(subject_\d+)(?![A-Za-z0-9_])")


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
        embed_model = str(task.params.get("embed_model") or self.default_embed_model)
        retry_max = int(task.params.get("retry_max") if task.params.get("retry_max") is not None else self.default_retry_max)
        embedding_cache: dict[str, list[float]] = {}

        self._prime_embedding_cache(
            texts=[
                self._appearance_signature_text(appearance)
                for appearance in list(appearance_lookup.values())
                + [
                    self._appearance_signature_text(str(item.get("ref_image_description", "")).strip())
                    for item in mapping_lookup.values()
                ]
            ],
            embed_model=embed_model,
            retry_max=retry_max,
            embedding_cache=embedding_cache,
        )

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
                embed_model=embed_model,
                retry_max=retry_max,
                embedding_cache=embedding_cache,
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

    def _build_mapping_lookup(self, character_bank: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
        lookup: dict[tuple[int, str], dict[str, Any]] = {}
        for character in character_bank.get("characters", []):
            ref_id = str(character.get("ref_id", "")).strip()
            ref_name = self._normalize_ref_name(
                ref_name=str(character.get("ref_name", "")).strip(),
                ref_id=ref_id,
            )
            if not ref_id:
                continue
            ref_image_description = str(character.get("ref_image_description", "")).strip()
            ref_image_features = [
                str(token).strip() for token in character.get("ref_image_features", []) if str(token).strip()
            ]
            for item in character.get("scene_presence", []):
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                key = (int(item[0]), str(item[1]).strip())
                if key in lookup and str(lookup[key].get("ref_id", "")).strip() != ref_id:
                    raise Stage1Error("subject_mapping_duplicate", f"主体映射重复: {key}", 500)
                lookup[key] = {
                    "ref_id": ref_id,
                    "ref_name": ref_name,
                    "ref_image_description": ref_image_description,
                    "ref_image_features": ref_image_features,
                }
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
        mapping_lookup: dict[tuple[int, str], dict[str, Any]],
        appearance_lookup: dict[tuple[int, str], str],
        embed_model: str,
        retry_max: int,
        embedding_cache: dict[str, list[float]],
    ) -> str:
        clauses = [item.strip() for item in re.split(r"[；;。]", desc) if item.strip()]
        normalized_clauses: list[str] = []
        for clause in clauses:
            subject_ids = [match.group(1) for match in SUBJECT_ID_PATTERN.finditer(clause)]
            if not subject_ids:
                raise Stage1Error("desc_subject_missing", f"分镜 {scene_id} 的子句缺少 subject_id: {clause}", 500)
            rewritten = clause
            ordered_subject_ids: list[str] = []
            for subject_id in subject_ids:
                if subject_id not in ordered_subject_ids:
                    ordered_subject_ids.append(subject_id)

            for subject_id in ordered_subject_ids:
                mapping = mapping_lookup.get((scene_id, subject_id))
                if not mapping:
                    raise Stage1Error("subject_mapping_missing", f"分镜 {scene_id} 缺少主体映射: {subject_id}", 500)

                appearance = appearance_lookup.get((scene_id, subject_id))
                if appearance is None:
                    raise Stage1Error("subject_appearance_missing", f"分镜 {scene_id} 缺少主体外貌: {subject_id}", 500)
                ref_name = self._normalize_ref_name(
                    ref_name=str(mapping.get("ref_name", "")).strip(),
                    ref_id=str(mapping.get("ref_id", "")).strip(),
                )
                if not ref_name:
                    raise Stage1Error("subject_ref_name_missing", f"分镜 {scene_id} 缺少参考图名称: {subject_id}", 500)
                appearance_matches_reference = self._appearance_matches_reference(
                    current_appearance=appearance,
                    ref_image_description=str(mapping.get("ref_image_description", "")).strip(),
                    embed_model=embed_model,
                    retry_max=retry_max,
                    embedding_cache=embedding_cache,
                )

                rewritten = self._inject_subject_clause(
                    clause=rewritten,
                    subject_id=subject_id,
                    ref_name=ref_name,
                    appearance=appearance,
                    appearance_matches_reference=appearance_matches_reference,
                )

            normalized_clauses.append(rewritten)

        if not normalized_clauses:
            return ""
        return "；".join(normalized_clauses) + "。"

    def _inject_subject_clause(
        self,
        *,
        clause: str,
        subject_id: str,
        ref_name: str,
        appearance: str,
        appearance_matches_reference: bool,
    ) -> str:
        appearance_text = str(appearance).strip()
        replacement = ref_name if appearance_matches_reference else f"（{appearance_text}）的{ref_name}" if appearance_text else ref_name
        rewritten = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(subject_id)}(?![A-Za-z0-9_])", replacement, clause)
        if rewritten == clause:
            raise Stage1Error("desc_rewrite_failed", f"主体注入失败: {clause}", 500)
        return rewritten

    def _normalize_ref_name(self, *, ref_name: str, ref_id: str) -> str:
        cleaned = str(ref_name).strip()
        if cleaned:
            return cleaned
        ref_id_text = str(ref_id).strip()
        match = re.fullmatch(r"Ref_(\d+)", ref_id_text)
        if match:
            return f"参考图{int(match.group(1))}"
        return ""
