from __future__ import annotations

"""
模块作用：
- 承载流水线各 Stage 共用的通用工具，避免编排器和业务 Stage 重复实现基础逻辑。

当前文件工具函数：
- `_persist_index_and_result`：汇总合同文件、产物文件与统计信息并写入索引。
- `_to_markdown_table`：把最终提示词结果转成 Markdown 表格。
- `_sec_to_hms`：把秒数格式化成 `HH:MM:SS.mmm`。
- `_assert_not_timeout`：统一执行超时检查。
- `_cosine` / `_mean_vector` / `_mean` / `_jaccard` / `_dominant_value`：聚类与相似度计算基础函数。
- `_merge_appearances` / `_derive_key_features`：角色描述压缩与特征提取工具。
"""

import math
import time
from typing import Any

from app.errors import Stage1Error
from app.models import TaskRecord


class PipelineCommonMixin:
    def _persist_index_and_result(
        self,
        task: TaskRecord,
        physical_manifest: dict[str, Any],
        raw_scene_descriptions: dict[str, Any],
        character_bank: dict[str, Any],
        normalized_scene_descriptions: dict[str, Any],
        final_table: dict[str, Any],
    ) -> dict[str, Any]:
        contracts = {
            "physical_manifest": str(self.store.contract_path(task.task_id, "physical_manifest").resolve()),
            "raw_scene_descriptions": str(self.store.contract_path(task.task_id, "raw_scene_descriptions").resolve()),
            "character_bank": str(self.store.contract_path(task.task_id, "character_bank").resolve()),
            "normalized_scene_descriptions": str(
                self.store.contract_path(task.task_id, "normalized_scene_descriptions").resolve()
            ),
            "final_production_table": str(self.store.contract_path(task.task_id, "final_production_table").resolve()),
        }

        artifacts = {
            "final_prompts.md": str(self.store.markdown_path(task.task_id).resolve()),
            "index.json": str(self.store.index_path(task.task_id).resolve()),
        }

        stats = {
            "scene_count": len(physical_manifest.get("scenes", [])),
            "character_count": len(character_bank.get("characters", [])),
            "prompt_count": len(final_table.get("prompts", [])),
        }

        index_payload = {
            "project_id": task.task_id,
            "contracts": contracts,
            "artifacts": artifacts,
            "stats": stats,
        }
        self.store.write_index(task.task_id, index_payload)

        return {
            "project_id": task.task_id,
            "contracts": contracts,
            "artifacts": artifacts,
            "stats": stats,
        }

    def _to_markdown_table(self, prompts: list[dict[str, Any]]) -> str:
        lines = [
            "| 分镜编号 | 画面提示词 | 视频提示词 |",
            "| --- | --- | --- |",
        ]
        for item in sorted(prompts, key=lambda x: int(x.get("shot_id", 0))):
            shot_id = int(item.get("shot_id", 0))
            image_prompt = str(item.get("image_prompt", "")).replace("|", "\\|")
            video_prompt = str(item.get("video_prompt", "")).replace("|", "\\|")
            lines.append(f"| {shot_id} | {image_prompt} | {video_prompt} |")
        return "\n".join(lines) + "\n"

    def _sec_to_hms(self, sec: float) -> str:
        total_ms = int(round(sec * 1000))
        hours = total_ms // 3_600_000
        rem = total_ms % 3_600_000
        minutes = rem // 60_000
        rem %= 60_000
        seconds = rem // 1000
        millis = rem % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"

    def _assert_not_timeout(self, started_at: float) -> None:
        if time.monotonic() - started_at > self.timeout_sec:
            raise Stage1Error("task_timeout", "任务执行超时", 504)

    def _cosine(self, a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return -1.0
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return -1.0
        return dot / (na * nb)

    def _mean_vector(self, vectors: list[list[float]]) -> list[float]:
        if not vectors:
            return []
        dim = len(vectors[0])
        out = [0.0] * dim
        for vec in vectors:
            for idx, value in enumerate(vec):
                out[idx] += value
        return [value / len(vectors) for value in out]

    def _merge_appearances(self, appearances: list[str], max_items: int | None = 3) -> str:
        unique: list[str] = []
        for text in appearances:
            normalized = text.strip()
            if normalized and normalized not in unique:
                unique.append(normalized)
        if not unique:
            return ""
        if max_items is None:
            return "；".join(unique)
        return "；".join(unique[: max(1, max_items)])

    def _derive_key_features(self, master_description: str) -> list[str]:
        if not master_description:
            return []
        parts = [item.strip() for item in master_description.replace("；", "，").split("，") if item.strip()]
        return parts[:3]

    def _jaccard(self, a: set[str], b: set[str]) -> float:
        if not a and not b:
            return 0.5
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        if union == 0:
            return 0.0
        return inter / union

    def _dominant_value(self, counter: dict[str, int], default: str) -> str:
        if not counter:
            return default
        return sorted(counter.items(), key=lambda item: item[1], reverse=True)[0][0]

    def _mean(self, values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _appearance_signature_text(self, text: str) -> str:
        raw_text = str(text).strip()
        if not raw_text:
            return ""

        parts: list[str] = []
        parse_features = getattr(self, "_parse_subject_features", None)
        if callable(parse_features):
            features = parse_features(raw_text, "")
            gender = str(features.get("gender", "unknown"))
            role = str(features.get("role", "human"))
            gender_label = {
                "female": "女性",
                "male": "男性",
            }.get(gender, "")
            role_label = {
                "doctor": "医生",
                "mermaid": "美人鱼",
            }.get(role, "")
            if gender_label:
                parts.append(gender_label)
            if role_label:
                parts.append(role_label)

        appearance_profile_from_text = getattr(self, "_appearance_profile_from_text", None)
        profile_to_description = getattr(self, "_profile_to_description", None)
        if callable(appearance_profile_from_text) and callable(profile_to_description):
            profile = appearance_profile_from_text(raw_text)
            profile_text = str(profile_to_description(profile)).strip()
            if profile_text:
                parts.append(profile_text)

        unique_parts: list[str] = []
        for part in parts:
            cleaned = str(part).strip()
            if cleaned and cleaned not in unique_parts:
                unique_parts.append(cleaned)
        return "，".join(unique_parts) or raw_text

    def _prime_embedding_cache(
        self,
        *,
        texts: list[str],
        embed_model: str,
        retry_max: int,
        embedding_cache: dict[str, list[float]],
    ) -> None:
        missing: list[str] = []
        for text in texts:
            cleaned = str(text).strip()
            if cleaned and cleaned not in embedding_cache and cleaned not in missing:
                missing.append(cleaned)
        if not missing:
            return

        vectors = self.embedding_provider.embed_texts(missing, model=embed_model, retry_max=retry_max)
        for text, vector in zip(missing, vectors, strict=False):
            embedding_cache[text] = vector

    def _appearance_match_score(
        self,
        *,
        current_appearance: str,
        ref_image_description: str,
        embed_model: str,
        retry_max: int,
        embedding_cache: dict[str, list[float]] | None = None,
    ) -> float:
        current_text = str(current_appearance).strip()
        ref_text = str(ref_image_description).strip()
        if not current_text or not ref_text:
            return 0.0

        parse_features = getattr(self, "_parse_subject_features", None)
        if not callable(parse_features):
            return 1.0 if current_text == ref_text else 0.0

        current_parsed = parse_features(current_text, "")
        ref_parsed = parse_features(ref_text, "")

        current_gender = str(current_parsed.get("gender", "unknown"))
        ref_gender = str(ref_parsed.get("gender", "unknown"))
        if current_gender != "unknown" and ref_gender != "unknown" and current_gender != ref_gender:
            return 0.0

        current_anchor = set(current_parsed.get("anchors", []))
        ref_anchor = set(ref_parsed.get("anchors", []))
        anchor_sim = self._jaccard(current_anchor, ref_anchor)

        struct_parts: list[float] = []
        for key in ["gender", "role", "hair_color", "hair_style", "body_type"]:
            left = str(current_parsed.get(key, "unknown"))
            right = str(ref_parsed.get(key, "unknown"))
            if left == "unknown" or right == "unknown":
                continue
            struct_parts.append(1.0 if left == right else 0.0)
        struct_sim = (sum(struct_parts) / len(struct_parts)) if struct_parts else 0.5

        current_signature = self._appearance_signature_text(current_text)
        ref_signature = self._appearance_signature_text(ref_text)
        vector_sim = 0.0
        cache = embedding_cache if embedding_cache is not None else {}
        if current_signature and ref_signature:
            self._prime_embedding_cache(
                texts=[current_signature, ref_signature],
                embed_model=embed_model,
                retry_max=retry_max,
                embedding_cache=cache,
            )
            vector_sim = max(
                0.0,
                self._cosine(
                    cache.get(current_signature, []),
                    cache.get(ref_signature, []),
                ),
            )

        score = (0.50 * vector_sim) + (0.30 * anchor_sim) + (0.20 * struct_sim)

        current_role = str(current_parsed.get("role", "human"))
        ref_role = str(ref_parsed.get("role", "human"))
        if current_role != ref_role and {current_role, ref_role} != {"human", "mermaid"}:
            score -= 0.15

        return max(0.0, min(1.0, score))

    def _appearance_matches_reference(
        self,
        *,
        current_appearance: str,
        ref_image_description: str,
        embed_model: str,
        retry_max: int,
        embedding_cache: dict[str, list[float]] | None = None,
        threshold: float = 0.80,
    ) -> bool:
        score = self._appearance_match_score(
            current_appearance=current_appearance,
            ref_image_description=ref_image_description,
            embed_model=embed_model,
            retry_max=retry_max,
            embedding_cache=embedding_cache,
        )
        return score >= threshold
