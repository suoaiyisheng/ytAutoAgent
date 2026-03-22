from __future__ import annotations

"""
模块作用：
- 负责 Stage 3 身份对齐，输出最小角色库。

当前文件工具函数：
- `_flatten_subjects`：把 Stage 2 分镜主体整理成统一聚类输入。
- `_cluster_subjects`：根据向量、锚点与时序信息做主体聚类。
- `_build_provisional_characters`：从聚类结果构建临时角色记录与主体到角色映射。
- `_build_character_bank`：把聚类结果转成文档约定的 `03_character_bank.json` 结构。
"""

import re
from collections import defaultdict
from typing import Any, Callable

from app.models import TaskRecord


class PipelineStage3Mixin:
    def _run_stage3(
        self,
        task: TaskRecord,
        params: dict[str, Any],
        physical_manifest: dict[str, Any],
        raw_scene_descriptions: dict[str, Any],
        progress_cb: Callable[[float, str], None],
        started_at: float,
    ) -> dict[str, Any]:
        retry_max = int(params.get("retry_max") if params.get("retry_max") is not None else self.default_retry_max)
        embed_model = str(params.get("embed_model") or self.default_embed_model)
        vlm_model = str(params.get("vlm_model") or self.default_vlm_model)

        scene_to_image = {int(item["scene_id"]): str(item["keyframe_path"]) for item in physical_manifest["scenes"]}
        subjects = self._flatten_subjects(raw_scene_descriptions, scene_to_image)

        if not subjects:
            character_bank = {
                "project_id": task.task_id,
                "characters": [],
                "global_style": "真实摄影风格",
            }
            self.store.write_contract(task.task_id, "character_bank", character_bank)
            return character_bank

        vectors = self.embedding_provider.embed_texts(
            texts=[item["appearance_norm"] or item["appearance"] or item["subject_id"] for item in subjects],
            model=embed_model,
            retry_max=retry_max,
        )
        progress_cb(62.0, "向量计算完成，开始身份聚类")
        self._assert_not_timeout(started_at)

        clusters = self._cluster_subjects(
            subjects=subjects,
            vectors=vectors,
            join_threshold=0.72,
            boundary_low=0.55,
            boundary_high=0.75,
        )
        provisional, ref_mapping = self._build_provisional_characters(clusters)

        boundary_candidates = [item for item in provisional if item.get("needs_review")]
        reviewed: list[dict[str, Any]] = []
        if boundary_candidates:
            reviewed = self.vlm_provider.review_character_candidates(
                candidates=boundary_candidates,
                model=vlm_model,
                retry_max=retry_max,
            )
        merged = self._merge_reviewed_characters(provisional, reviewed)
        character_bank = self._build_character_bank(task.task_id, merged, ref_mapping)

        self.store.write_contract(task.task_id, "character_bank", character_bank)
        progress_cb(70.0, "身份对齐完成")
        return character_bank

    def _flatten_subjects(
        self,
        raw_scene_descriptions: dict[str, Any],
        scene_to_image: dict[int, str],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for scene in raw_scene_descriptions.get("scenes", []):
            scene_id = int(scene.get("scene_id", 0))
            for subject in scene.get("subjects", []):
                appearance = str(subject.get("appearance", "")).strip()
                features = self._parse_subject_features(appearance=appearance, action="")
                rows.append(
                    {
                        "scene_id": scene_id,
                        "subject_id": str(subject.get("subject_id", "")).strip(),
                        "appearance": appearance,
                        "appearance_norm": features["normalized_text"],
                        "keyframe_path": scene_to_image.get(scene_id, ""),
                        "features": features,
                    }
                )
        return rows

    def _cluster_subjects(
        self,
        subjects: list[dict[str, Any]],
        vectors: list[list[float]],
        join_threshold: float,
        boundary_low: float,
        boundary_high: float,
    ) -> list[dict[str, Any]]:
        clusters: list[dict[str, Any]] = []
        for idx, subject in enumerate(subjects):
            vector = vectors[idx]
            if not clusters:
                clusters.append(
                    {
                        "members": [subject],
                        "vectors": [vector],
                        "centroid": vector,
                        "needs_review": False,
                        "assignment_scores": [1.0],
                    }
                )
                self._refresh_cluster_profile(clusters[-1])
                continue

            best_idx = -1
            best_score = -1.0
            for cluster_idx, cluster in enumerate(clusters):
                score = self._score_subject_to_cluster(subject=subject, vector=vector, cluster=cluster)
                if score > best_score:
                    best_score = score
                    best_idx = cluster_idx

            if best_idx >= 0 and best_score >= join_threshold:
                cluster = clusters[best_idx]
                cluster["members"].append(subject)
                cluster["vectors"].append(vector)
                cluster["assignment_scores"].append(best_score)
                if boundary_low <= best_score <= boundary_high:
                    cluster["needs_review"] = True
                self._refresh_cluster_profile(cluster)
            else:
                new_cluster = {
                    "members": [subject],
                    "vectors": [vector],
                    "centroid": vector,
                    "needs_review": boundary_low <= best_score <= boundary_high,
                    "assignment_scores": [max(0.0, best_score)],
                }
                self._refresh_cluster_profile(new_cluster)
                if best_idx >= 0 and boundary_low <= best_score <= boundary_high:
                    clusters[best_idx]["needs_review"] = True
                clusters.append(new_cluster)
        return clusters

    def _build_provisional_characters(
        self,
        clusters: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[tuple[int, str], str]]:
        characters: list[dict[str, Any]] = []
        mapping: dict[tuple[int, str], str] = {}
        main_idx = 1

        sorted_clusters = sorted(
            clusters,
            key=lambda cluster: min(int(member["scene_id"]) for member in cluster.get("members", []) or [{"scene_id": 0}]),
        )
        for cluster in sorted_clusters:
            if self._is_extra_cluster(cluster):
                continue

            ref_id = f"Ref_{main_idx}"
            main_idx += 1
            members = sorted(cluster.get("members", []), key=lambda item: (int(item.get("scene_id", 0)), item.get("subject_id", "")))
            appearances = [member["appearance"] for member in members if member["appearance"]]
            master_description = self._merge_appearances(appearances)
            key_features = self._derive_key_features(master_description)
            ref_image = members[0].get("keyframe_path", "") if members else ""
            confidence = round(self._mean(cluster.get("assignment_scores") or [0.0]), 3)
            scene_presence = [[int(member["scene_id"]), str(member["subject_id"])] for member in members]

            characters.append(
                {
                    "ref_id": ref_id,
                    "name": "",
                    "master_description": master_description,
                    "key_features": key_features,
                    "ref_image_path": ref_image,
                    "needs_review": bool(cluster.get("needs_review", False)),
                    "merge_confidence": confidence,
                    "scene_presence": scene_presence,
                    "members": members,
                }
            )

            for member in members:
                mapping[(int(member["scene_id"]), str(member["subject_id"]))] = ref_id

        return characters, mapping

    def _merge_reviewed_characters(
        self,
        provisional: list[dict[str, Any]],
        reviewed: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        by_ref = {str(item.get("ref_id", "")).strip(): item for item in reviewed if str(item.get("ref_id", "")).strip()}
        merged: list[dict[str, Any]] = []
        for item in provisional:
            ref_id = item["ref_id"]
            candidate = by_ref.get(ref_id)
            if candidate:
                updated = dict(item)
                updated["name"] = str(candidate.get("name", item.get("name", ""))).strip()
                updated["master_description"] = str(candidate.get("master_description", item.get("master_description", ""))).strip()
                updated["key_features"] = [
                    str(value).strip() for value in candidate.get("key_features", item.get("key_features", [])) if str(value).strip()
                ]
                updated["ref_image_path"] = str(candidate.get("ref_image_path", item.get("ref_image_path", ""))).strip()
                merged.append(updated)
            else:
                merged.append(item)
        return merged

    def _build_character_bank(
        self,
        project_id: str,
        characters: list[dict[str, Any]],
        ref_mapping: dict[tuple[int, str], str],
    ) -> dict[str, Any]:
        public_characters: list[dict[str, Any]] = []

        for item in characters:
            ref_id = str(item.get("ref_id", "")).strip()
            members = sorted(
                item.get("members", []),
                key=lambda member: (int(member.get("scene_id", 0)), str(member.get("subject_id", ""))),
            )
            if not ref_id or not members:
                continue

            canonical_description = str(item.get("master_description", "")).strip()
            if not canonical_description:
                canonical_description = self._merge_appearances(
                    [str(member.get("appearance", "")).strip() for member in members if str(member.get("appearance", "")).strip()],
                    max_items=1,
                )

            public_characters.append(
                {
                    "ref_id": ref_id,
                    "canonical_description": canonical_description,
                    "ref_image_path": str(item.get("ref_image_path", "")).strip(),
                    "scene_presence": [
                        [int(member.get("scene_id", 0)), str(member.get("subject_id", "")).strip()]
                        for member in members
                    ],
                }
            )

        self._validate_scene_presence(public_characters, ref_mapping)

        return {
            "project_id": project_id,
            "characters": public_characters,
            "global_style": "真实摄影风格",
        }

    def _validate_scene_presence(
        self,
        characters: list[dict[str, Any]],
        ref_mapping: dict[tuple[int, str], str],
    ) -> None:
        seen: set[tuple[int, str]] = set()
        for character in characters:
            for item in character.get("scene_presence", []):
                key = (int(item[0]), str(item[1]))
                if key in seen:
                    raise ValueError(f"duplicate scene presence: {key}")
                seen.add(key)

        if set(ref_mapping.keys()) != seen:
            missing = sorted(set(ref_mapping.keys()) - seen)
            extra = sorted(seen - set(ref_mapping.keys()))
            if missing or extra:
                raise ValueError(f"scene presence mismatch, missing={missing}, extra={extra}")

    def _appearance_profile_from_text(self, text: str) -> dict[str, Any]:
        phrases = self._appearance_phrases(text)
        profile = {
            "hair": self._pick_phrase(phrases, ["发", "辫", "马尾"]),
            "tops": [],
            "outerwear": [],
            "bottoms": [],
            "footwear": [],
            "accessories": [],
            "eyewear": "",
            "body_shape": self._body_shape_phrase(text),
        }

        for phrase in phrases:
            if any(token in phrase for token in ["外套", "夹克", "风衣", "大衣", "实验服", "白大褂"]):
                profile["outerwear"].append(phrase)
            elif any(token in phrase for token in ["衬衫", "T恤", "短袖", "上衣", "背心", "毛衣", "卫衣", "裙"]):
                profile["tops"].append(phrase)
            elif any(token in phrase for token in ["裤", "裙", "半身裙"]):
                profile["bottoms"].append(phrase)
            elif any(token in phrase for token in ["鞋", "靴"]):
                profile["footwear"].append(phrase)
            elif any(token in phrase for token in ["围裙", "帽", "项链", "耳环", "手套", "包", "背包"]):
                profile["accessories"].append(phrase)
            if "眼镜" in phrase:
                profile["eyewear"] = phrase if "未佩戴" not in phrase else "未佩戴眼镜"

        if not profile["eyewear"] and ("眼镜" not in text and "墨镜" not in text):
            profile["eyewear"] = "未佩戴眼镜"

        for key in ["tops", "outerwear", "bottoms", "footwear", "accessories"]:
            profile[key] = self._unique_texts(profile[key])
        return profile

    def _appearance_phrases(self, text: str) -> list[str]:
        normalized = str(text).replace("，", ",").replace("；", ",").replace("。", ",")
        parts = re.split(r"[,、]|和", normalized)
        return [part.strip() for part in parts if part.strip()]

    def _pick_phrase(self, phrases: list[str], tokens: list[str]) -> str:
        for phrase in phrases:
            if any(token in phrase for token in tokens):
                return phrase
        return ""

    def _body_shape_phrase(self, text: str) -> str:
        normalized = self._normalize_text(text)
        body_type = self._extract_body_type(normalized)
        mapping = {
            "overweight": "体型偏胖",
            "slim": "体型清瘦",
            "muscular": "体型健壮肌肉",
            "normal": "体型匀称",
        }
        return mapping.get(body_type, "")

    def _appearance_delta(self, base_profile: dict[str, Any], current_profile: dict[str, Any]) -> dict[str, Any]:
        delta: dict[str, Any] = {}
        for key in ["hair", "eyewear", "body_shape"]:
            base_value = str(base_profile.get(key, "")).strip()
            current_value = str(current_profile.get(key, "")).strip()
            if current_value and current_value != base_value:
                delta[key] = current_value

        for key in ["tops", "outerwear", "bottoms", "footwear", "accessories"]:
            base_value = base_profile.get(key, [])
            current_value = current_profile.get(key, [])
            if current_value and current_value != base_value:
                delta[key] = current_value
        return delta

    def _profile_to_description(self, profile: dict[str, Any]) -> str:
        parts: list[str] = []
        hair = str(profile.get("hair", "")).strip()
        if hair:
            parts.append(hair)

        clothing_items = []
        for key in ["tops", "outerwear", "bottoms", "footwear", "accessories"]:
            clothing_items.extend([str(item).strip() for item in profile.get(key, []) if str(item).strip()])
        if clothing_items:
            normalized_items = self._unique_texts(clothing_items)
            clothing_text = "、".join(normalized_items)
            if clothing_text.startswith("穿"):
                parts.append(clothing_text)
            else:
                parts.append("穿" + clothing_text)

        eyewear = str(profile.get("eyewear", "")).strip()
        if eyewear:
            parts.append(eyewear)

        body_shape = str(profile.get("body_shape", "")).strip()
        if body_shape:
            parts.append(body_shape)
        return "，".join(parts)

    def _state_label_from_delta(self, delta: dict[str, Any]) -> str:
        body_shape = str(delta.get("body_shape", "")).strip()
        if body_shape:
            if body_shape.startswith("体型"):
                return body_shape.replace("体型", "体型更", 1) if "更" not in body_shape else body_shape
            return f"{body_shape}状态"

        first_key = next(iter(delta.keys()), "")
        value = delta.get(first_key)
        if isinstance(value, list):
            value_text = "、".join(str(item).strip() for item in value if str(item).strip())
        else:
            value_text = str(value).strip()
        return value_text or "外貌变化"

    def _state_description_from_delta(self, delta: dict[str, Any], current_profile: dict[str, Any]) -> str:
        delta_parts: list[str] = []
        for key, value in delta.items():
            if isinstance(value, list):
                if value:
                    delta_parts.append("、".join(str(item).strip() for item in value if str(item).strip()))
            else:
                text = str(value).strip()
                if text:
                    delta_parts.append(text)
        if not delta_parts:
            return self._profile_to_description(current_profile)
        return f"相对基准外貌，{'，'.join(delta_parts)}"

    def _unique_texts(self, values: list[str]) -> list[str]:
        seen: list[str] = []
        for value in values:
            text = str(value).strip()
            if text and text not in seen:
                seen.append(text)
        return seen

    def _parse_subject_features(self, appearance: str, action: str) -> dict[str, Any]:
        text = self._normalize_text(f"{appearance} {action}")
        gender = "unknown"
        if any(token in text for token in ["男性", "男人", "男", "male", "man"]):
            gender = "male"
        if any(token in text for token in ["女性", "女孩", "女人", "女", "female", "woman"]):
            gender = "female"

        role = "human"
        if any(token in text for token in ["医生", "白大褂", "lab coat", "doctor"]):
            role = "doctor"
        elif any(token in text for token in ["鱼尾", "美人鱼", "mermaid"]):
            role = "mermaid"

        hair_color = "unknown"
        if any(token in text for token in ["紫色", "purple"]):
            hair_color = "purple"
        elif any(token in text for token in ["红色", "red"]):
            hair_color = "red"
        elif any(token in text for token in ["金发", "blonde"]):
            hair_color = "blonde"
        elif any(token in text for token in ["黑发", "black hair", "黑色头发"]):
            hair_color = "black"

        hair_style = "unknown"
        if any(token in text for token in ["辫", "braid"]):
            hair_style = "braid"
        elif any(token in text for token in ["双马尾", "pigtail"]):
            hair_style = "pigtail"
        elif any(token in text for token in ["短发", "short hair"]):
            hair_style = "short"
        elif any(token in text for token in ["长发", "long hair"]):
            hair_style = "long"

        body_type = self._extract_body_type(text)

        anchors: set[str] = set()
        if gender != "unknown":
            anchors.add(f"gender:{gender}")
        if role != "human":
            anchors.add(f"role:{role}")
        if hair_color != "unknown":
            anchors.add(f"hair_color:{hair_color}")
        if hair_style != "unknown":
            anchors.add(f"hair_style:{hair_style}")
        if body_type != "unknown":
            anchors.add(f"body:{body_type}")
        for token, alias in [
            ("黄色", "top:yellow"),
            ("黑色短裤", "bottom:black_shorts"),
            ("紫色长辫", "hair:purple_braid"),
            ("粉色", "style:pink"),
            ("白大褂", "prop:labcoat"),
            ("眼镜", "prop:glasses"),
            ("肌肉", "body:muscular"),
            ("胖", "body:overweight"),
            ("瘦", "body:slim"),
            ("运动鞋", "shoes:sneakers"),
        ]:
            if token in text:
                anchors.add(alias)

        visibility = "normal"
        if any(token in text for token in ["部分可见", "partially visible", "partially", "右侧可见"]):
            visibility = "partial"

        return {
            "normalized_text": text,
            "gender": gender,
            "role": role,
            "hair_color": hair_color,
            "hair_style": hair_style,
            "body_type": body_type,
            "anchors": sorted(anchors),
            "visibility": visibility,
        }

    def _normalize_text(self, text: str) -> str:
        normalized = text.lower()
        replacements = {
            "mermaid": "美人鱼",
            "male": "男性",
            "female": "女性",
            "woman": "女性",
            "man": "男性",
            "doctor": "医生",
            "lab coat": "白大褂",
            "blonde": "金发",
            "purple": "紫色",
            "pink": "粉色",
            "red": "红色",
            "black hair": "黑发",
            "yellow": "黄色",
            "black shorts": "黑色短裤",
            "shorts": "短裤",
            "t-shirt": "t恤",
            "braid": "辫子",
            "pigtail": "双马尾",
            "short hair": "短发",
            "long hair": "长发",
            "partially visible": "部分可见",
            "partially": "部分可见",
            "overweight": "偏胖",
            "plus size": "大码",
            "obese": "肥胖",
            "chubby": "偏胖",
            "heavyset": "壮实",
            "slim": "纤细",
            "slender": "纤细",
            "lean": "清瘦",
            "muscular": "健壮肌肉",
            "athletic": "健壮肌肉",
            "fit": "匀称",
            "toned": "健壮肌肉",
        }
        for source, target in replacements.items():
            normalized = normalized.replace(source, target)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _extract_body_type(self, text: str) -> str:
        token_map: dict[str, list[str]] = {
            "overweight": ["偏胖", "肥胖", "大码", "丰满", "圆润", "胖", "壮实"],
            "slim": ["纤细", "苗条", "清瘦", "修长", "瘦"],
            "muscular": ["健壮肌肉", "肌肉", "壮硕", "结实"],
            "normal": ["匀称", "普通体型", "标准体型"],
        }
        for body_type, tokens in token_map.items():
            if any(token in text for token in tokens):
                return body_type
        return "unknown"

    def _refresh_cluster_profile(self, cluster: dict[str, Any]) -> None:
        cluster["centroid"] = self._mean_vector(cluster.get("vectors") or [])
        anchors: set[str] = set()
        scene_ids: set[int] = set()
        gender_counter: dict[str, int] = defaultdict(int)
        role_counter: dict[str, int] = defaultdict(int)
        visibility_counter: dict[str, int] = defaultdict(int)
        for member in cluster.get("members", []):
            features = member.get("features", {})
            anchors.update(features.get("anchors", []))
            scene_ids.add(int(member.get("scene_id", 0)))
            gender_counter[str(features.get("gender", "unknown"))] += 1
            role_counter[str(features.get("role", "human"))] += 1
            visibility_counter[str(features.get("visibility", "normal"))] += 1

        cluster["anchor_tokens"] = anchors
        cluster["scene_ids"] = scene_ids
        cluster["dominant_gender"] = self._dominant_value(gender_counter, default="unknown")
        cluster["dominant_role"] = self._dominant_value(role_counter, default="human")
        cluster["partial_ratio"] = visibility_counter.get("partial", 0) / max(1, len(cluster.get("members", [])))

    def _score_subject_to_cluster(
        self,
        subject: dict[str, Any],
        vector: list[float],
        cluster: dict[str, Any],
    ) -> float:
        anchor_subject = set(subject.get("features", {}).get("anchors", []))
        anchor_cluster = set(cluster.get("anchor_tokens", set()))
        anchor_sim = self._jaccard(anchor_subject, anchor_cluster)

        vector_sim = max(0.0, self._cosine(vector, cluster.get("centroid", [])))
        struct_sim = self._structured_similarity(subject, cluster)
        temporal = self._temporal_continuity(int(subject.get("scene_id", 0)), cluster.get("scene_ids", set()))
        cooccur = self._cooccurrence_score(subject, cluster)

        score = (0.45 * ((anchor_sim + vector_sim) / 2.0)) + (0.30 * struct_sim) + (0.15 * temporal) + (0.10 * cooccur)

        subject_gender = subject.get("features", {}).get("gender", "unknown")
        cluster_gender = cluster.get("dominant_gender", "unknown")
        if subject_gender != "unknown" and cluster_gender != "unknown" and subject_gender != cluster_gender:
            score -= 0.25

        subject_role = subject.get("features", {}).get("role", "human")
        cluster_role = cluster.get("dominant_role", "human")
        if {"doctor", "mermaid"} <= {subject_role, cluster_role}:
            score -= 0.2

        return max(0.0, min(1.0, score))

    def _structured_similarity(self, subject: dict[str, Any], cluster: dict[str, Any]) -> float:
        members = cluster.get("members", [])
        if not members:
            return 0.5

        subject_features = subject.get("features", {})
        scores: list[float] = []
        for member in members:
            member_features = member.get("features", {})
            local_parts: list[float] = []
            for key in ["gender", "role", "hair_color", "hair_style"]:
                left = str(subject_features.get(key, "unknown"))
                right = str(member_features.get(key, "unknown"))
                if left == "unknown" or right == "unknown":
                    continue
                local_parts.append(1.0 if left == right else 0.0)
            if local_parts:
                scores.append(sum(local_parts) / len(local_parts))
        if not scores:
            return 0.5
        return max(scores)

    def _temporal_continuity(self, scene_id: int, scene_ids: set[int]) -> float:
        if not scene_ids:
            return 0.5
        min_gap = min(abs(scene_id - other_scene_id) for other_scene_id in scene_ids)
        if min_gap == 0:
            return 0.2
        if min_gap == 1:
            return 1.0
        if min_gap <= 3:
            return 0.8
        if min_gap <= 6:
            return 0.6
        return 0.4

    def _cooccurrence_score(self, subject: dict[str, Any], cluster: dict[str, Any]) -> float:
        scene_id = int(subject.get("scene_id", 0))
        if scene_id in cluster.get("scene_ids", set()):
            return 0.0
        return 1.0

    def _is_extra_cluster(self, cluster: dict[str, Any]) -> bool:
        members = cluster.get("members", [])
        if not members:
            return True

        scene_count = len(cluster.get("scene_ids", set()))
        dominant_role = cluster.get("dominant_role", "human")
        partial_ratio = float(cluster.get("partial_ratio", 0.0))
        avg_anchor = self._mean([len(member.get("features", {}).get("anchors", [])) for member in members])

        if dominant_role == "doctor":
            return False
        if partial_ratio >= 0.5 and scene_count <= 2:
            return True
        if scene_count < 2 and avg_anchor < 2.0 and dominant_role == "human":
            return True
        return False
