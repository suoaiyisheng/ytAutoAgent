from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from app.errors import Stage1Error
from app.models import TaskRecord
from app.services.downloader import VideoDownloader
from app.services.frame_extractor import FrameExtractor
from app.services.media_probe import MediaProbe
from app.services.providers import EmbeddingProvider, VLMProvider
from app.services.scene_detection import PySceneDetectProvider, SceneBoundary, SceneDetector
from app.services.store import TaskStore

ProgressCallback = Callable[[float, str], None]


class Stage1Pipeline:
    def __init__(
        self,
        store: TaskStore,
        timeout_sec: int,
        vlm_provider: VLMProvider,
        embedding_provider: EmbeddingProvider,
        default_vlm_model: str = "gemini-1.5-pro",
        default_embed_model: str = "text-embedding-004",
        default_retry_max: int = 2,
        default_batch_size: int = 4,
        architect_prompt_path: Path | None = None,
        downloader: VideoDownloader | None = None,
        scene_detector: SceneDetector | None = None,
        frame_extractor: FrameExtractor | None = None,
        media_probe: MediaProbe | None = None,
    ) -> None:
        self.store = store
        self.timeout_sec = timeout_sec
        self.vlm_provider = vlm_provider
        self.embedding_provider = embedding_provider
        self.default_vlm_model = default_vlm_model
        self.default_embed_model = default_embed_model
        self.default_retry_max = max(0, default_retry_max)
        self.default_batch_size = max(1, default_batch_size)
        self.architect_prompt_path = architect_prompt_path or (Path(__file__).resolve().parents[2] / "dosc" / "V4.2 架构师提示词.md")

        self.downloader = downloader or VideoDownloader()
        self.scene_detector = scene_detector or PySceneDetectProvider()
        self.frame_extractor = frame_extractor or FrameExtractor()
        self.media_probe = media_probe or MediaProbe()

    def validate_ready(self) -> None:
        for provider in (self.vlm_provider, self.embedding_provider):
            ensure = getattr(provider, "ensure_ready", None)
            if callable(ensure):
                ensure()

    def run(self, task: TaskRecord, progress_cb: ProgressCallback) -> dict[str, Any]:
        self.validate_ready()

        started_at = time.monotonic()
        params = task.params

        progress_cb(3.0, "开始结构化提取")
        physical_manifest = self._run_stage1(task, params, progress_cb, started_at)
        self._assert_not_timeout(started_at)

        progress_cb(30.0, "开始原子级感知")
        raw_scene_descriptions = self._run_stage2(task, params, physical_manifest, progress_cb, started_at)
        self._assert_not_timeout(started_at)

        progress_cb(55.0, "开始身份对齐与归一化")
        character_bank, aligned_storyboard = self._run_stage3(
            task,
            params,
            physical_manifest,
            raw_scene_descriptions,
            progress_cb,
            started_at,
        )
        self._assert_not_timeout(started_at)

        progress_cb(80.0, "开始指令合成")
        final_table = self._run_stage4(
            task,
            params,
            character_bank,
            aligned_storyboard,
            progress_cb,
            started_at,
        )
        self._assert_not_timeout(started_at)

        result = self._persist_index_and_result(
            task=task,
            physical_manifest=physical_manifest,
            raw_scene_descriptions=raw_scene_descriptions,
            character_bank=character_bank,
            aligned_storyboard=aligned_storyboard,
            final_table=final_table,
        )
        progress_cb(100.0, "任务完成")
        return result

    def _run_stage1(
        self,
        task: TaskRecord,
        params: dict[str, Any],
        progress_cb: ProgressCallback,
        started_at: float,
    ) -> dict[str, Any]:
        task_dir = self.store.task_dir(task.task_id)
        source_dir = task_dir / "source"
        frames_dir = task_dir / "frames"
        clips_dir = task_dir / "clips"

        video_path = self.downloader.obtain_video(
            source_url=params.get("source_url"),
            local_video_path=params.get("local_video_path"),
            output_dir=source_dir,
            download_format=params["download_format"],
        )
        self._assert_not_timeout(started_at)

        progress_cb(10.0, "视频获取完成，提取元数据")
        metadata = self.media_probe.probe(video_path)
        self._assert_not_timeout(started_at)

        progress_cb(15.0, "开始检测分镜")
        scenes = self.scene_detector.detect(
            video_path=video_path,
            threshold=float(params["threshold"]),
            min_scene_len=float(params["min_scene_len"]),
        )
        if not scenes:
            raise Stage1Error("no_scene_detected", "未检测到可用分镜", 422)
        self._assert_not_timeout(started_at)

        scene_items = self._extract_scene_assets(
            video_path=video_path,
            frames_dir=frames_dir,
            clips_dir=clips_dir,
            scenes=scenes,
            quality=int(params["frame_quality"]),
            progress_cb=progress_cb,
            started_at=started_at,
        )

        contract = {
            "project_id": task.task_id,
            "video_metadata": {
                "source_url": params.get("source_url") or params.get("local_video_path") or "",
                "fps": metadata.fps,
                "resolution": metadata.resolution,
            },
            "scenes": scene_items,
        }
        self.store.write_contract(task.task_id, "physical_manifest", contract)
        return contract

    def _extract_scene_assets(
        self,
        video_path: Path,
        frames_dir: Path,
        clips_dir: Path,
        scenes: list[SceneBoundary],
        quality: int,
        progress_cb: ProgressCallback,
        started_at: float,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        total = len(scenes)
        for idx, scene in enumerate(scenes, start=1):
            frame_path = self.frame_extractor.extract_first_frame(
                video_path=video_path,
                output_path=frames_dir / f"frame_{idx:03d}.jpg",
                timestamp_sec=scene.start,
                quality=quality,
            )
            clip_path = self.frame_extractor.extract_clip(
                video_path=video_path,
                output_path=clips_dir / f"clip_{idx:03d}.mp4",
                start_sec=scene.start,
                end_sec=scene.end,
            )
            items.append(
                {
                    "scene_id": idx,
                    "start_time": self._sec_to_hms(scene.start),
                    "end_time": self._sec_to_hms(scene.end),
                    "keyframe_path": str(frame_path),
                    "clip_path": str(clip_path),
                }
            )
            pct = 15.0 + (idx / total) * 13.0
            progress_cb(pct, f"结构化提取进度 {idx}/{total}")
            self._assert_not_timeout(started_at)
        return items

    def _run_stage2(
        self,
        task: TaskRecord,
        params: dict[str, Any],
        physical_manifest: dict[str, Any],
        progress_cb: ProgressCallback,
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
        vlm_model = str(params.get("vlm_model") or self.default_vlm_model)

        all_rows: list[dict[str, Any]] = []
        total_batches = max(1, math.ceil(len(scene_inputs) / batch_size))

        for batch_idx, start in enumerate(range(0, len(scene_inputs), batch_size), start=1):
            batch = scene_inputs[start : start + batch_size]
            rows = self.vlm_provider.describe_scenes(
                scene_inputs=batch,
                model=vlm_model,
                retry_max=retry_max,
            )
            all_rows.extend(rows)
            pct = 30.0 + (batch_idx / total_batches) * 20.0
            progress_cb(pct, f"原子感知批次 {batch_idx}/{total_batches}")
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
            sid = int(row.get("scene_id", 0))
            if sid > 0:
                by_scene[sid] = row

        normalized: list[dict[str, Any]] = []
        for item in scene_inputs:
            sid = int(item["scene_id"])
            row = by_scene.get(sid) or {
                "scene_id": sid,
                "visual_analysis": {
                    "subjects": [],
                    "environment": {"location": "", "lighting": "", "atmosphere": ""},
                    "camera": {"shot_size": "", "angle": "", "movement": ""},
                },
            }

            va = row.get("visual_analysis") or {}
            raw_subjects = va.get("subjects") or []
            subjects = []
            for idx, subject in enumerate(raw_subjects, start=1):
                subjects.append(
                    {
                        "temp_id": str(subject.get("temp_id") or f"subject_{idx}"),
                        "appearance": str(subject.get("appearance", "")).strip(),
                        "action": str(subject.get("action", "")).strip(),
                        "expression": str(subject.get("expression", "")).strip(),
                    }
                )

            env = va.get("environment") or {}
            camera = va.get("camera") or {}
            normalized.append(
                {
                    "scene_id": sid,
                    "visual_analysis": {
                        "subjects": subjects,
                        "environment": {
                            "location": str(env.get("location", "")).strip(),
                            "lighting": str(env.get("lighting", "")).strip(),
                            "atmosphere": str(env.get("atmosphere", "")).strip(),
                        },
                        "camera": {
                            "shot_size": str(camera.get("shot_size", "")).strip(),
                            "angle": str(camera.get("angle", "")).strip(),
                            "movement": str(camera.get("movement", "")).strip(),
                        },
                    },
                }
            )
        return normalized

    def _run_stage3(
        self,
        task: TaskRecord,
        params: dict[str, Any],
        physical_manifest: dict[str, Any],
        raw_scene_descriptions: dict[str, Any],
        progress_cb: ProgressCallback,
        started_at: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        retry_max = int(params.get("retry_max") if params.get("retry_max") is not None else self.default_retry_max)
        embed_model = str(params.get("embed_model") or self.default_embed_model)
        vlm_model = str(params.get("vlm_model") or self.default_vlm_model)

        scene_to_image = {int(s["scene_id"]): str(s["keyframe_path"]) for s in physical_manifest["scenes"]}
        subjects = self._flatten_subjects(raw_scene_descriptions, scene_to_image)

        if not subjects:
            character_bank = {
                "project_id": task.task_id,
                "characters": [],
                "global_style": "真实摄影风格",
            }
            aligned_storyboard = self._build_aligned_storyboard(
                raw_scene_descriptions,
                {},
                character_bank.get("characters", []),
            )
            self.store.write_contract(task.task_id, "character_bank", character_bank)
            self.store.write_contract(task.task_id, "aligned_storyboard", aligned_storyboard)
            return character_bank, aligned_storyboard

        vectors = self.embedding_provider.embed_texts(
            texts=[x["appearance_norm"] or x["appearance"] or x["temp_id"] for x in subjects],
            model=embed_model,
            retry_max=retry_max,
        )
        progress_cb(62.0, "向量计算完成，开始聚类")
        self._assert_not_timeout(started_at)

        clusters = self._cluster_subjects(
            subjects=subjects,
            vectors=vectors,
            join_threshold=0.72,
            boundary_low=0.55,
            boundary_high=0.75,
        )
        provisional, mapping = self._build_provisional_characters(clusters)

        boundary_candidates = [x for x in provisional if x.get("needs_review")]
        reviewed: list[dict[str, Any]] = []
        if boundary_candidates:
            reviewed = self.vlm_provider.review_character_candidates(
                candidates=boundary_candidates,
                model=vlm_model,
                retry_max=retry_max,
            )
        merged = self._merge_reviewed_characters(provisional, reviewed)

        main_characters = [self._public_character_record(x) for x in merged]

        character_bank = {
            "project_id": task.task_id,
            "characters": main_characters,
            "global_style": "真实摄影风格",
        }
        aligned_storyboard = self._build_aligned_storyboard(raw_scene_descriptions, mapping, main_characters)

        self.store.write_contract(task.task_id, "character_bank", character_bank)
        self.store.write_contract(task.task_id, "aligned_storyboard", aligned_storyboard)
        progress_cb(74.0, "身份对齐完成")
        return character_bank, aligned_storyboard

    def _flatten_subjects(
        self,
        raw_scene_descriptions: dict[str, Any],
        scene_to_image: dict[int, str],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for scene in raw_scene_descriptions.get("scenes", []):
            scene_id = int(scene.get("scene_id", 0))
            subjects = scene.get("visual_analysis", {}).get("subjects", [])
            for subject in subjects:
                appearance = str(subject.get("appearance", "")).strip()
                action = str(subject.get("action", "")).strip()
                features = self._parse_subject_features(appearance=appearance, action=action)
                rows.append(
                    {
                        "scene_id": scene_id,
                        "temp_id": str(subject.get("temp_id", "")).strip(),
                        "appearance": appearance,
                        "appearance_norm": features["normalized_text"],
                        "action": action,
                        "expression": str(subject.get("expression", "")).strip(),
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
            for cidx, cluster in enumerate(clusters):
                score = self._score_subject_to_cluster(subject=subject, vector=vector, cluster=cluster)
                if score > best_score:
                    best_score = score
                    best_idx = cidx

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
                clusters.append(
                    new_cluster
                )
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
            key=lambda x: min(int(m["scene_id"]) for m in x.get("members", []) or [{"scene_id": 0}]),
        )
        for cluster in sorted_clusters:
            is_extra = self._is_extra_cluster(cluster)
            if is_extra:
                # extra 角色不进入角色库，也不写入全局 ref 映射。
                continue

            ref_id = f"Ref_{main_idx}"
            main_idx += 1
            appearances = [m["appearance"] for m in cluster["members"] if m["appearance"]]
            master_description = self._merge_appearances(appearances)
            key_features = self._derive_key_features(master_description)
            ref_image = cluster["members"][0].get("keyframe_path", "")
            confidence = round(self._mean(cluster.get("assignment_scores") or [0.0]), 3)
            scene_presence: set[tuple[int, str]] = set()
            for member in cluster["members"]:
                scene_id = int(member.get("scene_id", 0))
                temp_id = str(member.get("temp_id", "")).strip()
                scene_presence.add((scene_id, temp_id))

            characters.append(
                {
                    "ref_id": ref_id,
                    "name": "",
                    "master_description": master_description,
                    "key_features": key_features,
                    "ref_image_path": ref_image,
                    "needs_review": bool(cluster.get("needs_review", False)),
                    "merge_confidence": confidence,
                    "scene_presence": [[sid, pid] for sid, pid in sorted(scene_presence, key=lambda x: (x[0], x[1]))],
                }
            )

            for member in cluster["members"]:
                mapping[(int(member["scene_id"]), member["temp_id"])] = ref_id

        return characters, mapping

    def _merge_reviewed_characters(
        self,
        provisional: list[dict[str, Any]],
        reviewed: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        by_ref = {item["ref_id"]: item for item in reviewed if item.get("ref_id")}
        merged: list[dict[str, Any]] = []
        for item in provisional:
            ref_id = item["ref_id"]
            candidate = by_ref.get(ref_id)
            if candidate:
                updated = dict(item)
                updated["name"] = str(candidate.get("name", item.get("name", ""))).strip()
                updated["master_description"] = str(candidate.get("master_description", item["master_description"])).strip()
                updated["key_features"] = [
                    str(x).strip() for x in candidate.get("key_features", item["key_features"]) if str(x).strip()
                ]
                updated["ref_image_path"] = str(candidate.get("ref_image_path", item["ref_image_path"])).strip()
                merged.append(updated)
            else:
                merged.append(item)
        return merged

    def _public_character_record(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "ref_id": item["ref_id"],
            "name": str(item.get("name", "")).strip(),
            "master_description": str(item.get("master_description", "")).strip(),
            "key_features": [str(x).strip() for x in item.get("key_features", []) if str(x).strip()],
            "ref_image_path": item.get("ref_image_path", ""),
            "scene_presence": self._normalize_scene_presence(item.get("scene_presence", [])),
        }

    def _build_aligned_storyboard(
        self,
        raw_scene_descriptions: dict[str, Any],
        mapping: dict[tuple[int, str], str],
        characters: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        profiles = self._build_character_description_profiles(characters or [])
        scenes: list[dict[str, Any]] = []
        for scene in sorted(raw_scene_descriptions.get("scenes", []), key=lambda x: int(x.get("scene_id", 0))):
            sid = int(scene.get("scene_id", 0))
            va = scene.get("visual_analysis", {})
            subjects = va.get("subjects", [])

            aligned_subjects: list[dict[str, Any]] = []
            for subject in subjects:
                temp_id = str(subject.get("temp_id", "")).strip()
                ref_id = str(mapping.get((sid, temp_id), "")).strip()
                if not ref_id:
                    ref_id = self._match_subject_to_character(
                        scene_id=sid,
                        subject=subject,
                        profiles=profiles,
                    )
                aligned_subjects.append(
                    {
                        "id": ref_id,
                        "appearance": str(subject.get("appearance", "")).strip(),
                        "action": str(subject.get("action", "")).strip(),
                        "expression": str(subject.get("expression", "")).strip(),
                    }
                )

            env = va.get("environment", {})
            camera = va.get("camera", {})
            scenes.append(
                {
                    "scene_id": sid,
                    "visual_analysis": {
                        "subjects": aligned_subjects,
                        "environment": {
                            "location": str(env.get("location", "")).strip(),
                            "lighting": str(env.get("lighting", "")).strip(),
                            "atmosphere": str(env.get("atmosphere", "")).strip(),
                        },
                        "camera": {
                            "shot_size": str(camera.get("shot_size", "")).strip(),
                            "angle": str(camera.get("angle", "")).strip(),
                            "movement": str(camera.get("movement", "")).strip(),
                        },
                    },
                }
            )

        return {
            "project_id": raw_scene_descriptions.get("project_id", ""),
            "scenes": scenes,
        }

    def _build_character_description_profiles(self, characters: list[dict[str, Any]]) -> list[dict[str, Any]]:
        profiles: list[dict[str, Any]] = []
        for item in characters:
            ref_id = str(item.get("ref_id", "")).strip()
            if not ref_id:
                continue

            scene_presence: set[int] = set()
            for presence in item.get("scene_presence", []):
                scene_id = self._scene_id_from_scene_presence(presence)
                if scene_id is not None:
                    scene_presence.add(scene_id)

            description_parts: list[str] = []
            for value in [
                str(item.get("name", "")).strip(),
                str(item.get("master_description", "")).strip(),
            ]:
                if value:
                    description_parts.append(value)
            for feature in item.get("key_features", []):
                text = str(feature).strip()
                if text:
                    description_parts.append(text)

            description = "，".join(description_parts)
            features = self._parse_subject_features(appearance=description, action="")
            profiles.append(
                {
                    "ref_id": ref_id,
                    "scene_presence": scene_presence,
                    "anchors": set(features.get("anchors", [])),
                    "tokens": self._description_tokens(description),
                    "gender": str(features.get("gender", "unknown")),
                    "role": str(features.get("role", "human")),
                }
            )
        return profiles

    def _match_subject_to_character(
        self,
        scene_id: int,
        subject: dict[str, Any],
        profiles: list[dict[str, Any]],
    ) -> str:
        if not profiles:
            return ""

        scene_candidates = [item for item in profiles if scene_id in item.get("scene_presence", set())]
        candidates = scene_candidates or profiles

        appearance = str(subject.get("appearance", "")).strip()
        action = str(subject.get("action", "")).strip()
        features = self._parse_subject_features(appearance=appearance, action=action)
        subject_anchors = set(features.get("anchors", []))
        subject_tokens = self._description_tokens(f"{appearance} {action}".strip())

        subject_gender = str(features.get("gender", "unknown"))
        subject_role = str(features.get("role", "human"))

        best_ref = ""
        best_score = 0.0
        for item in candidates:
            score = (0.65 * self._jaccard(subject_anchors, item.get("anchors", set()))) + (
                0.35 * self._jaccard(subject_tokens, item.get("tokens", set()))
            )
            if scene_id in item.get("scene_presence", set()):
                score += 0.05

            candidate_gender = str(item.get("gender", "unknown"))
            candidate_role = str(item.get("role", "human"))
            if (
                subject_gender != "unknown"
                and candidate_gender != "unknown"
                and subject_gender != candidate_gender
            ):
                score -= 0.25
            if {"doctor", "mermaid"} <= {subject_role, candidate_role}:
                score -= 0.2

            if score > best_score:
                best_score = score
                best_ref = str(item.get("ref_id", "")).strip()

        if best_score < 0.18:
            return ""
        return best_ref

    def _description_tokens(self, text: str) -> set[str]:
        normalized = self._normalize_text(text)
        if not normalized:
            return set()
        parts = re.split(r"[，,。；;、\s/（）()]+", normalized)
        return {part for part in parts if len(part) >= 2}

    def _scene_id_from_scene_presence(self, presence: Any) -> int | None:
        raw = presence
        if isinstance(presence, (list, tuple)):
            if not presence:
                return None
            raw = presence[0]
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    def _normalize_scene_presence(self, raw_presence: Any) -> list[list[Any]]:
        if not isinstance(raw_presence, list):
            return []
        normalized: set[tuple[int, str]] = set()
        for item in raw_presence:
            if isinstance(item, (list, tuple)):
                if not item:
                    continue
                scene_id = self._scene_id_from_scene_presence(item)
                if scene_id is None:
                    continue
                person_id = str(item[1]).strip() if len(item) > 1 else ""
                normalized.add((scene_id, person_id))
                continue
            scene_id = self._scene_id_from_scene_presence(item)
            if scene_id is None:
                continue
            normalized.add((scene_id, ""))
        return [[scene_id, person_id] for scene_id, person_id in sorted(normalized, key=lambda x: (x[0], x[1]))]

    def _project_aligned_storyboard_to_stage4_storyboard(
        self,
        aligned_storyboard: dict[str, Any],
    ) -> dict[str, Any]:
        storyboard_payload = aligned_storyboard.get("storyboard")
        if isinstance(storyboard_payload, list):
            normalized_storyboard: list[dict[str, Any]] = []
            for shot in storyboard_payload:
                shot_id = int(shot.get("shot_id", 0))
                mappings = []
                for mapping in shot.get("character_mappings", []):
                    ref_id = str(mapping.get("ref_id", "")).strip()
                    if ref_id:
                        normalized_mapping: dict[str, Any] = {"ref_id": ref_id}
                        appearance = str(mapping.get("appearance", "")).strip()
                        action_in_shot = str(mapping.get("action_in_shot", "")).strip()
                        expression_range = self._normalize_expression_range(mapping.get("expression_range"))
                        if appearance:
                            normalized_mapping["appearance"] = appearance
                        if action_in_shot:
                            normalized_mapping["action_in_shot"] = action_in_shot
                        if expression_range:
                            normalized_mapping["expression_range"] = expression_range
                        mappings.append(normalized_mapping)
                normalized_storyboard.append(
                    {
                        "shot_id": shot_id,
                        "character_mappings": mappings,
                        "environment_context": str(shot.get("environment_context", "")).strip(),
                        "camera_instruction": str(shot.get("camera_instruction", "")).strip(),
                    }
                )
            return {
                "project_id": aligned_storyboard.get("project_id", ""),
                "storyboard": sorted(normalized_storyboard, key=lambda x: int(x.get("shot_id", 0))),
            }

        scenes_payload = aligned_storyboard.get("scenes")
        if not isinstance(scenes_payload, list):
            return {"project_id": aligned_storyboard.get("project_id", ""), "storyboard": []}

        storyboard: list[dict[str, Any]] = []
        for scene in sorted(scenes_payload, key=lambda x: int(x.get("scene_id", 0))):
            scene_id = int(scene.get("scene_id", 0))
            va = scene.get("visual_analysis", {})
            subjects = va.get("subjects", [])

            char_mappings: list[dict[str, Any]] = []
            for subject in subjects:
                ref_id = str(subject.get("id", "")).strip()
                if ref_id:
                    mapping = {
                        "ref_id": ref_id,
                        "appearance": str(subject.get("appearance", "")).strip(),
                        "action_in_shot": str(subject.get("action", "")).strip(),
                    }
                    expression = str(subject.get("expression", "")).strip()
                    if expression:
                        mapping["expression_range"] = [expression]
                    char_mappings.append(mapping)

            env = va.get("environment", {})
            camera = va.get("camera", {})
            env_text = "，".join([x for x in [env.get("location", ""), env.get("lighting", ""), env.get("atmosphere", "")] if x])
            cam_text = "，".join([x for x in [camera.get("shot_size", ""), camera.get("angle", ""), camera.get("movement", "")] if x])
            storyboard.append(
                {
                    "shot_id": scene_id,
                    "character_mappings": char_mappings,
                    "environment_context": env_text,
                    "camera_instruction": cam_text,
                }
            )

        return {
            "project_id": aligned_storyboard.get("project_id", ""),
            "storyboard": storyboard,
        }

    def _build_stage5_generation_input(
        self,
        *,
        aligned_storyboard: dict[str, Any],
        stage4_storyboard: dict[str, Any],
        character_bank: dict[str, Any],
        ref_image_url_by_id: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        ref_image_url_by_id = ref_image_url_by_id or {}
        ref_catalog = []
        for item in character_bank.get("characters", []):
            ref_id = str(item.get("ref_id", "")).strip()
            if not ref_id:
                continue
            ref_path = str(item.get("ref_image_path", "")).strip()
            ref_image_url = str(ref_image_url_by_id.get(ref_id, "")).strip()
            ref_catalog.append(
                {
                    "ref_id": ref_id,
                    "ref_image_path": ref_path,
                    "ref_image_url": ref_image_url,
                    "scene_presence": item.get("scene_presence", []),
                }
            )

        return {
            "project_id": stage4_storyboard.get("project_id", "") or aligned_storyboard.get("project_id", ""),
            "storyboard": [dict(x) for x in stage4_storyboard.get("storyboard", [])],
            "reference_catalog": ref_catalog,
        }

    def _normalize_expression_range(self, raw: Any) -> list[str]:
        if isinstance(raw, list):
            return [str(item).strip() for item in raw if str(item).strip()]
        if isinstance(raw, str):
            text = raw.strip()
            if text:
                return [text]
        return []

    def _parse_subject_features(self, appearance: str, action: str) -> dict[str, Any]:
        text = self._normalize_text(f"{appearance} {action}")
        gender = "unknown"
        if any(x in text for x in ["男性", "男人", "男", "male", "man"]):
            gender = "male"
        if any(x in text for x in ["女性", "女孩", "女人", "女", "female", "woman"]):
            gender = "female"

        role = "human"
        if any(x in text for x in ["医生", "白大褂", "lab coat", "doctor"]):
            role = "doctor"
        elif any(x in text for x in ["鱼尾", "美人鱼", "mermaid"]):
            role = "mermaid"

        hair_color = "unknown"
        if any(x in text for x in ["紫色", "purple"]):
            hair_color = "purple"
        elif any(x in text for x in ["红色", "red"]):
            hair_color = "red"
        elif any(x in text for x in ["金发", "blonde"]):
            hair_color = "blonde"
        elif any(x in text for x in ["黑发", "black hair", "黑色头发"]):
            hair_color = "black"

        hair_style = "unknown"
        if any(x in text for x in ["辫", "braid"]):
            hair_style = "braid"
        elif any(x in text for x in ["双马尾", "pigtail"]):
            hair_style = "pigtail"
        elif any(x in text for x in ["短发", "short hair"]):
            hair_style = "short"

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
        ]:
            if token in text:
                anchors.add(alias)

        visibility = "normal"
        if any(x in text for x in ["部分可见", "partially visible", "partially", "右侧可见", "partially visible to the right"]):
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
            "t-shirt": "短袖",
            "t恤": "短袖",
            "braid": "辫子",
            "pigtail": "双马尾",
            "short hair": "短发",
            "partially visible": "部分可见",
            "partially": "部分可见",
            "overweight": "偏胖",
            "plus size": "大码",
            "obese": "肥胖",
            "chubby": "偏胖",
            "heavyset": "壮实",
            "slim": "纤细",
            "slender": "纤细",
            "lean": "精瘦",
            "muscular": "健壮肌肉",
            "athletic": "健壮肌肉",
            "fit": "匀称",
            "toned": "健壮肌肉",
        }
        for src, dst in replacements.items():
            normalized = normalized.replace(src, dst)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _extract_body_type(self, text: str) -> str:
        token_map: dict[str, list[str]] = {
            "overweight": ["偏胖", "肥胖", "大码", "丰满", "圆润", "胖", "壮实"],
            "slim": ["纤细", "苗条", "精瘦", "修长", "瘦"],
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
        cluster["partial_ratio"] = (
            visibility_counter.get("partial", 0) / max(1, len(cluster.get("members", [])))
        )

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

        s = subject.get("features", {})
        scores: list[float] = []
        for member in members:
            m = member.get("features", {})
            local_parts: list[float] = []
            for key in ["gender", "role", "hair_color", "hair_style"]:
                a = str(s.get(key, "unknown"))
                b = str(m.get(key, "unknown"))
                if a == "unknown" or b == "unknown":
                    continue
                local_parts.append(1.0 if a == b else 0.0)
            if local_parts:
                scores.append(sum(local_parts) / len(local_parts))
        if not scores:
            return 0.5
        return max(scores)

    def _temporal_continuity(self, scene_id: int, scene_ids: set[int]) -> float:
        if not scene_ids:
            return 0.5
        min_gap = min(abs(scene_id - sid) for sid in scene_ids)
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
        sid = int(subject.get("scene_id", 0))
        if sid in cluster.get("scene_ids", set()):
            return 0.0
        return 1.0

    def _is_extra_cluster(self, cluster: dict[str, Any]) -> bool:
        members = cluster.get("members", [])
        if not members:
            return True

        scene_count = len(cluster.get("scene_ids", set()))
        dominant_role = cluster.get("dominant_role", "human")
        partial_ratio = float(cluster.get("partial_ratio", 0.0))
        avg_anchor = self._mean([len(m.get("features", {}).get("anchors", [])) for m in members])

        if dominant_role == "doctor":
            return False
        if partial_ratio >= 0.5 and scene_count <= 2:
            return True
        if scene_count < 2 and avg_anchor < 2.0 and dominant_role == "human":
            return True
        return False

    def _build_cluster_states(self, members: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for member in members:
            state_id = self._infer_state(member)
            if state_id not in grouped:
                grouped[state_id] = {
                    "scene_presence": set(),
                    "candidates": [],
                }

            scene_id = int(member.get("scene_id", 0))
            grouped[state_id]["scene_presence"].add(scene_id)
            grouped[state_id]["candidates"].append(
                {
                    "scene_id": scene_id,
                    "appearance": str(member.get("appearance", "")).strip(),
                    "appearance_norm": str(member.get("appearance_norm", "")).strip(),
                }
            )

        states: list[dict[str, Any]] = []
        for state_id, info in grouped.items():
            description = self._pick_state_description(
                state_id=state_id,
                candidates=info.get("candidates") or [],
            )
            states.append(
                {
                    "state_id": state_id,
                    "scene_presence": sorted(info["scene_presence"]),
                    "description": description,
                }
            )
        return sorted(states, key=lambda x: x["state_id"])

    def _pick_state_description(self, state_id: str, candidates: list[dict[str, Any]]) -> str:
        if not candidates:
            return ""

        expects_mermaid = state_id.startswith("mermaid_")
        expects_human = state_id.startswith("human_")

        def _role_match_score(item: dict[str, Any]) -> int:
            appearance = str(item.get("appearance", "")).strip()
            appearance_norm = str(item.get("appearance_norm", "")).strip()
            has_mermaid = self._contains_mermaid_token(appearance) or self._contains_mermaid_token(appearance_norm)
            if expects_mermaid:
                return 1 if has_mermaid else 0
            if expects_human:
                return 1 if not has_mermaid else 0
            return 1

        sorted_candidates = sorted(
            candidates,
            key=lambda x: (
                -_role_match_score(x),
                int(x.get("scene_id", 10**9)),
            ),
        )
        chosen = sorted_candidates[0]
        description = str(chosen.get("appearance", "")).strip() or str(chosen.get("appearance_norm", "")).strip()
        return self._force_state_role_hint(state_id=state_id, description=description)

    def _contains_mermaid_token(self, text: str) -> bool:
        lower = text.lower()
        return any(token in lower for token in ["鱼尾", "美人鱼", "人鱼", "mermaid"])

    def _force_state_role_hint(self, state_id: str, description: str) -> str:
        text = description.strip()
        if not text:
            return text

        has_mermaid = self._contains_mermaid_token(text)
        if state_id.startswith("mermaid_") and not has_mermaid:
            return f"{text}，下半身为鱼尾（美人鱼形态）"
        if state_id.startswith("human_") and has_mermaid:
            return f"{text}（人类形态）"
        return text

    def _infer_state(self, member: dict[str, Any]) -> str:
        text = str(member.get("appearance_norm") or member.get("appearance") or "").lower()
        body_type = str(member.get("features", {}).get("body_type", "unknown"))

        if any(x in text for x in ["医生", "白大褂", "doctor"]):
            return "doctor"
        if any(x in text for x in ["鱼尾", "美人鱼", "mermaid"]):
            if body_type == "overweight":
                return "mermaid_overweight"
            if body_type == "slim":
                return "mermaid_slim"
            if body_type == "muscular":
                return "mermaid_muscular"
            if body_type == "normal":
                return "mermaid_normal"
            return "mermaid_form"
        if body_type == "overweight":
            return "human_overweight"
        if body_type == "slim":
            return "human_slim"
        if body_type == "muscular":
            return "human_muscular"
        if body_type == "normal":
            return "human_normal"
        return "human_form"

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
        return sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]

    def _mean(self, values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _run_stage4(
        self,
        task: TaskRecord,
        params: dict[str, Any],
        character_bank: dict[str, Any],
        aligned_storyboard: dict[str, Any],
        progress_cb: ProgressCallback,
        started_at: float,
    ) -> dict[str, Any]:
        vlm_model = str(params.get("vlm_model") or self.default_vlm_model)
        retry_max = int(params.get("retry_max") if params.get("retry_max") is not None else self.default_retry_max)

        if not self.architect_prompt_path.exists():
            raise Stage1Error("prompt_template_missing", "缺少 V4.2 架构师提示词模板", 500)

        stage4_base_storyboard = self._project_aligned_storyboard_to_stage4_storyboard(aligned_storyboard)
        expected_bindings = self._build_reference_bindings_by_shot(stage4_base_storyboard)
        stage4_storyboard = self._attach_reference_bindings_to_storyboard(
            aligned_storyboard=stage4_base_storyboard,
            bindings_by_shot=expected_bindings,
        )
        ref_image_url_by_id = self._load_ref_image_url_map(task.task_id)
        stage5_input = self._build_stage5_generation_input(
            aligned_storyboard=aligned_storyboard,
            stage4_storyboard=stage4_storyboard,
            character_bank=character_bank,
            ref_image_url_by_id=ref_image_url_by_id,
        )

        architect_prompt = self.architect_prompt_path.read_text(encoding="utf-8")
        stage5_dump_path = str(params.get("stage5_dump_path", "") or "").strip()
        debug_context: dict[str, Any] | None = {} if stage5_dump_path else None
        table = self.vlm_provider.generate_production_table(
            character_bank=character_bank,
            aligned_storyboard=stage5_input,
            architect_prompt=architect_prompt,
            model=vlm_model,
            retry_max=retry_max,
            debug_context=debug_context,
        )
        if isinstance(debug_context, dict):
            debug_context["provider_table_output"] = json.loads(json.dumps(table, ensure_ascii=False))
        table["project_id"] = task.task_id

        prompts = table.get("prompts") or []
        if not isinstance(prompts, list):
            raise Stage1Error("prompt_invalid", "最终提示词表结构无效", 500)

        normalized_prompts = []
        for item in prompts:
            shot_id = int(item.get("shot_id", 0))
            expected = expected_bindings.get(shot_id, [])
            actual = self._normalize_reference_bindings(item.get("reference_bindings"))
            bindings = actual if self._reference_bindings_match(expected, actual) else expected
            normalized_prompts.append(
                {
                    "shot_id": shot_id,
                    "reference_bindings": bindings,
                    "image_prompt": str(item.get("image_prompt", "")).strip(),
                    "video_prompt": str(item.get("video_prompt", "")).strip(),
                }
            )
        table["prompts"] = normalized_prompts
        if isinstance(debug_context, dict):
            self._dump_stage5_context(
                task_id=task.task_id,
                raw_path=stage5_dump_path,
                debug_context=debug_context,
                final_table=table,
            )

        self.store.write_contract(task.task_id, "final_production_table", table)
        md = self._to_markdown_table(normalized_prompts)
        self.store.write_markdown(task.task_id, md)

        progress_cb(92.0, "指令合成完成")
        self._assert_not_timeout(started_at)
        return table

    def _load_ref_image_url_map(self, task_id: str) -> dict[str, str]:
        path = self.store.ref_oss_map_path(task_id)
        if not path.exists():
            return {}
        try:
            payload = self.store.read_json(path)
        except Exception:  # noqa: BLE001
            return {}

        refs = payload.get("refs")
        if not isinstance(refs, list):
            return {}

        ref_image_url_by_id: dict[str, str] = {}
        for item in refs:
            if not isinstance(item, dict):
                continue
            ref_id = str(item.get("ref_id", "")).strip()
            oss_url = str(item.get("oss_url", "")).strip()
            if ref_id and oss_url:
                ref_image_url_by_id[ref_id] = oss_url
        return ref_image_url_by_id

    def _build_reference_bindings_by_shot(
        self,
        aligned_storyboard: dict[str, Any],
    ) -> dict[int, list[dict[str, Any]]]:
        by_shot: dict[int, list[dict[str, Any]]] = {}
        for shot in aligned_storyboard.get("storyboard", []):
            shot_id = int(shot.get("shot_id", 0))
            bindings: list[dict[str, Any]] = []
            for idx, mapping in enumerate(shot.get("character_mappings", []), start=1):
                ref_id = str(mapping.get("ref_id", "")).strip()
                bindings.append(
                    {
                        "reference_index": idx,
                        "ref_id": ref_id,
                    }
                )
            by_shot[shot_id] = bindings
        return by_shot

    def _attach_reference_bindings_to_storyboard(
        self,
        aligned_storyboard: dict[str, Any],
        bindings_by_shot: dict[int, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        storyboard = []
        for shot in aligned_storyboard.get("storyboard", []):
            shot_id = int(shot.get("shot_id", 0))
            enriched = dict(shot)
            enriched["reference_bindings"] = [dict(x) for x in bindings_by_shot.get(shot_id, [])]
            storyboard.append(enriched)
        return {
            "project_id": aligned_storyboard.get("project_id", ""),
            "storyboard": storyboard,
        }

    def _normalize_reference_bindings(self, raw: Any) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            return []

        normalized: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                reference_index = int(item.get("reference_index", 0))
            except (TypeError, ValueError):
                continue
            if reference_index <= 0:
                continue
            ref_id = str(item.get("ref_id", "")).strip()
            normalized.append(
                {
                    "reference_index": reference_index,
                    "ref_id": ref_id,
                }
            )
        return normalized

    def _reference_bindings_match(self, expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> bool:
        if len(expected) != len(actual):
            return False
        for exp, act in zip(expected, actual, strict=False):
            if int(exp.get("reference_index", 0)) != int(act.get("reference_index", 0)):
                return False
            if str(exp.get("ref_id", "")) != str(act.get("ref_id", "")):
                return False
        return True

    def _resolve_stage5_dump_path(self, task_id: str, raw_path: str) -> Path:
        resolved_raw = raw_path.strip().replace("{task_id}", task_id)
        path = Path(resolved_raw).expanduser()
        if (path.exists() and path.is_dir()) or resolved_raw.endswith(("/", "\\")):
            return path / f"stage05_context_{task_id}.json"
        return path

    def _dump_stage5_context(
        self,
        task_id: str,
        raw_path: str,
        debug_context: dict[str, Any],
        final_table: dict[str, Any],
    ) -> None:
        dump_path = self._resolve_stage5_dump_path(task_id, raw_path)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_id": task_id,
            "architect_prompt": str(debug_context.get("architect_prompt", "")),
            "stage5_protocol_prompt": str(debug_context.get("stage5_protocol_prompt", "")),
            "character_bank": debug_context.get("character_bank", {}),
            "aligned_storyboard": debug_context.get("aligned_storyboard", {}),
            "provider_request": debug_context.get("provider_request", {}),
            "provider_raw_output": debug_context.get("provider_raw_output", {}),
            "provider_table_output": debug_context.get("provider_table_output", {}),
            "final_output": final_table,
        }
        dump_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _persist_index_and_result(
        self,
        task: TaskRecord,
        physical_manifest: dict[str, Any],
        raw_scene_descriptions: dict[str, Any],
        character_bank: dict[str, Any],
        aligned_storyboard: dict[str, Any],
        final_table: dict[str, Any],
    ) -> dict[str, Any]:
        contracts = {
            "physical_manifest": str(self.store.contract_path(task.task_id, "physical_manifest").resolve()),
            "raw_scene_descriptions": str(self.store.contract_path(task.task_id, "raw_scene_descriptions").resolve()),
            "character_bank": str(self.store.contract_path(task.task_id, "character_bank").resolve()),
            "aligned_storyboard": str(self.store.contract_path(task.task_id, "aligned_storyboard").resolve()),
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
            for i, v in enumerate(vec):
                out[i] += v
        return [x / len(vectors) for x in out]

    def _merge_appearances(self, appearances: list[str], max_items: int | None = 3) -> str:
        unique: list[str] = []
        for text in appearances:
            s = text.strip()
            if s and s not in unique:
                unique.append(s)
        if not unique:
            return ""
        if max_items is None:
            return "；".join(unique)
        return "；".join(unique[: max(1, max_items)])

    def _derive_key_features(self, master_description: str) -> list[str]:
        if not master_description:
            return []
        parts = [x.strip() for x in master_description.replace("；", "，").split("，") if x.strip()]
        return parts[:3]
