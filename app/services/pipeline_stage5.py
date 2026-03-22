from __future__ import annotations

"""
模块作用：
- 负责 Stage 5 指令合成，包括构造最小生成输入并输出最终提示词表。

当前文件工具函数：
- `_build_stage5_input`：把标准化描述和角色库收敛为最终生成输入。
- `_build_stage5_protocol_prompt`：统一定义 Stage 5 投喂给模型的协议提示词。
- `_load_ref_image_url_map` / `_resolve_stage5_dump_path` / `_dump_stage5_context`：加载参考图 URL 并写入调试上下文。
"""

import json
import re
from pathlib import Path
from typing import Any, Callable

from app.errors import Stage1Error
from app.models import TaskRecord


class PipelineStage5Mixin:
    def _run_stage5(
        self,
        task: TaskRecord,
        params: dict[str, Any],
        character_bank: dict[str, Any],
        normalized_scene_descriptions: dict[str, Any],
        progress_cb: Callable[[float, str], None],
        started_at: float,
    ) -> dict[str, Any]:
        vlm_model = str(params.get("vlm_model") or self.default_vlm_model)
        retry_max = int(params.get("retry_max") if params.get("retry_max") is not None else self.default_retry_max)

        if not self.architect_prompt_path.exists():
            raise Stage1Error("prompt_template_missing", "缺少 V4.2 架构师提示词模板", 500)

        ref_image_url_by_id = self._load_ref_image_url_map(task.task_id)
        stage5_input = self._build_stage5_input(
            character_bank=character_bank,
            normalized_scene_descriptions=normalized_scene_descriptions,
            ref_image_url_by_id=ref_image_url_by_id,
        )
        stage5_protocol_prompt = self._build_stage5_protocol_prompt()

        architect_prompt = self.architect_prompt_path.read_text(encoding="utf-8")
        stage5_dump_path = str(params.get("stage5_dump_path", "") or "").strip()
        debug_context: dict[str, Any] | None = {} if stage5_dump_path else None
        if isinstance(debug_context, dict):
            debug_context["character_bank"] = json.loads(json.dumps(character_bank, ensure_ascii=False))
            debug_context["stage5_protocol_prompt"] = stage5_protocol_prompt

        table = self.vlm_provider.generate_production_table(
            stage5_input=stage5_input,
            stage5_protocol_prompt=stage5_protocol_prompt,
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
            normalized_prompts.append(
                {
                    "shot_id": shot_id,
                    "image_prompt": str(item.get("image_prompt", "")).strip(),
                    "video_prompt": str(item.get("video_prompt", "")).strip(),
                }
            )
        table["prompts"] = sorted(normalized_prompts, key=lambda item: int(item.get("shot_id", 0)))

        if isinstance(debug_context, dict):
            self._dump_stage5_context(
                task_id=task.task_id,
                raw_path=stage5_dump_path,
                debug_context=debug_context,
                final_table=table,
            )

        self.store.write_contract(task.task_id, "final_production_table", table)
        self.store.write_markdown(task.task_id, self._to_markdown_table(table["prompts"]))
        progress_cb(94.0, "指令合成完成")
        self._assert_not_timeout(started_at)
        return table

    def _build_stage5_input(
        self,
        *,
        character_bank: dict[str, Any],
        normalized_scene_descriptions: dict[str, Any],
        ref_image_url_by_id: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        ref_image_url_by_id = ref_image_url_by_id or {}
        embed_model = str(getattr(self, "default_embed_model", "") or "")
        retry_max = int(getattr(self, "default_retry_max", 0) or 0)
        embedding_cache: dict[str, list[float]] = {}

        reference_catalog_by_id: dict[str, dict[str, Any]] = {}
        reference_catalog_by_name: dict[str, dict[str, Any]] = {}
        for item in character_bank.get("characters", []):
            ref_id = str(item.get("ref_id", "")).strip()
            if not ref_id:
                continue
            ref_name = str(item.get("ref_name", "")).strip() or f"参考图{ref_id.split('_')[-1]}"
            catalog_item = {
                "ref_id": ref_id,
                "ref_name": ref_name,
                "ref_image_path": str(item.get("ref_image_path", "")).strip(),
                "ref_image_url": str(ref_image_url_by_id.get(ref_id, "")).strip(),
                "ref_image_description": str(item.get("ref_image_description", "")).strip(),
                "ref_image_features": [
                    str(token).strip()
                    for token in item.get("ref_image_features", [])
                    if str(token).strip()
                ],
            }
            reference_catalog_by_id[ref_id] = catalog_item
            reference_catalog_by_name[ref_name] = catalog_item

        self._prime_embedding_cache(
            texts=[
                self._appearance_signature_text(str(item.get("ref_image_description", "")).strip())
                for item in reference_catalog_by_id.values()
            ],
            embed_model=embed_model,
            retry_max=retry_max,
            embedding_cache=embedding_cache,
        )

        shots: list[dict[str, Any]] = []
        for scene in sorted(normalized_scene_descriptions.get("scenes", []), key=lambda item: int(item.get("scene_id", 0))):
            shot_id = int(scene.get("scene_id", 0))
            desc = str(scene.get("desc", "")).strip()
            reference_groundings = self._build_reference_groundings(
                desc=desc,
                reference_catalog_by_id=reference_catalog_by_id,
                reference_catalog_by_name=reference_catalog_by_name,
                embed_model=embed_model,
                retry_max=retry_max,
                embedding_cache=embedding_cache,
            )
            shots.append(
                {
                    "shot_id": shot_id,
                    "grounded_desc": self._build_grounded_desc(desc=desc, reference_groundings=reference_groundings),
                    "references": self._build_stage5_shot_references(reference_groundings, reference_catalog_by_id),
                }
            )

        return {
            "project_id": normalized_scene_descriptions.get("project_id", "") or character_bank.get("project_id", ""),
            "shots": shots,
        }

    def _build_stage5_protocol_prompt(self) -> str:
        return (
            "请只输出合法 JSON，对象结构固定为 "
            "{\"project_id\":\"string\",\"prompts\":[{\"shot_id\":number,\"image_prompt\":\"string\",\"video_prompt\":\"string\"}]}。"
            "不要输出 markdown，不要输出解释。"
            "所有自然语言内容必须使用简体中文，字段名保持英文。"
            "输入 JSON 中，每个 shot 只提供 grounded_desc 与当前镜头对应的 references。"
            "你必须仅依据 grounded_desc 描述镜头内容，仅依据 references 中给出的参考图来理解“参考图n”对应的人物。"
            "在 image_prompt 与 video_prompt 中引用人物时，必须沿用 grounded_desc 里已经出现的“参考图n”或“（...）的参考图n”写法，不得改写成 Ref_n，也不得改写成 subject_n。"
            "如果 grounded_desc 中已经写明性别、体型、发型/发色、服装、鱼尾、配饰、道具、位置、环境或动作，你必须尽量完整保留，不能压缩成模糊代称。"
            "references 只用于告诉你该镜头有哪些参考图及其对应图片，不要在输出中泄露文件路径或 URL。"
            "不要输出 reference_bindings 或任何额外字段。"
        )

    def _build_stage5_shot_references(
        self,
        reference_groundings: list[dict[str, Any]],
        reference_catalog_by_id: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        references: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in reference_groundings:
            ref_id = str(item.get("ref_id", "")).strip()
            if not ref_id:
                continue
            catalog_item = reference_catalog_by_id.get(ref_id, {})
            ref_name = str(item.get("ref_name", "")).strip() or str(catalog_item.get("ref_name", "")).strip()
            if not ref_name or ref_name in seen:
                continue
            seen.add(ref_name)
            references.append(
                {
                    "ref_name": ref_name,
                    "ref_image_path": str(catalog_item.get("ref_image_path", "")).strip(),
                    "ref_image_url": str(catalog_item.get("ref_image_url", "")).strip(),
                }
            )
        return references

    def _build_reference_groundings(
        self,
        *,
        desc: str,
        reference_catalog_by_id: dict[str, dict[str, Any]],
        reference_catalog_by_name: dict[str, dict[str, Any]],
        embed_model: str,
        retry_max: int,
        embedding_cache: dict[str, list[float]],
    ) -> list[dict[str, Any]]:
        groundings: list[dict[str, Any]] = []
        seen: set[str] = set()
        pattern = re.compile(r"(?:（(?P<appearance>[^）]+)）的)?(?:(?P<ref_name>参考图\d+)|(?P<ref_id>Ref_\d+))")
        for match in pattern.finditer(desc):
            ref_id = str(match.group("ref_id") or "").strip()
            ref_name = str(match.group("ref_name") or "").strip()
            if not ref_id and ref_name:
                ref_id = str((reference_catalog_by_name.get(ref_name) or {}).get("ref_id", "")).strip()
            if not ref_name and ref_id:
                ref_name = str((reference_catalog_by_id.get(ref_id) or {}).get("ref_name", "")).strip()
            if not ref_id or ref_id in seen:
                continue
            seen.add(ref_id)
            current_appearance = str(match.group("appearance") or "").strip()
            catalog_item = reference_catalog_by_id.get(ref_id, {})
            ref_name = ref_name or str(catalog_item.get("ref_name", "")).strip()
            reference_index = self._reference_index_from_name(ref_name)
            ref_image_description = str(catalog_item.get("ref_image_description", "")).strip()
            ref_image_features = [
                str(token).strip() for token in catalog_item.get("ref_image_features", []) if str(token).strip()
            ]
            appearance_matches_reference = not current_appearance or self._appearance_matches_reference(
                current_appearance=current_appearance,
                ref_image_description=ref_image_description,
                embed_model=embed_model,
                retry_max=retry_max,
                embedding_cache=embedding_cache,
            )
            preferred_reference_phrase = (
                ref_name
                if appearance_matches_reference and ref_name
                else f"（{current_appearance}）的{ref_name}"
                if ref_name and current_appearance
                else ref_name or ref_id
            )
            groundings.append(
                {
                    "reference_index": reference_index,
                    "ref_id": ref_id,
                    "ref_name": ref_name,
                    "current_appearance": current_appearance,
                    "current_features": self._appearance_feature_tokens(current_appearance),
                    "ref_image_description": ref_image_description,
                    "ref_image_features": ref_image_features,
                    "appearance_matches_reference": appearance_matches_reference,
                    "preferred_reference_phrase": preferred_reference_phrase,
                }
            )
        return groundings

    def _build_grounded_desc(
        self,
        *,
        desc: str,
        reference_groundings: list[dict[str, Any]],
    ) -> str:
        replacement_by_name = {
            str(item.get("ref_name", "")).strip(): str(item.get("preferred_reference_phrase", "")).strip()
            for item in reference_groundings
            if str(item.get("ref_name", "")).strip() and str(item.get("preferred_reference_phrase", "")).strip()
        }
        replacement_by_id = {
            str(item.get("ref_id", "")).strip(): str(item.get("preferred_reference_phrase", "")).strip()
            for item in reference_groundings
            if str(item.get("ref_id", "")).strip() and str(item.get("preferred_reference_phrase", "")).strip()
        }

        def _replace(match: re.Match[str]) -> str:
            appearance = str(match.group("appearance") or "").strip()
            ref_name = str(match.group("ref_name") or "").strip()
            ref_id = str(match.group("ref_id") or "").strip()
            if ref_name in replacement_by_name:
                return replacement_by_name[ref_name]
            if ref_id in replacement_by_id:
                return replacement_by_id[ref_id]
            reference_token = ref_name or ref_id
            return f"（{appearance}）的{reference_token}" if appearance else reference_token

        return re.sub(r"(?:（(?P<appearance>[^）]+)）的)?(?:(?P<ref_name>参考图\d+)|(?P<ref_id>Ref_\d+))", _replace, desc)

    def _appearance_feature_tokens(self, text: str) -> list[str]:
        normalized = str(text).replace("，", ",").replace("；", ",").replace("。", ",")
        parts = re.split(r"[,、]|和", normalized)
        tokens: list[str] = []
        for raw in parts:
            cleaned = raw.strip()
            cleaned = re.sub(r"^(一名|一个|一位)", "", cleaned).strip()
            cleaned = re.sub(r"^(身穿|穿着|穿)", "", cleaned).strip()
            cleaned = re.sub(r"^(佩戴着|佩戴|戴着|戴)", "", cleaned).strip()
            if cleaned and cleaned not in tokens:
                tokens.append(cleaned)
        return tokens

    def _appearance_feature_match(
        self,
        *,
        current_features: list[str],
        ref_features: list[str],
        current_appearance: str,
        ref_image_description: str,
    ) -> bool:
        return self._appearance_matches_reference(
            current_appearance=current_appearance or "".join(current_features),
            ref_image_description=ref_image_description or "".join(ref_features),
            embed_model=str(getattr(self, "default_embed_model", "") or ""),
            retry_max=int(getattr(self, "default_retry_max", 0) or 0),
        )

    def _reference_index_from_name(self, ref_name: str) -> int:
        cleaned = str(ref_name).strip()
        match = re.fullmatch(r"参考图(\d+)", cleaned)
        return int(match.group(1)) if match else 0

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
            "stage5_input": debug_context.get("stage5_input", {}),
            "provider_request": debug_context.get("provider_request", {}),
            "provider_raw_output": debug_context.get("provider_raw_output", {}),
            "provider_table_output": debug_context.get("provider_table_output", {}),
            "final_output": final_table,
        }
        dump_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
