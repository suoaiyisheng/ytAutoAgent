from __future__ import annotations

"""
模块作用：
- 负责 Stage 5 指令合成，包括构造生成输入、校验引用绑定并输出最终提示词表。

当前文件工具函数：
- `_build_stage5_input`：把标准化描述和角色库收敛为最终生成输入。
- `_build_reference_bindings_by_shot`：按镜头中的 `Ref_x` 顺序生成参考绑定关系。
- `_normalize_reference_bindings` / `_reference_bindings_match`：校验模型返回的绑定是否合法。
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
        expected_bindings = self._build_reference_bindings_by_shot(stage5_input)

        architect_prompt = self.architect_prompt_path.read_text(encoding="utf-8")
        stage5_dump_path = str(params.get("stage5_dump_path", "") or "").strip()
        debug_context: dict[str, Any] | None = {} if stage5_dump_path else None

        table = self.vlm_provider.generate_production_table(
            character_bank=character_bank,
            stage5_input=stage5_input,
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

        shots: list[dict[str, Any]] = []
        for scene in sorted(normalized_scene_descriptions.get("scenes", []), key=lambda item: int(item.get("scene_id", 0))):
            shot_id = int(scene.get("scene_id", 0))
            desc = str(scene.get("desc", "")).strip()
            shots.append(
                {
                    "shot_id": shot_id,
                    "desc": desc,
                    "reference_bindings": self._bindings_from_desc(desc),
                }
            )

        reference_catalog = []
        for item in character_bank.get("characters", []):
            ref_id = str(item.get("ref_id", "")).strip()
            if not ref_id:
                continue
            reference_catalog.append(
                {
                    "ref_id": ref_id,
                    "ref_image_path": str(item.get("ref_image_path", "")).strip(),
                    "ref_image_url": str(ref_image_url_by_id.get(ref_id, "")).strip(),
                    "canonical_description": str(item.get("canonical_description", "")).strip(),
                }
            )

        return {
            "project_id": normalized_scene_descriptions.get("project_id", "") or character_bank.get("project_id", ""),
            "shots": shots,
            "reference_catalog": reference_catalog,
        }

    def _bindings_from_desc(self, desc: str) -> list[dict[str, Any]]:
        seen: list[str] = []
        for ref_id in re.findall(r"Ref_\d+", desc):
            if ref_id not in seen:
                seen.append(ref_id)
        return [
            {
                "reference_index": idx,
                "ref_id": ref_id,
            }
            for idx, ref_id in enumerate(seen, start=1)
        ]

    def _build_reference_bindings_by_shot(
        self,
        stage5_input: dict[str, Any],
    ) -> dict[int, list[dict[str, Any]]]:
        return {
            int(shot.get("shot_id", 0)): [dict(item) for item in shot.get("reference_bindings", [])]
            for shot in stage5_input.get("shots", [])
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
        for expected_item, actual_item in zip(expected, actual, strict=False):
            if int(expected_item.get("reference_index", 0)) != int(actual_item.get("reference_index", 0)):
                return False
            if str(expected_item.get("ref_id", "")) != str(actual_item.get("ref_id", "")):
                return False
        return True

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
