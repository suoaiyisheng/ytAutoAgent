from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _scene_id_from_presence_item(item: Any) -> int | None:
    raw = item
    if isinstance(item, (list, tuple)):
        if not item:
            return None
        raw = item[0]
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _build_state_scene_lookup(characters: list[dict[str, Any]]) -> dict[tuple[str, int], str]:
    lookup: dict[tuple[str, int], str] = {}
    for character in characters:
        ref_id = str(character.get("ref_id", "")).strip()
        if not ref_id:
            continue
        states = character.get("states")
        if not isinstance(states, list):
            continue
        for state in states:
            state_id = str((state or {}).get("state_id", "")).strip()
            if not state_id:
                continue
            scene_presence = (state or {}).get("scene_presence") or []
            for sid in scene_presence:
                try:
                    scene_id = int(sid)
                except (TypeError, ValueError):
                    continue
                key = (ref_id, scene_id)
                if key not in lookup:
                    lookup[key] = state_id
    return lookup


def _build_state_text_lookup(characters: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    for character in characters:
        ref_id = str(character.get("ref_id", "")).strip()
        if not ref_id:
            continue
        states = character.get("states")
        if not isinstance(states, list):
            continue
        for state in states:
            state_id = str((state or {}).get("state_id", "")).strip()
            if not state_id:
                continue
            state_text = str((state or {}).get("description", "")).strip()
            lookup[(ref_id, state_id)] = state_text
    return lookup


def _build_reference_bindings_by_shot(
    storyboard: list[dict[str, Any]],
    characters: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    state_lookup = _build_state_scene_lookup(characters)
    state_text_lookup = _build_state_text_lookup(characters)
    single_state_lookup: dict[str, str] = {}
    for character in characters:
        ref_id = str(character.get("ref_id", "")).strip()
        if not ref_id:
            continue
        states = character.get("states")
        if not isinstance(states, list):
            continue
        ids = [str((s or {}).get("state_id", "")).strip() for s in states]
        ids = [x for x in ids if x]
        if len(ids) == 1:
            single_state_lookup[ref_id] = ids[0]

    by_shot: dict[int, list[dict[str, Any]]] = {}
    for shot in storyboard:
        shot_id = int(shot.get("shot_id", 0))
        bindings: list[dict[str, Any]] = []
        for idx, mapping in enumerate(shot.get("character_mappings", []), start=1):
            ref_id = str(mapping.get("ref_id", "")).strip()
            raw_state = mapping.get("state_id")
            state_id = str(raw_state).strip() if raw_state is not None else ""
            if not state_id and ref_id:
                state_id = state_lookup.get((ref_id, shot_id), "")
            if not state_id and ref_id in single_state_lookup:
                state_id = single_state_lookup[ref_id]
            bindings.append(
                {
                    "reference_index": idx,
                    "ref_id": ref_id,
                    "state_id": state_id or None,
                    "state_text": state_text_lookup.get((ref_id, state_id), "") if state_id else "",
                }
            )
        by_shot[shot_id] = bindings
    return by_shot


def _inject_state_text(image_prompt: str, bindings: list[dict[str, Any]]) -> str:
    text = image_prompt
    for binding in bindings:
        state_text = str(binding.get("state_text", "")).strip()
        if not state_text:
            continue
        index = int(binding.get("reference_index", 0))
        if index <= 0:
            continue
        pattern = rf"参考图{index}(?!（)"
        text = re.sub(pattern, f"参考图{index}（{state_text}状态）", text)
    return text


def _scene_assets(physical_manifest: dict[str, Any]) -> dict[int, dict[str, str]]:
    assets: dict[int, dict[str, str]] = {}
    for item in physical_manifest.get("scenes", []):
        try:
            sid = int(item.get("scene_id", 0))
        except (TypeError, ValueError):
            continue
        assets[sid] = {
            "keyframe_path": str(item.get("keyframe_path", "")).strip(),
            "clip_path": str(item.get("clip_path", "")).strip(),
        }
    return assets


def _select_unique_ref_image_paths(
    characters: list[dict[str, Any]],
    storyboard: list[dict[str, Any]],
    scene_assets: dict[int, dict[str, str]],
) -> dict[str, str]:
    shot_presence: dict[str, list[int]] = defaultdict(list)
    solo_presence: dict[str, list[int]] = defaultdict(list)
    for shot in storyboard:
        try:
            shot_id = int(shot.get("shot_id", 0))
        except (TypeError, ValueError):
            continue
        mappings = shot.get("character_mappings", [])
        ref_ids = [str(m.get("ref_id", "")).strip() for m in mappings if str(m.get("ref_id", "")).strip()]
        for ref_id in ref_ids:
            shot_presence[ref_id].append(shot_id)
        if len(ref_ids) == 1:
            solo_presence[ref_ids[0]].append(shot_id)

    current_paths: dict[str, str] = {}
    for character in characters:
        ref_id = str(character.get("ref_id", "")).strip()
        if not ref_id:
            continue
        current_paths[ref_id] = str(character.get("ref_image_path", "")).strip()

    path_to_refs: dict[str, list[str]] = defaultdict(list)
    for ref_id, path in current_paths.items():
        path_to_refs[path].append(ref_id)

    resolved = dict(current_paths)
    used_paths = set(path for path in current_paths.values() if path)
    for path, refs in path_to_refs.items():
        if not path or len(refs) <= 1:
            continue
        refs_sorted = sorted(refs)
        keeper = refs_sorted[0]
        for ref_id in refs_sorted[1:]:
            candidates: list[str] = []
            for sid in solo_presence.get(ref_id, []):
                keyframe = scene_assets.get(sid, {}).get("keyframe_path", "")
                if keyframe:
                    candidates.append(keyframe)
            for sid in shot_presence.get(ref_id, []):
                keyframe = scene_assets.get(sid, {}).get("keyframe_path", "")
                if keyframe:
                    candidates.append(keyframe)
            for sid in (next((c for c in characters if str(c.get("ref_id", "")).strip() == ref_id), {}) or {}).get(
                "scene_presence", []
            ):
                scene_id = _scene_id_from_presence_item(sid)
                if scene_id is None:
                    continue
                keyframe = scene_assets.get(scene_id, {}).get("keyframe_path", "")
                if keyframe:
                    candidates.append(keyframe)

            new_path = ""
            for candidate in candidates:
                if candidate != path and candidate not in used_paths:
                    new_path = candidate
                    break
            if not new_path:
                for candidate in candidates:
                    if candidate != path:
                        new_path = candidate
                        break
            if new_path:
                resolved[ref_id] = new_path
                used_paths.add(new_path)
        used_paths.add(resolved.get(keeper, path))
    return resolved


def repair_job(job_dir: Path, write_generation_inputs: bool = True) -> dict[str, Any]:
    physical_manifest_path = job_dir / "01_physical_manifest.json"
    character_bank_path = job_dir / "03_character_bank.json"
    aligned_storyboard_path = job_dir / "04_aligned_storyboard.json"
    final_table_path = job_dir / "05_final_production_table.json"
    generation_inputs_path = job_dir / "06_generation_inputs.json"

    if not all(path.exists() for path in [physical_manifest_path, character_bank_path, aligned_storyboard_path, final_table_path]):
        missing = [str(p) for p in [physical_manifest_path, character_bank_path, aligned_storyboard_path, final_table_path] if not p.exists()]
        raise FileNotFoundError(f"缺少必要文件: {missing}")

    physical_manifest = _load_json(physical_manifest_path)
    character_bank = _load_json(character_bank_path)
    aligned_storyboard = _load_json(aligned_storyboard_path)
    final_table = _load_json(final_table_path)

    project_id = str(final_table.get("project_id") or character_bank.get("project_id") or job_dir.name)
    characters = character_bank.get("characters", [])
    storyboard = aligned_storyboard.get("storyboard", [])
    scene_assets = _scene_assets(physical_manifest)

    resolved_ref_paths = _select_unique_ref_image_paths(characters=characters, storyboard=storyboard, scene_assets=scene_assets)
    for character in characters:
        ref_id = str(character.get("ref_id", "")).strip()
        if ref_id in resolved_ref_paths:
            character["ref_image_path"] = resolved_ref_paths[ref_id]

    expected_bindings = _build_reference_bindings_by_shot(storyboard=storyboard, characters=characters)
    prompts_by_shot: dict[int, dict[str, Any]] = {}
    for item in final_table.get("prompts", []):
        try:
            shot_id = int(item.get("shot_id", 0))
        except (TypeError, ValueError):
            continue
        prompts_by_shot[shot_id] = item

    repaired_prompts: list[dict[str, Any]] = []
    for shot in storyboard:
        shot_id = int(shot.get("shot_id", 0))
        old_item = prompts_by_shot.get(shot_id, {})
        bindings = expected_bindings.get(shot_id, [])
        image_prompt = _inject_state_text(str(old_item.get("image_prompt", "")).strip(), bindings)
        repaired_prompts.append(
            {
                "shot_id": shot_id,
                "reference_bindings": bindings,
                "image_prompt": image_prompt,
                "video_prompt": str(old_item.get("video_prompt", "")).strip(),
            }
        )

    final_table["project_id"] = project_id
    final_table["prompts"] = repaired_prompts
    character_bank["project_id"] = project_id
    aligned_storyboard["project_id"] = project_id
    character_bank["characters"] = characters

    _write_json(final_table_path, final_table)
    _write_json(character_bank_path, character_bank)
    _write_json(aligned_storyboard_path, aligned_storyboard)

    generation_inputs: dict[str, Any] = {}
    if write_generation_inputs:
        ref_path_by_id = {
            str(item.get("ref_id", "")).strip(): str(item.get("ref_image_path", "")).strip()
            for item in characters
            if str(item.get("ref_id", "")).strip()
        }
        tasks = []
        for item in repaired_prompts:
            shot_id = int(item.get("shot_id", 0))
            bindings = item.get("reference_bindings", [])
            reference_images = []
            for binding in bindings:
                ref_id = str(binding.get("ref_id", "")).strip()
                reference_images.append(
                    {
                        "reference_index": int(binding.get("reference_index", 0)),
                        "ref_id": ref_id,
                        "state_id": binding.get("state_id"),
                        "state_text": str(binding.get("state_text", "")).strip(),
                        "image_path": ref_path_by_id.get(ref_id, ""),
                    }
                )
            tasks.append(
                {
                    "shot_id": shot_id,
                    "image_prompt": str(item.get("image_prompt", "")).strip(),
                    "video_prompt": str(item.get("video_prompt", "")).strip(),
                    "reference_images": reference_images,
                    "keyframe_path": scene_assets.get(shot_id, {}).get("keyframe_path", ""),
                    "clip_path": scene_assets.get(shot_id, {}).get("clip_path", ""),
                }
            )

        generation_inputs = {
            "project_id": project_id,
            "upstream_project_id": str(physical_manifest.get("project_id", "")).strip(),
            "global_assets": {
                "references": [
                    {
                        "ref_id": str(item.get("ref_id", "")).strip(),
                        "ref_image_path": str(item.get("ref_image_path", "")).strip(),
                    }
                    for item in characters
                    if str(item.get("ref_id", "")).strip()
                ]
            },
            "tasks": tasks,
        }
        _write_json(generation_inputs_path, generation_inputs)

    return {
        "project_id": project_id,
        "job_dir": str(job_dir.resolve()),
        "prompts_count": len(repaired_prompts),
        "ref_paths": resolved_ref_paths,
        "generation_inputs_path": str(generation_inputs_path.resolve()) if write_generation_inputs else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair stage3/4 outputs and generate downstream inputs.")
    parser.add_argument("--job-dir", required=True, help="job 目录，需包含 01/03/04/05 四个 JSON 文件")
    parser.add_argument("--skip-generation-inputs", action="store_true", help="不生成 06_generation_inputs.json")
    args = parser.parse_args()

    result = repair_job(
        job_dir=Path(args.job_dir).expanduser().resolve(),
        write_generation_inputs=not args.skip_generation_inputs,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
