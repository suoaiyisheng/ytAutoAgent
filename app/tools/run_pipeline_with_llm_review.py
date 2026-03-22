#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from app.cli import _build_pipeline, _validate_source
from app.config import load_settings
from app.errors import Stage1Error
from app.models import TaskRecord
from app.services.providers import GeminiVLMProvider, QwenVLMProvider, VLMProvider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行完整流水线并使用多模态模型评审最终提示词还原度。")
    parser.add_argument("--source-url", default=None)
    parser.add_argument("--local-video-path", default=None)
    parser.add_argument("--threshold", type=float, default=27.0)
    parser.add_argument("--min-scene-len", type=float, default=1.0)
    parser.add_argument("--frame-quality", type=int, default=2)
    parser.add_argument("--download-format", default="bestvideo+bestaudio/best")
    parser.add_argument("--vlm-model", default=None)
    parser.add_argument("--embed-model", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--retry-max", type=int, default=None)
    parser.add_argument("--stage5-dump-path", default=None)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--review-provider", choices=["auto", "gemini", "qwen"], default="auto")
    parser.add_argument("--review-model", default=None)
    parser.add_argument("--review-retry-max", type=int, default=1)
    parser.add_argument("--max-review-frames", type=int, default=8)
    parser.add_argument("--skip-review", action="store_true")
    return parser


def _select_review_provider(requested: str) -> tuple[VLMProvider, str]:
    settings = load_settings()
    requested = requested.strip().lower()

    if requested == "gemini":
        if not settings.gemini_api_key and not settings.openrouter_api_key:
            raise Stage1Error("config_error", "当前环境未配置 GEMINI_API_KEY 或 OPENROUTER_API_KEY，无法使用 Gemini 评审", 400)
        return (
            GeminiVLMProvider(
                api_key=settings.gemini_api_key,
                openrouter_api_key=settings.openrouter_api_key,
                openrouter_base_url=settings.openrouter_base_url,
            ),
            settings.gemini_vlm_model,
        )

    if requested == "qwen":
        if not settings.qwen_api_key:
            raise Stage1Error("config_error", "当前环境未配置 QWEN_API_KEY，无法使用 Qwen 评审", 400)
        return QwenVLMProvider(api_key=settings.qwen_api_key, base_url=settings.qwen_base_url), settings.qwen_vlm_model

    if settings.gemini_api_key or settings.openrouter_api_key:
        return (
            GeminiVLMProvider(
                api_key=settings.gemini_api_key,
                openrouter_api_key=settings.openrouter_api_key,
                openrouter_base_url=settings.openrouter_base_url,
            ),
            settings.gemini_vlm_model,
        )
    if settings.qwen_api_key:
        return QwenVLMProvider(api_key=settings.qwen_api_key, base_url=settings.qwen_base_url), settings.qwen_vlm_model
    raise Stage1Error("config_error", "未找到可用的评审模型配置", 400)


def _load_contracts(result: dict[str, Any]) -> dict[str, Any]:
    contracts = {}
    for name, path in result.get("contracts", {}).items():
        contracts[name] = json.loads(Path(path).read_text(encoding="utf-8"))
    return contracts


def _build_review_prompt(
    *,
    project_id: str,
    source_url: str,
    raw_scene_descriptions: dict[str, Any],
    normalized_scene_descriptions: dict[str, Any],
    final_table: dict[str, Any],
) -> str:
    per_shot_rows: list[str] = []
    raw_by_scene = {
        int(item.get("scene_id", 0)): item
        for item in raw_scene_descriptions.get("scenes", [])
    }
    normalized_by_scene = {
        int(item.get("scene_id", 0)): item
        for item in normalized_scene_descriptions.get("scenes", [])
    }
    prompt_by_shot = {
        int(item.get("shot_id", 0)): item
        for item in final_table.get("prompts", [])
    }

    shot_ids = sorted(set(raw_by_scene) | set(normalized_by_scene) | set(prompt_by_shot))
    for shot_id in shot_ids:
        raw_desc = str((raw_by_scene.get(shot_id) or {}).get("desc", "")).strip()
        normalized_desc = str((normalized_by_scene.get(shot_id) or {}).get("desc", "")).strip()
        prompt_item = prompt_by_shot.get(shot_id) or {}
        image_prompt = str(prompt_item.get("image_prompt", "")).strip()
        video_prompt = str(prompt_item.get("video_prompt", "")).strip()
        per_shot_rows.append(
            "\n".join(
                [
                    f"Shot {shot_id}:",
                    f"- 原始场景描述: {raw_desc}",
                    f"- 标准化描述: {normalized_desc}",
                    f"- image_prompt: {image_prompt}",
                    f"- video_prompt: {video_prompt}",
                ]
            )
        )

    return (
        "你是一位严格的视频提示词评审员。你会看到一个真实视频的分镜关键帧，以及该视频经过流水线生成的最终提示词。"
        "请先根据关键帧总结视频本身内容，再判断最终提示词是否还原了视频内容。"
        "评审时以关键帧证据为主，以原始场景描述和标准化描述为辅助，不要被提示词本身带偏。"
        "请只输出 JSON，不要输出 Markdown。\n\n"
        "输出结构必须是：\n"
        "{\n"
        '  "project_id": "string",\n'
        '  "source_url": "string",\n'
        '  "video_summary": "string",\n'
        '  "overall_score": 0,\n'
        '  "overall_verdict": "pass|warn|fail",\n'
        '  "summary_findings": ["string"],\n'
        '  "per_shot": [\n'
        "    {\n"
        '      "shot_id": 1,\n'
        '      "observed_content": "string",\n'
        '      "image_prompt_score": 0,\n'
        '      "video_prompt_score": 0,\n'
        '      "matches": ["string"],\n'
        '      "missing_details": ["string"],\n'
        '      "hallucinations": ["string"],\n'
        '      "verdict": "pass|warn|fail"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"project_id: {project_id}\n"
        f"source_url: {source_url}\n\n"
        "请评审以下分镜：\n"
        + "\n\n".join(per_shot_rows)
    )


def _review_with_gemini(
    *,
    provider: GeminiVLMProvider,
    model: str,
    retry_max: int,
    prompt: str,
    physical_manifest: dict[str, Any],
    max_review_frames: int,
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [{"text": prompt}]
    for item in sorted(physical_manifest.get("scenes", []), key=lambda row: int(row.get("scene_id", 0)))[:max_review_frames]:
        scene_id = int(item.get("scene_id", 0))
        keyframe_path = Path(str(item.get("keyframe_path", "")).strip())
        if not keyframe_path.exists():
            continue
        parts.append({"text": f"Shot {scene_id} 关键帧"})
        parts.append({"inline_data": provider._to_inline_data(keyframe_path)})  # noqa: SLF001
    payload = provider._generate_json(parts=parts, model=model, retry_max=retry_max)  # noqa: SLF001
    if not isinstance(payload, dict):
        raise Stage1Error("review_invalid_output", "Gemini 评审结果不是 JSON 对象", 502)
    return payload


def _review_with_qwen(
    *,
    provider: QwenVLMProvider,
    model: str,
    retry_max: int,
    prompt: str,
    physical_manifest: dict[str, Any],
    max_review_frames: int,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for item in sorted(physical_manifest.get("scenes", []), key=lambda row: int(row.get("scene_id", 0)))[:max_review_frames]:
        scene_id = int(item.get("scene_id", 0))
        keyframe_path = Path(str(item.get("keyframe_path", "")).strip())
        if not keyframe_path.exists():
            continue
        content.append({"type": "text", "text": f"Shot {scene_id} 关键帧"})
        content.append({"type": "image_url", "image_url": {"url": provider._to_data_uri(keyframe_path)}})  # noqa: SLF001
    payload = provider._chat_completion_json(  # noqa: SLF001
        messages=[{"role": "user", "content": content}],
        model=model,
        retry_max=retry_max,
    )
    if not isinstance(payload, dict):
        raise Stage1Error("review_invalid_output", "Qwen 评审结果不是 JSON 对象", 502)
    return payload


def _run_review(
    *,
    result: dict[str, Any],
    source_url: str,
    review_provider_name: str,
    review_model: str | None,
    review_retry_max: int,
    max_review_frames: int,
) -> tuple[dict[str, Any], Path]:
    contracts = _load_contracts(result)
    provider, default_model = _select_review_provider(review_provider_name)
    model = str(review_model or default_model).strip() or default_model

    prompt = _build_review_prompt(
        project_id=str(result.get("project_id", "")).strip(),
        source_url=source_url,
        raw_scene_descriptions=contracts["raw_scene_descriptions"],
        normalized_scene_descriptions=contracts["normalized_scene_descriptions"],
        final_table=contracts["final_production_table"],
    )

    if isinstance(provider, GeminiVLMProvider):
        review_payload = _review_with_gemini(
            provider=provider,
            model=model,
            retry_max=review_retry_max,
            prompt=prompt,
            physical_manifest=contracts["physical_manifest"],
            max_review_frames=max_review_frames,
        )
        actual_provider = "gemini"
    elif isinstance(provider, QwenVLMProvider):
        review_payload = _review_with_qwen(
            provider=provider,
            model=model,
            retry_max=review_retry_max,
            prompt=prompt,
            physical_manifest=contracts["physical_manifest"],
            max_review_frames=max_review_frames,
        )
        actual_provider = "qwen"
    else:
        raise Stage1Error("review_provider_unsupported", "当前评审脚本只支持 Gemini/Qwen", 500)

    review_payload["project_id"] = str(review_payload.get("project_id") or result.get("project_id") or "")
    review_payload["source_url"] = str(review_payload.get("source_url") or source_url)
    review_payload["review_provider"] = actual_provider
    review_payload["review_model"] = model

    task_dir = Path(result["contracts"]["final_production_table"]).resolve().parent
    review_json_path = task_dir / "06_llm_prompt_review.json"
    review_md_path = task_dir / "06_llm_prompt_review.md"
    review_json_path.write_text(json.dumps(review_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    review_md_path.write_text(_review_to_markdown(review_payload), encoding="utf-8")
    return review_payload, review_json_path


def _review_to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# LLM Prompt Review",
        "",
        f"- project_id: {payload.get('project_id', '')}",
        f"- source_url: {payload.get('source_url', '')}",
        f"- review_provider: {payload.get('review_provider', '')}",
        f"- review_model: {payload.get('review_model', '')}",
        f"- overall_score: {payload.get('overall_score', '')}",
        f"- overall_verdict: {payload.get('overall_verdict', '')}",
        "",
        "## Video Summary",
        "",
        str(payload.get("video_summary", "")),
        "",
        "## Findings",
        "",
    ]
    for item in payload.get("summary_findings", []) or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Per Shot", ""])

    for item in payload.get("per_shot", []) or []:
        lines.append(f"### Shot {item.get('shot_id', '')}")
        lines.append(f"- observed_content: {item.get('observed_content', '')}")
        lines.append(f"- image_prompt_score: {item.get('image_prompt_score', '')}")
        lines.append(f"- video_prompt_score: {item.get('video_prompt_score', '')}")
        lines.append(f"- verdict: {item.get('verdict', '')}")
        for key in ["matches", "missing_details", "hallucinations"]:
            values = item.get(key, []) or []
            if not values:
                continue
            lines.append(f"- {key}:")
            for value in values:
                lines.append(f"  - {value}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        _validate_source(args.source_url, args.local_video_path)
        pipeline = _build_pipeline()
        pipeline.validate_ready()

        task_id = str(args.task_id or uuid.uuid4().hex).strip()
        working_dir = str(pipeline.store.task_dir(task_id))
        params = {
            "source_url": args.source_url,
            "local_video_path": str(Path(args.local_video_path).expanduser().resolve()) if args.local_video_path else None,
            "threshold": args.threshold,
            "min_scene_len": args.min_scene_len,
            "frame_quality": args.frame_quality,
            "download_format": args.download_format,
            "vlm_model": args.vlm_model,
            "embed_model": args.embed_model,
            "batch_size": args.batch_size,
            "retry_max": args.retry_max,
            "stage5_dump_path": args.stage5_dump_path,
        }
        task = TaskRecord.new(task_id=task_id, params=params, working_dir=working_dir)
        pipeline.store.save_manifest(task)

        def on_progress(progress: float, message: str) -> None:
            line = f"[{progress:6.2f}%] {message}"
            print(line, flush=True)
            pipeline.store.append_log(task_id, line)

        result = pipeline.run(task, on_progress)
        output = {
            "pipeline_result": result,
        }

        if not args.skip_review:
            review_payload, review_path = _run_review(
                result=result,
                source_url=str(args.source_url or args.local_video_path or ""),
                review_provider_name=args.review_provider,
                review_model=args.review_model,
                review_retry_max=int(args.review_retry_max),
                max_review_frames=max(1, int(args.max_review_frames)),
            )
            output["review_result"] = review_payload
            output["review_json_path"] = str(review_path.resolve())

        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except Stage1Error as exc:
        print(json.dumps({"code": exc.code, "message": exc.message}, ensure_ascii=False), file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"code": "internal_error", "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
