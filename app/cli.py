from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import load_settings
from app.errors import Stage1Error
from app.models import ImageGenerationRunRecord, TaskRecord
from app.services.image_generation import ImageGenerationService
from app.services.pipeline import Stage1Pipeline
from app.services.providers import (
    GeminiEmbeddingProvider,
    GeminiVLMProvider,
    QwenEmbeddingProvider,
    QwenVLMProvider,
)
from app.services.store import TaskStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ytAuto 全链路流水线 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run-full", help="运行五阶段完整算法")
    run.add_argument("--source-url", default=None)
    run.add_argument("--local-video-path", default=None)
    run.add_argument("--threshold", type=float, default=27.0)
    run.add_argument("--min-scene-len", type=float, default=1.0)
    run.add_argument("--frame-quality", type=int, default=2)
    run.add_argument("--download-format", default="bestvideo+bestaudio/best")
    run.add_argument("--vlm-model", default=None)
    run.add_argument("--embed-model", default=None)
    run.add_argument("--batch-size", type=int, default=None)
    run.add_argument("--retry-max", type=int, default=None)
    run.add_argument("--stage5-dump-path", default=None)

    gen = sub.add_parser("generate-images", help="运行生图阶段（OSS+模型提供方）")
    gen.add_argument("--job-id", required=True)
    gen.add_argument("--shot-range", default=None)
    gen.add_argument("--candidates-per-shot", type=int, default=4)
    gen.add_argument("--aspect-ratio", default="9:16")
    gen.add_argument("--concurrency", type=int, default=2)
    return parser


def _validate_source(source_url: str | None, local_video_path: str | None) -> None:
    if bool(source_url) == bool(local_video_path):
        raise Stage1Error("invalid_source", "--source-url 与 --local-video-path 必须且只能提供一个", 422)


def _build_pipeline() -> Stage1Pipeline:
    settings = load_settings()
    store = TaskStore(settings.data_dir)

    if settings.vlm_provider == "qwen":
        vlm_provider = QwenVLMProvider(
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
        )
        default_vlm_model = settings.qwen_vlm_model
    else:
        vlm_provider = GeminiVLMProvider(
            api_key=settings.gemini_api_key,
            openrouter_api_key=settings.openrouter_api_key,
            openrouter_base_url=settings.openrouter_base_url,
        )
        default_vlm_model = settings.gemini_vlm_model

    if settings.embedding_provider == "qwen":
        embedding_provider = QwenEmbeddingProvider(
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_embed_base_url,
        )
        default_embed_model = settings.qwen_embed_model
    else:
        embedding_provider = GeminiEmbeddingProvider(settings.gemini_api_key)
        default_embed_model = settings.gemini_embed_model

    return Stage1Pipeline(
        store=store,
        timeout_sec=settings.task_timeout_sec,
        vlm_provider=vlm_provider,
        embedding_provider=embedding_provider,
        default_vlm_model=default_vlm_model,
        default_embed_model=default_embed_model,
        default_retry_max=settings.pipeline_retry_max,
        default_batch_size=settings.pipeline_batch_size,
    )


def run_full(args: argparse.Namespace) -> int:
    _validate_source(args.source_url, args.local_video_path)

    pipeline = _build_pipeline()
    pipeline.validate_ready()

    task_id = uuid.uuid4().hex
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
        pipeline.store.append_log(task_id, f"[{progress:.2f}%] {message}")

    result = pipeline.run(task, on_progress)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _save_image_generation_manifest(record: ImageGenerationRunRecord, pipeline_store: TaskStore) -> None:
    pipeline_store.write_json(
        pipeline_store.image_generation_manifest_path(record.task_id),
        record.to_manifest(),
    )


def run_generate_images(args: argparse.Namespace) -> int:
    settings = load_settings()
    store = TaskStore(settings.data_dir)
    service = ImageGenerationService(store=store, settings=settings)
    service.validate_ready()

    task_id = str(args.job_id).strip()
    task_dir = store.root_dir / task_id
    if not task_dir.exists() or not task_dir.is_dir():
        raise Stage1Error("task_not_found", "任务不存在", 404)

    params = {
        "shot_range": args.shot_range,
        "candidates_per_shot": int(args.candidates_per_shot),
        "aspect_ratio": str(args.aspect_ratio),
        "concurrency": int(args.concurrency),
    }
    record = ImageGenerationRunRecord.new(task_id=task_id, params=params)
    _save_image_generation_manifest(record, store)

    def on_progress(progress: float, message: str) -> None:
        record.progress = progress
        record.status = "running"
        record.updated_at = datetime.now(timezone.utc)
        _save_image_generation_manifest(record, store)
        store.append_log(task_id, f"[生图][{progress:.2f}%] {message}")

    try:
        result = service.run(task_id=task_id, params=params, progress_cb=on_progress)
    except Stage1Error as exc:
        record.status = "failed"
        record.error = {"code": exc.code, "message": exc.message}
        record.updated_at = datetime.now(timezone.utc)
        _save_image_generation_manifest(record, store)
        raise

    record.status = "succeeded"
    record.progress = 100.0
    record.result = result
    record.updated_at = datetime.now(timezone.utc)
    _save_image_generation_manifest(record, store)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "run-full":
            return run_full(args)
        if args.command == "generate-images":
            return run_generate_images(args)
    except Stage1Error as exc:
        print(json.dumps({"code": exc.code, "message": exc.message}, ensure_ascii=False), file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"code": "internal_error", "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
