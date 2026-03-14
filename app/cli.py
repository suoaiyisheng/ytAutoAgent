from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from app.config import load_settings
from app.errors import Stage1Error
from app.models import TaskRecord
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
        vlm_provider = GeminiVLMProvider(settings.gemini_api_key)
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
    }

    task = TaskRecord.new(task_id=task_id, params=params, working_dir=working_dir)
    pipeline.store.save_manifest(task)

    def on_progress(progress: float, message: str) -> None:
        pipeline.store.append_log(task_id, f"[{progress:.2f}%] {message}")

    result = pipeline.run(task, on_progress)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "run-full":
            return run_full(args)
    except Stage1Error as exc:
        print(json.dumps({"code": exc.code, "message": exc.message}, ensure_ascii=False), file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"code": "internal_error", "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
