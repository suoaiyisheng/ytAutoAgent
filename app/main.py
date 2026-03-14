from __future__ import annotations

import importlib.util
from contextlib import asynccontextmanager
from shutil import which

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.config import Settings, load_settings
from app.errors import Stage1Error
from app.models import (
    ErrorDetail,
    HealthDependency,
    HealthResponse,
    JobCreateRequest,
    JobResult,
    JobResultResponse,
    JobStatusResponse,
    JobSubmitResponse,
)
from app.services.job_manager import JobManager
from app.services.pipeline import Stage1Pipeline
from app.services.providers import (
    GeminiEmbeddingProvider,
    GeminiVLMProvider,
    QwenEmbeddingProvider,
    QwenVLMProvider,
)
from app.services.store import TaskStore


def _api_error(code: str, message: str, status_code: int) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


def _to_status_response(task) -> JobStatusResponse:
    error = ErrorDetail(**task.error) if task.error else None
    return JobStatusResponse(
        task_id=task.task_id,
        status=task.status,
        progress=round(task.progress, 2),
        created_at=task.created_at,
        updated_at=task.updated_at,
        error=error,
    )


def build_manager(settings: Settings) -> JobManager:
    store = TaskStore(settings.data_dir)

    if settings.vlm_provider == "qwen":
        vlm_provider = QwenVLMProvider(
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
        )
        default_vlm_model = settings.qwen_vlm_model
    else:
        vlm_provider = GeminiVLMProvider(api_key=settings.gemini_api_key)
        default_vlm_model = settings.gemini_vlm_model

    if settings.embedding_provider == "qwen":
        embedding_provider = QwenEmbeddingProvider(
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_embed_base_url,
        )
        default_embed_model = settings.qwen_embed_model
    else:
        embedding_provider = GeminiEmbeddingProvider(api_key=settings.gemini_api_key)
        default_embed_model = settings.gemini_embed_model

    pipeline = Stage1Pipeline(
        store=store,
        timeout_sec=settings.task_timeout_sec,
        vlm_provider=vlm_provider,
        embedding_provider=embedding_provider,
        default_vlm_model=default_vlm_model,
        default_embed_model=default_embed_model,
        default_retry_max=settings.pipeline_retry_max,
        default_batch_size=settings.pipeline_batch_size,
    )
    return JobManager(store=store, pipeline=pipeline, max_workers=settings.max_workers)


def create_app(settings: Settings | None = None, manager: JobManager | None = None) -> FastAPI:
    settings = settings or load_settings()
    manager = manager or build_manager(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        manager.start()
        try:
            yield
        finally:
            manager.stop()

    app = FastAPI(title="ytAuto Full Pipeline API", version="0.2.0", lifespan=lifespan)
    app.state.manager = manager

    @app.exception_handler(Stage1Error)
    async def stage1_error_handler(_, exc: Stage1Error):
        return JSONResponse(
            status_code=exc.http_status,
            content={"detail": {"code": exc.code, "message": exc.message}},
        )

    @app.post("/api/v1/stage1/jobs", response_model=JobSubmitResponse)
    async def create_job(payload: JobCreateRequest):
        try:
            app.state.manager.pipeline.validate_ready()
        except Stage1Error as exc:
            raise _api_error(exc.code, exc.message, exc.http_status) from exc

        params = payload.model_dump()
        params["source_url"] = str(payload.source_url) if payload.source_url else None

        task = app.state.manager.submit(params)
        return JobSubmitResponse(task_id=task.task_id, status="queued", created_at=task.created_at)

    @app.get("/api/v1/stage1/jobs/{task_id}", response_model=JobStatusResponse)
    async def get_job(task_id: str):
        task = app.state.manager.get(task_id)
        if not task:
            raise _api_error("task_not_found", "任务不存在", 404)
        return _to_status_response(task)

    @app.get("/api/v1/stage1/jobs/{task_id}/result", response_model=JobResultResponse)
    async def get_job_result(task_id: str):
        task = app.state.manager.get(task_id)
        if not task:
            raise _api_error("task_not_found", "任务不存在", 404)
        if task.status != "succeeded" or not task.result:
            code = "result_not_ready"
            message = "任务尚未完成"
            if task.status == "failed" and task.error:
                code = task.error.get("code", "result_not_ready")
                message = task.error.get("message", "任务失败")
            raise _api_error(code, message, 409)

        result = JobResult(**task.result)
        return JobResultResponse(task_id=task_id, status="succeeded", result=result)

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health():
        ffmpeg_path = which("ffmpeg")
        ffprobe_path = which("ffprobe")
        yt_dlp_path = which("yt-dlp")
        yt_dlp_python = importlib.util.find_spec("yt_dlp") is not None

        dependencies = {
            "ffmpeg": HealthDependency(available=ffmpeg_path is not None, path=ffmpeg_path),
            "ffprobe": HealthDependency(available=ffprobe_path is not None, path=ffprobe_path),
            "yt-dlp": HealthDependency(
                available=(yt_dlp_path is not None) or yt_dlp_python,
                path=yt_dlp_path,
            ),
            "vlm_provider": HealthDependency(available=True, path=settings.vlm_provider),
            "embedding_provider": HealthDependency(available=True, path=settings.embedding_provider),
            "gemini_api_key": HealthDependency(
                available=(
                    bool(settings.gemini_api_key)
                    or (settings.vlm_provider != "gemini" and settings.embedding_provider != "gemini")
                ),
                path="configured" if settings.gemini_api_key else None,
            ),
            "qwen_api_key": HealthDependency(
                available=(
                    bool(settings.qwen_api_key)
                    or (settings.vlm_provider != "qwen" and settings.embedding_provider != "qwen")
                ),
                path="configured" if settings.qwen_api_key else None,
            ),
        }
        overall = "ok" if all(dep.available for dep in dependencies.values()) else "degraded"
        return HealthResponse(status=overall, dependencies=dependencies)

    return app


app = create_app()
