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
    ImageGenerationCreateRequest,
    ImageGenerationResult,
    ImageGenerationResultResponse,
    ImageGenerationStatusResponse,
    ImageGenerationSubmitResponse,
    JobCreateRequest,
    JobResult,
    JobResultResponse,
    JobStatusResponse,
    JobSubmitResponse,
)
from app.services.image_generation import ImageGenerationService
from app.services.image_generation_manager import ImageGenerationManager
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


def _to_image_status_response(record) -> ImageGenerationStatusResponse:
    error = ErrorDetail(**record.error) if record.error else None
    return ImageGenerationStatusResponse(
        task_id=record.task_id,
        status=record.status,
        progress=round(record.progress, 2),
        created_at=record.created_at,
        updated_at=record.updated_at,
        error=error,
    )


def build_manager(settings: Settings) -> JobManager:
    store = TaskStore(settings.data_dir)
    gemini_provider = GeminiVLMProvider(
        api_key=settings.gemini_api_key,
        openrouter_api_key=settings.openrouter_api_key,
        openrouter_base_url=settings.openrouter_base_url,
    )

    if settings.vlm_provider == "qwen":
        vlm_provider = QwenVLMProvider(
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
        )
        default_vlm_model = settings.qwen_vlm_model
    else:
        vlm_provider = gemini_provider
        default_vlm_model = settings.gemini_vlm_model

    stage2_vlm_provider = gemini_provider if settings.openrouter_api_key and settings.openrouter_base_url else vlm_provider
    default_stage2_vlm_model = settings.gemini_vlm_model if stage2_vlm_provider is gemini_provider else default_vlm_model

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
        stage2_vlm_provider=stage2_vlm_provider,
        embedding_provider=embedding_provider,
        default_vlm_model=default_vlm_model,
        default_stage2_vlm_model=default_stage2_vlm_model,
        default_embed_model=default_embed_model,
        default_retry_max=settings.pipeline_retry_max,
        default_batch_size=settings.pipeline_batch_size,
    )
    return JobManager(store=store, pipeline=pipeline, max_workers=settings.max_workers)


def create_app(settings: Settings | None = None, manager: JobManager | None = None) -> FastAPI:
    settings = settings or load_settings()
    manager = manager or build_manager(settings)
    image_manager = ImageGenerationManager(
        store=manager.store,
        service=ImageGenerationService(store=manager.store, settings=settings),
        max_workers=settings.max_workers,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        manager.start()
        image_manager.start()
        try:
            yield
        finally:
            image_manager.stop()
            manager.stop()

    app = FastAPI(title="ytAuto Full Pipeline API", version="0.2.0", lifespan=lifespan)
    app.state.manager = manager
    app.state.image_manager = image_manager

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

    @app.post(
        "/api/v1/stage1/jobs/{task_id}/image-generation",
        response_model=ImageGenerationSubmitResponse,
    )
    async def create_image_generation(
        task_id: str,
        payload: ImageGenerationCreateRequest | None = None,
    ):
        params = (payload or ImageGenerationCreateRequest()).model_dump()
        record = app.state.image_manager.submit(task_id, params)
        return ImageGenerationSubmitResponse(task_id=record.task_id, status="queued", created_at=record.created_at)

    @app.get(
        "/api/v1/stage1/jobs/{task_id}/image-generation",
        response_model=ImageGenerationStatusResponse,
    )
    async def get_image_generation_status(task_id: str):
        record = app.state.image_manager.get(task_id)
        if not record:
            raise _api_error("image_generation_not_found", "生图任务不存在", 404)
        return _to_image_status_response(record)

    @app.get(
        "/api/v1/stage1/jobs/{task_id}/image-generation/result",
        response_model=ImageGenerationResultResponse,
    )
    async def get_image_generation_result(task_id: str):
        record = app.state.image_manager.get(task_id)
        if not record:
            raise _api_error("image_generation_not_found", "生图任务不存在", 404)
        if record.status != "succeeded" or not record.result:
            code = "result_not_ready"
            message = "生图任务尚未完成"
            if record.status == "failed" and record.error:
                code = record.error.get("code", "result_not_ready")
                message = record.error.get("message", "生图任务失败")
            raise _api_error(code, message, 409)
        result = ImageGenerationResult(**record.result)
        return ImageGenerationResultResponse(task_id=task_id, status="succeeded", result=result)

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
                    or settings.embedding_provider != "gemini"
                ),
                path="configured" if settings.gemini_api_key else None,
            ),
            "openrouter_api_key": HealthDependency(
                available=(
                    bool(settings.openrouter_api_key)
                    or settings.vlm_provider != "gemini"
                    or bool(settings.gemini_api_key)
                ),
                path="configured" if settings.openrouter_api_key else None,
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
