from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import AnyHttpUrl, BaseModel, Field, model_validator

JobStatus = Literal["queued", "running", "succeeded", "failed"]


class JobCreateRequest(BaseModel):
    source_url: AnyHttpUrl | None = None
    local_video_path: str | None = None
    threshold: float = Field(default=27.0, gt=0.0, le=100.0)
    min_scene_len: float = Field(default=1.0, gt=0.0, le=60.0)
    frame_quality: int = Field(default=2, ge=2, le=31)
    download_format: str = Field(default="bestvideo+bestaudio/best", min_length=1, max_length=200)
    vlm_model: str | None = Field(default=None, min_length=1, max_length=120)
    embed_model: str | None = Field(default=None, min_length=1, max_length=120)
    batch_size: int | None = Field(default=None, ge=1, le=32)
    retry_max: int | None = Field(default=None, ge=0, le=8)

    @model_validator(mode="after")
    def validate_source(self) -> "JobCreateRequest":
        has_url = self.source_url is not None
        has_local = bool(self.local_video_path)
        if has_url == has_local:
            raise ValueError("source_url 与 local_video_path 必须且只能提供一个")

        if self.local_video_path:
            path = Path(self.local_video_path).expanduser().resolve()
            self.local_video_path = str(path)
        return self


class ErrorDetail(BaseModel):
    code: str
    message: str


class JobSubmitResponse(BaseModel):
    task_id: str
    status: Literal["queued"]
    created_at: datetime


class JobStatusResponse(BaseModel):
    task_id: str
    status: JobStatus
    progress: float
    created_at: datetime
    updated_at: datetime
    error: ErrorDetail | None = None


class JobStats(BaseModel):
    scene_count: int
    character_count: int
    prompt_count: int


class JobResult(BaseModel):
    project_id: str
    contracts: dict[str, str]
    artifacts: dict[str, str]
    stats: JobStats


class JobResultResponse(BaseModel):
    task_id: str
    status: Literal["succeeded"]
    result: JobResult


class HealthDependency(BaseModel):
    available: bool
    path: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    dependencies: dict[str, HealthDependency]


@dataclass
class TaskRecord:
    task_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    progress: float
    params: dict[str, Any]
    error: dict[str, str] | None = None
    result: dict[str, Any] | None = None
    working_dir: str = ""
    logs: list[str] = field(default_factory=list)

    @classmethod
    def new(cls, task_id: str, params: dict[str, Any], working_dir: str) -> "TaskRecord":
        now = datetime.now(timezone.utc)
        return cls(
            task_id=task_id,
            status="queued",
            created_at=now,
            updated_at=now,
            progress=0.0,
            params=params,
            working_dir=working_dir,
        )

    def to_manifest(self) -> dict[str, Any]:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_manifest(cls, payload: dict[str, Any]) -> "TaskRecord":
        return cls(
            task_id=payload["task_id"],
            status=payload["status"],
            created_at=datetime.fromisoformat(payload["created_at"]),
            updated_at=datetime.fromisoformat(payload["updated_at"]),
            progress=float(payload.get("progress", 0.0)),
            params=payload.get("params", {}),
            error=payload.get("error"),
            result=payload.get("result"),
            working_dir=payload.get("working_dir", ""),
            logs=payload.get("logs", []),
        )
