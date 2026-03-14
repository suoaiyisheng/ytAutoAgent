from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    max_workers: int
    task_timeout_sec: int
    vlm_provider: str
    embedding_provider: str
    gemini_api_key: str
    qwen_api_key: str
    qwen_base_url: str
    qwen_embed_base_url: str
    qwen_vlm_model: str
    qwen_embed_model: str
    gemini_vlm_model: str
    gemini_embed_model: str
    pipeline_retry_max: int
    pipeline_batch_size: int
    image_generation_provider: str
    openrouter_api_key: str
    openrouter_base_url: str
    openrouter_image_model: str
    volcengine_access_key_id: str
    volcengine_access_key_secret: str
    volcengine_session_token: str
    volcengine_visual_host: str
    volcengine_region: str
    volcengine_service: str
    volcengine_jimeng_req_key: str
    volcengine_jimeng_version: str
    volcengine_poll_interval_sec: float
    volcengine_poll_timeout_sec: int
    aliyun_oss_region: str
    aliyun_oss_access_key_id: str
    aliyun_oss_access_key_secret: str
    aliyun_oss_bucket: str
    aliyun_oss_public_domain: str


def _load_dotenv_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def load_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[1]
    dotenv_path = Path(os.getenv("STAGE1_DOTENV_PATH", str(project_root / ".env"))).expanduser().resolve()
    _load_dotenv_file(dotenv_path)

    data_dir = Path(os.getenv("STAGE1_DATA_DIR", "runtime/jobs")).resolve()
    max_workers = int(os.getenv("STAGE1_MAX_WORKERS", "2"))
    task_timeout_sec = int(os.getenv("STAGE1_TASK_TIMEOUT_SEC", "3600"))
    pipeline_retry_max = int(os.getenv("PIPELINE_RETRY_MAX", "2"))
    pipeline_batch_size = int(os.getenv("PIPELINE_BATCH_SIZE", "4"))

    if max_workers < 1:
        max_workers = 1
    if task_timeout_sec < 60:
        task_timeout_sec = 60
    if pipeline_retry_max < 0:
        pipeline_retry_max = 0
    if pipeline_batch_size < 1:
        pipeline_batch_size = 1

    vlm_provider = os.getenv("VLM_PROVIDER", "qwen").strip().lower() or "qwen"
    if vlm_provider not in {"qwen", "gemini"}:
        vlm_provider = "qwen"

    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
    if embedding_provider not in {"qwen", "gemini"}:
        embedding_provider = vlm_provider

    image_generation_provider = os.getenv("IMAGE_GENERATION_PROVIDER", "openrouter").strip().lower()
    if image_generation_provider not in {"openrouter", "volcengine_jimeng30"}:
        image_generation_provider = "openrouter"

    return Settings(
        data_dir=data_dir,
        max_workers=max_workers,
        task_timeout_sec=task_timeout_sec,
        vlm_provider=vlm_provider,
        embedding_provider=embedding_provider,
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        qwen_api_key=os.getenv("QWEN_API_KEY", "").strip(),
        qwen_base_url=(
            os.getenv(
                "QWEN_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            ).strip()
            or "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        ),
        qwen_embed_base_url=(
            os.getenv(
                "QWEN_EMBED_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
            ).strip()
            or "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
        ),
        qwen_vlm_model=os.getenv("QWEN_VLM_MODEL", "qwen3-vl-flash").strip() or "qwen3-vl-flash",
        qwen_embed_model=os.getenv("QWEN_EMBED_MODEL", "text-embedding-v3").strip() or "text-embedding-v3",
        gemini_vlm_model=os.getenv("GEMINI_VLM_MODEL", "gemini-1.5-pro").strip() or "gemini-1.5-pro",
        gemini_embed_model=os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004").strip() or "text-embedding-004",
        pipeline_retry_max=pipeline_retry_max,
        pipeline_batch_size=pipeline_batch_size,
        image_generation_provider=image_generation_provider,
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
        or "https://openrouter.ai/api/v1",
        openrouter_image_model=os.getenv("OPENROUTER_IMAGE_MODEL", "google/gemini-2.5-flash-image").strip()
        or "google/gemini-2.5-flash-image",
        volcengine_access_key_id=os.getenv("VOLCENGINE_ACCESS_KEY_ID", "").strip(),
        volcengine_access_key_secret=os.getenv("VOLCENGINE_ACCESS_KEY_SECRET", "").strip(),
        volcengine_session_token=os.getenv("VOLCENGINE_SESSION_TOKEN", "").strip(),
        volcengine_visual_host=os.getenv("VOLCENGINE_VISUAL_HOST", "visual.volcengineapi.com").strip()
        or "visual.volcengineapi.com",
        volcengine_region=os.getenv("VOLCENGINE_REGION", "cn-north-1").strip() or "cn-north-1",
        volcengine_service=os.getenv("VOLCENGINE_SERVICE", "cv").strip() or "cv",
        volcengine_jimeng_req_key=os.getenv("VOLCENGINE_JIMENG_REQ_KEY", "jimeng_t2i_v30").strip()
        or "jimeng_t2i_v30",
        volcengine_jimeng_version=os.getenv("VOLCENGINE_JIMENG_VERSION", "2022-08-31").strip()
        or "2022-08-31",
        volcengine_poll_interval_sec=float(os.getenv("VOLCENGINE_POLL_INTERVAL_SEC", "2").strip() or "2"),
        volcengine_poll_timeout_sec=int(os.getenv("VOLCENGINE_POLL_TIMEOUT_SEC", "120").strip() or "120"),
        aliyun_oss_region=os.getenv("ALIYUN_OSS_REGION", "").strip(),
        aliyun_oss_access_key_id=os.getenv("ALIYUN_OSS_ACCESS_KEY_ID", "").strip(),
        aliyun_oss_access_key_secret=os.getenv("ALIYUN_OSS_ACCESS_KEY_SECRET", "").strip(),
        aliyun_oss_bucket=os.getenv("ALIYUN_OSS_BUCKET", "").strip(),
        aliyun_oss_public_domain=os.getenv("ALIYUN_OSS_PUBLIC_DOMAIN", "").strip(),
    )
