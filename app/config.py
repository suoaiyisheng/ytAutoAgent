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
    )
