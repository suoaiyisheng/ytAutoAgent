from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.services.job_manager import JobManager
from app.services.store import TaskStore


@pytest.fixture
def build_client(tmp_path):
    def _factory(pipeline, max_workers: int = 1) -> TestClient:
        store = TaskStore(tmp_path / "jobs")
        manager = JobManager(store=store, pipeline=pipeline, max_workers=max_workers)
        settings = Settings(
            data_dir=tmp_path / "jobs",
            max_workers=max_workers,
            task_timeout_sec=300,
            vlm_provider="gemini",
            embedding_provider="gemini",
            gemini_api_key="test-key",
            qwen_api_key="",
            qwen_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            qwen_embed_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
            qwen_vlm_model="qwen3-vl-flash",
            qwen_embed_model="text-embedding-v3",
            gemini_vlm_model="gemini-1.5-pro",
            gemini_embed_model="text-embedding-004",
            pipeline_retry_max=2,
            pipeline_batch_size=4,
        )
        app = create_app(settings=settings, manager=manager)
        return TestClient(app)

    return _factory
