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
            image_generation_provider="openrouter",
            openrouter_api_key="",
            openrouter_base_url="https://openrouter.ai/api/v1",
            openrouter_image_model="google/gemini-2.5-flash-image",
            volcengine_access_key_id="",
            volcengine_access_key_secret="",
            volcengine_session_token="",
            volcengine_visual_host="visual.volcengineapi.com",
            volcengine_region="cn-north-1",
            volcengine_service="cv",
            volcengine_jimeng_req_key="jimeng_t2i_v30",
            volcengine_jimeng_version="2022-08-31",
            volcengine_poll_interval_sec=2.0,
            volcengine_poll_timeout_sec=120,
            aliyun_oss_region="",
            aliyun_oss_access_key_id="",
            aliyun_oss_access_key_secret="",
            aliyun_oss_bucket="",
            aliyun_oss_public_domain="",
        )
        app = create_app(settings=settings, manager=manager)
        return TestClient(app)

    return _factory
