from __future__ import annotations

from pathlib import Path

import pytest

from app.cli import run_full, run_generate_images
from app.config import Settings
from app.errors import Stage1Error
from app.services.store import TaskStore


class DummyStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def task_dir(self, task_id: str) -> Path:
        path = self.root / task_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_manifest(self, task) -> None:  # noqa: ANN001
        return None

    def append_log(self, task_id: str, message: str) -> None:  # noqa: ARG002
        return None


class DummyPipeline:
    def __init__(self, root: Path) -> None:
        self.store = DummyStore(root)

    def validate_ready(self) -> None:
        return None

    def run(self, task, progress_cb):  # noqa: ANN001
        progress_cb(100.0, "done")
        return {
            "project_id": task.task_id,
            "contracts": {},
            "artifacts": {},
            "stats": {"scene_count": 1, "character_count": 1, "prompt_count": 1},
        }


class DummyImageService:
    def __init__(self) -> None:
        self.called = {}

    def validate_ready(self) -> None:
        return None

    def run(self, task_id: str, params: dict, progress_cb):  # noqa: ANN001
        self.called["task_id"] = task_id
        self.called["params"] = params
        progress_cb(50.0, "half")
        return {
            "project_id": task_id,
            "shot_count": 3,
            "total_candidates": 24,
            "failed_shots": [],
            "artifacts": {},
        }


def test_run_full_with_local_path(monkeypatch, tmp_path, capsys):
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"x")

    monkeypatch.setattr("app.cli._build_pipeline", lambda: DummyPipeline(tmp_path))

    class Args:
        source_url = None
        local_video_path = str(video)
        threshold = 27.0
        min_scene_len = 1.0
        frame_quality = 2
        download_format = "bestvideo+bestaudio/best"
        vlm_model = None
        embed_model = None
        batch_size = None
        retry_max = None

    code = run_full(Args)
    assert code == 0
    out = capsys.readouterr().out
    assert "project_id" in out


def test_run_full_requires_exactly_one_source(monkeypatch):
    monkeypatch.setattr("app.cli._build_pipeline", lambda: DummyPipeline(Path("/tmp")))

    class Args:
        source_url = None
        local_video_path = None
        threshold = 27.0
        min_scene_len = 1.0
        frame_quality = 2
        download_format = "bestvideo+bestaudio/best"
        vlm_model = None
        embed_model = None
        batch_size = None
        retry_max = None

    with pytest.raises(Stage1Error) as exc:
        run_full(Args)
    assert exc.value.code == "invalid_source"


def test_run_generate_images(monkeypatch, tmp_path, capsys):
    store = TaskStore(tmp_path / "jobs")
    task_id = "job_a"
    store.task_dir(task_id)

    settings = Settings(
        data_dir=tmp_path / "jobs",
        max_workers=1,
        task_timeout_sec=300,
        vlm_provider="gemini",
        embedding_provider="gemini",
        gemini_api_key="x",
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
        openrouter_api_key="k",
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
        aliyun_oss_region="oss-cn-hangzhou",
        aliyun_oss_access_key_id="ak",
        aliyun_oss_access_key_secret="sk",
        aliyun_oss_bucket="bucket",
        aliyun_oss_public_domain="",
    )
    monkeypatch.setattr("app.cli.load_settings", lambda: settings)
    dummy = DummyImageService()
    monkeypatch.setattr("app.cli.ImageGenerationService", lambda **kwargs: dummy)

    class Args:
        job_id = task_id
        shot_range = "1-3"
        candidates_per_shot = 8
        aspect_ratio = "9:16"
        concurrency = 2

    code = run_generate_images(Args)
    assert code == 0
    out = capsys.readouterr().out
    assert '"shot_count": 3' in out
    assert dummy.called["params"]["shot_range"] == "1-3"
