from __future__ import annotations

import time
from datetime import datetime, timezone

from app.models import ImageGenerationRunRecord
from tests.helpers import FakeSuccessPipeline


def _wait_until_done(client, task_id: str, timeout_sec: float = 3.0) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        payload = client.get(f"/api/v1/stage1/jobs/{task_id}").json()
        if payload["status"] == "succeeded":
            return
        time.sleep(0.05)
    raise AssertionError("stage1 任务未完成")


def test_image_generation_api_routes(build_client, monkeypatch):
    client = build_client(FakeSuccessPipeline(result_scene_count=3))
    with client:
        create_resp = client.post(
            "/api/v1/stage1/jobs",
            json={"source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        )
        task_id = create_resp.json()["task_id"]
        _wait_until_done(client, task_id)

        submitted = {}

        def fake_submit(tid: str, params: dict):
            submitted["tid"] = tid
            submitted["params"] = params
            return ImageGenerationRunRecord.new(task_id=tid, params=params)

        running = ImageGenerationRunRecord.new(task_id=task_id, params={"shot_range": "1-3"})
        running.status = "running"
        running.progress = 60.0
        running.updated_at = datetime.now(timezone.utc)

        succeeded = ImageGenerationRunRecord.new(task_id=task_id, params={"shot_range": "1-3"})
        succeeded.status = "succeeded"
        succeeded.progress = 100.0
        succeeded.updated_at = datetime.now(timezone.utc)
        succeeded.result = {
            "project_id": task_id,
            "shot_count": 3,
            "total_candidates": 24,
            "failed_shots": [],
            "artifacts": {
                "ref_oss_map": "/tmp/06_ref_oss_map.json",
                "image_candidates": "/tmp/08_image_candidates.json",
                "image_output_dir": "/tmp/images",
                "generation_manifest": "/tmp/07_image_generation_manifest.json",
            },
        }

        monkeypatch.setattr(client.app.state.image_manager, "submit", fake_submit)
        status_state = {"mode": "running"}

        def fake_get(tid: str):
            if status_state["mode"] == "running":
                return running
            return succeeded

        monkeypatch.setattr(client.app.state.image_manager, "get", fake_get)

        start_resp = client.post(
            f"/api/v1/stage1/jobs/{task_id}/image-generation",
            json={"shot_range": "1-3", "candidates_per_shot": 8, "aspect_ratio": "9:16", "concurrency": 2},
        )
        assert start_resp.status_code == 200
        assert start_resp.json()["status"] == "queued"
        assert submitted["tid"] == task_id
        assert submitted["params"]["shot_range"] == "1-3"

        status_resp = client.get(f"/api/v1/stage1/jobs/{task_id}/image-generation")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "running"

        result_not_ready = client.get(f"/api/v1/stage1/jobs/{task_id}/image-generation/result")
        assert result_not_ready.status_code == 409

        status_state["mode"] = "succeeded"
        result_resp = client.get(f"/api/v1/stage1/jobs/{task_id}/image-generation/result")
        assert result_resp.status_code == 200
        payload = result_resp.json()["result"]
        assert payload["shot_count"] == 3
        assert payload["total_candidates"] == 24

