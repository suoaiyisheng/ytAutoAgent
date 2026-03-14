from __future__ import annotations

import time

from tests.helpers import FakeConfigErrorPipeline, FakeSuccessPipeline


def _wait_for_status(client, task_id: str, expected: str, timeout_sec: float = 3.0) -> dict:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        resp = client.get(f"/api/v1/stage1/jobs/{task_id}")
        payload = resp.json()
        if payload["status"] == expected:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"任务未在 {timeout_sec}s 内达到 {expected}")


def test_create_job_defaults_with_source_url(build_client):
    client = build_client(FakeSuccessPipeline(result_scene_count=2))
    with client:
        resp = client.post(
            "/api/v1/stage1/jobs",
            json={"source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "queued"

        task_id = body["task_id"]
        _wait_for_status(client, task_id, "succeeded")

        result_resp = client.get(f"/api/v1/stage1/jobs/{task_id}/result")
        assert result_resp.status_code == 200
        result = result_resp.json()["result"]
        assert result["project_id"] == task_id
        assert "physical_manifest" in result["contracts"]
        assert "final_prompts.md" in result["artifacts"]
        assert result["stats"]["scene_count"] == 2


def test_create_job_with_local_video_path(build_client, tmp_path):
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"fake")

    client = build_client(FakeSuccessPipeline(result_scene_count=1))
    with client:
        resp = client.post(
            "/api/v1/stage1/jobs",
            json={"local_video_path": str(video)},
        )
        assert resp.status_code == 200


def test_invalid_source_choice_returns_422(build_client, tmp_path):
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"fake")

    client = build_client(FakeSuccessPipeline())
    with client:
        resp_both = client.post(
            "/api/v1/stage1/jobs",
            json={
                "source_url": "https://example.com/video",
                "local_video_path": str(video),
            },
        )
        assert resp_both.status_code == 422

        resp_none = client.post(
            "/api/v1/stage1/jobs",
            json={"threshold": 20},
        )
        assert resp_none.status_code == 422


def test_config_error_returns_400(build_client):
    client = build_client(FakeConfigErrorPipeline())
    with client:
        resp = client.post(
            "/api/v1/stage1/jobs",
            json={"source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert detail["code"] == "config_error"
