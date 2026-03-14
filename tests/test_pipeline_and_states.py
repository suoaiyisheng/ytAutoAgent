from __future__ import annotations

import json
import time
from pathlib import Path

from app.models import TaskRecord
from app.services.media_probe import VideoMetadata
from app.services.pipeline import Stage1Pipeline
from app.services.scene_detection import SceneBoundary
from app.services.store import TaskStore
from tests.helpers import FakeEmbeddingProvider, FakeFailedPipeline, FakeSuccessPipeline, FakeVLMProvider


class LocalFileDownloader:
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path

    def obtain_video(self, source_url, local_video_path, output_dir, download_format):  # noqa: ANN001, ARG002
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / self.file_path.name
        target.write_bytes(self.file_path.read_bytes())
        return target.resolve()


class FixedSceneDetector:
    def detect(self, video_path: Path, threshold: float, min_scene_len: float):  # noqa: ARG002
        return [
            SceneBoundary(start=0.0, end=1.2),
            SceneBoundary(start=1.2, end=2.6),
            SceneBoundary(start=2.6, end=4.0),
        ]


class TouchFrameExtractor:
    def extract_first_frame(self, video_path: Path, output_path: Path, timestamp_sec: float, quality: int):  # noqa: ARG002
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"frame")
        return output_path.resolve()

    def extract_clip(self, video_path: Path, output_path: Path, start_sec: float, end_sec: float):  # noqa: ARG002
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"clip")
        return output_path.resolve()


class FakeMediaProbe:
    def probe(self, video_path: Path):  # noqa: ARG002
        return VideoMetadata(fps=30.0, resolution="1920x1080")


def _wait_until_done(client, task_id: str, timeout: float = 4.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        resp = client.get(f"/api/v1/stage1/jobs/{task_id}")
        last = resp.json()
        if last["status"] in {"succeeded", "failed"}:
            return last
        time.sleep(0.05)
    raise AssertionError(f"任务未结束，最后状态: {last}")


def test_full_pipeline_contracts_and_alignment(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake-video")

    store = TaskStore(tmp_path / "jobs")
    pipeline = Stage1Pipeline(
        store=store,
        timeout_sec=120,
        vlm_provider=FakeVLMProvider(),
        embedding_provider=FakeEmbeddingProvider(),
        downloader=LocalFileDownloader(source),
        scene_detector=FixedSceneDetector(),
        frame_extractor=TouchFrameExtractor(),
        media_probe=FakeMediaProbe(),
    )

    task = TaskRecord.new(
        task_id="task_local",
        params={
            "source_url": "https://example.com/test",
            "local_video_path": None,
            "threshold": 27.0,
            "min_scene_len": 1.0,
            "frame_quality": 2,
            "download_format": "bestvideo+bestaudio/best",
            "vlm_model": None,
            "embed_model": None,
            "batch_size": 2,
            "retry_max": 0,
        },
        working_dir=str(store.task_dir("task_local")),
    )

    result = pipeline.run(task, lambda *_: None)
    assert result["stats"]["scene_count"] == 3
    assert result["stats"]["character_count"] == 2
    assert result["stats"]["prompt_count"] == 3

    for contract_path in result["contracts"].values():
        assert Path(contract_path).exists()

    aligned = json.loads(Path(result["contracts"]["aligned_storyboard"]).read_text(encoding="utf-8"))
    assert aligned["storyboard"][0]["character_mappings"][0]["ref_id"] == "Ref_1"
    assert aligned["storyboard"][1]["character_mappings"][0]["ref_id"] == "Ref_1"
    assert aligned["storyboard"][0]["character_mappings"][0]["state_id"] == "human_form"
    assert aligned["storyboard"][1]["character_mappings"][0]["state_id"] == "human_form"
    assert aligned["storyboard"][2]["character_mappings"][0]["state_id"] is None

    final_table = json.loads(Path(result["contracts"]["final_production_table"]).read_text(encoding="utf-8"))
    first_bindings = final_table["prompts"][0]["reference_bindings"]
    assert first_bindings[0]["reference_index"] == 1
    assert first_bindings[0]["ref_id"] == "Ref_1"
    assert first_bindings[0]["state_id"] == "human_form"
    assert "state_text" in first_bindings[0]


def test_job_state_flow_success_and_failure(build_client):
    success_client = build_client(FakeSuccessPipeline(result_scene_count=1, delay_sec=0.2))
    with success_client:
        submit = success_client.post(
            "/api/v1/stage1/jobs",
            json={"source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        )
        task_id = submit.json()["task_id"]

        first_status = success_client.get(f"/api/v1/stage1/jobs/{task_id}").json()["status"]
        assert first_status in {"queued", "running", "succeeded"}

        done = _wait_until_done(success_client, task_id)
        assert done["status"] == "succeeded"

    failed_client = build_client(FakeFailedPipeline())
    with failed_client:
        submit = failed_client.post(
            "/api/v1/stage1/jobs",
            json={"source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        )
        task_id = submit.json()["task_id"]
        done = _wait_until_done(failed_client, task_id)
        assert done["status"] == "failed"

        result_resp = failed_client.get(f"/api/v1/stage1/jobs/{task_id}/result")
        assert result_resp.status_code == 409
        detail = result_resp.json()["detail"]
        assert detail["code"] == "download_failed"
        assert "模拟下载失败" in detail["message"]


def test_not_found_error_shape(build_client):
    client = build_client(FakeSuccessPipeline())
    with client:
        resp = client.get("/api/v1/stage1/jobs/not-exist")
        assert resp.status_code == 404
        detail = resp.json()["detail"]
        assert detail["code"] == "task_not_found"
        assert detail["message"] == "任务不存在"
