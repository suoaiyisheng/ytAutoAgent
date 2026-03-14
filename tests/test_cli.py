from __future__ import annotations

from pathlib import Path

import pytest

from app.cli import run_full
from app.errors import Stage1Error


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
