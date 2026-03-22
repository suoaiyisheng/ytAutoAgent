from __future__ import annotations

"""
模块作用：
- 负责 Stage 1 结构化提取，包括视频获取、媒体探测、分镜检测以及首帧与 clip 导出。

当前文件工具函数：
- `_extract_scene_assets`：按分镜批量导出首帧和 clip，并同步更新进度。
"""

from pathlib import Path
from typing import Any, Callable

from app.errors import Stage1Error
from app.models import TaskRecord
from app.services.scene_detection import SceneBoundary


class PipelineStage1Mixin:
    def _run_stage1(
        self,
        task: TaskRecord,
        params: dict[str, Any],
        progress_cb: Callable[[float, str], None],
        started_at: float,
    ) -> dict[str, Any]:
        task_dir = self.store.task_dir(task.task_id)
        source_dir = task_dir / "source"
        frames_dir = task_dir / "frames"
        clips_dir = task_dir / "clips"

        video_path = self.downloader.obtain_video(
            source_url=params.get("source_url"),
            local_video_path=params.get("local_video_path"),
            output_dir=source_dir,
            download_format=params["download_format"],
        )
        self._assert_not_timeout(started_at)

        progress_cb(10.0, "视频获取完成，提取元数据")
        metadata = self.media_probe.probe(video_path)
        self._assert_not_timeout(started_at)

        progress_cb(15.0, "开始检测分镜")
        scenes = self.scene_detector.detect(
            video_path=video_path,
            threshold=float(params["threshold"]),
            min_scene_len=float(params["min_scene_len"]),
        )
        if not scenes:
            raise Stage1Error("no_scene_detected", "未检测到可用分镜", 422)
        self._assert_not_timeout(started_at)

        scene_items = self._extract_scene_assets(
            video_path=video_path,
            frames_dir=frames_dir,
            clips_dir=clips_dir,
            scenes=scenes,
            quality=int(params["frame_quality"]),
            progress_cb=progress_cb,
            started_at=started_at,
        )

        contract = {
            "project_id": task.task_id,
            "video_metadata": {
                "source_url": params.get("source_url") or params.get("local_video_path") or "",
                "fps": metadata.fps,
                "resolution": metadata.resolution,
            },
            "scenes": scene_items,
        }
        self.store.write_contract(task.task_id, "physical_manifest", contract)
        return contract

    def _extract_scene_assets(
        self,
        video_path: Path,
        frames_dir: Path,
        clips_dir: Path,
        scenes: list[SceneBoundary],
        quality: int,
        progress_cb: Callable[[float, str], None],
        started_at: float,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        total = len(scenes)
        for idx, scene in enumerate(scenes, start=1):
            frame_path = self.frame_extractor.extract_first_frame(
                video_path=video_path,
                output_path=frames_dir / f"frame_{idx:03d}.jpg",
                timestamp_sec=scene.start,
                quality=quality,
            )
            clip_path = self.frame_extractor.extract_clip(
                video_path=video_path,
                output_path=clips_dir / f"clip_{idx:03d}.mp4",
                start_sec=scene.start,
                end_sec=scene.end,
            )
            items.append(
                {
                    "scene_id": idx,
                    "start_time": self._sec_to_hms(scene.start),
                    "end_time": self._sec_to_hms(scene.end),
                    "keyframe_path": str(frame_path),
                    "clip_path": str(clip_path),
                }
            )
            pct = 15.0 + (idx / total) * 13.0
            progress_cb(pct, f"结构化提取进度 {idx}/{total}")
            self._assert_not_timeout(started_at)
        return items
