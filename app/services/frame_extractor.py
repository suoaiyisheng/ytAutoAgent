from __future__ import annotations

import subprocess
from pathlib import Path

from app.errors import Stage1Error


class FrameExtractor:
    def extract_first_frame(
        self,
        video_path: Path,
        output_path: Path,
        timestamp_sec: float,
        quality: int,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp_sec:.3f}",
            "-i",
            str(video_path),
            "-vframes",
            "1",
            "-q:v",
            str(quality),
            str(output_path),
        ]
        self._run_ffmpeg(cmd=cmd, timeout=120, err_code="ffmpeg_failed", err_msg="ffmpeg 抽帧失败")

        if not output_path.exists():
            raise Stage1Error("frame_missing", "抽帧完成但输出文件不存在", 500)
        return output_path.resolve()

    def extract_clip(self, video_path: Path, output_path: Path, start_sec: float, end_sec: float) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        duration = max(0.05, end_sec - start_sec)
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{duration:.3f}",
            "-c",
            "copy",
            str(output_path),
        ]
        self._run_ffmpeg(cmd=cmd, timeout=180, err_code="ffmpeg_clip_failed", err_msg="ffmpeg 分镜导出失败")

        if not output_path.exists():
            raise Stage1Error("clip_missing", "导出分镜完成但输出文件不存在", 500)
        return output_path.resolve()

    def _run_ffmpeg(self, cmd: list[str], timeout: int, err_code: str, err_msg: str) -> None:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
        except FileNotFoundError as exc:
            raise Stage1Error("dependency_missing", "未找到 ffmpeg 可执行文件", 500) from exc
        except subprocess.TimeoutExpired as exc:
            raise Stage1Error("ffmpeg_timeout", "ffmpeg 执行超时", 504) from exc

        if proc.returncode != 0:
            message = proc.stderr.strip() or err_msg
            raise Stage1Error(err_code, message, 500)
