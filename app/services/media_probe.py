from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.errors import Stage1Error


@dataclass
class VideoMetadata:
    fps: float
    resolution: str


class MediaProbe:
    def probe(self, video_path: Path) -> VideoMetadata:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-of",
            "json",
            str(video_path),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=30)
        except FileNotFoundError as exc:
            raise Stage1Error("dependency_missing", "未找到 ffprobe 可执行文件", 500) from exc
        except subprocess.TimeoutExpired as exc:
            raise Stage1Error("ffprobe_timeout", "ffprobe 执行超时", 504) from exc

        if proc.returncode != 0:
            msg = proc.stderr.strip() or "ffprobe 执行失败"
            raise Stage1Error("ffprobe_failed", msg, 500)

        try:
            payload = json.loads(proc.stdout or "{}")
            stream = (payload.get("streams") or [])[0]
            width = int(stream.get("width", 0))
            height = int(stream.get("height", 0))
            fps = self._parse_fps(str(stream.get("r_frame_rate", "0/1")))
            if width <= 0 or height <= 0:
                raise ValueError("invalid resolution")
            return VideoMetadata(fps=fps, resolution=f"{width}x{height}")
        except Exception as exc:  # noqa: BLE001
            raise Stage1Error("ffprobe_parse_failed", f"ffprobe 输出解析失败: {exc}", 500) from exc

    def _parse_fps(self, value: str) -> float:
        if "/" in value:
            num, den = value.split("/", 1)
            n = float(num)
            d = float(den)
            if d == 0:
                return 0.0
            return round(n / d, 3)
        return round(float(value), 3)
