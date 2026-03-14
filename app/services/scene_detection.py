from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from app.errors import Stage1Error


@dataclass
class SceneBoundary:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class SceneDetector(ABC):
    @abstractmethod
    def detect(self, video_path: Path, threshold: float, min_scene_len: float) -> list[SceneBoundary]:
        raise NotImplementedError


class PySceneDetectProvider(SceneDetector):
    def detect(self, video_path: Path, threshold: float, min_scene_len: float) -> list[SceneBoundary]:
        try:
            from scenedetect import ContentDetector, detect
        except ModuleNotFoundError as exc:
            raise Stage1Error(
                code="dependency_missing",
                message="缺少 PySceneDetect，请先安装 scenedetect[opencv]",
                http_status=500,
            ) from exc

        detected = detect(str(video_path), ContentDetector(threshold=threshold))
        scenes: list[SceneBoundary] = []
        for start, end in detected:
            start_sec = float(start.get_seconds())
            end_sec = float(end.get_seconds())
            if end_sec - start_sec >= min_scene_len:
                scenes.append(SceneBoundary(start=start_sec, end=end_sec))

        if not scenes and detected:
            first_start, first_end = detected[0]
            scenes.append(
                SceneBoundary(
                    start=float(first_start.get_seconds()),
                    end=float(first_end.get_seconds()),
                )
            )
        return scenes
