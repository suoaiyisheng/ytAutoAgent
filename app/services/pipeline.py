from __future__ import annotations

"""
模块作用：
- 作为主流水线编排入口，按顺序驱动 Stage 1 到 Stage 4，并维护运行期依赖注入配置。

当前文件工具函数：
- `validate_ready`：在任务开始前校验 VLM/Embedding provider 已具备运行条件。
- `run`：统一编排四个 Stage，串联进度回调、超时检查和最终结果落盘。
"""

import time
from pathlib import Path
from typing import Any, Callable

from app.models import TaskRecord
from app.services.downloader import VideoDownloader
from app.services.frame_extractor import FrameExtractor
from app.services.media_probe import MediaProbe
from app.services.pipeline_common import PipelineCommonMixin
from app.services.pipeline_stage1 import PipelineStage1Mixin
from app.services.pipeline_stage2 import PipelineStage2Mixin
from app.services.pipeline_stage3 import PipelineStage3Mixin
from app.services.pipeline_stage4 import PipelineStage4Mixin
from app.services.pipeline_stage5 import PipelineStage5Mixin
from app.services.providers import EmbeddingProvider, VLMProvider
from app.services.scene_detection import PySceneDetectProvider, SceneDetector
from app.services.store import TaskStore

ProgressCallback = Callable[[float, str], None]


class Stage1Pipeline(
    PipelineStage1Mixin,
    PipelineStage2Mixin,
    PipelineStage3Mixin,
    PipelineStage4Mixin,
    PipelineStage5Mixin,
    PipelineCommonMixin,
):
    def __init__(
        self,
        store: TaskStore,
        timeout_sec: int,
        vlm_provider: VLMProvider,
        embedding_provider: EmbeddingProvider,
        default_vlm_model: str = "gemini-1.5-pro",
        default_embed_model: str = "text-embedding-004",
        default_retry_max: int = 2,
        default_batch_size: int = 4,
        architect_prompt_path: Path | None = None,
        downloader: VideoDownloader | None = None,
        scene_detector: SceneDetector | None = None,
        frame_extractor: FrameExtractor | None = None,
        media_probe: MediaProbe | None = None,
    ) -> None:
        self.store = store
        self.timeout_sec = timeout_sec
        self.vlm_provider = vlm_provider
        self.embedding_provider = embedding_provider
        self.default_vlm_model = default_vlm_model
        self.default_embed_model = default_embed_model
        self.default_retry_max = max(0, default_retry_max)
        self.default_batch_size = max(1, default_batch_size)
        self.architect_prompt_path = architect_prompt_path or (Path(__file__).resolve().parents[2] / "dosc" / "V4.2 架构师提示词.md")

        self.downloader = downloader or VideoDownloader()
        self.scene_detector = scene_detector or PySceneDetectProvider()
        self.frame_extractor = frame_extractor or FrameExtractor()
        self.media_probe = media_probe or MediaProbe()

    def validate_ready(self) -> None:
        for provider in (self.vlm_provider, self.embedding_provider):
            ensure = getattr(provider, "ensure_ready", None)
            if callable(ensure):
                ensure()

    def run(self, task: TaskRecord, progress_cb: ProgressCallback) -> dict[str, Any]:
        self.validate_ready()

        started_at = time.monotonic()
        params = task.params

        progress_cb(3.0, "开始结构化提取")
        physical_manifest = self._run_stage1(task, params, progress_cb, started_at)
        self._assert_not_timeout(started_at)

        progress_cb(30.0, "开始原子级感知")
        raw_scene_descriptions = self._run_stage2(task, params, physical_manifest, progress_cb, started_at)
        self._assert_not_timeout(started_at)

        progress_cb(55.0, "开始身份对齐")
        character_bank = self._run_stage3(
            task,
            params,
            physical_manifest,
            raw_scene_descriptions,
            progress_cb,
            started_at,
        )
        self._assert_not_timeout(started_at)

        progress_cb(72.0, "开始归一化")
        normalized_scene_descriptions = self._run_stage4(
            task,
            raw_scene_descriptions,
            character_bank,
            progress_cb,
            started_at,
        )
        self._assert_not_timeout(started_at)

        progress_cb(84.0, "开始指令合成")
        final_table = self._run_stage5(
            task,
            params,
            character_bank,
            normalized_scene_descriptions,
            progress_cb,
            started_at,
        )
        self._assert_not_timeout(started_at)

        result = self._persist_index_and_result(
            task=task,
            physical_manifest=physical_manifest,
            raw_scene_descriptions=raw_scene_descriptions,
            character_bank=character_bank,
            normalized_scene_descriptions=normalized_scene_descriptions,
            final_table=final_table,
        )
        progress_cb(100.0, "任务完成")
        return result
