from __future__ import annotations

import time
from pathlib import Path

from app.errors import Stage1Error
from app.services.providers import EmbeddingProvider, VLMProvider


class FakeSuccessPipeline:
    def __init__(self, result_scene_count: int = 1, delay_sec: float = 0.0) -> None:
        self.result_scene_count = result_scene_count
        self.delay_sec = delay_sec

    def validate_ready(self) -> None:
        return None

    def run(self, task, progress_cb):  # noqa: ANN001
        progress_cb(20.0, "fake-start")
        if self.delay_sec > 0:
            time.sleep(self.delay_sec)

        task_dir = Path(task.working_dir)
        task_dir.mkdir(parents=True, exist_ok=True)
        contracts = {
            "physical_manifest": str((task_dir / "01_physical_manifest.json").resolve()),
            "raw_scene_descriptions": str((task_dir / "02_raw_scene_descriptions.json").resolve()),
            "character_bank": str((task_dir / "03_character_bank.json").resolve()),
            "normalized_scene_descriptions": str((task_dir / "04_normalized_scene_descriptions.json").resolve()),
            "final_production_table": str((task_dir / "05_final_production_table.json").resolve()),
        }
        artifacts = {
            "final_prompts.md": str((task_dir / "final_prompts.md").resolve()),
            "index.json": str((task_dir / "index.json").resolve()),
        }
        stats = {
            "scene_count": self.result_scene_count,
            "character_count": 1,
            "prompt_count": self.result_scene_count,
        }
        return {
            "project_id": task.task_id,
            "contracts": contracts,
            "artifacts": artifacts,
            "stats": stats,
        }


class FakeFailedPipeline:
    def validate_ready(self) -> None:
        return None

    def run(self, task, progress_cb):  # noqa: ANN001
        progress_cb(10.0, "fake-failed")
        raise Stage1Error("download_failed", "模拟下载失败", 502)


class FakeConfigErrorPipeline:
    def validate_ready(self) -> None:
        raise Stage1Error("config_error", "缺少 GEMINI_API_KEY 配置", 400)

    def run(self, task, progress_cb):  # noqa: ANN001
        raise AssertionError("should not run")


class FakeVLMProvider(VLMProvider):
    def describe_scenes(self, scene_inputs, model: str, retry_max: int):  # noqa: ARG002
        out = []
        for item in scene_inputs:
            sid = int(item["scene_id"])
            if sid in {1, 2}:
                appearance = "穿红领带的黄色方块"
            else:
                appearance = "戴墨镜的蓝色机器人"

            out.append(
                {
                    "scene_id": sid,
                    "subjects": [
                        {
                            "subject_id": "subject_1",
                            "appearance": appearance,
                        }
                    ],
                    "desc": "subject_1 在街道上正在向前走。",
                }
            )
        return out

    def review_character_candidates(self, candidates, model: str, retry_max: int):  # noqa: ARG002
        reviewed = []
        for item in candidates:
            record = dict(item)
            if record["ref_id"] == "Ref_1":
                record["name"] = "海绵宝宝"
            reviewed.append(record)
        return reviewed

    def generate_production_table(
        self,
        character_bank,
        stage5_input,
        architect_prompt: str,
        model: str,
        retry_max: int,
        debug_context=None,
    ):  # noqa: ARG002
        if isinstance(debug_context, dict):
            debug_context["architect_prompt"] = architect_prompt
            debug_context["stage5_protocol_prompt"] = "fake_stage5_protocol"
            debug_context["character_bank"] = character_bank
            debug_context["stage5_input"] = stage5_input
            debug_context["provider_request"] = {"provider": "fake", "model": model}
            debug_context["provider_raw_output"] = {"ok": True}
        prompts = []
        for shot in stage5_input.get("shots", []):
            sid = int(shot["shot_id"])
            prompts.append(
                {
                    "shot_id": sid,
                    "reference_bindings": shot.get("reference_bindings", []),
                    "image_prompt": f"{sid}，真实摄影风格，中景，平视。Ref_1站在街道上。",
                    "video_prompt": "跟随镜头，主体向前走，表情由专注变为坚定。",
                }
            )
        return {
            "project_id": stage5_input.get("project_id", ""),
            "prompts": prompts,
        }


class FakeEmbeddingProvider(EmbeddingProvider):
    def embed_texts(self, texts, model: str, retry_max: int):  # noqa: ARG002
        vectors = []
        for text in texts:
            if "红领带" in text or "黄色方块" in text:
                vectors.append([1.0, 0.0, 0.0])
            else:
                vectors.append([0.0, 1.0, 0.0])
        return vectors
