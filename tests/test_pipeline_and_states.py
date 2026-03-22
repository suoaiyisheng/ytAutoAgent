from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from app.errors import Stage1Error
from app.models import TaskRecord
from app.services.media_probe import VideoMetadata
from app.services.pipeline import Stage1Pipeline
from app.services.providers import EmbeddingProvider, VLMProvider
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


def _build_unit_pipeline(tmp_path: Path, vlm_provider: VLMProvider | None = None, embedding_provider: EmbeddingProvider | None = None) -> Stage1Pipeline:
    return Stage1Pipeline(
        store=TaskStore(tmp_path / "jobs"),
        timeout_sec=120,
        vlm_provider=vlm_provider or FakeVLMProvider(),
        embedding_provider=embedding_provider or FakeEmbeddingProvider(),
    )


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


class WeightLossVLMProvider(VLMProvider):
    def describe_scenes(self, scene_inputs, model: str, retry_max: int):  # noqa: ARG002
        return [
            {
                "scene_id": 1,
                "subjects": [
                    {
                        "subject_id": "subject_1",
                        "appearance": "女性，长发，穿浅蓝色运动T恤、黑色紧身裤和白色运动鞋，体型偏胖",
                    }
                ],
                "desc": "subject_1 在卧室穿衣镜前正在看着镜子哭泣。",
            },
            {
                "scene_id": 2,
                "subjects": [
                    {
                        "subject_id": "subject_1",
                        "appearance": "女性，长发，穿浅蓝色运动T恤、黑色紧身裤和白色运动鞋，体型偏胖",
                    }
                ],
                "desc": "subject_1 在客厅瑜伽垫上正在运动。",
            },
            {
                "scene_id": 3,
                "subjects": [
                    {
                        "subject_id": "subject_1",
                        "appearance": "女性，长发，穿浅蓝色运动T恤、黑色紧身裤和白色运动鞋，体型清瘦",
                    }
                ],
                "desc": "subject_1 在卧室穿衣镜前正在看向镜中的自己微笑。",
            },
        ]

    def review_character_candidates(self, candidates, model: str, retry_max: int):  # noqa: ARG002
        return candidates

    def generate_production_table(
        self,
        stage5_input,
        stage5_protocol_prompt: str,
        architect_prompt: str,
        model: str,
        retry_max: int,
        debug_context=None,
    ):  # noqa: ARG002
        if isinstance(debug_context, dict):
            debug_context["architect_prompt"] = architect_prompt
            debug_context["stage5_protocol_prompt"] = stage5_protocol_prompt
            debug_context["stage5_input"] = stage5_input
            debug_context["provider_request"] = {"provider": "weight_loss", "model": model}
            debug_context["provider_raw_output"] = {"ok": True}
        return {
            "project_id": stage5_input.get("project_id", ""),
            "prompts": [
                {
                    "shot_id": int(shot["shot_id"]),
                    "image_prompt": f"镜头{shot['shot_id']}，参考图1。",
                    "video_prompt": "固定镜头。",
                }
                for shot in stage5_input.get("shots", [])
            ],
        }


class WeightLossEmbeddingProvider(EmbeddingProvider):
    def embed_texts(self, texts, model: str, retry_max: int):  # noqa: ARG002
        return [[1.0, 0.0, 0.0] for _ in texts]


class SplitRefVLMProvider(VLMProvider):
    def describe_scenes(self, scene_inputs, model: str, retry_max: int):  # noqa: ARG002
        return [
            {
                "scene_id": 1,
                "subjects": [{"subject_id": "subject_1", "appearance": "女性，紫色长辫子，黄色短袖，体型偏胖"}],
                "desc": "subject_1 在室内正在站立。",
            },
            {
                "scene_id": 2,
                "subjects": [{"subject_id": "subject_1", "appearance": "女性，紫色长辫子，黄色短袖，体型偏胖"}],
                "desc": "subject_1 在室内正在行走。",
            },
            {
                "scene_id": 3,
                "subjects": [{"subject_id": "subject_1", "appearance": "男性，金色短发，白色实验服，佩戴眼镜，体型匀称"}],
                "desc": "subject_1 在实验室正在讲解。",
            },
        ]

    def review_character_candidates(self, candidates, model: str, retry_max: int):  # noqa: ARG002
        return candidates

    def generate_production_table(
        self,
        stage5_input,
        stage5_protocol_prompt: str,
        architect_prompt: str,
        model: str,
        retry_max: int,
        debug_context=None,
    ):  # noqa: ARG002
        return {
            "project_id": stage5_input.get("project_id", ""),
            "prompts": [
                {
                    "shot_id": int(shot["shot_id"]),
                    "image_prompt": "参考图1。",
                    "video_prompt": "固定镜头。",
                }
                for shot in stage5_input.get("shots", [])
            ],
        }


class SplitRefEmbeddingProvider(EmbeddingProvider):
    def embed_texts(self, texts, model: str, retry_max: int):  # noqa: ARG002
        vectors = []
        for text in texts:
            if "实验服" in str(text) or "眼镜" in str(text):
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([1.0, 0.0, 0.0])
        return vectors


def test_full_pipeline_few_shot_weight_loss_contracts(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake-video")

    store = TaskStore(tmp_path / "jobs")
    pipeline = Stage1Pipeline(
        store=store,
        timeout_sec=120,
        vlm_provider=WeightLossVLMProvider(),
        embedding_provider=WeightLossEmbeddingProvider(),
        downloader=LocalFileDownloader(source),
        scene_detector=FixedSceneDetector(),
        frame_extractor=TouchFrameExtractor(),
        media_probe=FakeMediaProbe(),
    )

    task = TaskRecord.new(
        task_id="task_weight_loss",
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
        working_dir=str(store.task_dir("task_weight_loss")),
    )

    result = pipeline.run(task, lambda *_: None)
    assert result["stats"]["scene_count"] == 3
    assert result["stats"]["character_count"] == 1
    assert result["stats"]["prompt_count"] == 3

    raw_scene = json.loads(Path(result["contracts"]["raw_scene_descriptions"]).read_text(encoding="utf-8"))
    assert set(raw_scene["scenes"][0].keys()) == {"scene_id", "subjects", "desc"}
    assert raw_scene["scenes"][0]["subjects"][0]["subject_id"] == "subject_1"
    assert "Ref_" not in raw_scene["scenes"][0]["desc"]

    character_bank = json.loads(Path(result["contracts"]["character_bank"]).read_text(encoding="utf-8"))
    character = character_bank["characters"][0]
    assert set(character.keys()) == {
        "ref_id",
        "ref_name",
        "canonical_description",
        "ref_image_path",
        "ref_image_description",
        "ref_image_features",
        "scene_presence",
    }
    assert character["ref_id"] == "Ref_1"
    assert character["ref_name"] == "参考图1"
    assert character["ref_image_description"] == "女性，长发，穿浅蓝色运动T恤、黑色紧身裤和白色运动鞋，体型偏胖"
    assert "长发" in character["ref_image_features"]
    assert character["scene_presence"] == [[1, "subject_1"], [2, "subject_1"], [3, "subject_1"]]
    assert "体型偏胖" in character["canonical_description"]
    assert "体型清瘦" in character["canonical_description"]
    assert "subject_mappings" not in character_bank

    normalized = json.loads(Path(result["contracts"]["normalized_scene_descriptions"]).read_text(encoding="utf-8"))
    assert set(normalized.keys()) == {"project_id", "scenes"}
    assert normalized["scenes"][0]["desc"] == "参考图1 在卧室穿衣镜前正在看着镜子哭泣。"
    assert normalized["scenes"][1]["desc"] == "参考图1 在客厅瑜伽垫上正在运动。"
    assert normalized["scenes"][2]["desc"] == "参考图1 在卧室穿衣镜前正在看向镜中的自己微笑。"
    assert "客厅瑜伽垫上正在运动" in normalized["scenes"][1]["desc"]

    final_table = json.loads(Path(result["contracts"]["final_production_table"]).read_text(encoding="utf-8"))
    assert set(final_table["prompts"][0].keys()) == {"shot_id", "image_prompt", "video_prompt"}


def test_stage3_can_split_different_people_into_different_refs(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake-video")

    store = TaskStore(tmp_path / "jobs")
    pipeline = Stage1Pipeline(
        store=store,
        timeout_sec=120,
        vlm_provider=SplitRefVLMProvider(),
        embedding_provider=SplitRefEmbeddingProvider(),
        downloader=LocalFileDownloader(source),
        scene_detector=FixedSceneDetector(),
        frame_extractor=TouchFrameExtractor(),
        media_probe=FakeMediaProbe(),
    )

    task = TaskRecord.new(
        task_id="task_split_ref",
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
        working_dir=str(store.task_dir("task_split_ref")),
    )

    result = pipeline.run(task, lambda *_: None)
    character_bank = json.loads(Path(result["contracts"]["character_bank"]).read_text(encoding="utf-8"))
    assert [item["ref_id"] for item in character_bank["characters"]] == ["Ref_1", "Ref_2"]
    assert character_bank["characters"][0]["scene_presence"] == [[1, "subject_1"], [2, "subject_1"]]
    assert character_bank["characters"][1]["scene_presence"] == [[3, "subject_1"]]


def test_stage4_raises_when_subject_mapping_missing(tmp_path):
    pipeline = _build_unit_pipeline(tmp_path)
    raw_scene_descriptions = {
        "project_id": "task_missing_mapping",
        "scenes": [
            {
                "scene_id": 1,
                "subjects": [{"subject_id": "subject_1", "appearance": "女性，长发，黄色短袖，体型偏胖"}],
                "desc": "subject_1 在卧室正在哭泣。",
            }
        ],
    }
    character_bank = {
        "project_id": "task_missing_mapping",
        "characters": [],
        "global_style": "真实摄影风格",
    }

    with pytest.raises(Stage1Error) as exc:
        pipeline._run_stage4(  # noqa: SLF001
            TaskRecord.new(task_id="task_missing_mapping", params={}, working_dir=str(tmp_path)),
            raw_scene_descriptions,
            character_bank,
            lambda *_: None,
            time.monotonic(),
    )
    assert exc.value.code == "subject_mapping_missing"


def test_stage4_uses_reference_name_when_appearance_similarity_exceeds_threshold(tmp_path):
    pipeline = _build_unit_pipeline(tmp_path)
    raw_scene_descriptions = {
        "project_id": "task_similarity_reference_name",
        "scenes": [
            {
                "scene_id": 1,
                "subjects": [
                    {
                        "subject_id": "subject_1",
                        "appearance": "女性，偏胖，紫色长辫，身穿黄色短袖和黑色短裤，坐在椅子上",
                    }
                ],
                "desc": "subject_1 坐在椅子上，手里拿着一件破损的美人鱼尾巴。",
            }
        ],
    }
    character_bank = {
        "project_id": "task_similarity_reference_name",
        "characters": [
            {
                "ref_id": "Ref_1",
                "ref_name": "参考图1",
                "ref_image_path": "/tmp/ref_1.jpg",
                "ref_image_description": "女性，偏胖，紫色长辫，身穿黄色短袖上衣和黑色短裤",
                "ref_image_features": ["女性", "偏胖", "紫色长辫", "黄色短袖上衣", "黑色短裤"],
                "canonical_description": "女性，偏胖，紫色长辫，身穿黄色短袖上衣和黑色短裤",
                "scene_presence": [[1, "subject_1"]],
            }
        ],
        "global_style": "真实摄影风格",
    }

    normalized = pipeline._run_stage4(  # noqa: SLF001
        TaskRecord.new(task_id="task_similarity_reference_name", params={}, working_dir=str(tmp_path)),
        raw_scene_descriptions,
        character_bank,
        lambda *_: None,
        time.monotonic(),
    )
    assert normalized["scenes"][0]["desc"] == "参考图1 坐在椅子上，手里拿着一件破损的美人鱼尾巴。"


def test_stage4_replaces_all_subject_tokens_with_reference_names(tmp_path):
    pipeline = _build_unit_pipeline(tmp_path)
    raw_scene_descriptions = {
        "project_id": "task_replace_all_subject_tokens",
        "scenes": [
            {
                "scene_id": 1,
                "subjects": [
                    {
                        "subject_id": "subject_1",
                        "appearance": "女性，偏胖，紫色长辫，身穿黄色短袖和黑色短裤",
                    },
                    {
                        "subject_id": "subject_2",
                        "appearance": "男性，健壮肌肉，赤裸上身，下身是紫色鱼尾",
                    },
                ],
                "desc": "subject_1 在水族馆的玻璃前，惊讶地看着水中的subject_2，subject_2在水中向subject_1挥手。",
            }
        ],
    }
    character_bank = {
        "project_id": "task_replace_all_subject_tokens",
        "characters": [
            {
                "ref_id": "Ref_1",
                "ref_name": "参考图1",
                "ref_image_path": "/tmp/ref_1.jpg",
                "ref_image_description": "女性，偏胖，紫色长辫，身穿黄色短袖和黑色短裤",
                "ref_image_features": ["女性", "偏胖", "紫色长辫", "黄色短袖", "黑色短裤"],
                "canonical_description": "女性，偏胖，紫色长辫，身穿黄色短袖和黑色短裤",
                "scene_presence": [[1, "subject_1"]],
            },
            {
                "ref_id": "Ref_2",
                "ref_name": "参考图2",
                "ref_image_path": "/tmp/ref_2.jpg",
                "ref_image_description": "男性，健壮肌肉，赤裸上身，下身是紫色鱼尾",
                "ref_image_features": ["男性", "健壮肌肉", "赤裸上身", "下身是紫色鱼尾"],
                "canonical_description": "男性，健壮肌肉，赤裸上身，下身是紫色鱼尾",
                "scene_presence": [[1, "subject_2"]],
            },
        ],
        "global_style": "真实摄影风格",
    }

    normalized = pipeline._run_stage4(  # noqa: SLF001
        TaskRecord.new(task_id="task_replace_all_subject_tokens", params={}, working_dir=str(tmp_path)),
        raw_scene_descriptions,
        character_bank,
        lambda *_: None,
        time.monotonic(),
    )
    assert normalized["scenes"][0]["desc"] == "参考图1 在水族馆的玻璃前，惊讶地看着水中的参考图2，参考图2在水中向参考图1挥手。"
    assert "subject_" not in normalized["scenes"][0]["desc"]


def test_stage4_derives_ref_name_from_ref_id_instead_of_outputting_ref_id(tmp_path):
    pipeline = _build_unit_pipeline(tmp_path)
    raw_scene_descriptions = {
        "project_id": "task_stage4_ref_name_only",
        "scenes": [
            {
                "scene_id": 1,
                "subjects": [{"subject_id": "subject_1", "appearance": "女性，偏胖，紫色长辫，身穿黄色短袖和黑色短裤"}],
                "desc": "subject_1 在室内向前走。",
            }
        ],
    }
    character_bank = {
        "project_id": "task_stage4_ref_name_only",
        "characters": [
            {
                "ref_id": "Ref_1",
                "ref_name": "",
                "ref_image_path": "/tmp/ref_1.jpg",
                "ref_image_description": "女性，偏胖，紫色长辫，身穿黄色短袖和黑色短裤",
                "ref_image_features": ["女性", "偏胖", "紫色长辫", "黄色短袖", "黑色短裤"],
                "canonical_description": "女性，偏胖，紫色长辫，身穿黄色短袖和黑色短裤",
                "scene_presence": [[1, "subject_1"]],
            }
        ],
        "global_style": "真实摄影风格",
    }

    normalized = pipeline._run_stage4(  # noqa: SLF001
        TaskRecord.new(task_id="task_stage4_ref_name_only", params={}, working_dir=str(tmp_path)),
        raw_scene_descriptions,
        character_bank,
        lambda *_: None,
        time.monotonic(),
    )
    assert normalized["scenes"][0]["desc"] == "参考图1 在室内向前走。"
    assert "Ref_" not in normalized["scenes"][0]["desc"]


def test_stage5_input_uses_reference_only_when_appearance_matches(tmp_path):
    pipeline = _build_unit_pipeline(tmp_path)
    character_bank = {
        "project_id": "task_stage5_grounding_match",
        "characters": [
            {
                "ref_id": "Ref_1",
                "ref_image_path": "/tmp/ref_1.jpg",
                "ref_image_description": "女性，匀称，紫色长辫子，穿着紫粉色亮片美人鱼尾和珍珠项链",
                "ref_image_features": ["女性", "匀称", "紫色长辫子", "紫粉色亮片美人鱼尾", "珍珠项链"],
                "canonical_description": "女性，匀称，紫色长辫子，穿着紫粉色亮片美人鱼尾和珍珠项链",
                "scene_presence": [[1, "subject_1"]],
            }
        ],
    }
    normalized_scene_descriptions = {
        "project_id": "task_stage5_grounding_match",
        "scenes": [
            {
                "scene_id": 1,
                "desc": "（女性，匀称，紫色长辫子，穿着紫粉色亮片美人鱼尾和珍珠项链）的Ref_1 在水下游动。",
            }
        ],
    }

    stage5_input = pipeline._build_stage5_input(  # noqa: SLF001
        character_bank=character_bank,
        normalized_scene_descriptions=normalized_scene_descriptions,
    )
    assert "reference_catalog" not in stage5_input
    assert set(stage5_input["shots"][0].keys()) == {"shot_id", "grounded_desc", "references"}
    assert "reference_bindings" not in stage5_input["shots"][0]
    assert "reference_groundings" not in stage5_input["shots"][0]
    assert stage5_input["shots"][0]["grounded_desc"] == "参考图1 在水下游动。"
    assert stage5_input["shots"][0]["references"] == [
        {"ref_name": "参考图1", "ref_image_path": "/tmp/ref_1.jpg", "ref_image_url": ""}
    ]


def test_stage5_input_uses_reference_only_when_appearance_is_similar_enough(tmp_path):
    pipeline = _build_unit_pipeline(tmp_path)
    character_bank = {
        "project_id": "task_stage5_grounding_similarity",
        "characters": [
            {
                "ref_id": "Ref_1",
                "ref_name": "参考图1",
                "ref_image_path": "/tmp/ref_1.jpg",
                "ref_image_description": "女性，偏胖，紫色长辫，身穿黄色短袖上衣和黑色短裤",
                "ref_image_features": ["女性", "偏胖", "紫色长辫", "黄色短袖上衣", "黑色短裤"],
                "canonical_description": "女性，偏胖，紫色长辫，身穿黄色短袖上衣和黑色短裤",
                "scene_presence": [[1, "subject_1"]],
            }
        ],
    }
    normalized_scene_descriptions = {
        "project_id": "task_stage5_grounding_similarity",
        "scenes": [
            {
                "scene_id": 1,
                "desc": "（女性，偏胖，紫色长辫，身穿黄色短袖和黑色短裤，坐在椅子上）的参考图1 在房间里看着前方。",
            }
        ],
    }

    stage5_input = pipeline._build_stage5_input(  # noqa: SLF001
        character_bank=character_bank,
        normalized_scene_descriptions=normalized_scene_descriptions,
    )
    assert "reference_catalog" not in stage5_input
    assert set(stage5_input["shots"][0].keys()) == {"shot_id", "grounded_desc", "references"}
    assert "reference_bindings" not in stage5_input["shots"][0]
    assert "reference_groundings" not in stage5_input["shots"][0]
    assert stage5_input["shots"][0]["grounded_desc"] == "参考图1 在房间里看着前方。"


def test_stage5_input_keeps_current_appearance_when_different_from_reference(tmp_path):
    pipeline = _build_unit_pipeline(tmp_path)
    character_bank = {
        "project_id": "task_stage5_grounding_delta",
        "characters": [
            {
                "ref_id": "Ref_1",
                "ref_image_path": "/tmp/ref_1.jpg",
                "ref_image_description": "女性，偏胖，紫色长辫子，身穿黄色短袖上衣和黑色运动短裤",
                "ref_image_features": ["女性", "偏胖", "紫色长辫子", "黄色短袖上衣", "黑色运动短裤"],
                "canonical_description": "女性，偏胖，紫色长辫子，身穿黄色短袖上衣和黑色运动短裤",
                "scene_presence": [[1, "subject_1"]],
            }
        ],
    }
    normalized_scene_descriptions = {
        "project_id": "task_stage5_grounding_delta",
        "scenes": [
            {
                "scene_id": 1,
                "desc": "（女性，匀称，紫色长辫子，穿着紫粉色亮片美人鱼尾和珍珠项链）的Ref_1 在水下游动。",
            }
        ],
    }

    stage5_input = pipeline._build_stage5_input(  # noqa: SLF001
        character_bank=character_bank,
        normalized_scene_descriptions=normalized_scene_descriptions,
    )
    assert "reference_catalog" not in stage5_input
    assert set(stage5_input["shots"][0].keys()) == {"shot_id", "grounded_desc", "references"}
    assert "reference_bindings" not in stage5_input["shots"][0]
    assert "reference_groundings" not in stage5_input["shots"][0]
    assert (
        stage5_input["shots"][0]["grounded_desc"]
        == "（女性，匀称，紫色长辫子，穿着紫粉色亮片美人鱼尾和珍珠项链）的参考图1 在水下游动。"
    )


def test_stage5_dump_context_contains_stage5_input(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake-video")
    dump_dir = tmp_path / "dump_ctx"
    dump_dir.mkdir(parents=True, exist_ok=True)

    store = TaskStore(tmp_path / "jobs")
    pipeline = Stage1Pipeline(
        store=store,
        timeout_sec=120,
        vlm_provider=WeightLossVLMProvider(),
        embedding_provider=WeightLossEmbeddingProvider(),
        downloader=LocalFileDownloader(source),
        scene_detector=FixedSceneDetector(),
        frame_extractor=TouchFrameExtractor(),
        media_probe=FakeMediaProbe(),
    )

    task = TaskRecord.new(
        task_id="task_dump_ctx",
        params={
            "source_url": "https://example.com/test",
            "threshold": 27.0,
            "min_scene_len": 1.0,
            "frame_quality": 2,
            "download_format": "bestvideo+bestaudio/best",
            "batch_size": 2,
            "retry_max": 0,
            "stage5_dump_path": str(dump_dir),
        },
        working_dir=str(store.task_dir("task_dump_ctx")),
    )

    pipeline.run(task, lambda *_: None)
    dump_path = dump_dir / "stage05_context_task_dump_ctx.json"
    payload = json.loads(dump_path.read_text(encoding="utf-8"))
    assert "stage5_input" in payload
    assert "aligned_storyboard" not in payload
    assert "base_state_references" not in payload["stage5_input"]
    assert "reference_catalog" not in payload["stage5_input"]
    assert payload["stage5_input"]["shots"][0]["grounded_desc"].startswith("参考图1 ")
    assert set(payload["stage5_input"]["shots"][0].keys()) == {"shot_id", "grounded_desc", "references"}
    assert "reference_groundings" not in payload["stage5_input"]["shots"][0]


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
