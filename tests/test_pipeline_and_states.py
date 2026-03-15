from __future__ import annotations

import json
import time
from pathlib import Path

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


def _build_unit_pipeline(tmp_path: Path) -> Stage1Pipeline:
    return Stage1Pipeline(
        store=TaskStore(tmp_path / "jobs"),
        timeout_sec=120,
        vlm_provider=FakeVLMProvider(),
        embedding_provider=FakeEmbeddingProvider(),
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
    assert result["stats"]["character_count"] == 1
    assert result["stats"]["prompt_count"] == 3

    for contract_path in result["contracts"].values():
        assert Path(contract_path).exists()

    character_bank = json.loads(Path(result["contracts"]["character_bank"]).read_text(encoding="utf-8"))
    for character in character_bank["characters"]:
        assert {"ref_id", "ref_image_path", "scene_presence", "name", "master_description", "key_features"}.issubset(
            set(character.keys())
        )
        assert isinstance(character["key_features"], list)
        for presence in character["scene_presence"]:
            assert isinstance(presence, list)
            assert len(presence) == 2
            assert isinstance(presence[0], int)
            assert isinstance(presence[1], str)

    aligned = json.loads(Path(result["contracts"]["aligned_storyboard"]).read_text(encoding="utf-8"))
    assert aligned["scenes"][0]["visual_analysis"]["subjects"][0]["id"] == "Ref_1"
    assert aligned["scenes"][1]["visual_analysis"]["subjects"][0]["id"] == "Ref_1"
    for scene in aligned["scenes"]:
        for subject in scene["visual_analysis"]["subjects"]:
            assert set(subject.keys()) == {"id", "appearance", "action", "expression"}
            assert "temp_id" not in subject

    final_table = json.loads(Path(result["contracts"]["final_production_table"]).read_text(encoding="utf-8"))
    first_bindings = final_table["prompts"][0]["reference_bindings"]
    assert first_bindings[0]["reference_index"] == 1
    assert first_bindings[0]["ref_id"] == "Ref_1"
    assert set(first_bindings[0].keys()) == {"reference_index", "ref_id"}


def test_stage5_dump_context_file_generated(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake-video")
    dump_dir = tmp_path / "dump_ctx"
    dump_dir.mkdir(parents=True, exist_ok=True)

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
        task_id="task_dump_ctx",
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
            "stage5_dump_path": str(dump_dir),
        },
        working_dir=str(store.task_dir("task_dump_ctx")),
    )

    pipeline.run(task, lambda *_: None)
    dump_path = dump_dir / "stage05_context_task_dump_ctx.json"
    assert dump_path.exists()
    payload = json.loads(dump_path.read_text(encoding="utf-8"))
    assert payload["task_id"] == "task_dump_ctx"
    for key in [
        "architect_prompt",
        "stage5_protocol_prompt",
        "character_bank",
        "aligned_storyboard",
        "provider_request",
        "provider_raw_output",
        "provider_table_output",
        "final_output",
    ]:
        assert key in payload
    assert payload["final_output"]["project_id"] == "task_dump_ctx"
    assert isinstance(payload["final_output"].get("prompts"), list)


class SplitTempIdVLMProvider(VLMProvider):
    def describe_scenes(self, scene_inputs, model: str, retry_max: int):  # noqa: ARG002
        out = []
        for item in scene_inputs:
            sid = int(item["scene_id"])
            if sid == 1:
                subjects = [
                    {"temp_id": "P1", "appearance": "女性，偏胖，紫色长辫子", "action": "站立", "expression": "平静"},
                    {"temp_id": "P2", "appearance": "男性，匀称，上身赤裸，下半身鱼尾", "action": "游动", "expression": "微笑"},
                ]
            elif sid == 2:
                subjects = [
                    {"temp_id": "P1", "appearance": "女性，偏胖，紫色长辫子", "action": "行走", "expression": "平静"},
                ]
            elif sid == 3:
                subjects = [
                    {"temp_id": "P1", "appearance": "女性，偏胖，紫色长辫子", "action": "说话", "expression": "严肃"},
                    {
                        "temp_id": "P2",
                        "appearance": "男性，匀称，金色短发，戴黑框眼镜，身穿白色实验服",
                        "action": "讲解",
                        "expression": "认真",
                    },
                ]
            else:
                raise Stage1Error("scene_invalid", f"unexpected scene {sid}", 500)
            out.append(
                {
                    "scene_id": sid,
                    "visual_analysis": {
                        "subjects": subjects,
                        "environment": {"location": "室内", "lighting": "明亮", "atmosphere": "自然"},
                        "camera": {"shot_size": "中景", "angle": "平视", "movement": "固定镜头"},
                    },
                }
            )
        return out

    def review_character_candidates(self, candidates, model: str, retry_max: int):  # noqa: ARG002
        return candidates

    def generate_production_table(
        self,
        character_bank,
        aligned_storyboard,
        architect_prompt: str,
        model: str,
        retry_max: int,
        debug_context=None,
    ):  # noqa: ARG002
        if isinstance(debug_context, dict):
            debug_context["architect_prompt"] = architect_prompt
            debug_context["stage5_protocol_prompt"] = "split_test_protocol"
            debug_context["character_bank"] = character_bank
            debug_context["aligned_storyboard"] = aligned_storyboard
            debug_context["provider_request"] = {"provider": "split_test", "model": model}
            debug_context["provider_raw_output"] = {"ok": True}
        prompts = []
        for shot in aligned_storyboard.get("storyboard", []):
            sid = int(shot["shot_id"])
            prompts.append(
                {
                    "shot_id": sid,
                    "reference_bindings": shot.get("reference_bindings", []),
                    "image_prompt": f"{sid}，参考图1。",
                    "video_prompt": "固定镜头。",
                }
            )
        return {"project_id": aligned_storyboard.get("project_id", ""), "prompts": prompts}


class SplitTempIdEmbeddingProvider(EmbeddingProvider):
    def embed_texts(self, texts, model: str, retry_max: int):  # noqa: ARG002
        vectors = []
        for text in texts:
            normalized = str(text)
            if "紫色长辫子" in normalized:
                vectors.append([1.0, 0.0, 0.0])
            elif "鱼尾" in normalized:
                vectors.append([0.0, 1.0, 0.0])
            elif "白色实验服" in normalized or "黑框眼镜" in normalized:
                vectors.append([0.0, 0.0, 1.0])
            else:
                vectors.append([0.2, 0.2, 0.2])
        return vectors


def test_stage3_can_split_same_temp_id_into_different_refs(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake-video")

    store = TaskStore(tmp_path / "jobs")
    pipeline = Stage1Pipeline(
        store=store,
        timeout_sec=120,
        vlm_provider=SplitTempIdVLMProvider(),
        embedding_provider=SplitTempIdEmbeddingProvider(),
        downloader=LocalFileDownloader(source),
        scene_detector=FixedSceneDetector(),
        frame_extractor=TouchFrameExtractor(),
        media_probe=FakeMediaProbe(),
    )

    task = TaskRecord.new(
        task_id="task_split_temp_id",
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
        working_dir=str(store.task_dir("task_split_temp_id")),
    )
    result = pipeline.run(task, lambda *_: None)
    aligned = json.loads(Path(result["contracts"]["aligned_storyboard"]).read_text(encoding="utf-8"))

    scene1_subjects = aligned["scenes"][0]["visual_analysis"]["subjects"]
    scene3_subjects = aligned["scenes"][2]["visual_analysis"]["subjects"]
    p2_ref_scene1 = scene1_subjects[1]["id"]
    p2_ref_scene3 = scene3_subjects[1]["id"]
    assert p2_ref_scene1.startswith("Ref_")
    assert p2_ref_scene3.startswith("Ref_")
    assert p2_ref_scene1 != p2_ref_scene3


def test_aligned_storyboard_can_fallback_map_with_description_and_scene_presence(tmp_path):
    store = TaskStore(tmp_path / "jobs")
    pipeline = Stage1Pipeline(
        store=store,
        timeout_sec=120,
        vlm_provider=FakeVLMProvider(),
        embedding_provider=FakeEmbeddingProvider(),
    )

    raw_scene_descriptions = {
        "project_id": "task_fallback_mapping",
        "scenes": [
            {
                "scene_id": 1,
                "visual_analysis": {
                    "subjects": [
                        {
                            "temp_id": "P1",
                            "appearance": "女性，偏胖，紫色长辫子，身穿黄色短袖上衣",
                            "action": "站立",
                            "expression": "平静",
                        }
                    ],
                    "environment": {"location": "室内", "lighting": "明亮", "atmosphere": "自然"},
                    "camera": {"shot_size": "中景", "angle": "平视", "movement": "固定镜头"},
                },
            },
            {
                "scene_id": 2,
                "visual_analysis": {
                    "subjects": [
                        {
                            "temp_id": "P1",
                            "appearance": "女性，偏胖，紫色长辫子，身穿黄色短袖上衣",
                            "action": "行走",
                            "expression": "平静",
                        }
                    ],
                    "environment": {"location": "室内", "lighting": "明亮", "atmosphere": "自然"},
                    "camera": {"shot_size": "中景", "angle": "平视", "movement": "固定镜头"},
                },
            },
        ],
    }
    characters = [
        {
            "ref_id": "Ref_1",
            "name": "角色A",
            "master_description": "女性，偏胖，紫色长辫子，身穿黄色短袖上衣",
            "key_features": ["gender:female", "hair:purple_braid", "top:yellow", "body:overweight"],
            "ref_image_path": "",
            "scene_presence": [[1, "P1"]],
        },
        {
            "ref_id": "Ref_2",
            "name": "角色B",
            "master_description": "女性，偏胖，紫色长辫子，身穿黄色短袖上衣",
            "key_features": ["gender:female", "hair:purple_braid", "top:yellow", "body:overweight"],
            "ref_image_path": "",
            "scene_presence": [[2, "P1"]],
        },
    ]

    aligned = pipeline._build_aligned_storyboard(raw_scene_descriptions, mapping={}, characters=characters)
    assert aligned["scenes"][0]["visual_analysis"]["subjects"][0]["id"] == "Ref_1"
    assert aligned["scenes"][1]["visual_analysis"]["subjects"][0]["id"] == "Ref_2"


def test_stage5_generation_input_uses_aligned_storyboard_subject_facts(tmp_path):
    pipeline = _build_unit_pipeline(tmp_path)
    aligned_storyboard = {
        "project_id": "task_stage5_appearance_priority",
        "scenes": [
            {
                "scene_id": 1,
                "visual_analysis": {
                    "subjects": [
                        {
                            "id": "Ref_1",
                            "appearance": "男性，黑发，红色外套",
                            "action": "奔跑",
                            "expression": "紧张",
                        }
                    ],
                    "environment": {"location": "街道", "lighting": "阴天", "atmosphere": "压抑"},
                    "camera": {"shot_size": "中景", "angle": "平视", "movement": "固定镜头"},
                },
            }
        ],
    }
    character_bank = {
        "project_id": "task_stage5_appearance_priority",
        "characters": [
            {
                "ref_id": "Ref_1",
                "ref_image_path": "/tmp/ref_1.jpg",
                "master_description": "女性，紫色长辫子，黄色短袖",
                "key_features": ["女性", "紫色长辫子", "黄色短袖"],
                "states": [
                    {
                        "state_id": "state_scene_1",
                        "description": "女性，金发，白色外套",
                        "scene_presence": [1],
                    }
                ],
            }
        ],
    }

    stage4_base_storyboard = pipeline._project_aligned_storyboard_to_stage4_storyboard(aligned_storyboard)
    expected_bindings = pipeline._build_reference_bindings_by_shot(stage4_base_storyboard)
    stage4_storyboard = pipeline._attach_reference_bindings_to_storyboard(
        aligned_storyboard=stage4_base_storyboard,
        bindings_by_shot=expected_bindings,
    )
    stage5_input = pipeline._build_stage5_generation_input(
        aligned_storyboard=aligned_storyboard,
        stage4_storyboard=stage4_storyboard,
        character_bank=character_bank,
    )

    shot = stage5_input["storyboard"][0]
    mapping = shot["character_mappings"][0]
    assert mapping["appearance"] == "男性，黑发，红色外套"
    assert mapping["action_in_shot"] == "奔跑"
    assert mapping["expression_range"] == ["紧张"]
    assert shot["reference_bindings"] == [{"reference_index": 1, "ref_id": "Ref_1"}]
    assert stage5_input["reference_catalog"][0]["ref_image_path"] == "/tmp/ref_1.jpg"


def test_stage5_generation_input_can_use_storyboard_only_and_keep_binding_order(tmp_path):
    pipeline = _build_unit_pipeline(tmp_path)
    aligned_storyboard = {
        "project_id": "task_stage5_storyboard_only",
        "storyboard": [
            {
                "shot_id": 2,
                "character_mappings": [
                    {
                        "ref_id": "Ref_1",
                        "appearance": "04外观A",
                        "action_in_shot": "动作B",
                        "expression_range": ["表情B"],
                    }
                ],
                "environment_context": "环境B",
                "camera_instruction": "机位B",
            },
            {
                "shot_id": 1,
                "character_mappings": [
                    {
                        "ref_id": "Ref_2",
                        "appearance": "04外观2",
                        "action_in_shot": "动作2",
                        "expression_range": ["表情2"],
                    },
                    {
                        "ref_id": "Ref_1",
                        "appearance": "04外观1",
                        "action_in_shot": "动作1",
                        "expression_range": ["表情1"],
                    },
                ],
                "environment_context": "环境A",
                "camera_instruction": "机位A",
            },
        ],
    }
    character_bank = {
        "project_id": "task_stage5_storyboard_only",
        "characters": [
            {
                "ref_id": "Ref_1",
                "ref_image_path": "/tmp/ref_1.jpg",
                "master_description": "03外观1",
                "key_features": ["角色1"],
            },
            {
                "ref_id": "Ref_2",
                "ref_image_path": "/tmp/ref_2.jpg",
                "master_description": "03外观2",
                "key_features": ["角色2"],
            },
        ],
    }

    stage4_base_storyboard = pipeline._project_aligned_storyboard_to_stage4_storyboard(aligned_storyboard)
    expected_bindings = pipeline._build_reference_bindings_by_shot(stage4_base_storyboard)
    stage4_storyboard = pipeline._attach_reference_bindings_to_storyboard(
        aligned_storyboard=stage4_base_storyboard,
        bindings_by_shot=expected_bindings,
    )
    stage5_input = pipeline._build_stage5_generation_input(
        aligned_storyboard=aligned_storyboard,
        stage4_storyboard=stage4_storyboard,
        character_bank=character_bank,
    )

    assert [int(x["shot_id"]) for x in stage5_input["storyboard"]] == [1, 2]
    shot1 = stage5_input["storyboard"][0]
    assert [int(x["reference_index"]) for x in shot1["reference_bindings"]] == [1, 2]
    assert [str(x["ref_id"]) for x in shot1["character_mappings"]] == ["Ref_2", "Ref_1"]
    assert shot1["character_mappings"][0]["appearance"] == "04外观2"
    assert shot1["character_mappings"][1]["appearance"] == "04外观1"
    assert shot1["character_mappings"][0]["action_in_shot"] == "动作2"
    assert shot1["character_mappings"][1]["action_in_shot"] == "动作1"
    assert shot1["character_mappings"][0]["expression_range"] == ["表情2"]
    assert shot1["character_mappings"][1]["expression_range"] == ["表情1"]
    assert shot1["environment_context"] == "环境A"
    assert shot1["camera_instruction"] == "机位A"


def test_stage5_generation_input_allows_duplicate_ref_image_paths(tmp_path):
    pipeline = _build_unit_pipeline(tmp_path)
    aligned_storyboard = {
        "project_id": "task_stage5_duplicate_ref_path",
        "storyboard": [
            {
                "shot_id": 1,
                "character_mappings": [
                    {"ref_id": "Ref_1", "action_in_shot": "动作1", "expression_range": ["表情1"]},
                    {"ref_id": "Ref_2", "action_in_shot": "动作2", "expression_range": ["表情2"]},
                ],
                "environment_context": "环境",
                "camera_instruction": "机位",
            }
        ],
    }
    character_bank = {
        "project_id": "task_stage5_duplicate_ref_path",
        "characters": [
            {
                "ref_id": "Ref_1",
                "ref_image_path": "/tmp/shared_ref.jpg",
                "master_description": "角色1外观",
                "key_features": ["角色1"],
            },
            {
                "ref_id": "Ref_2",
                "ref_image_path": "/tmp/shared_ref.jpg",
                "master_description": "角色2外观",
                "key_features": ["角色2"],
            },
        ],
    }

    stage4_base_storyboard = pipeline._project_aligned_storyboard_to_stage4_storyboard(aligned_storyboard)
    expected_bindings = pipeline._build_reference_bindings_by_shot(stage4_base_storyboard)
    stage4_storyboard = pipeline._attach_reference_bindings_to_storyboard(
        aligned_storyboard=stage4_base_storyboard,
        bindings_by_shot=expected_bindings,
    )
    stage5_input = pipeline._build_stage5_generation_input(
        aligned_storyboard=aligned_storyboard,
        stage4_storyboard=stage4_storyboard,
        character_bank=character_bank,
    )

    reference_catalog = stage5_input["reference_catalog"]
    assert len(reference_catalog) == 2
    assert reference_catalog[0]["ref_image_path"] == "/tmp/shared_ref.jpg"
    assert reference_catalog[1]["ref_image_path"] == "/tmp/shared_ref.jpg"


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
