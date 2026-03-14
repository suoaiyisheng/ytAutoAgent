from __future__ import annotations

import base64
from pathlib import Path

import pytest

from app.config import Settings
from app.services.image_generation import (
    AliyunOSSUploader,
    ImageGenerationService,
    OpenRouterHTTPError,
    OpenRouterImageClient,
    parse_shot_range,
)
from app.services.store import TaskStore


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "jobs",
        max_workers=1,
        task_timeout_sec=300,
        vlm_provider="gemini",
        embedding_provider="gemini",
        gemini_api_key="test",
        qwen_api_key="",
        qwen_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        qwen_embed_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
        qwen_vlm_model="qwen3-vl-flash",
        qwen_embed_model="text-embedding-v3",
        gemini_vlm_model="gemini-1.5-pro",
        gemini_embed_model="text-embedding-004",
        pipeline_retry_max=2,
        pipeline_batch_size=4,
        image_generation_provider="openrouter",
        openrouter_api_key="test-openrouter-key",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_image_model="google/gemini-2.5-flash-image",
        volcengine_access_key_id="",
        volcengine_access_key_secret="",
        volcengine_session_token="",
        volcengine_visual_host="visual.volcengineapi.com",
        volcengine_region="cn-north-1",
        volcengine_service="cv",
        volcengine_jimeng_req_key="jimeng_t2i_v30",
        volcengine_jimeng_version="2022-08-31",
        volcengine_poll_interval_sec=2.0,
        volcengine_poll_timeout_sec=120,
        aliyun_oss_region="oss-cn-hangzhou",
        aliyun_oss_access_key_id="ak",
        aliyun_oss_access_key_secret="sk",
        aliyun_oss_bucket="bucket",
        aliyun_oss_public_domain="https://cdn.example.com",
    )


def test_parse_shot_range_supports_ranges_and_list():
    assert parse_shot_range("1-3,5", [1, 2, 3, 4, 5, 6]) == [1, 2, 3, 5]
    assert parse_shot_range(None, [3, 1, 2]) == [1, 2, 3]


def test_openrouter_fallback_to_text_prompt(monkeypatch):
    client = OpenRouterImageClient(
        api_key="k",
        base_url="https://openrouter.ai/api/v1",
        model="bytedance/seedream-4",
        retry_max=0,
    )
    b64 = base64.b64encode(b"mock-image").decode("utf-8")
    calls = []

    def fake_request(payload):
        calls.append(payload)
        if len(calls) == 1:
            raise OpenRouterHTTPError(400, "unsupported param")
        return {
            "choices": [
                {
                    "message": {
                        "images": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            }
                        ]
                    }
                }
            ]
        }

    monkeypatch.setattr(client, "_request", fake_request)
    items = client.generate_images(
        prompt="hello",
        n=1,
        aspect_ratio="9:16",
        reference_urls=["https://oss/a.jpg"],
    )
    assert len(items) == 1
    assert items[0]["url"].startswith("data:image/png;base64,")
    first_content = calls[0]["messages"][0]["content"]
    assert any(x.get("type") == "image_url" for x in first_content if isinstance(x, dict))
    second_content = calls[1]["messages"][0]["content"]
    assert not any(x.get("type") == "image_url" for x in second_content if isinstance(x, dict))
    text_parts = [x.get("text", "") for x in second_content if isinstance(x, dict) and x.get("type") == "text"]
    assert any("参考图链接如下" in x for x in text_parts)


def test_openrouter_model_fallback_when_invalid(monkeypatch):
    client = OpenRouterImageClient(
        api_key="k",
        base_url="https://openrouter.ai/api/v1",
        model="bytedance/seedream-4",
        retry_max=0,
    )
    b64 = base64.b64encode(b"fallback-image").decode("utf-8")
    called_models: list[str] = []

    def fake_request_with_image_compat(payload):
        called_models.append(str(payload.get("model", "")))
        if payload.get("model") == "bytedance/seedream-4":
            raise OpenRouterHTTPError(400, "invalid model id")
        return {
            "choices": [
                {
                    "message": {
                        "images": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            }
                        ]
                    }
                }
            ]
        }

    monkeypatch.setattr(client, "_request_with_image_compat", fake_request_with_image_compat)
    items = client.generate_images(
        prompt="hello",
        n=1,
        aspect_ratio="9:16",
        reference_urls=[],
    )

    assert len(items) == 1
    assert items[0]["url"].startswith("data:image/png;base64,")
    assert called_models[:2] == ["bytedance/seedream-4", "google/gemini-2.5-flash-image"]


def test_oss_upload_builds_authorization_header(monkeypatch, tmp_path):
    uploader = AliyunOSSUploader(
        region="oss-cn-hangzhou",
        access_key_id="ak",
        access_key_secret="sk",
        bucket="bucket",
        public_domain="https://cdn.example.com",
    )
    image_path = tmp_path / "ref.jpg"
    image_path.write_bytes(b"abc")

    captured = {}

    class DummyResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout):  # noqa: ANN001
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["method"] = request.get_method()
        captured["timeout"] = timeout
        return DummyResp()

    monkeypatch.setattr("app.services.image_generation.urllib.request.urlopen", fake_urlopen)
    url = uploader.upload_file(local_path=image_path, key="x/y.jpg", content_type="image/jpeg")

    assert url == "https://cdn.example.com/x/y.jpg"
    assert captured["method"] == "PUT"
    assert captured["url"] == "https://bucket.oss-cn-hangzhou.aliyuncs.com/x/y.jpg"
    assert "Authorization" in captured["headers"]
    assert captured["headers"]["Authorization"].startswith("OSS ak:")
    assert "Date" in captured["headers"]


def test_service_generates_only_first_three_shots(tmp_path):
    settings = _settings(tmp_path)
    store = TaskStore(settings.data_dir)
    task_id = "job123"
    task_dir = store.task_dir(task_id)

    ref1 = task_dir / "ref1.jpg"
    ref2 = task_dir / "ref2.jpg"
    ref1.write_bytes(b"ref1")
    ref2.write_bytes(b"ref2")

    scenes = []
    for sid in range(1, 5):
        keyframe = task_dir / f"frame_{sid:03d}.jpg"
        clip = task_dir / f"clip_{sid:03d}.mp4"
        keyframe.write_bytes(f"frame{sid}".encode("utf-8"))
        clip.write_bytes(f"clip{sid}".encode("utf-8"))
        scenes.append(
            {
                "scene_id": sid,
                "keyframe_path": str(keyframe),
                "clip_path": str(clip),
            }
        )
    store.write_json(store.contract_path(task_id, "physical_manifest"), {"project_id": task_id, "scenes": scenes})
    store.write_json(
        store.contract_path(task_id, "character_bank"),
        {
            "project_id": task_id,
            "characters": [
                {"ref_id": "Ref_1", "ref_image_path": str(ref1), "states": []},
                {"ref_id": "Ref_2", "ref_image_path": str(ref2), "states": []},
            ],
        },
    )
    store.write_json(
        store.contract_path(task_id, "aligned_storyboard"),
        {
            "project_id": task_id,
            "storyboard": [
                {"shot_id": 1, "character_mappings": [{"ref_id": "Ref_1"}, {"ref_id": "Ref_2"}]},
                {"shot_id": 2, "character_mappings": [{"ref_id": "Ref_1"}]},
                {"shot_id": 3, "character_mappings": [{"ref_id": "Ref_2"}]},
                {"shot_id": 4, "character_mappings": [{"ref_id": "Ref_1"}]},
            ],
        },
    )
    store.write_json(
        store.contract_path(task_id, "final_production_table"),
        {
            "project_id": task_id,
            "prompts": [
                {"shot_id": 1, "image_prompt": "p1", "video_prompt": "v1"},
                {"shot_id": 2, "image_prompt": "p2", "video_prompt": "v2"},
                {"shot_id": 3, "image_prompt": "p3", "video_prompt": "v3"},
                {"shot_id": 4, "image_prompt": "p4", "video_prompt": "v4"},
            ],
        },
    )

    class DummyUploader:
        def upload_file(self, *, local_path: Path, key: str, content_type: str, timeout_sec: int = 60) -> str:  # noqa: ARG002
            return f"https://oss.example.com/{key}"

    class DummyClient:
        def __init__(self) -> None:
            self.counter = 0

        def generate_images(self, *, prompt: str, n: int, aspect_ratio: str, reference_urls: list[str]):  # noqa: ARG002
            out = []
            for _ in range(n):
                self.counter += 1
                raw = f"{prompt}-{self.counter}".encode("utf-8")
                out.append({"url": "", "b64_json": base64.b64encode(raw).decode("utf-8")})
            return out

    class DummyService(ImageGenerationService):
        def validate_ready(self) -> None:
            return None

        def _create_oss_uploader(self):
            return DummyUploader()

        def _create_image_client(self):
            return DummyClient()

    service = DummyService(store=store, settings=settings)
    progress = []
    result = service.run(
        task_id=task_id,
        params={
            "shot_range": "1-3",
            "candidates_per_shot": 8,
            "aspect_ratio": "9:16",
            "concurrency": 2,
        },
        progress_cb=lambda p, m: progress.append((p, m)),
    )

    assert result["shot_count"] == 3
    assert result["total_candidates"] == 24
    assert result["failed_shots"] == []
    assert store.ref_oss_map_path(task_id).exists()
    assert store.image_candidates_path(task_id).exists()
    payload = store.read_json(store.image_candidates_path(task_id))
    assert [int(x["shot_id"]) for x in payload["shots"]] == [1, 2, 3]
    for shot in payload["shots"]:
        assert len(shot["candidates"]) == 8
        for item in shot["candidates"]:
            assert Path(item["local_path"]).exists()
    assert not (task_dir / "outputs" / "images" / "shot_004").exists()
    assert any("镜头 3 生图完成" in msg for _, msg in progress)
