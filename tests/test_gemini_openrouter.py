from __future__ import annotations

from pathlib import Path

from app.cli import _build_pipeline
from app.config import Settings
from app.services.providers import GeminiVLMProvider
from app.tools.run_pipeline_with_llm_review import _select_review_provider


def _settings() -> Settings:
    return Settings(
        data_dir=Path("/tmp/ytauto-tests"),
        max_workers=1,
        task_timeout_sec=300,
        vlm_provider="gemini",
        embedding_provider="qwen",
        gemini_api_key="",
        qwen_api_key="qwen-key",
        qwen_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        qwen_embed_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
        qwen_vlm_model="qwen3-vl-flash",
        qwen_embed_model="text-embedding-v3",
        gemini_vlm_model="gemini-2.5-flash",
        gemini_embed_model="text-embedding-004",
        pipeline_retry_max=2,
        pipeline_batch_size=4,
        image_generation_provider="openrouter",
        openrouter_api_key="openrouter-key",
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
        aliyun_oss_region="",
        aliyun_oss_access_key_id="",
        aliyun_oss_access_key_secret="",
        aliyun_oss_bucket="",
        aliyun_oss_public_domain="",
    )


def test_gemini_vlm_provider_uses_openrouter_protocol(monkeypatch):
    provider = GeminiVLMProvider(
        api_key="",
        openrouter_api_key="openrouter-key",
        openrouter_base_url="https://openrouter.ai/api/v1",
    )
    captured: dict[str, object] = {}

    def fake_post_openrouter_chat_completion(body):  # noqa: ANN001
        captured["body"] = body
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"ok": true}',
                    }
                }
            ]
        }

    monkeypatch.setattr(provider, "_post_openrouter_chat_completion", fake_post_openrouter_chat_completion)

    payload = provider._generate_json(parts=[{"text": "请输出 JSON"}], model="gemini-2.5-flash", retry_max=0)  # noqa: SLF001

    assert payload == {"ok": True}
    assert provider.is_using_openrouter() is True
    assert captured["body"]["model"] == "google/gemini-2.5-flash"


def test_select_review_provider_accepts_openrouter_gemini(monkeypatch):
    settings = _settings()
    monkeypatch.setattr("app.tools.run_pipeline_with_llm_review.load_settings", lambda: settings)

    provider, model = _select_review_provider("gemini")

    assert isinstance(provider, GeminiVLMProvider)
    assert provider.is_using_openrouter() is True
    assert model == "gemini-2.5-flash"


def test_build_pipeline_wires_gemini_via_openrouter(monkeypatch):
    settings = _settings()
    monkeypatch.setattr("app.cli.load_settings", lambda: settings)

    pipeline = _build_pipeline()

    assert isinstance(pipeline.vlm_provider, GeminiVLMProvider)
    assert pipeline.vlm_provider.is_using_openrouter() is True
    assert pipeline.default_vlm_model == "gemini-2.5-flash"
