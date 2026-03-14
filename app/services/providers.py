from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.errors import Stage1Error

SCENE_DESCRIBE_PROMPT = """# Role
你是一位专业的视觉结构化分析专家，负责将文本或图像描述转化为高度精确的场景解析 JSON 数据。

# Output Format
请仅输出一个合法的 JSON 对象，严禁包含任何 Markdown 格式块、解释性文字或引言。

# JSON Structure
{
  "scenes": [
    {
      "scene_id": "场景序号",
      "visual_analysis": {
        "subjects": [
          {
            "temp_id": "角色唯一标识(如P1, P2)",
            "appearance": "性别(男性/女性/人物)，体型(偏胖/匀称/纤细/健壮肌肉)，外貌细节与服装描述",
            "action": "具体肢体动作或行为",
            "expression": "面部神情细节"
          }
        ],
        "environment": {
          "location": "空间场景/地理位置",
          "lighting": "光影性质(如: 侧逆光, 弥散光, 霓虹灯光)",
          "atmosphere": "氛围基调(如: 赛博朋克, 肃穆, 温馨)"
        },
        "camera": {
          "shot_size": "景别(如: 特写, 全景)",
          "angle": "拍摄角度(如: 俯拍, 平视)",
          "movement": "镜头运动(如: 推拉, 摇移, 固定)"
        }
      }
    }
  ]
}

# Constraint Rules
1. **语言限制**：除 temp_id 外，所有字段严禁出现英文，必须使用简体中文。
2. **性别必填**：subjects.appearance 必须以“男性”或“女性”开头，无法判断则使用“人物”。
3. **体型必填**：必须从 [偏胖、匀称、纤细、健壮肌肉] 中选择其一或组合，严禁遗漏。
4. **动态追踪**：若同一角色在不同 scene_id 中出现，必须保持 temp_id 一致；若其体型发生变化，必须在 appearance 中体现。
5. **空值处理**：若画面无人，subjects 必须设为空数组 []。
6. **视觉深度**：描述应具备可拍摄性，避免抽象情感词，多使用具象视觉描述。"""


def _ensure_gender_prefix(appearance: str) -> str:
    text = appearance.strip()
    if not text:
        return ""
    if text.startswith(("女性，", "男性，", "人物，")):
        return text

    lower = text.lower()
    female_tokens = ["女性", "女人", "女孩", "女生", "woman", "female", "lady", "girl"]
    male_tokens = ["男性", "男人", "男孩", "男生", "man", "male", "boy", "gentleman"]

    if any(token in text for token in female_tokens) or any(token in lower for token in female_tokens):
        return f"女性，{text}"
    if any(token in text for token in male_tokens) or any(token in lower for token in male_tokens):
        return f"男性，{text}"
    return text


class VLMProvider(ABC):
    @abstractmethod
    def describe_scenes(
        self,
        scene_inputs: list[dict[str, Any]],
        model: str,
        retry_max: int,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def review_character_candidates(
        self,
        candidates: list[dict[str, Any]],
        model: str,
        retry_max: int,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def generate_production_table(
        self,
        character_bank: dict[str, Any],
        aligned_storyboard: dict[str, Any],
        architect_prompt: str,
        model: str,
        retry_max: int,
    ) -> dict[str, Any]:
        raise NotImplementedError


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str], model: str, retry_max: int) -> list[list[float]]:
        raise NotImplementedError


class GeminiVLMProvider(VLMProvider):
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key.strip()

    def describe_scenes(
        self,
        scene_inputs: list[dict[str, Any]],
        model: str,
        retry_max: int,
    ) -> list[dict[str, Any]]:
        if not scene_inputs:
            return []

        blocks: list[str] = []
        image_parts: list[dict[str, Any]] = []
        for item in scene_inputs:
            scene_id = int(item["scene_id"])
            keyframe_path = Path(item["keyframe_path"])
            image_parts.append({"text": f"Scene {scene_id}"})
            image_parts.append({"inline_data": self._to_inline_data(keyframe_path)})
            blocks.append(f"Scene {scene_id} -> image: {keyframe_path}")

        prompt = SCENE_DESCRIBE_PROMPT
        parts = [{"text": prompt}, *image_parts]
        payload = self._generate_json(parts=parts, model=model, retry_max=retry_max)

        if isinstance(payload, list):
            return [self._normalize_scene_description(x) for x in payload]

        if isinstance(payload, dict) and isinstance(payload.get("scenes"), list):
            return [self._normalize_scene_description(x) for x in payload["scenes"]]

        raise Stage1Error("vlm_invalid_output", "Gemini 返回的分镜描述格式无效", 502)

    def review_character_candidates(
        self,
        candidates: list[dict[str, Any]],
        model: str,
        retry_max: int,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        prompt = (
            "你是角色归一化审核器。输入是候选角色列表，请仅输出 JSON 数组。"
            "每个元素必须包含 ref_id,name,master_description,key_features,ref_image_path。"
            "若候选角色可合并，请直接输出合并后的最终列表。"
        )
        parts = [{"text": prompt}, {"text": json.dumps(candidates, ensure_ascii=False)}]
        payload = self._generate_json(parts=parts, model=model, retry_max=retry_max)

        if isinstance(payload, list):
            return [self._normalize_character(item) for item in payload]
        if isinstance(payload, dict) and isinstance(payload.get("characters"), list):
            return [self._normalize_character(item) for item in payload["characters"]]
        return [self._normalize_character(item) for item in candidates]

    def generate_production_table(
        self,
        character_bank: dict[str, Any],
        aligned_storyboard: dict[str, Any],
        architect_prompt: str,
        model: str,
        retry_max: int,
    ) -> dict[str, Any]:
        prompt = (
            "请严格按照系统规范输出最终 JSON。结构："
            "{project_id, prompts:[{shot_id,image_prompt,video_prompt}]}。"
            "不要输出 markdown，不要解释。"
            "所有自然语言内容必须使用简体中文，"
            "image_prompt 与 video_prompt 必须全中文（字段名保持英文）。"
        )
        parts = [
            {"text": architect_prompt},
            {"text": prompt},
            {"text": json.dumps(character_bank, ensure_ascii=False)},
            {"text": json.dumps(aligned_storyboard, ensure_ascii=False)},
        ]
        payload = self._generate_json(parts=parts, model=model, retry_max=retry_max)

        if isinstance(payload, dict) and isinstance(payload.get("prompts"), list):
            prompts = []
            for item in payload["prompts"]:
                prompts.append(
                    {
                        "shot_id": int(item.get("shot_id", 0)),
                        "image_prompt": str(item.get("image_prompt", "")).strip(),
                        "video_prompt": str(item.get("video_prompt", "")).strip(),
                    }
                )
            return {
                "project_id": str(payload.get("project_id", aligned_storyboard.get("project_id", ""))),
                "prompts": prompts,
            }

        raise Stage1Error("vlm_invalid_output", "Gemini 返回的最终提示词表格式无效", 502)

    def _generate_json(self, parts: list[dict[str, Any]], model: str, retry_max: int) -> Any:
        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseMimeType": "application/json"},
        }

        last_error: Exception | None = None
        for _ in range(retry_max + 1):
            try:
                data = self._post_generate_content(model=model, body=body)
                text = self._extract_text(data)
                return self._parse_model_json(text)
            except Exception as exc:  # noqa: BLE001
                last_error = exc

        raise Stage1Error("gemini_failed", f"Gemini 调用失败: {last_error}", 502)

    def _post_generate_content(self, model: str, body: dict[str, Any]) -> dict[str, Any]:
        self._ensure_api_key()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
        req = urllib.request.Request(
            url=url,
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise Stage1Error("gemini_http_error", f"Gemini HTTP 错误: {detail}", 502) from exc
        except urllib.error.URLError as exc:
            raise Stage1Error("gemini_network_error", f"Gemini 网络错误: {exc}", 502) from exc

    def _extract_text(self, payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates") or []
        if not candidates:
            raise Stage1Error("gemini_empty", "Gemini 未返回候选结果", 502)

        parts = (
            candidates[0]
            .get("content", {})
            .get("parts", [])
        )
        text_chunks = [str(part.get("text", "")) for part in parts if part.get("text")]
        merged = "\n".join(text_chunks).strip()
        if not merged:
            raise Stage1Error("gemini_empty", "Gemini 返回内容为空", 502)
        return merged

    def _parse_model_json(self, text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
            if not match:
                raise
            return json.loads(match.group(1))

    def _to_inline_data(self, image_path: Path) -> dict[str, Any]:
        raw = image_path.read_bytes()
        encoded = base64.b64encode(raw).decode("ascii")
        return {"mime_type": "image/jpeg", "data": encoded}

    def _normalize_scene_description(self, item: dict[str, Any]) -> dict[str, Any]:
        subjects = []
        for idx, subject in enumerate(item.get("visual_analysis", {}).get("subjects", []), start=1):
            subjects.append(
                {
                    "temp_id": str(subject.get("temp_id") or f"subject_{idx}"),
                    "appearance": _ensure_gender_prefix(str(subject.get("appearance", "")).strip()),
                    "action": str(subject.get("action", "")).strip(),
                    "expression": str(subject.get("expression", "")).strip(),
                }
            )

        env = item.get("visual_analysis", {}).get("environment", {})
        camera = item.get("visual_analysis", {}).get("camera", {})

        return {
            "scene_id": self._parse_scene_id(item.get("scene_id", 0)),
            "visual_analysis": {
                "subjects": subjects,
                "environment": {
                    "location": str(env.get("location", "")).strip(),
                    "lighting": str(env.get("lighting", "")).strip(),
                    "atmosphere": str(env.get("atmosphere", "")).strip(),
                },
                "camera": {
                    "shot_size": str(camera.get("shot_size", "")).strip(),
                    "angle": str(camera.get("angle", "")).strip(),
                    "movement": str(camera.get("movement", "")).strip(),
                },
            },
        }

    def _parse_scene_id(self, raw: Any) -> int:
        if isinstance(raw, int):
            return raw
        text = str(raw).strip()
        if text.isdigit():
            return int(text)
        match = re.search(r"(\d+)", text)
        if match:
            return int(match.group(1))
        return 0

    def _normalize_character(self, item: dict[str, Any]) -> dict[str, Any]:
        raw_features = item.get("key_features") or []
        features = [str(x).strip() for x in raw_features if str(x).strip()]
        return {
            "ref_id": str(item.get("ref_id", "")).strip(),
            "name": str(item.get("name", "")).strip(),
            "master_description": str(item.get("master_description", "")).strip(),
            "key_features": features,
            "ref_image_path": str(item.get("ref_image_path", "")).strip(),
        }

    def _ensure_api_key(self) -> None:
        if not self.api_key:
            raise Stage1Error("config_error", "缺少 GEMINI_API_KEY 配置", 400)

    def ensure_ready(self) -> None:
        self._ensure_api_key()


class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key.strip()

    def embed_texts(self, texts: list[str], model: str, retry_max: int) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_api_key()

        vectors: list[list[float]] = []
        for text in texts:
            body = {
                "content": {
                    "parts": [{"text": text}],
                }
            }

            last_error: Exception | None = None
            for _ in range(retry_max + 1):
                try:
                    payload = self._post_embed_content(model=model, body=body)
                    values = payload.get("embedding", {}).get("values") or []
                    if not values:
                        raise Stage1Error("embedding_empty", "Embedding 返回为空", 502)
                    vectors.append([float(x) for x in values])
                    last_error = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
            if last_error is not None:
                raise Stage1Error("embedding_failed", f"Embedding 调用失败: {last_error}", 502)

        return vectors

    def _post_embed_content(self, model: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent?key={self.api_key}"
        req = urllib.request.Request(
            url=url,
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise Stage1Error("embedding_http_error", f"Embedding HTTP 错误: {detail}", 502) from exc
        except urllib.error.URLError as exc:
            raise Stage1Error("embedding_network_error", f"Embedding 网络错误: {exc}", 502) from exc

    def _ensure_api_key(self) -> None:
        if not self.api_key:
            raise Stage1Error("config_error", "缺少 GEMINI_API_KEY 配置", 400)

    def ensure_ready(self) -> None:
        self._ensure_api_key()


class QwenVLMProvider(VLMProvider):
    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url.strip()

    def describe_scenes(
        self,
        scene_inputs: list[dict[str, Any]],
        model: str,
        retry_max: int,
    ) -> list[dict[str, Any]]:
        if not scene_inputs:
            return []

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                    "text": SCENE_DESCRIBE_PROMPT,
                }
            ]
        for item in scene_inputs:
            scene_id = int(item["scene_id"])
            keyframe_path = Path(item["keyframe_path"])
            content.append({"type": "text", "text": f"Scene {scene_id}"})
            content.append({"type": "image_url", "image_url": {"url": self._to_data_uri(keyframe_path)}})

        messages = [{"role": "user", "content": content}]
        payload = self._chat_completion_json(messages=messages, model=model, retry_max=retry_max)

        if isinstance(payload, dict) and isinstance(payload.get("scenes"), list):
            return [self._normalize_scene_description(x) for x in payload["scenes"]]
        if isinstance(payload, list):
            return [self._normalize_scene_description(x) for x in payload]
        raise Stage1Error("vlm_invalid_output", "Qwen 返回的分镜描述格式无效", 502)

    def review_character_candidates(
        self,
        candidates: list[dict[str, Any]],
        model: str,
        retry_max: int,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "你是角色归一化审核器。请只输出 JSON 对象，结构："
                            "{\"characters\":[{ref_id,name,master_description,key_features,ref_image_path}]}"
                            "。若候选角色可合并，请输出合并后的最终列表。"
                        ),
                    },
                    {"type": "text", "text": json.dumps(candidates, ensure_ascii=False)},
                ],
            }
        ]
        payload = self._chat_completion_json(messages=messages, model=model, retry_max=retry_max)
        if isinstance(payload, dict) and isinstance(payload.get("characters"), list):
            return [self._normalize_character(item) for item in payload["characters"]]
        if isinstance(payload, list):
            return [self._normalize_character(item) for item in payload]
        return [self._normalize_character(item) for item in candidates]

    def generate_production_table(
        self,
        character_bank: dict[str, Any],
        aligned_storyboard: dict[str, Any],
        architect_prompt: str,
        model: str,
        retry_max: int,
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": architect_prompt},
                    {
                        "type": "text",
                        "text": (
                            "请严格按系统规范输出 JSON："
                            "{\"project_id\":\"string\",\"prompts\":[{\"shot_id\":number,\"image_prompt\":\"string\",\"video_prompt\":\"string\"}]}"
                            "。所有自然语言必须使用简体中文，不要输出 markdown，不要解释。"
                        ),
                    },
                    {"type": "text", "text": json.dumps(character_bank, ensure_ascii=False)},
                    {"type": "text", "text": json.dumps(aligned_storyboard, ensure_ascii=False)},
                ],
            }
        ]
        payload = self._chat_completion_json(messages=messages, model=model, retry_max=retry_max)
        if isinstance(payload, dict) and isinstance(payload.get("prompts"), list):
            prompts = []
            for item in payload["prompts"]:
                prompts.append(
                    {
                        "shot_id": int(item.get("shot_id", 0)),
                        "image_prompt": str(item.get("image_prompt", "")).strip(),
                        "video_prompt": str(item.get("video_prompt", "")).strip(),
                    }
                )
            return {
                "project_id": str(payload.get("project_id", aligned_storyboard.get("project_id", ""))),
                "prompts": prompts,
            }
        raise Stage1Error("vlm_invalid_output", "Qwen 返回的最终提示词表格式无效", 502)

    def ensure_ready(self) -> None:
        if not self.api_key:
            raise Stage1Error("config_error", "缺少 QWEN_API_KEY 配置", 400)
        if not self.base_url:
            raise Stage1Error("config_error", "缺少 QWEN_BASE_URL 配置", 400)

    def _chat_completion_json(
        self,
        messages: list[dict[str, Any]],
        model: str,
        retry_max: int,
    ) -> Any:
        self.ensure_ready()
        body = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }

        last_error: Exception | None = None
        for _ in range(retry_max + 1):
            try:
                payload = self._post_json(self.base_url, body)
                text = self._extract_text(payload)
                return self._parse_model_json(text)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise Stage1Error("qwen_failed", f"Qwen 调用失败: {last_error}", 502)

    def _post_json(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            url=url,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:  # noqa: S310
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise Stage1Error("qwen_http_error", f"Qwen HTTP 错误: {detail}", 502) from exc
        except urllib.error.URLError as exc:
            raise Stage1Error("qwen_network_error", f"Qwen 网络错误: {exc}", 502) from exc

    def _extract_text(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            raise Stage1Error("qwen_empty", "Qwen 未返回候选结果", 502)

        content = choices[0].get("message", {}).get("content")
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    chunks.append(str(item["text"]))
            text = "\n".join(chunks).strip()
        else:
            text = ""
        if not text:
            raise Stage1Error("qwen_empty", "Qwen 返回内容为空", 502)
        return text

    def _parse_model_json(self, text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
            if not match:
                raise
            return json.loads(match.group(1))

    def _to_data_uri(self, image_path: Path) -> str:
        raw = image_path.read_bytes()
        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _normalize_scene_description(self, item: dict[str, Any]) -> dict[str, Any]:
        subjects = []
        for idx, subject in enumerate(item.get("visual_analysis", {}).get("subjects", []), start=1):
            subjects.append(
                {
                    "temp_id": str(subject.get("temp_id") or f"subject_{idx}"),
                    "appearance": _ensure_gender_prefix(str(subject.get("appearance", "")).strip()),
                    "action": str(subject.get("action", "")).strip(),
                    "expression": str(subject.get("expression", "")).strip(),
                }
            )
        env = item.get("visual_analysis", {}).get("environment", {})
        camera = item.get("visual_analysis", {}).get("camera", {})

        return {
            "scene_id": self._parse_scene_id(item.get("scene_id", 0)),
            "visual_analysis": {
                "subjects": subjects,
                "environment": {
                    "location": str(env.get("location", "")).strip(),
                    "lighting": str(env.get("lighting", "")).strip(),
                    "atmosphere": str(env.get("atmosphere", "")).strip(),
                },
                "camera": {
                    "shot_size": str(camera.get("shot_size", "")).strip(),
                    "angle": str(camera.get("angle", "")).strip(),
                    "movement": str(camera.get("movement", "")).strip(),
                },
            },
        }

    def _normalize_character(self, item: dict[str, Any]) -> dict[str, Any]:
        raw_features = item.get("key_features") or []
        features = [str(x).strip() for x in raw_features if str(x).strip()]
        return {
            "ref_id": str(item.get("ref_id", "")).strip(),
            "name": str(item.get("name", "")).strip(),
            "master_description": str(item.get("master_description", "")).strip(),
            "key_features": features,
            "ref_image_path": str(item.get("ref_image_path", "")).strip(),
        }

    def _parse_scene_id(self, raw: Any) -> int:
        if isinstance(raw, int):
            return raw
        text = str(raw).strip()
        if text.isdigit():
            return int(text)
        match = re.search(r"(\d+)", text)
        if match:
            return int(match.group(1))
        return 0


class QwenEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url.strip()

    def ensure_ready(self) -> None:
        if not self.api_key:
            raise Stage1Error("config_error", "缺少 QWEN_API_KEY 配置", 400)
        if not self.base_url:
            raise Stage1Error("config_error", "缺少 QWEN_EMBED_BASE_URL 配置", 400)

    def embed_texts(self, texts: list[str], model: str, retry_max: int) -> list[list[float]]:
        if not texts:
            return []
        self.ensure_ready()
        vectors: list[list[float]] = []

        last_error: Exception | None = None
        for text in texts:
            body = {
                "model": model,
                "input": text,
            }
            for _ in range(retry_max + 1):
                try:
                    payload = self._post_json(self.base_url, body)
                    vectors.append(self._extract_embedding(payload))
                    last_error = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
            if last_error is not None:
                raise Stage1Error("embedding_failed", f"Qwen Embedding 调用失败: {last_error}", 502)

        return vectors

    def _post_json(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            url=url,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise Stage1Error("embedding_http_error", f"Qwen Embedding HTTP 错误: {detail}", 502) from exc
        except urllib.error.URLError as exc:
            raise Stage1Error("embedding_network_error", f"Qwen Embedding 网络错误: {exc}", 502) from exc

    def _extract_embedding(self, payload: dict[str, Any]) -> list[float]:
        data = payload.get("data") or []
        if not data:
            raise Stage1Error("embedding_empty", "Qwen Embedding 返回为空", 502)
        vec = data[0].get("embedding") if isinstance(data[0], dict) else None
        if not vec:
            raise Stage1Error("embedding_empty", "Qwen Embedding 返回为空", 502)
        return [float(x) for x in vec]
