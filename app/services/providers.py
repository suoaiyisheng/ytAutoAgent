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
你是一位专业的视觉结构化分析专家，负责将图像描述转化为原始分镜描述 JSON 数据。

# Output Format
请仅输出一个合法的 JSON 对象，严禁包含任何 Markdown 格式块、解释性文字或引言。

# JSON Structure
{
  "scenes": [
    {
      "scene_id": "场景序号",
      "subjects": [
        {
          "subject_id": "subject_1",
          "appearance": "性别(男性/女性/人物)，体型(偏胖/匀称/纤细/健壮肌肉)，外貌细节与服装描述"
        }
      ],
      "desc": "subject_1 在什么地方正在做什么"
    }
  ]
}

# Constraint Rules
1. **语言限制**：除 `subject_id` 外，所有字段严禁出现英文，必须使用简体中文。
2. **主体编号**：`subject_id` 必须使用 `subject_1`、`subject_2` 这类格式，并与 `desc` 中引用完全一致。
3. **性别必填**：`subjects.appearance` 必须以“男性”或“女性”开头，无法判断则使用“人物”。
4. **体型必填**：`subjects.appearance` 必须体现体型，至少从 [偏胖、匀称、纤细、健壮肌肉] 中选择其一。
5. **描述约束**：`desc` 只能描述“subject_x 在什么地方正在做什么”，不要写镜头语言，不要写 `ref_id`。
6. **空值处理**：若画面无人，`subjects` 必须设为空数组 []，`desc` 设为空字符串。"""


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
        stage5_input: dict[str, Any],
        architect_prompt: str,
        model: str,
        retry_max: int,
        debug_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str], model: str, retry_max: int) -> list[list[float]]:
        raise NotImplementedError


class GeminiVLMProvider(VLMProvider):
    def __init__(
        self,
        api_key: str,
        openrouter_api_key: str = "",
        openrouter_base_url: str = "",
    ) -> None:
        self.api_key = api_key.strip()
        self.openrouter_api_key = openrouter_api_key.strip()
        self.openrouter_base_url = openrouter_base_url.strip()

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
        stage5_input: dict[str, Any],
        architect_prompt: str,
        model: str,
        retry_max: int,
        debug_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        stage5_protocol_prompt = (
            "请严格按照系统规范输出最终 JSON。结构："
            "{project_id, prompts:[{shot_id,reference_bindings:[{reference_index,ref_id}],image_prompt,video_prompt}]}。"
            "不要输出 markdown，不要解释。"
            "所有自然语言内容必须使用简体中文，"
            "image_prompt 与 video_prompt 必须全中文（字段名保持英文）。"
            "输入 shots 中每个 shot 都有 reference_bindings，必须原样保留到对应 prompts[].reference_bindings。"
            "在 image_prompt 中引用角色时，必须写“参考图{reference_index}”，不得省略参考图编号。"
            "输入还包含 reference_catalog（ref_id到参考图路径映射）。"
            "你必须以 shots[].desc 为唯一场景事实来源，并以 reference_catalog 和 character_bank 为唯一身份事实来源来写提示词，禁止杜撰新增外貌、服装、配饰。"
            "你必须逐镜头、逐角色保留 shots[].desc 中已经出现的具体外貌与动作事实，包括但不限于性别、体型、发型/发色、衣物、鱼尾、裸露特征、配饰、手持道具、相对位置、地点和动作。"
            "若 shots[].desc 中已经写出具体特征，禁止在 image_prompt 或 video_prompt 中把这些特征压缩成模糊代称，例如“女孩”“男人”“角色”，除非同时保留关键特征短语。"
            "若某个角色在当前镜头的 desc 中带有多项外貌特征，image_prompt 必须尽量完整覆盖这些特征，不能只保留其中一两项。"
            "video_prompt 必须延续同一镜头 desc 中的角色特征和动作逻辑，不得只写抽象情绪变化而丢掉角色识别特征。"
            "如果提示词遗漏了 desc 中的关键外貌、服装、道具或动作事实，这视为错误。"
            "每个参考图编号必须严格对应 shots[].reference_bindings[].reference_index。"
            "你必须读取 reference_catalog 中全部 Ref_n 的身份定义与参考图，不得只关注单个 Ref。"
            "reference_catalog 仅用于编号与ref映射，不要在提示词中输出文件路径。"
        )
        reference_parts = self._build_reference_image_parts(stage5_input)
        parts = [
            {"text": stage5_protocol_prompt},
            {"text": json.dumps(stage5_input, ensure_ascii=False)},
            *reference_parts,
        ]
        if isinstance(debug_context, dict):
            debug_context["architect_prompt"] = architect_prompt
            debug_context["stage5_protocol_prompt"] = stage5_protocol_prompt
            debug_context["character_bank"] = json.loads(json.dumps(character_bank, ensure_ascii=False))
            debug_context["stage5_input"] = json.loads(json.dumps(stage5_input, ensure_ascii=False))
            if self.is_using_openrouter():
                debug_context["provider_request"] = {
                    "provider": "gemini_openrouter",
                    "model": self._normalize_openrouter_model(model),
                    "retry_max": retry_max,
                    "body": {
                        "model": self._normalize_openrouter_model(model),
                        "messages": self._build_openrouter_messages(parts=parts, system_instruction=architect_prompt),
                        "temperature": 0.2,
                        "response_format": {"type": "json_object"},
                    },
                }
            else:
                debug_context["provider_request"] = {
                    "provider": "gemini",
                    "model": model,
                    "retry_max": retry_max,
                    "body": {
                        "contents": [{"role": "user", "parts": parts}],
                        "systemInstruction": {"parts": [{"text": architect_prompt}]},
                        "generationConfig": {"responseMimeType": "application/json"},
                    },
                }
        payload = self._generate_json(parts=parts, model=model, retry_max=retry_max, system_instruction=architect_prompt)
        if isinstance(debug_context, dict):
            debug_context["provider_raw_output"] = json.loads(json.dumps(payload, ensure_ascii=False))

        if isinstance(payload, dict) and isinstance(payload.get("prompts"), list):
            prompts = []
            for item in payload["prompts"]:
                prompts.append(
                    {
                        "shot_id": int(item.get("shot_id", 0)),
                        "reference_bindings": item.get("reference_bindings", []),
                        "image_prompt": str(item.get("image_prompt", "")).strip(),
                        "video_prompt": str(item.get("video_prompt", "")).strip(),
                    }
                )
            return {
                "project_id": str(payload.get("project_id", stage5_input.get("project_id", ""))),
                "prompts": prompts,
            }

        raise Stage1Error("vlm_invalid_output", "Gemini 返回的最终提示词表格式无效", 502)

    def _build_reference_image_parts(self, stage5_input: dict[str, Any]) -> list[dict[str, Any]]:
        reference_catalog = stage5_input.get("reference_catalog")
        if not isinstance(reference_catalog, list):
            return []

        parts: list[dict[str, Any]] = []
        for idx, item in enumerate(reference_catalog, start=1):
            if not isinstance(item, dict):
                continue
            ref_id = str(item.get("ref_id", "")).strip()
            if not ref_id:
                continue
            parts.append({"text": f"参考图{idx} 对应 {ref_id}。"})

            ref_image_path = str(item.get("ref_image_path", "")).strip()
            if ref_image_path:
                try:
                    path = Path(ref_image_path).expanduser()
                    if path.exists() and path.is_file():
                        parts.append({"inline_data": self._to_inline_data(path)})
                        continue
                except Exception:  # noqa: BLE001
                    pass

            ref_image_url = str(item.get("ref_image_url", "")).strip()
            if ref_image_url:
                parts.append({"text": f"{ref_id} 参考图URL: {ref_image_url}"})
        return parts

    def _generate_json(
        self,
        parts: list[dict[str, Any]],
        model: str,
        retry_max: int,
        system_instruction: str = "",
    ) -> Any:
        if self.is_using_openrouter():
            return self._generate_json_via_openrouter(
                parts=parts,
                model=model,
                retry_max=retry_max,
                system_instruction=system_instruction,
            )

        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseMimeType": "application/json"},
        }
        if system_instruction.strip():
            body["systemInstruction"] = {
                "parts": [{"text": system_instruction}],
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

    def _generate_json_via_openrouter(
        self,
        parts: list[dict[str, Any]],
        model: str,
        retry_max: int,
        system_instruction: str = "",
    ) -> Any:
        self._ensure_openrouter_ready()
        body = {
            "model": self._normalize_openrouter_model(model),
            "messages": self._build_openrouter_messages(parts=parts, system_instruction=system_instruction),
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }

        last_error: Exception | None = None
        for _ in range(retry_max + 1):
            try:
                data = self._post_openrouter_chat_completion(body=body)
                text = self._extract_openrouter_text(data)
                return self._parse_model_json(text)
            except Exception as exc:  # noqa: BLE001
                last_error = exc

        raise Stage1Error("gemini_failed", f"Gemini(OpenRouter) 调用失败: {last_error}", 502)

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

    def _post_openrouter_chat_completion(self, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            url=self._openrouter_chat_url(),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openrouter_api_key}",
            },
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:  # noqa: S310
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise Stage1Error("gemini_http_error", f"Gemini(OpenRouter) HTTP 错误: {detail}", 502) from exc
        except urllib.error.URLError as exc:
            raise Stage1Error("gemini_network_error", f"Gemini(OpenRouter) 网络错误: {exc}", 502) from exc

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

    def _build_openrouter_messages(
        self,
        *,
        parts: list[dict[str, Any]],
        system_instruction: str = "",
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_instruction.strip():
            messages.append({"role": "system", "content": system_instruction})
        messages.append(
            {
                "role": "user",
                "content": self._parts_to_openrouter_content(parts),
            }
        )
        return messages

    def _parts_to_openrouter_content(self, parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = str(part.get("text", "")).strip()
            if text:
                content.append({"type": "text", "text": text})
                continue

            inline_data = part.get("inline_data")
            if not isinstance(inline_data, dict):
                continue
            mime_type = str(inline_data.get("mime_type", "")).strip() or "image/jpeg"
            data = str(inline_data.get("data", "")).strip()
            if not data:
                continue
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{data}"},
                }
            )
        return content

    def _extract_openrouter_text(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            raise Stage1Error("gemini_empty", "Gemini(OpenRouter) 未返回候选结果", 502)

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
            raise Stage1Error("gemini_empty", "Gemini(OpenRouter) 返回内容为空", 502)
        return text

    def _normalize_openrouter_model(self, model: str) -> str:
        name = model.strip()
        if "/" in name or not name:
            return name
        if name.startswith("gemini"):
            return f"google/{name}"
        return name

    def _openrouter_chat_url(self) -> str:
        base = self.openrouter_base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def _normalize_scene_description(self, item: dict[str, Any]) -> dict[str, Any]:
        raw_subjects = item.get("subjects")
        desc = str(item.get("desc", "")).strip()
        if isinstance(raw_subjects, list):
            subjects = []
            for idx, subject in enumerate(raw_subjects, start=1):
                subjects.append(
                    {
                        "subject_id": str(subject.get("subject_id") or subject.get("temp_id") or f"subject_{idx}"),
                        "appearance": _ensure_gender_prefix(str(subject.get("appearance", "")).strip()),
                    }
                )
            return {
                "scene_id": self._parse_scene_id(item.get("scene_id", 0)),
                "subjects": subjects,
                "desc": desc,
            }

        subjects = []
        visual_analysis = item.get("visual_analysis", {})
        for idx, subject in enumerate(visual_analysis.get("subjects", []), start=1):
            subject_id = str(subject.get("temp_id") or f"subject_{idx}")
            subjects.append(
                {
                    "subject_id": subject_id,
                    "appearance": _ensure_gender_prefix(str(subject.get("appearance", "")).strip()),
                }
            )

        env = visual_analysis.get("environment", {})
        location = str(env.get("location", "")).strip() or "画面中"
        if not desc:
            clauses = []
            for idx, subject in enumerate(visual_analysis.get("subjects", []), start=1):
                subject_id = str(subject.get("temp_id") or f"subject_{idx}")
                action = str(subject.get("action", "")).strip() or "停留"
                clauses.append(f"{subject_id} 在{location}正在{action}")
            desc = "；".join(clauses) + ("。" if clauses else "")

        return {
            "scene_id": self._parse_scene_id(item.get("scene_id", 0)),
            "subjects": subjects,
            "desc": desc,
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

    def _ensure_openrouter_ready(self) -> None:
        if not self.openrouter_api_key:
            raise Stage1Error("config_error", "缺少 OPENROUTER_API_KEY 配置，无法通过 OpenRouter 调用 Gemini", 400)
        if not self.openrouter_base_url:
            raise Stage1Error("config_error", "缺少 OPENROUTER_BASE_URL 配置，无法通过 OpenRouter 调用 Gemini", 400)

    def is_using_openrouter(self) -> bool:
        return bool(self.openrouter_api_key and self.openrouter_base_url)

    def ensure_ready(self) -> None:
        if self.is_using_openrouter():
            self._ensure_openrouter_ready()
            return
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
        stage5_input: dict[str, Any],
        architect_prompt: str,
        model: str,
        retry_max: int,
        debug_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        stage5_protocol_prompt = (
            "请严格按系统规范输出 JSON："
            "{\"project_id\":\"string\",\"prompts\":[{\"shot_id\":number,\"reference_bindings\":[{\"reference_index\":number,\"ref_id\":\"string\"}],\"image_prompt\":\"string\",\"video_prompt\":\"string\"}]}"
            "。所有自然语言必须使用简体中文，不要输出 markdown，不要解释。"
            "输入 shots 中每个 shot 都有 reference_bindings，必须原样保留到对应 prompts[].reference_bindings。"
            "在 image_prompt 中引用角色时，必须写“参考图{reference_index}”，不得省略参考图编号。"
            "输入还包含 reference_catalog（ref_id到参考图路径映射）。"
            "你必须以 shots[].desc 为唯一场景事实来源，并以 reference_catalog 和 character_bank 为唯一身份事实来源来写提示词，禁止杜撰新增外貌、服装、配饰。"
            "你必须逐镜头、逐角色保留 shots[].desc 中已经出现的具体外貌与动作事实，包括但不限于性别、体型、发型/发色、衣物、鱼尾、裸露特征、配饰、手持道具、相对位置、地点和动作。"
            "若 shots[].desc 中已经写出具体特征，禁止在 image_prompt 或 video_prompt 中把这些特征压缩成模糊代称，例如“女孩”“男人”“角色”，除非同时保留关键特征短语。"
            "若某个角色在当前镜头的 desc 中带有多项外貌特征，image_prompt 必须尽量完整覆盖这些特征，不能只保留其中一两项。"
            "video_prompt 必须延续同一镜头 desc 中的角色特征和动作逻辑，不得只写抽象情绪变化而丢掉角色识别特征。"
            "如果提示词遗漏了 desc 中的关键外貌、服装、道具或动作事实，这视为错误。"
            "每个参考图编号必须严格对应 shots[].reference_bindings[].reference_index。"
            "你必须读取 reference_catalog 中全部 Ref_n 的身份定义与参考图，不得只关注单个 Ref。"
            "reference_catalog 仅用于编号与ref映射，不要在提示词中输出文件路径。"
        )
        reference_contents = self._build_reference_image_contents(stage5_input)
        messages = [
            {
                "role": "system",
                "content": architect_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": stage5_protocol_prompt,
                    },
                    {"type": "text", "text": json.dumps(stage5_input, ensure_ascii=False)},
                    *reference_contents,
                ],
            }
        ]
        if isinstance(debug_context, dict):
            debug_context["architect_prompt"] = architect_prompt
            debug_context["stage5_protocol_prompt"] = stage5_protocol_prompt
            debug_context["character_bank"] = json.loads(json.dumps(character_bank, ensure_ascii=False))
            debug_context["stage5_input"] = json.loads(json.dumps(stage5_input, ensure_ascii=False))
            debug_context["provider_request"] = {
                "provider": "qwen",
                "model": model,
                "retry_max": retry_max,
                "body": {
                    "model": model,
                    "messages": messages,
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"},
                },
            }
        payload = self._chat_completion_json(messages=messages, model=model, retry_max=retry_max)
        if isinstance(debug_context, dict):
            debug_context["provider_raw_output"] = json.loads(json.dumps(payload, ensure_ascii=False))
        if isinstance(payload, dict) and isinstance(payload.get("prompts"), list):
            prompts = []
            for item in payload["prompts"]:
                prompts.append(
                    {
                        "shot_id": int(item.get("shot_id", 0)),
                        "reference_bindings": item.get("reference_bindings", []),
                        "image_prompt": str(item.get("image_prompt", "")).strip(),
                        "video_prompt": str(item.get("video_prompt", "")).strip(),
                    }
                )
            return {
                "project_id": str(payload.get("project_id", stage5_input.get("project_id", ""))),
                "prompts": prompts,
            }
        raise Stage1Error("vlm_invalid_output", "Qwen 返回的最终提示词表格式无效", 502)

    def _build_reference_image_contents(self, stage5_input: dict[str, Any]) -> list[dict[str, Any]]:
        reference_catalog = stage5_input.get("reference_catalog")
        if not isinstance(reference_catalog, list):
            return []

        content: list[dict[str, Any]] = []
        for idx, item in enumerate(reference_catalog, start=1):
            if not isinstance(item, dict):
                continue
            ref_id = str(item.get("ref_id", "")).strip()
            if not ref_id:
                continue
            content.append({"type": "text", "text": f"参考图{idx} 对应 {ref_id}。"})

            ref_image_url = str(item.get("ref_image_url", "")).strip()
            if ref_image_url:
                content.append({"type": "image_url", "image_url": {"url": ref_image_url}})
                continue

            ref_image_path = str(item.get("ref_image_path", "")).strip()
            if not ref_image_path:
                continue
            try:
                path = Path(ref_image_path).expanduser()
                if path.exists() and path.is_file():
                    content.append({"type": "image_url", "image_url": {"url": self._to_data_uri(path)}})
            except Exception:  # noqa: BLE001
                continue
        return content

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
        raw_subjects = item.get("subjects")
        desc = str(item.get("desc", "")).strip()
        if isinstance(raw_subjects, list):
            subjects = []
            for idx, subject in enumerate(raw_subjects, start=1):
                subjects.append(
                    {
                        "subject_id": str(subject.get("subject_id") or subject.get("temp_id") or f"subject_{idx}"),
                        "appearance": _ensure_gender_prefix(str(subject.get("appearance", "")).strip()),
                    }
                )
            return {
                "scene_id": self._parse_scene_id(item.get("scene_id", 0)),
                "subjects": subjects,
                "desc": desc,
            }

        subjects = []
        visual_analysis = item.get("visual_analysis", {})
        for idx, subject in enumerate(visual_analysis.get("subjects", []), start=1):
            subject_id = str(subject.get("temp_id") or f"subject_{idx}")
            subjects.append(
                {
                    "subject_id": subject_id,
                    "appearance": _ensure_gender_prefix(str(subject.get("appearance", "")).strip()),
                }
            )
        env = visual_analysis.get("environment", {})
        location = str(env.get("location", "")).strip() or "画面中"
        if not desc:
            clauses = []
            for idx, subject in enumerate(visual_analysis.get("subjects", []), start=1):
                subject_id = str(subject.get("temp_id") or f"subject_{idx}")
                action = str(subject.get("action", "")).strip() or "停留"
                clauses.append(f"{subject_id} 在{location}正在{action}")
            desc = "；".join(clauses) + ("。" if clauses else "")

        return {
            "scene_id": self._parse_scene_id(item.get("scene_id", 0)),
            "subjects": subjects,
            "desc": desc,
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
