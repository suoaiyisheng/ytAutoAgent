from __future__ import annotations

import base64
import hashlib
import hmac
import json
import mimetypes
import re
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.config import Settings
from app.errors import Stage1Error
from app.services.store import TaskStore

ProgressCallback = Callable[[float, str], None]


class OpenRouterHTTPError(Exception):
    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = message
        super().__init__(f"openrouter_http_{status}: {message}")


class VolcengineHTTPError(Exception):
    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = message
        super().__init__(f"volcengine_http_{status}: {message}")


def parse_shot_range(shot_range: str | None, available_shots: list[int]) -> list[int]:
    sorted_available = sorted(set(int(x) for x in available_shots))
    if not sorted_available:
        return []
    if not shot_range:
        return sorted_available

    selected: set[int] = set()
    chunks = [x.strip() for x in shot_range.split(",") if x.strip()]
    if not chunks:
        raise Stage1Error("invalid_shot_range", "shot_range 不能为空字符串", 422)

    for chunk in chunks:
        if "-" in chunk:
            parts = chunk.split("-", 1)
            if len(parts) != 2:
                raise Stage1Error("invalid_shot_range", f"shot_range 片段无效: {chunk}", 422)
            try:
                start = int(parts[0].strip())
                end = int(parts[1].strip())
            except ValueError as exc:
                raise Stage1Error("invalid_shot_range", f"shot_range 片段无效: {chunk}", 422) from exc
            if start <= 0 or end <= 0 or start > end:
                raise Stage1Error("invalid_shot_range", f"shot_range 片段无效: {chunk}", 422)
            for sid in range(start, end + 1):
                selected.add(sid)
            continue
        try:
            selected.add(int(chunk))
        except ValueError as exc:
            raise Stage1Error("invalid_shot_range", f"shot_range 片段无效: {chunk}", 422) from exc

    filtered = sorted(x for x in selected if x in sorted_available)
    if not filtered:
        available_text = ",".join(str(x) for x in sorted_available)
        raise Stage1Error("invalid_shot_range", f"shot_range 无有效镜头，可用镜头: {available_text}", 422)
    return filtered


class AliyunOSSUploader:
    def __init__(
        self,
        *,
        region: str,
        access_key_id: str,
        access_key_secret: str,
        bucket: str,
        public_domain: str = "",
    ) -> None:
        self.region = region.strip()
        self.access_key_id = access_key_id.strip()
        self.access_key_secret = access_key_secret.strip()
        self.bucket = bucket.strip()
        self.public_domain = public_domain.strip().rstrip("/")

    def endpoint(self) -> str:
        return f"https://{self.bucket}.{self.region}.aliyuncs.com"

    def public_url(self, key: str) -> str:
        if self.public_domain:
            return f"{self.public_domain}/{key}"
        return f"{self.endpoint()}/{key}"

    def build_signature(self, method: str, content_type: str, date_text: str, resource: str) -> str:
        plain = f"{method}\n\n{content_type}\n{date_text}\n{resource}"
        return base64.b64encode(
            hmac.new(self.access_key_secret.encode("utf-8"), plain.encode("utf-8"), hashlib.sha1).digest()
        ).decode("utf-8")

    def upload_file(self, *, local_path: Path, key: str, content_type: str, timeout_sec: int = 60) -> str:
        if not local_path.exists() or not local_path.is_file():
            raise Stage1Error("ref_image_not_found", f"参考图不存在: {local_path}", 422)

        body = local_path.read_bytes()
        date_text = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
        resource = f"/{self.bucket}/{key}"
        signature = self.build_signature("PUT", content_type, date_text, resource)
        url = f"{self.endpoint()}/{key}"

        request = urllib.request.Request(
            url=url,
            data=body,
            method="PUT",
            headers={
                "Content-Type": content_type,
                "Date": date_text,
                "Authorization": f"OSS {self.access_key_id}:{signature}",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:  # noqa: S310
                status = int(getattr(response, "status", 200))
                if status < 200 or status >= 300:
                    raise Stage1Error("oss_upload_failed", f"OSS 上传失败: HTTP {status}", 502)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise Stage1Error("oss_upload_failed", f"OSS 上传失败: HTTP {exc.code} {detail}", 502) from exc
        except urllib.error.URLError as exc:
            raise Stage1Error("oss_upload_failed", f"OSS 上传网络错误: {exc}", 502) from exc
        return self.public_url(key)


class OpenRouterImageClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        add_watermark: bool = False,
        retry_max: int = 2,
        timeout_sec: int = 90,
    ) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.add_watermark = bool(add_watermark)
        self.retry_max = max(0, retry_max)
        self.timeout_sec = timeout_sec

    def _compact_prompt(self, prompt: str) -> str:
        # 避免超长提示词在部分模型上导致空响应，保留前部语义用于降级重试。
        compact = " ".join(prompt.split())
        if len(compact) <= 240:
            return compact
        return compact[:240].rstrip() + "。"

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        last_error: OpenRouterHTTPError | None = None
        for attempt in range(self.retry_max + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:  # noqa: S310
                    raw = response.read().decode("utf-8")
                    return json.loads(raw)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore")
                err = OpenRouterHTTPError(exc.code, detail or exc.reason or "http error")
                if exc.code == 429 or exc.code >= 500:
                    last_error = err
                    if attempt < self.retry_max:
                        time.sleep(2**attempt)
                        continue
                raise err from exc
            except urllib.error.URLError as exc:
                if attempt < self.retry_max:
                    time.sleep(2**attempt)
                    continue
                raise Stage1Error("openrouter_network_error", f"OpenRouter 网络错误: {exc}", 502) from exc
            except json.JSONDecodeError as exc:
                raise Stage1Error("openrouter_invalid_json", f"OpenRouter 返回非 JSON: {exc}", 502) from exc
        if last_error:
            raise last_error
        raise Stage1Error("openrouter_failed", "OpenRouter 调用失败", 502)

    def _parse_candidates(self, payload: dict[str, Any]) -> list[dict[str, str]]:
        # OpenRouter chat completions multimodal image output.
        choices = payload.get("choices")
        if isinstance(choices, list):
            out: list[dict[str, str]] = []
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message")
                if not isinstance(message, dict):
                    continue

                images = message.get("images")
                if isinstance(images, list):
                    for image in images:
                        if not isinstance(image, dict):
                            continue
                        image_url = image.get("image_url")
                        if isinstance(image_url, dict):
                            url = str(image_url.get("url", "")).strip()
                        else:
                            url = str(image_url or image.get("url") or "").strip()
                        if url:
                            out.append({"url": url, "b64_json": ""})

                content = message.get("content")
                if isinstance(content, list):
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        if str(part.get("type", "")).strip() != "image_url":
                            continue
                        image_url = part.get("image_url")
                        if isinstance(image_url, dict):
                            url = str(image_url.get("url", "")).strip()
                        else:
                            url = str(image_url or "").strip()
                        if url:
                            out.append({"url": url, "b64_json": ""})
            if out:
                return out

        data = payload.get("data")
        if data is None:
            data = payload.get("images")
        if data is None:
            data = payload.get("output")

        if isinstance(data, dict):
            raw_items = [data]
        elif isinstance(data, list):
            raw_items = data
        elif isinstance(data, str):
            raw_items = [{"url": data}]
        else:
            raw_items = []

        candidates: list[dict[str, str]] = []
        for item in raw_items:
            if isinstance(item, str):
                candidates.append({"url": item, "b64_json": ""})
                continue
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or item.get("image_url") or "").strip()
            b64 = str(item.get("b64_json") or item.get("b64") or "").strip()
            if not url and not b64:
                continue
            candidates.append({"url": url, "b64_json": b64})
        return candidates

    def _build_text_fallback_prompt(self, prompt: str, reference_urls: list[str]) -> str:
        refs_text = "\n".join(f"{idx + 1}. {url}" for idx, url in enumerate(reference_urls))
        return (
            f"{prompt}\n\n"
            "请保持角色与参考图一致。参考图链接如下：\n"
            f"{refs_text}"
        )

    def _build_chat_payload(
        self,
        *,
        prompt: str,
        aspect_ratio: str,
        reference_urls: list[str],
    ) -> dict[str, Any]:
        content = [{"type": "text", "text": prompt}]
        for url in reference_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
        return {
            "model": self.model,
            "modalities": ["image", "text"],
            "messages": [{"role": "user", "content": content}],
            "image_config": {
                "aspect_ratio": aspect_ratio,
                "add_watermark": self.add_watermark,
            },
        }

    def _request_with_image_compat(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._request(payload)
        except OpenRouterHTTPError as exc:
            low = exc.message.lower()
            if exc.status in {400, 422} and "image" in low and "model id" not in low:
                fallback_payload = dict(payload)
                fallback_payload.pop("image", None)
                fallback_payload.pop("image_config", None)
                return self._request(fallback_payload)
            raise

    def _request_with_model_fallback(self, payload: dict[str, Any]) -> dict[str, Any]:
        models = [self.model]
        fallback_model = "google/gemini-2.5-flash-image"
        if fallback_model not in models:
            models.append(fallback_model)

        for idx, model_name in enumerate(models):
            try_payload = dict(payload)
            try_payload["model"] = model_name
            try:
                return self._request_with_image_compat(try_payload)
            except OpenRouterHTTPError as exc:
                msg = exc.message.lower()
                is_model_invalid = (
                    exc.status in {400, 404, 422}
                    and "model" in msg
                    and ("invalid" in msg or "not found" in msg or "unknown" in msg or "unrecognized" in msg)
                )
                if is_model_invalid and idx < len(models) - 1:
                    continue
                raise
        raise Stage1Error("openrouter_failed", "OpenRouter 调用失败", 502)

    def _single_generate(
        self,
        *,
        prompt: str,
        aspect_ratio: str,
        reference_urls: list[str],
    ) -> dict[str, str]:
        native_payload = self._build_chat_payload(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            reference_urls=reference_urls,
        )
        try:
            response = self._request_with_model_fallback(native_payload)
            items = self._parse_candidates(response)
            if items:
                return items[0]
        except OpenRouterHTTPError as exc:
            # 模型不支持图参时降级到纯文本提示。
            if not reference_urls or exc.status not in {400, 404, 422}:
                msg = exc.message.strip().replace("\n", " ")
                raise Stage1Error("openrouter_http_error", f"OpenRouter 调用失败: HTTP {exc.status} {msg[:240]}", 502) from exc

        fallback_prompt = self._build_text_fallback_prompt(prompt, reference_urls) if reference_urls else prompt
        fallback_payload = self._build_chat_payload(
            prompt=fallback_prompt,
            aspect_ratio=aspect_ratio,
            reference_urls=[],
        )
        try:
            response = self._request_with_model_fallback(fallback_payload)
            items = self._parse_candidates(response)
        except OpenRouterHTTPError as exc:
            msg = exc.message.strip().replace("\n", " ")
            raise Stage1Error("openrouter_http_error", f"OpenRouter 调用失败: HTTP {exc.status} {msg[:240]}", 502) from exc
        if not items:
            raise Stage1Error("openrouter_empty", "OpenRouter 未返回任何图片", 502)
        return items[0]

    def generate_images(
        self,
        *,
        prompt: str,
        n: int,
        aspect_ratio: str,
        reference_urls: list[str],
    ) -> list[dict[str, str]]:
        target = max(1, int(n))
        out: list[dict[str, str]] = []
        compact_prompt = self._compact_prompt(prompt)

        fallback_strategies: list[tuple[str, list[str]]] = [
            (prompt, reference_urls),
            (prompt, []),
        ]
        if compact_prompt != prompt:
            fallback_strategies.append((compact_prompt, []))

        for _ in range(target):
            generated: dict[str, str] | None = None
            saw_empty = False
            for strategy_prompt, strategy_refs in fallback_strategies:
                try:
                    generated = self._single_generate(
                        prompt=strategy_prompt,
                        aspect_ratio=aspect_ratio,
                        reference_urls=strategy_refs,
                    )
                    break
                except Stage1Error as exc:
                    if exc.code == "openrouter_empty":
                        saw_empty = True
                        continue
                    raise

            if generated:
                out.append(generated)
                continue
            if not out and saw_empty:
                raise Stage1Error("openrouter_empty", "OpenRouter 未返回任何图片", 502)
            break
        return out


class VolcengineJimengClient:
    def __init__(
        self,
        *,
        access_key_id: str,
        access_key_secret: str,
        session_token: str,
        host: str,
        region: str,
        service: str,
        version: str,
        req_key: str,
        retry_max: int = 2,
        timeout_sec: int = 60,
        poll_interval_sec: float = 2.0,
        poll_timeout_sec: int = 120,
    ) -> None:
        self.access_key_id = access_key_id.strip()
        self.access_key_secret = access_key_secret.strip()
        self.session_token = session_token.strip()
        self.host = host.strip()
        self.region = region.strip() or "cn-north-1"
        self.service = service.strip() or "cv"
        self.version = version.strip() or "2022-08-31"
        self.req_key = req_key.strip() or "jimeng_t2i_v30"
        self.retry_max = max(0, int(retry_max))
        self.timeout_sec = max(10, int(timeout_sec))
        self.poll_interval_sec = max(0.2, float(poll_interval_sec))
        self.poll_timeout_sec = max(10, int(poll_timeout_sec))

    def _hmac_sha256(self, key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    def _sha256_hex(self, raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    def _percent_encode(self, val: str) -> str:
        return urllib.parse.quote(val, safe="-_.~")

    def _build_query(self, action: str) -> str:
        pairs = [("Action", action), ("Version", self.version)]
        pairs.sort(key=lambda x: (x[0], x[1]))
        return "&".join(f"{self._percent_encode(k)}={self._percent_encode(v)}" for k, v in pairs)

    def _build_auth_headers(self, *, query: str, body: bytes) -> dict[str, str]:
        payload_hash = self._sha256_hex(body)
        now = datetime.now(timezone.utc)
        x_date = now.strftime("%Y%m%dT%H%M%SZ")
        short_date = now.strftime("%Y%m%d")

        canonical_headers: dict[str, str] = {
            "content-type": "application/json",
            "host": self.host,
            "x-content-sha256": payload_hash,
            "x-date": x_date,
        }
        if self.session_token:
            canonical_headers["x-security-token"] = self.session_token

        signed_header_keys = sorted(canonical_headers.keys())
        signed_headers = ";".join(signed_header_keys)
        canonical_header_text = "".join(f"{key}:{canonical_headers[key]}\n" for key in signed_header_keys)
        canonical_request = (
            f"POST\n"
            f"/\n"
            f"{query}\n"
            f"{canonical_header_text}\n"
            f"{signed_headers}\n"
            f"{payload_hash}"
        )
        scope = f"{short_date}/{self.region}/{self.service}/request"
        string_to_sign = (
            f"HMAC-SHA256\n"
            f"{x_date}\n"
            f"{scope}\n"
            f"{self._sha256_hex(canonical_request.encode('utf-8'))}"
        )

        k_date = self._hmac_sha256(self.access_key_secret.encode("utf-8"), short_date)
        k_region = self._hmac_sha256(k_date, self.region)
        k_service = self._hmac_sha256(k_region, self.service)
        k_signing = self._hmac_sha256(k_service, "request")
        signature = hmac.new(k_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

        authorization = (
            f"HMAC-SHA256 Credential={self.access_key_id}/{scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )
        headers = {
            "Content-Type": "application/json",
            "Host": self.host,
            "X-Date": x_date,
            "X-Content-Sha256": payload_hash,
            "Authorization": authorization,
        }
        if self.session_token:
            headers["X-Security-Token"] = self.session_token
        return headers

    def _signed_post(self, *, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        query = self._build_query(action)
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        url = f"https://{self.host}/?{query}"
        headers = self._build_auth_headers(query=query, body=body)
        request = urllib.request.Request(url=url, data=body, method="POST", headers=headers)

        last_http_error: VolcengineHTTPError | None = None
        for attempt in range(self.retry_max + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:  # noqa: S310
                    raw = response.read().decode("utf-8")
                    return json.loads(raw)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore")
                err = VolcengineHTTPError(exc.code, detail or exc.reason or "http error")
                if exc.code == 429 or exc.code >= 500:
                    last_http_error = err
                    if attempt < self.retry_max:
                        time.sleep(2**attempt)
                        continue
                raise err from exc
            except urllib.error.URLError as exc:
                if attempt < self.retry_max:
                    time.sleep(2**attempt)
                    continue
                raise Stage1Error("volcengine_network_error", f"火山引擎网络错误: {exc}", 502) from exc
            except json.JSONDecodeError as exc:
                raise Stage1Error("volcengine_invalid_json", f"火山引擎返回非 JSON: {exc}", 502) from exc
        if last_http_error:
            raise last_http_error
        raise Stage1Error("volcengine_failed", "火山引擎调用失败", 502)

    def _size_for_aspect_ratio(self, aspect_ratio: str) -> tuple[int | None, int | None]:
        if ":" not in aspect_ratio:
            return None, None
        left, right = aspect_ratio.split(":", 1)
        try:
            w_ratio = int(left.strip())
            h_ratio = int(right.strip())
        except ValueError:
            return None, None
        if w_ratio <= 0 or h_ratio <= 0:
            return None, None

        ratio = w_ratio / h_ratio
        ratio = max(1.0 / 3.0, min(3.0, ratio))
        target_area = 1328 * 1328
        height = int(round((target_area / ratio) ** 0.5))
        width = int(round(height * ratio))
        width = max(512, min(2048, (width // 16) * 16))
        height = max(512, min(2048, (height // 16) * 16))
        if width <= 0 or height <= 0:
            return None, None
        return width, height

    def _submit_task(self, *, prompt: str, width: int | None, height: int | None) -> str:
        payload: dict[str, Any] = {
            "req_key": self.req_key,
            "prompt": prompt,
            "seed": -1,
            "use_pre_llm": True,
        }
        if width and height:
            payload["width"] = int(width)
            payload["height"] = int(height)

        try:
            response = self._signed_post(action="CVSync2AsyncSubmitTask", payload=payload)
        except VolcengineHTTPError as exc:
            msg = exc.message.strip().replace("\n", " ")
            raise Stage1Error("volcengine_http_error", f"火山引擎提交失败: HTTP {exc.status} {msg[:240]}", 502) from exc

        code = int(response.get("code", 0))
        if code != 10000:
            raise Stage1Error(
                "volcengine_submit_failed",
                f"火山引擎提交失败: code={code}, message={response.get('message', '')}",
                502,
            )
        task_id = str(((response.get("data") or {}).get("task_id") or "")).strip()
        if not task_id:
            raise Stage1Error("volcengine_submit_failed", "火山引擎提交成功但未返回 task_id", 502)
        return task_id

    def _poll_result(self, *, task_id: str) -> dict[str, str]:
        started = time.time()
        while time.time() - started <= self.poll_timeout_sec:
            payload: dict[str, Any] = {
                "req_key": self.req_key,
                "task_id": task_id,
                "req_json": json.dumps({"return_url": True}, ensure_ascii=False, separators=(",", ":")),
            }
            try:
                response = self._signed_post(action="CVSync2AsyncGetResult", payload=payload)
            except VolcengineHTTPError as exc:
                msg = exc.message.strip().replace("\n", " ")
                raise Stage1Error("volcengine_http_error", f"火山引擎查询失败: HTTP {exc.status} {msg[:240]}", 502) from exc

            code = int(response.get("code", 0))
            message = str(response.get("message", "")).strip()
            data = response.get("data") or {}
            if not isinstance(data, dict):
                data = {}
            status = str(data.get("status", "")).strip().lower()

            urls = data.get("image_urls")
            if isinstance(urls, list):
                for url in urls:
                    as_text = str(url).strip()
                    if as_text:
                        return {"url": as_text, "b64_json": ""}

            b64s = data.get("binary_data_base64")
            if isinstance(b64s, list):
                for raw in b64s:
                    as_text = str(raw).strip()
                    if as_text:
                        return {"url": "", "b64_json": as_text}

            if status in {"in_queue", "generating", ""} and code == 10000:
                time.sleep(self.poll_interval_sec)
                continue
            if status in {"not_found", "expired"}:
                raise Stage1Error("volcengine_task_expired", f"火山引擎任务状态异常: {status}", 502)
            if status == "done" and code == 10000:
                raise Stage1Error("volcengine_empty", "火山引擎任务完成但未返回图片", 502)
            if status == "done" and code != 10000:
                raise Stage1Error("volcengine_result_failed", f"火山引擎任务失败: code={code}, message={message}", 502)
            if code != 10000:
                raise Stage1Error("volcengine_result_failed", f"火山引擎查询失败: code={code}, message={message}", 502)
            time.sleep(self.poll_interval_sec)

        raise Stage1Error("volcengine_timeout", f"火山引擎任务超时: {task_id}", 504)

    def _build_prompt(self, prompt: str, reference_urls: list[str]) -> str:
        if not reference_urls:
            return prompt
        refs_text = "\n".join(f"{idx + 1}. {url}" for idx, url in enumerate(reference_urls))
        return (
            f"{prompt}\n\n"
            "请保持角色与参考图一致。参考图链接如下：\n"
            f"{refs_text}"
        )

    def generate_images(
        self,
        *,
        prompt: str,
        n: int,
        aspect_ratio: str,
        reference_urls: list[str],
    ) -> list[dict[str, str]]:
        target = max(1, int(n))
        width, height = self._size_for_aspect_ratio(aspect_ratio)
        effective_prompt = self._build_prompt(prompt, reference_urls)
        out: list[dict[str, str]] = []
        for _ in range(target):
            task_id = self._submit_task(prompt=effective_prompt, width=width, height=height)
            out.append(self._poll_result(task_id=task_id))
        return out


class ImageGenerationService:
    def __init__(self, *, store: TaskStore, settings: Settings) -> None:
        self.store = store
        self.settings = settings

    def _create_oss_uploader(self) -> AliyunOSSUploader:
        return AliyunOSSUploader(
            region=self.settings.aliyun_oss_region,
            access_key_id=self.settings.aliyun_oss_access_key_id,
            access_key_secret=self.settings.aliyun_oss_access_key_secret,
            bucket=self.settings.aliyun_oss_bucket,
            public_domain=self.settings.aliyun_oss_public_domain,
        )

    def _create_image_client(self) -> OpenRouterImageClient | VolcengineJimengClient:
        if self.settings.image_generation_provider == "volcengine_jimeng30":
            return VolcengineJimengClient(
                access_key_id=self.settings.volcengine_access_key_id,
                access_key_secret=self.settings.volcengine_access_key_secret,
                session_token=self.settings.volcengine_session_token,
                host=self.settings.volcengine_visual_host,
                region=self.settings.volcengine_region,
                service=self.settings.volcengine_service,
                version=self.settings.volcengine_jimeng_version,
                req_key=self.settings.volcengine_jimeng_req_key,
                retry_max=2,
                poll_interval_sec=self.settings.volcengine_poll_interval_sec,
                poll_timeout_sec=self.settings.volcengine_poll_timeout_sec,
            )
        return OpenRouterImageClient(
            api_key=self.settings.openrouter_api_key,
            base_url=self.settings.openrouter_base_url,
            model=self.settings.openrouter_image_model,
            add_watermark=self.settings.openrouter_image_add_watermark,
            retry_max=2,
        )

    def validate_ready(self) -> None:
        missing: list[str] = []
        if self.settings.image_generation_provider == "volcengine_jimeng30":
            if not self.settings.volcengine_access_key_id:
                missing.append("VOLCENGINE_ACCESS_KEY_ID")
            if not self.settings.volcengine_access_key_secret:
                missing.append("VOLCENGINE_ACCESS_KEY_SECRET")
        else:
            if not self.settings.openrouter_api_key:
                missing.append("OPENROUTER_API_KEY")
        if not self.settings.aliyun_oss_region:
            missing.append("ALIYUN_OSS_REGION")
        if not self.settings.aliyun_oss_access_key_id:
            missing.append("ALIYUN_OSS_ACCESS_KEY_ID")
        if not self.settings.aliyun_oss_access_key_secret:
            missing.append("ALIYUN_OSS_ACCESS_KEY_SECRET")
        if not self.settings.aliyun_oss_bucket:
            missing.append("ALIYUN_OSS_BUCKET")
        if missing:
            raise Stage1Error("config_error", f"缺少配置: {', '.join(missing)}", 400)

    def _guess_content_type(self, path: Path) -> str:
        mime, _ = mimetypes.guess_type(str(path))
        return mime or "application/octet-stream"

    def _oss_key_for_ref(self, task_id: str, ref_id: str, path: Path) -> str:
        suffix = path.suffix.lower() or ".bin"
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        safe_ref = re.sub(r"[^a-zA-Z0-9_-]+", "_", ref_id)
        return f"ytauto/{task_id}/refs/{safe_ref}-{digest}{suffix}"

    def _load_or_build_generation_inputs(self, task_id: str) -> dict[str, Any]:
        path = self.store.generation_inputs_path(task_id)
        if path.exists():
            return self.store.read_json(path)

        physical_manifest_path = self.store.contract_path(task_id, "physical_manifest")
        character_bank_path = self.store.contract_path(task_id, "character_bank")
        aligned_storyboard_path = self.store.contract_path(task_id, "aligned_storyboard")
        final_table_path = self.store.contract_path(task_id, "final_production_table")
        for file_path in [physical_manifest_path, character_bank_path, aligned_storyboard_path, final_table_path]:
            if not file_path.exists():
                raise Stage1Error("artifact_missing", f"缺少输入文件: {file_path}", 404)

        physical_manifest = self.store.read_json(physical_manifest_path)
        character_bank = self.store.read_json(character_bank_path)
        aligned_storyboard = self.store.read_json(aligned_storyboard_path)
        final_table = self.store.read_json(final_table_path)

        scene_assets: dict[int, dict[str, str]] = {}
        for item in physical_manifest.get("scenes", []):
            try:
                sid = int(item.get("scene_id", 0))
            except (TypeError, ValueError):
                continue
            scene_assets[sid] = {
                "keyframe_path": str(item.get("keyframe_path", "")).strip(),
                "clip_path": str(item.get("clip_path", "")).strip(),
            }

        ref_path_by_id = {}
        for item in character_bank.get("characters", []):
            ref_id = str(item.get("ref_id", "")).strip()
            if not ref_id:
                continue
            ref_path_by_id[ref_id] = str(item.get("ref_image_path", "")).strip()

        prompt_by_shot = {}
        for item in final_table.get("prompts", []):
            try:
                sid = int(item.get("shot_id", 0))
            except (TypeError, ValueError):
                continue
            prompt_by_shot[sid] = item

        storyboard_shots = self._normalize_aligned_storyboard_shots(aligned_storyboard)
        tasks = []
        for shot in storyboard_shots:
            shot_id = int(shot.get("shot_id", 0))
            prompt_item = prompt_by_shot.get(shot_id, {})

            references = []
            for idx, mapping in enumerate(shot.get("character_mappings", []), start=1):
                ref_id = str(mapping.get("ref_id", "")).strip()
                references.append(
                    {
                        "reference_index": idx,
                        "ref_id": ref_id,
                        "image_path": ref_path_by_id.get(ref_id, ""),
                    }
                )

            tasks.append(
                {
                    "shot_id": shot_id,
                    "image_prompt": str(prompt_item.get("image_prompt", "")).strip(),
                    "video_prompt": str(prompt_item.get("video_prompt", "")).strip(),
                    "reference_images": references,
                    "keyframe_path": scene_assets.get(shot_id, {}).get("keyframe_path", ""),
                    "clip_path": scene_assets.get(shot_id, {}).get("clip_path", ""),
                }
            )

        payload = {
            "project_id": str(final_table.get("project_id") or task_id),
            "tasks": sorted(tasks, key=lambda x: int(x["shot_id"])),
        }
        self.store.write_json(path, payload)
        return payload

    def _normalize_aligned_storyboard_shots(self, aligned_storyboard: dict[str, Any]) -> list[dict[str, Any]]:
        storyboard_payload = aligned_storyboard.get("storyboard")
        if isinstance(storyboard_payload, list):
            return storyboard_payload

        scenes_payload = aligned_storyboard.get("scenes")
        if not isinstance(scenes_payload, list):
            return []

        shots: list[dict[str, Any]] = []
        for scene in scenes_payload:
            try:
                shot_id = int(scene.get("scene_id", 0))
            except (TypeError, ValueError):
                continue
            subjects = (scene.get("visual_analysis") or {}).get("subjects") or []
            character_mappings = []
            for subject in subjects:
                ref_id = str((subject or {}).get("id", "")).strip()
                if not ref_id:
                    continue
                character_mappings.append({"ref_id": ref_id})
            shots.append({"shot_id": shot_id, "character_mappings": character_mappings})

        return sorted(shots, key=lambda x: int(x.get("shot_id", 0)))

    def _download_candidate_url(self, url: str, timeout_sec: int = 60) -> tuple[bytes, str]:
        request = urllib.request.Request(url=url, method="GET")
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:  # noqa: S310
            raw = response.read()
            content_type = str(response.headers.get("Content-Type", "")).split(";")[0].strip()
        ext = mimetypes.guess_extension(content_type) or Path(urllib.parse.urlparse(url).path).suffix or ".png"
        return raw, ext

    def _save_candidate(
        self,
        *,
        output_dir: Path,
        candidate_index: int,
        data: dict[str, str],
    ) -> tuple[Path, str]:
        url = data.get("url", "")
        b64 = data.get("b64_json", "")

        raw = b""
        ext = ".png"
        if b64:
            raw = base64.b64decode(b64)
        elif url.startswith("data:image/"):
            header, _, encoded = url.partition(",")
            mime = header[5:].split(";")[0].strip() if ";" in header else "image/png"
            raw = base64.b64decode(encoded)
            ext = mimetypes.guess_extension(mime) or ".png"
        elif url:
            raw, ext = self._download_candidate_url(url)
        else:
            raise Stage1Error("image_candidate_invalid", "候选图既无 url 也无 b64 数据", 502)

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"candidate_{candidate_index:02d}{ext}"
        out_path.write_bytes(raw)
        return out_path, url

    def _generate_for_shot(
        self,
        *,
        task_id: str,
        shot_task: dict[str, Any],
        oss_url_by_ref: dict[str, str],
        client: OpenRouterImageClient | VolcengineJimengClient,
        candidates_per_shot: int,
        aspect_ratio: str,
    ) -> dict[str, Any]:
        shot_id = int(shot_task.get("shot_id", 0))
        image_prompt = str(shot_task.get("image_prompt", "")).strip()
        references = []
        for ref in shot_task.get("reference_images", []):
            ref_id = str((ref or {}).get("ref_id", "")).strip()
            references.append(
                {
                    "reference_index": int((ref or {}).get("reference_index", 0)),
                    "ref_id": ref_id,
                    "image_path": str((ref or {}).get("image_path", "")).strip(),
                    "oss_url": oss_url_by_ref.get(ref_id, ""),
                }
            )
        reference_urls = [x["oss_url"] for x in references if x.get("oss_url")]
        output_dir = self.store.task_dir(task_id) / "outputs" / "images" / f"shot_{shot_id:03d}"
        if output_dir.exists():
            shutil.rmtree(output_dir)

        generated: list[dict[str, Any]] = []
        remaining = candidates_per_shot
        round_idx = 0
        max_rounds = max(2, (candidates_per_shot + 3) // 4)
        while remaining > 0 and round_idx < max_rounds:
            round_idx += 1
            request_n = min(4, remaining)
            try:
                items = client.generate_images(
                    prompt=image_prompt,
                    n=request_n,
                    aspect_ratio=aspect_ratio,
                    reference_urls=reference_urls,
                )
            except Stage1Error as exc:
                return {
                    "shot_id": shot_id,
                    "status": "failed",
                    "error": {"code": exc.code, "message": exc.message},
                    "image_prompt": image_prompt,
                    "video_prompt": str(shot_task.get("video_prompt", "")).strip(),
                    "reference_images": references,
                    "candidates": [],
                }

            if not items:
                break
            for item in items:
                if remaining <= 0:
                    break
                candidate_index = len(generated) + 1
                try:
                    local_path, source_url = self._save_candidate(
                        output_dir=output_dir,
                        candidate_index=candidate_index,
                        data=item,
                    )
                except Exception as exc:  # noqa: BLE001
                    return {
                        "shot_id": shot_id,
                        "status": "failed",
                        "error": {"code": "image_save_failed", "message": str(exc)},
                        "image_prompt": image_prompt,
                        "video_prompt": str(shot_task.get("video_prompt", "")).strip(),
                        "reference_images": references,
                        "candidates": generated,
                    }
                generated.append(
                    {
                        "candidate_index": candidate_index,
                        "round": round_idx,
                        "local_path": str(local_path.resolve()),
                        "source_url": source_url,
                    }
                )
                remaining -= 1

        if not generated:
            return {
                "shot_id": shot_id,
                "status": "failed",
                "error": {"code": "openrouter_empty", "message": "未返回任何候选图"},
                "image_prompt": image_prompt,
                "video_prompt": str(shot_task.get("video_prompt", "")).strip(),
                "reference_images": references,
                "candidates": [],
            }

        return {
            "shot_id": shot_id,
            "status": "succeeded",
            "error": None,
            "image_prompt": image_prompt,
            "video_prompt": str(shot_task.get("video_prompt", "")).strip(),
            "reference_images": references,
            "candidates": generated,
        }

    def run(self, task_id: str, params: dict[str, Any], progress_cb: ProgressCallback) -> dict[str, Any]:
        self.validate_ready()
        task_dir = self.store.root_dir / task_id
        if not task_dir.exists() or not task_dir.is_dir():
            raise Stage1Error("task_not_found", "任务不存在", 404)

        generation_inputs = self._load_or_build_generation_inputs(task_id)
        tasks = generation_inputs.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise Stage1Error("generation_inputs_invalid", "06_generation_inputs.json 缺少 tasks", 422)

        available = []
        for item in tasks:
            try:
                available.append(int(item.get("shot_id", 0)))
            except (TypeError, ValueError):
                continue
        selected_shots = parse_shot_range(str(params.get("shot_range") or "").strip() or None, available)
        selected_tasks = [item for item in tasks if int(item.get("shot_id", 0)) in selected_shots]
        selected_tasks.sort(key=lambda x: int(x.get("shot_id", 0)))

        candidates_per_shot = int(params.get("candidates_per_shot", 4))
        aspect_ratio = str(params.get("aspect_ratio", "9:16")).strip() or "9:16"
        concurrency = int(params.get("concurrency", 2))

        progress_cb(5.0, "生图任务初始化完成")

        uploader = self._create_oss_uploader()
        client = self._create_image_client()

        ref_local_path_by_id: dict[str, str] = {}
        for task in selected_tasks:
            for ref in task.get("reference_images", []):
                ref_id = str((ref or {}).get("ref_id", "")).strip()
                image_path = str((ref or {}).get("image_path", "")).strip()
                if ref_id and image_path and ref_id not in ref_local_path_by_id:
                    ref_local_path_by_id[ref_id] = image_path

        ref_oss_map: list[dict[str, str]] = []
        oss_url_by_ref: dict[str, str] = {}
        total_refs = max(1, len(ref_local_path_by_id))
        uploaded_refs = 0
        for ref_id, image_path in sorted(ref_local_path_by_id.items()):
            local_path = Path(image_path).expanduser().resolve()
            content_type = self._guess_content_type(local_path)
            key = self._oss_key_for_ref(task_id, ref_id, local_path)
            oss_url = uploader.upload_file(local_path=local_path, key=key, content_type=content_type)
            oss_url_by_ref[ref_id] = oss_url
            ref_oss_map.append(
                {
                    "ref_id": ref_id,
                    "local_path": str(local_path),
                    "oss_key": key,
                    "oss_url": oss_url,
                }
            )
            uploaded_refs += 1
            progress_cb(5.0 + uploaded_refs / total_refs * 15.0, f"参考图上传完成: {ref_id}")

        ref_map_payload = {
            "project_id": generation_inputs.get("project_id", task_id),
            "task_id": task_id,
            "refs": ref_oss_map,
        }
        self.store.write_json(self.store.ref_oss_map_path(task_id), ref_map_payload)

        results: list[dict[str, Any]] = []
        lock = threading.Lock()
        completed = 0
        total_shots = max(1, len(selected_tasks))

        def _on_completed(shot_id: int) -> None:
            nonlocal completed
            with lock:
                completed += 1
                progress = 20.0 + completed / total_shots * 70.0
            progress_cb(progress, f"镜头 {shot_id} 生图完成 ({completed}/{total_shots})")

        if concurrency <= 1:
            for task in selected_tasks:
                result = self._generate_for_shot(
                    task_id=task_id,
                    shot_task=task,
                    oss_url_by_ref=oss_url_by_ref,
                    client=client,
                    candidates_per_shot=candidates_per_shot,
                    aspect_ratio=aspect_ratio,
                )
                results.append(result)
                _on_completed(int(task.get("shot_id", 0)))
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_map = {
                    executor.submit(
                        self._generate_for_shot,
                        task_id=task_id,
                        shot_task=task,
                        oss_url_by_ref=oss_url_by_ref,
                        client=client,
                        candidates_per_shot=candidates_per_shot,
                        aspect_ratio=aspect_ratio,
                    ): int(task.get("shot_id", 0))
                    for task in selected_tasks
                }
                for future in as_completed(future_map):
                    shot_id = future_map[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # noqa: BLE001
                        result = {
                            "shot_id": shot_id,
                            "status": "failed",
                            "error": {"code": "image_generation_failed", "message": str(exc)},
                            "image_prompt": "",
                            "video_prompt": "",
                            "reference_images": [],
                            "candidates": [],
                        }
                    results.append(result)
                    _on_completed(shot_id)

        results.sort(key=lambda x: int(x.get("shot_id", 0)))
        failed_shots = [
            {"shot_id": int(item.get("shot_id", 0)), "error": item.get("error")}
            for item in results
            if item.get("status") != "succeeded"
        ]
        total_candidates = sum(len(item.get("candidates", [])) for item in results)

        candidates_payload = {
            "project_id": generation_inputs.get("project_id", task_id),
            "task_id": task_id,
            "shot_range": str(params.get("shot_range") or ""),
            "candidates_per_shot": candidates_per_shot,
            "aspect_ratio": aspect_ratio,
            "shots": results,
            "failed_shots": failed_shots,
        }
        self.store.write_json(self.store.image_candidates_path(task_id), candidates_payload)

        if len(failed_shots) >= len(results):
            raise Stage1Error("image_generation_failed", "所有镜头生图失败", 502)

        result = {
            "project_id": str(generation_inputs.get("project_id") or task_id),
            "shot_count": len(results),
            "total_candidates": total_candidates,
            "failed_shots": failed_shots,
            "artifacts": {
                "ref_oss_map": str(self.store.ref_oss_map_path(task_id).resolve()),
                "image_candidates": str(self.store.image_candidates_path(task_id).resolve()),
                "image_output_dir": str((self.store.task_dir(task_id) / "outputs" / "images").resolve()),
                "generation_manifest": str(self.store.image_generation_manifest_path(task_id).resolve()),
            },
        }
        progress_cb(95.0, "候选图清单已生成")
        return result
