#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def _normalize_api_base(base_url: str) -> str:
    raw = base_url.strip().rstrip("/")
    if raw.endswith("/v1"):
        return raw
    return f"{raw}/v1"


def _http_json(
    method: str,
    url: str,
    *,
    api_key: str,
    payload: dict[str, Any] | None = None,
    timeout_sec: float = 90.0,
) -> dict[str, Any]:
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        method=method.upper(),
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8")
        return json.loads(raw)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"网络错误: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"服务端返回非 JSON: {exc}") from exc


def _list_models(api_base: str, api_key: str) -> list[str]:
    payload = _http_json("GET", f"{api_base}/models", api_key=api_key)
    data = payload.get("data", [])
    ids: list[str] = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id", "")).strip()
            if model_id:
                ids.append(model_id)
    return ids


def _pick_models(all_models: list[str], requested_model: str) -> list[str]:
    req = requested_model.strip()
    if req and req.lower() != "auto":
        return [req]

    if all_models:
        keywords = ("banana", "gemini", "image", "seed", "flux", "sdxl")
        ranked = sorted(
            all_models,
            key=lambda mid: (
                0
                if any(key in mid.lower() for key in keywords)
                else 1,
                len(mid),
                mid,
            ),
        )
        return ranked[:8]

    return [
        "banana",
        "gemini-2.5-flash-image",
        "google/gemini-2.5-flash-image",
    ]


def _parse_chat_image_candidates(payload: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    choices = payload.get("choices", [])
    if not isinstance(choices, list):
        return out

    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue

        images = message.get("images", [])
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

    return out


def _decode_data_url(url: str) -> tuple[bytes, str] | None:
    if not url.startswith("data:image/"):
        return None
    header, _, encoded = url.partition(",")
    if not encoded:
        return None
    mime = header[5:].split(";")[0].strip() if ";" in header else "image/png"
    ext = mimetypes.guess_extension(mime) or ".png"
    return base64.b64decode(encoded), ext


def _download_image(url: str, timeout_sec: float = 90.0) -> tuple[bytes, str]:
    data_url = _decode_data_url(url)
    if data_url:
        return data_url

    req = urllib.request.Request(url=url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # noqa: S310
        raw = resp.read()
        content_type = str(resp.headers.get("Content-Type", "")).split(";")[0].strip()
    ext = mimetypes.guess_extension(content_type) or Path(urllib.parse.urlparse(url).path).suffix or ".png"
    return raw, ext


def _save_image_candidate(candidate: dict[str, str], output_dir: Path, stem: str) -> Path:
    b64 = str(candidate.get("b64_json", "")).strip()
    url = str(candidate.get("url", "")).strip()

    if b64:
        raw = base64.b64decode(b64)
        ext = ".png"
    elif url:
        raw, ext = _download_image(url)
    else:
        raise RuntimeError("候选图片既没有 b64_json，也没有 url")

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{stem}{ext}"
    out.write_bytes(raw)
    return out


def _try_images_generations(
    *,
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    size: str,
) -> tuple[dict[str, str], str]:
    payload = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "n": 1,
        "response_format": "b64_json",
    }
    data = _http_json("POST", f"{api_base}/images/generations", api_key=api_key, payload=payload)
    items = data.get("data", [])
    if not isinstance(items, list) or not items:
        raise RuntimeError(f"images/generations 未返回图片: {json.dumps(data, ensure_ascii=False)[:300]}")
    first = items[0]
    if not isinstance(first, dict):
        raise RuntimeError("images/generations 返回格式不正确")
    candidate = {
        "url": str(first.get("url", "")).strip(),
        "b64_json": str(first.get("b64_json", "")).strip(),
    }
    if not candidate["url"] and not candidate["b64_json"]:
        raise RuntimeError(f"images/generations 无可用图片字段: {json.dumps(first, ensure_ascii=False)[:300]}")
    return candidate, "images/generations"


def _try_chat_completions(
    *,
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    aspect_ratio: str,
) -> tuple[dict[str, str], str]:
    payload = {
        "model": model,
        "modalities": ["image", "text"],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
        "image": {"aspect_ratio": aspect_ratio},
    }
    data = _http_json("POST", f"{api_base}/chat/completions", api_key=api_key, payload=payload)
    candidates = _parse_chat_image_candidates(data)
    if not candidates:
        raise RuntimeError(f"chat/completions 未返回图片: {json.dumps(data, ensure_ascii=False)[:300]}")
    return candidates[0], "chat/completions"


def run_test(args: argparse.Namespace) -> int:
    api_key = (args.api_key or "").strip()
    if not api_key:
        raise RuntimeError("必须传入 --api-key 或设置 TABCODE_API_KEY 环境变量")

    api_base = _normalize_api_base(args.base_url)
    print(f"[INFO] API Base: {api_base}")

    models: list[str] = []
    try:
        models = _list_models(api_base, api_key)
        print(f"[INFO] /models 返回 {len(models)} 个模型")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] 拉取模型列表失败，改用兜底模型: {exc}")

    model_candidates = _pick_models(models, args.model)
    print(f"[INFO] 待测试模型: {model_candidates}")

    errors: list[str] = []
    t0 = time.time()
    for model in model_candidates:
        for method in ("images", "chat"):
            try:
                if method == "images":
                    candidate, used_endpoint = _try_images_generations(
                        api_base=api_base,
                        api_key=api_key,
                        model=model,
                        prompt=args.prompt,
                        size=args.size,
                    )
                else:
                    candidate, used_endpoint = _try_chat_completions(
                        api_base=api_base,
                        api_key=api_key,
                        model=model,
                        prompt=args.prompt,
                        aspect_ratio=args.aspect_ratio,
                    )
                elapsed = time.time() - t0
                output = _save_image_candidate(candidate, Path(args.output_dir), "tabcode_test")
                print("[OK] 生图成功")
                print(f"[OK] 模型: {model}")
                print(f"[OK] 接口: {used_endpoint}")
                print(f"[OK] 耗时: {elapsed:.2f}s")
                print(f"[OK] 文件: {output}")
                return 0
            except Exception as exc:  # noqa: BLE001
                errors.append(f"model={model}, method={method}, error={exc}")

    print("[FAIL] 全部尝试均失败")
    for idx, err in enumerate(errors, start=1):
        print(f"{idx}. {err}")
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="测试 chat.tabcode.cc 是否可生图（OpenAI 兼容接口）。")
    parser.add_argument("--base-url", default="https://chat.tabcode.cc", help="API 域名，默认 https://chat.tabcode.cc")
    parser.add_argument("--api-key", default="", help="API Key；不传则读取 TABCODE_API_KEY")
    parser.add_argument("--model", default="auto", help="模型名；默认 auto 自动探测")
    parser.add_argument("--prompt", default="一只戴护目镜的香蕉站在月球上，电影级光影，高清细节", help="生图提示词")
    parser.add_argument("--size", default="1024x1024", help="images/generations 的 size，如 1024x1024 或 2048x2048")
    parser.add_argument("--aspect-ratio", default="1:1", help="chat/completions 兜底接口的画幅比例")
    parser.add_argument("--output-dir", default="runtime/manual_tests", help="输出目录")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.api_key:
        args.api_key = os.getenv("TABCODE_API_KEY", "")
    try:
        return run_test(args)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
