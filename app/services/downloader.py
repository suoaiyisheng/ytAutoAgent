from __future__ import annotations

import shutil
from pathlib import Path

import yt_dlp

from app.errors import Stage1Error


class VideoDownloader:
    def obtain_video(
        self,
        source_url: str | None,
        local_video_path: str | None,
        output_dir: Path,
        download_format: str,
    ) -> Path:
        if bool(source_url) == bool(local_video_path):
            raise Stage1Error("invalid_source", "source_url 与 local_video_path 必须且只能提供一个", 422)

        if source_url:
            return self.download(url=source_url, output_dir=output_dir, download_format=download_format)
        return self.copy_local(local_video_path=str(local_video_path), output_dir=output_dir)

    def copy_local(self, local_video_path: str, output_dir: Path) -> Path:
        src = Path(local_video_path).expanduser().resolve()
        if not src.exists() or not src.is_file():
            raise Stage1Error("local_video_not_found", "local_video_path 不存在或不是文件", 422)

        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / src.name
        try:
            shutil.copy2(src, target)
        except Exception as exc:  # noqa: BLE001
            raise Stage1Error("local_copy_failed", f"复制本地视频失败: {exc}", 500) from exc
        return target.resolve()

    def download(self, url: str, output_dir: Path, download_format: str) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        opts = {
            "format": download_format,
            "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
            "noplaylist": True,
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": 30,
        }
        node_path = shutil.which("node")
        if node_path:
            opts["js_runtimes"] = {"node": {"path": node_path}}
            opts["remote_components"] = ["ejs:github"]

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                resolved = self._resolve_file_path(ydl, info, output_dir)
                if not resolved.exists():
                    raise Stage1Error(
                        code="download_failed",
                        message="下载完成后未找到视频文件",
                        http_status=502,
                    )
                return resolved
        except Stage1Error:
            raise
        except Exception as exc:  # noqa: BLE001
            raise self._map_download_error(exc) from exc

    def _resolve_file_path(self, ydl: yt_dlp.YoutubeDL, info: dict, output_dir: Path) -> Path:
        candidates: list[Path] = []

        filename = info.get("_filename")
        if filename:
            candidates.append(Path(filename))

        requested = info.get("requested_downloads") or []
        for item in requested:
            filepath = item.get("filepath")
            if filepath:
                candidates.append(Path(filepath))

        prepared = ydl.prepare_filename(info)
        if prepared:
            candidates.append(Path(prepared))

        video_id = info.get("id")
        if video_id:
            candidates.extend(sorted(output_dir.glob(f"{video_id}.*")))

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()

        matches = sorted(output_dir.glob("*"))
        for match in matches:
            if match.is_file():
                return match.resolve()

        return (output_dir / "missing.mp4").resolve()

    def _map_download_error(self, exc: Exception) -> Stage1Error:
        text = str(exc).lower()
        if "unsupported url" in text or "not a valid url" in text:
            return Stage1Error("invalid_url", "URL 无法被解析", 400)
        if "private" in text or "sign in" in text or "members-only" in text:
            return Stage1Error("protected_content", "受保护内容，无法下载", 403)
        if "country" in text or "geo" in text or "region" in text:
            return Stage1Error("geo_restricted", "该内容受地区限制", 451)
        if "timed out" in text or "timeout" in text:
            return Stage1Error("download_timeout", "下载超时", 504)
        return Stage1Error("download_failed", f"下载失败: {exc}", 502)
