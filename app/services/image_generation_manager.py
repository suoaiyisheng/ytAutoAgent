from __future__ import annotations

import copy
import queue
import threading
from datetime import datetime, timezone

from app.errors import Stage1Error
from app.models import ImageGenerationRunRecord
from app.services.image_generation import ImageGenerationService
from app.services.store import TaskStore


class ImageGenerationManager:
    def __init__(self, *, store: TaskStore, service: ImageGenerationService, max_workers: int = 1) -> None:
        self.store = store
        self.service = service
        self.max_workers = max(1, max_workers)

        self._runs: dict[str, ImageGenerationRunRecord] = {}
        self._lock = threading.RLock()
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

    def start(self) -> None:
        self._load_persisted_runs()
        for idx in range(self.max_workers):
            thread = threading.Thread(target=self._worker_loop, name=f"image-generation-worker-{idx}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._stop.set()
        for _ in self._threads:
            self._queue.put(None)
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads.clear()

    def submit(self, task_id: str, params: dict) -> ImageGenerationRunRecord:
        task_dir = self.store.root_dir / task_id
        if not task_dir.exists() or not task_dir.is_dir():
            raise Stage1Error("task_not_found", "任务不存在", 404)

        with self._lock:
            current = self._runs.get(task_id)
            if current and current.status in {"queued", "running"}:
                raise Stage1Error("image_generation_running", "该任务已有生图任务在执行", 409)
            record = ImageGenerationRunRecord.new(task_id=task_id, params=params)
            self._runs[task_id] = record
            self._save_manifest(record)

        self.store.append_log(task_id, "生图任务已创建，进入队列")
        self._queue.put(task_id)
        return copy.deepcopy(record)

    def get(self, task_id: str) -> ImageGenerationRunRecord | None:
        with self._lock:
            record = self._runs.get(task_id)
            if record:
                return copy.deepcopy(record)
        path = self.store.image_generation_manifest_path(task_id)
        if not path.exists():
            return None
        try:
            payload = self.store.read_json(path)
            record = ImageGenerationRunRecord.from_manifest(payload)
        except Exception:  # noqa: BLE001
            return None
        with self._lock:
            self._runs[task_id] = record
        return copy.deepcopy(record)

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                task_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if task_id is None:
                self._queue.task_done()
                return
            try:
                self._run(task_id)
            finally:
                self._queue.task_done()

    def _run(self, task_id: str) -> None:
        record = self.get(task_id)
        if not record:
            return

        self._update(task_id, status="running", progress=1.0)
        self.store.append_log(task_id, "生图任务开始执行")

        def progress_cb(progress: float, message: str) -> None:
            self._update(task_id, progress=progress)
            self.store.append_log(task_id, f"[生图] {message}")

        try:
            result = self.service.run(task_id, record.params, progress_cb)
        except Stage1Error as exc:
            self._update(task_id, status="failed", error={"code": exc.code, "message": exc.message})
            self.store.append_log(task_id, f"生图任务失败: {exc.code} {exc.message}")
            return
        except Exception as exc:  # noqa: BLE001
            self._update(task_id, status="failed", error={"code": "internal_error", "message": str(exc)})
            self.store.append_log(task_id, f"生图任务失败: internal_error {exc}")
            return

        self._update(task_id, status="succeeded", progress=100.0, result=result)
        self.store.append_log(task_id, "生图任务成功结束")

    def _update(
        self,
        task_id: str,
        *,
        status: str | None = None,
        progress: float | None = None,
        error: dict | None = None,
        result: dict | None = None,
    ) -> None:
        with self._lock:
            record = self._runs.get(task_id)
            if not record:
                return
            if status is not None:
                record.status = status
            if progress is not None:
                record.progress = max(0.0, min(100.0, progress))
            if error is not None:
                record.error = error
            if result is not None:
                record.result = result
            record.updated_at = datetime.now(timezone.utc)
            self._save_manifest(record)

    def _save_manifest(self, record: ImageGenerationRunRecord) -> None:
        self.store.write_json(self.store.image_generation_manifest_path(record.task_id), record.to_manifest())

    def _load_persisted_runs(self) -> None:
        for path in sorted(self.store.root_dir.glob("*/07_image_generation_manifest.json")):
            try:
                payload = self.store.read_json(path)
                record = ImageGenerationRunRecord.from_manifest(payload)
            except Exception:  # noqa: BLE001
                continue
            if record.status in {"queued", "running"}:
                record.status = "failed"
                record.error = {
                    "code": "restart_interrupted",
                    "message": "服务重启导致生图任务中断，请重新提交",
                }
                record.updated_at = datetime.now(timezone.utc)
                self._save_manifest(record)
            self._runs[record.task_id] = record

