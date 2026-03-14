from __future__ import annotations

import copy
import queue
import threading
import uuid
from datetime import datetime, timezone

from app.errors import Stage1Error
from app.models import TaskRecord
from app.services.pipeline import Stage1Pipeline
from app.services.store import TaskStore


class JobManager:
    def __init__(self, store: TaskStore, pipeline: Stage1Pipeline, max_workers: int = 2) -> None:
        self.store = store
        self.pipeline = pipeline
        self.max_workers = max(1, max_workers)

        self._tasks: dict[str, TaskRecord] = {}
        self._lock = threading.RLock()
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

    def start(self) -> None:
        self._load_persisted_tasks()
        for idx in range(self.max_workers):
            t = threading.Thread(target=self._worker_loop, name=f"stage1-worker-{idx}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        for _ in self._threads:
            self._queue.put(None)
        for t in self._threads:
            t.join(timeout=1.0)
        self._threads.clear()

    def submit(self, params: dict) -> TaskRecord:
        task_id = uuid.uuid4().hex
        working_dir = str(self.store.task_dir(task_id))
        task = TaskRecord.new(task_id=task_id, params=params, working_dir=working_dir)

        with self._lock:
            self._tasks[task_id] = task
            self.store.save_manifest(task)

        self.store.append_log(task_id, "任务已创建，进入队列")
        self._queue.put(task_id)
        return copy.deepcopy(task)

    def get(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            return copy.deepcopy(task)

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
                self._run_task(task_id)
            finally:
                self._queue.task_done()

    def _run_task(self, task_id: str) -> None:
        task = self.get(task_id)
        if not task:
            return

        self._update(task_id, status="running", progress=1.0)
        self.store.append_log(task_id, "任务开始执行")

        def on_progress(progress: float, message: str) -> None:
            self._update(task_id, progress=progress)
            self.store.append_log(task_id, message)

        try:
            result = self.pipeline.run(task, on_progress)
        except Stage1Error as exc:
            self._update(
                task_id,
                status="failed",
                error={"code": exc.code, "message": exc.message},
            )
            self.store.append_log(task_id, f"任务失败: {exc.code} {exc.message}")
            return
        except Exception as exc:  # noqa: BLE001
            self._update(
                task_id,
                status="failed",
                error={"code": "internal_error", "message": f"未预期错误: {exc}"},
            )
            self.store.append_log(task_id, f"任务失败: internal_error {exc}")
            return

        self._update(task_id, status="succeeded", progress=100.0, result=result)
        self.store.append_log(task_id, "任务成功结束")

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
            task = self._tasks.get(task_id)
            if not task:
                return
            if status is not None:
                task.status = status
            if progress is not None:
                task.progress = max(0.0, min(100.0, progress))
            if error is not None:
                task.error = error
            if result is not None:
                task.result = result
            task.updated_at = datetime.now(timezone.utc)
            self.store.save_manifest(task)

    def _load_persisted_tasks(self) -> None:
        for task in self.store.load_all():
            if task.status in {"queued", "running"}:
                task.status = "failed"
                task.error = {
                    "code": "restart_interrupted",
                    "message": "服务重启导致任务中断，请重新提交",
                }
                task.updated_at = datetime.now(timezone.utc)
                self.store.save_manifest(task)
            self._tasks[task.task_id] = task
