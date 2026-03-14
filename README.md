# ytAuto Full Pipeline Service

五阶段自动化服务（结构化提取 -> 原子感知 -> 身份对齐 -> 指令合成），基于 FastAPI。

## 运行 API

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
uvicorn app.main:app --reload
```

依赖：
- `ffmpeg`、`ffprobe` 需要在系统 PATH 中可执行。
- `yt-dlp` 推荐同时安装 CLI（若仅安装 Python 包，健康检查也会识别）。
- 需要配置视觉模型 API Key（默认使用 Qwen）。

## 使用 .env

```bash
cp .env.example .env
# 然后编辑 .env，填入 QWEN_API_KEY 或 GEMINI_API_KEY
```

配置加载规则：
- 启动时自动读取项目根目录 `.env`（不会覆盖已存在的系统环境变量）。
- 可通过 `STAGE1_DOTENV_PATH` 指定其他 `.env` 路径。

## API

- `POST /api/v1/stage1/jobs`: 提交完整流水线任务
- `GET /api/v1/stage1/jobs/{task_id}`: 查询任务状态
- `GET /api/v1/stage1/jobs/{task_id}/result`: 查询任务结果
- `GET /api/v1/health`: 健康检查

提交任务示例（URL）：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/stage1/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "threshold": 27.0,
    "min_scene_len": 1.0,
    "frame_quality": 2
  }'
```

提交任务示例（本地视频）：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/stage1/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "local_video_path": "/absolute/path/demo.mp4"
  }'
```

## CLI

```bash
python -m app.cli run-full --source-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
python -m app.cli run-full --local-video-path "/absolute/path/demo.mp4"
```

## 环境变量

- `STAGE1_DATA_DIR`：任务产物目录（默认 `./runtime/jobs`）
- `STAGE1_MAX_WORKERS`：并发 worker 数（默认 `2`）
- `STAGE1_TASK_TIMEOUT_SEC`：单任务最大执行秒数（默认 `3600`）
- `VLM_PROVIDER`：视觉模型提供方（`qwen`/`gemini`，默认 `qwen`）
- `EMBEDDING_PROVIDER`：向量提供方（`qwen`/`gemini`，默认跟随 `VLM_PROVIDER`）
- `QWEN_API_KEY`：Qwen API Key
- `QWEN_BASE_URL`：Qwen Chat Completions Endpoint（默认 DashScope compatible endpoint）
- `QWEN_EMBED_BASE_URL`：Qwen Embedding Endpoint（默认 DashScope compatible endpoint）
- `QWEN_VLM_MODEL`：视觉模型名（默认 `qwen3-vl-flash`）
- `QWEN_EMBED_MODEL`：向量模型名（默认 `text-embedding-v3`）
- `GEMINI_API_KEY`：Gemini API Key（必需）
- `GEMINI_VLM_MODEL`：VLM 模型名（默认 `gemini-1.5-pro`）
- `GEMINI_EMBED_MODEL`：Embedding 模型名（默认 `text-embedding-004`）
- `PIPELINE_RETRY_MAX`：模型调用重试次数（默认 `2`）
- `PIPELINE_BATCH_SIZE`：Stage2 批大小（默认 `4`）
- `STAGE1_DOTENV_PATH`：可选，指定 `.env` 文件路径
