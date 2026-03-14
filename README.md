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
- `POST /api/v1/stage1/jobs/{task_id}/image-generation`: 启动生图任务
- `GET /api/v1/stage1/jobs/{task_id}/image-generation`: 查询生图状态
- `GET /api/v1/stage1/jobs/{task_id}/image-generation/result`: 查询生图结果
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
python -m app.cli generate-images --job-id "your_job_id" --shot-range "1-3"
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
- `IMAGE_GENERATION_PROVIDER`：生图提供方（`openrouter`/`volcengine_jimeng30`，默认 `openrouter`）
- `OPENROUTER_API_KEY`：OpenRouter Key（生图）
- `OPENROUTER_BASE_URL`：OpenRouter API Base URL（默认 `https://openrouter.ai/api/v1`）
- `OPENROUTER_IMAGE_MODEL`：生图模型（默认 `google/gemini-2.5-flash-image`）
- `VOLCENGINE_ACCESS_KEY_ID`：火山引擎 AccessKey ID（即梦3.0）
- `VOLCENGINE_ACCESS_KEY_SECRET`：火山引擎 AccessKey Secret（即梦3.0）
- `VOLCENGINE_SESSION_TOKEN`：可选，STS 临时凭证 Token
- `VOLCENGINE_VISUAL_HOST`：即梦接口域名（默认 `visual.volcengineapi.com`）
- `VOLCENGINE_REGION`：签名 Region（默认 `cn-north-1`）
- `VOLCENGINE_SERVICE`：签名 Service（默认 `cv`）
- `VOLCENGINE_JIMENG_REQ_KEY`：即梦文生图 req_key（默认 `jimeng_t2i_v30`）
- `VOLCENGINE_JIMENG_VERSION`：接口版本（默认 `2022-08-31`）
- `VOLCENGINE_POLL_INTERVAL_SEC`：查询轮询间隔秒数（默认 `2`）
- `VOLCENGINE_POLL_TIMEOUT_SEC`：查询超时秒数（默认 `120`）
- `ALIYUN_OSS_REGION`：阿里云 OSS Region
- `ALIYUN_OSS_ACCESS_KEY_ID`：阿里云 OSS Access Key ID
- `ALIYUN_OSS_ACCESS_KEY_SECRET`：阿里云 OSS Access Key Secret
- `ALIYUN_OSS_BUCKET`：阿里云 OSS Bucket
- `ALIYUN_OSS_PUBLIC_DOMAIN`：可选，OSS 自定义公网域名
