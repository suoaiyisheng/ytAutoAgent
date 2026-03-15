# 当前算法 Stage 与 JSON Schema 梳理

## 1. 范围与结论

当前“主算法”对应 `Stage1Pipeline`，实际执行是 4 个 Stage：

1. 结构化提取（Stage 1）
2. 原子级感知（Stage 2）
3. 身份对齐与归一化（Stage 3）
4. 指令合成（Stage 4）

产出 5 份核心契约 JSON（`01` 到 `05`），由 `TaskStore.CONTRACT_FILES` 固定命名。

说明：

- 代码中出现的 `stage5_dump_path` 只是 Stage 4 的调试上下文导出，不是独立执行 Stage。
- `06_*` / `07_*` / `08_*` 属于独立的生图流水线（`ImageGenerationService`），不属于上述 4 个主算法 Stage。

---

## 2. Stage 总览（输入 / 处理 / 输出）

| Stage | 主要输入 | 主要处理 | 主要输出 |
| --- | --- | --- | --- |
| Stage 1 结构化提取 | 任务参数（source_url/local_video_path、阈值、抽帧质量等） | 视频获取、媒体探测、分镜检测、每镜抽首帧和 clip | `01_physical_manifest.json` |
| Stage 2 原子级感知 | Stage1 scenes 的 keyframe + 分镜时间 | 批量 VLM 视觉描述 + 标准化 | `02_raw_scene_descriptions.json` |
| Stage 3 身份对齐与归一化 | Stage1 keyframe 映射 + Stage2 描述 + Embedding | 人物展平、向量聚类、边界复核、Ref 映射、分镜对齐 | `03_character_bank.json`、`04_aligned_storyboard.json` |
| Stage 4 指令合成 | Stage3 输出 + 架构师提示词模板 | 构建 shot grounding、调用 VLM 合成最终提示词、reference_bindings 强校验 | `05_final_production_table.json`、`final_prompts.md` |

---

## 3. 任务入口参数 Schema（主算法输入）

对应 `POST /api/v1/stage1/jobs` 的请求体（`JobCreateRequest`）。

| 字段路径 | 类型 | 必填 | 可空 | 来源 |
| --- | --- | --- | --- | --- |
| `source_url` | string(url) | 条件必填（二选一） | 是 | API 请求体；与 `local_video_path` 互斥 |
| `local_video_path` | string(path) | 条件必填（二选一） | 是 | API 请求体；与 `source_url` 互斥 |
| `threshold` | number | 否（有默认值） | 否 | API 请求体，默认 `27.0` |
| `min_scene_len` | number | 否（有默认值） | 否 | API 请求体，默认 `1.0` |
| `frame_quality` | integer | 否（有默认值） | 否 | API 请求体，默认 `2` |
| `download_format` | string | 否（有默认值） | 否 | API 请求体，默认 `bestvideo+bestaudio/best` |
| `vlm_model` | string | 否 | 是 | API 请求体，空时回退到配置默认模型 |
| `embed_model` | string | 否 | 是 | API 请求体，空时回退到配置默认模型 |
| `batch_size` | integer | 否 | 是 | API 请求体，空时回退到配置默认值 |
| `retry_max` | integer | 否 | 是 | API 请求体，空时回退到配置默认值 |

---

## 4. Stage 1 输出 Schema：`01_physical_manifest.json`

### 4.1 顶层结构

| 字段路径 | 类型 | 必填 | 可空 | 来源 |
| --- | --- | --- | --- | --- |
| `project_id` | string | 是 | 否 | `task.task_id` |
| `video_metadata` | object | 是 | 否 | Stage1 组装 |
| `video_metadata.source_url` | string | 是 | 是（兜底空串） | `params.source_url` 或 `params.local_video_path` |
| `video_metadata.fps` | number | 是 | 否 | `MediaProbe.probe(video_path)` |
| `video_metadata.resolution` | string | 是 | 否 | `MediaProbe.probe(video_path)` |
| `scenes` | array<object> | 是 | 否（至少 1 个，否则直接报错） | `SceneDetector.detect` + 资产提取 |

### 4.2 `scenes[]` 子结构

| 字段路径 | 类型 | 必填 | 可空 | 来源 |
| --- | --- | --- | --- | --- |
| `scenes[].scene_id` | integer | 是 | 否 | 分镜序号（从 1 开始） |
| `scenes[].start_time` | string(HH:MM:SS.mmm) | 是 | 否 | SceneBoundary.start 经 `_sec_to_hms` |
| `scenes[].end_time` | string(HH:MM:SS.mmm) | 是 | 否 | SceneBoundary.end 经 `_sec_to_hms` |
| `scenes[].keyframe_path` | string(path) | 是 | 否 | `FrameExtractor.extract_first_frame` |
| `scenes[].clip_path` | string(path) | 是 | 否 | `FrameExtractor.extract_clip` |

---

## 5. Stage 2 输出 Schema：`02_raw_scene_descriptions.json`

### 5.1 顶层结构

| 字段路径 | 类型 | 必填 | 可空 | 来源 |
| --- | --- | --- | --- | --- |
| `project_id` | string | 是 | 否 | `task.task_id` |
| `scenes` | array<object> | 是 | 否 | 对 Stage1 全部分镜逐镜归一化 |

### 5.2 `scenes[]` 子结构

| 字段路径 | 类型 | 必填 | 可空 | 来源 |
| --- | --- | --- | --- | --- |
| `scenes[].scene_id` | integer | 是 | 否 | 对应 Stage1 `scene_id` |
| `scenes[].visual_analysis` | object | 是 | 否 | Stage2 归一化结果 |
| `scenes[].visual_analysis.subjects` | array<object> | 是 | 是（无人镜头可空数组） | VLM 输出，归一化补齐 |
| `scenes[].visual_analysis.environment` | object | 是 | 否 | VLM 输出，缺失时补空字段 |
| `scenes[].visual_analysis.camera` | object | 是 | 否 | VLM 输出，缺失时补空字段 |

### 5.3 `subjects[]` 子结构

| 字段路径 | 类型 | 必填 | 可空 | 来源 |
| --- | --- | --- | --- | --- |
| `subjects[].temp_id` | string | 是 | 否（最少自动补 `subject_n`） | VLM `temp_id` 或归一化默认值 |
| `subjects[].appearance` | string | 是 | 是 | VLM 输出（trim） |
| `subjects[].action` | string | 是 | 是 | VLM 输出（trim） |
| `subjects[].expression` | string | 是 | 是 | VLM 输出（trim） |

### 5.4 `environment` / `camera` 子结构

| 字段路径 | 类型 | 必填 | 可空 | 来源 |
| --- | --- | --- | --- | --- |
| `environment.location` | string | 是 | 是 | VLM 输出或空串兜底 |
| `environment.lighting` | string | 是 | 是 | VLM 输出或空串兜底 |
| `environment.atmosphere` | string | 是 | 是 | VLM 输出或空串兜底 |
| `camera.shot_size` | string | 是 | 是 | VLM 输出或空串兜底 |
| `camera.angle` | string | 是 | 是 | VLM 输出或空串兜底 |
| `camera.movement` | string | 是 | 是 | VLM 输出或空串兜底 |

---

## 6. Stage 3 输出 Schema（一）：`03_character_bank.json`

### 6.1 顶层结构

| 字段路径 | 类型 | 必填 | 可空 | 来源 |
| --- | --- | --- | --- | --- |
| `project_id` | string | 是 | 否 | `task.task_id` |
| `characters` | array<object> | 是 | 是（可能为空数组） | 聚类 + 复核 + 公共字段投影 |
| `global_style` | string | 是 | 否 | 当前固定值：`真实摄影风格` |

### 6.2 `characters[]` 子结构

| 字段路径 | 类型 | 必填 | 可空 | 来源 |
| --- | --- | --- | --- | --- |
| `characters[].ref_id` | string (`Ref_n`) | 是 | 否 | Stage3 分配 |
| `characters[].name` | string | 是 | 是 | 复核结果或空串 |
| `characters[].master_description` | string | 是 | 是 | 多 appearance 合并 |
| `characters[].key_features` | array<string> | 是 | 是（可空数组） | 从 `master_description` 派生 |
| `characters[].ref_image_path` | string(path) | 是 | 是 | 聚类首成员 keyframe 或复核回写 |
| `characters[].scene_presence` | array<[integer,string]> | 是 | 是（理论可空，通常非空） | 聚类成员中的 `(scene_id,temp_id)` |

---

## 7. Stage 3 输出 Schema（二）：`04_aligned_storyboard.json`

### 7.1 顶层结构

| 字段路径 | 类型 | 必填 | 可空 | 来源 |
| --- | --- | --- | --- | --- |
| `project_id` | string | 是 | 否 | 取自 Stage2 `project_id` |
| `scenes` | array<object> | 是 | 否 | 对 Stage2 场景按 `scene_id` 排序重建 |

### 7.2 `scenes[]` 子结构

| 字段路径 | 类型 | 必填 | 可空 | 来源 |
| --- | --- | --- | --- | --- |
| `scenes[].scene_id` | integer | 是 | 否 | Stage2 `scene_id` |
| `scenes[].visual_analysis` | object | 是 | 否 | Stage2 内容 + 主体 ID 对齐 |
| `scenes[].visual_analysis.subjects` | array<object> | 是 | 是 | 由 `temp_id -> Ref_n` 映射后生成 |
| `scenes[].visual_analysis.environment` | object | 是 | 否 | 从 Stage2 透传 |
| `scenes[].visual_analysis.camera` | object | 是 | 否 | 从 Stage2 透传 |

### 7.3 `subjects[]` 子结构（对齐后）

| 字段路径 | 类型 | 必填 | 可空 | 来源 |
| --- | --- | --- | --- | --- |
| `subjects[].id` | string (`Ref_n`) | 是 | 是（匹配失败时空串） | 显式映射或描述匹配 |
| `subjects[].appearance` | string | 是 | 是 | Stage2 透传 |
| `subjects[].action` | string | 是 | 是 | Stage2 透传 |
| `subjects[].expression` | string | 是 | 是 | Stage2 透传 |

### 7.4 `environment` / `camera` 子结构

| 字段路径 | 类型 | 必填 | 可空 | 来源 |
| --- | --- | --- | --- | --- |
| `environment.location` | string | 是 | 是 | Stage2 透传 |
| `environment.lighting` | string | 是 | 是 | Stage2 透传 |
| `environment.atmosphere` | string | 是 | 是 | Stage2 透传 |
| `camera.shot_size` | string | 是 | 是 | Stage2 透传 |
| `camera.angle` | string | 是 | 是 | Stage2 透传 |
| `camera.movement` | string | 是 | 是 | Stage2 透传 |

---

## 8. Stage 4 输出 Schema：`05_final_production_table.json`

### 8.1 顶层结构

| 字段路径 | 类型 | 必填 | 可空 | 来源 |
| --- | --- | --- | --- | --- |
| `project_id` | string | 是 | 否 | 最终强制写为 `task.task_id` |
| `prompts` | array<object> | 是 | 是（边界场景可空数组） | VLM 输出后归一化 |

### 8.2 `prompts[]` 子结构

| 字段路径 | 类型 | 必填 | 可空 | 来源 |
| --- | --- | --- | --- | --- |
| `prompts[].shot_id` | integer | 是 | 否 | Stage4 归一化（从模型返回读取） |
| `prompts[].reference_bindings` | array<object> | 是 | 是 | 若模型返回不一致，则回退到系统期望绑定 |
| `prompts[].image_prompt` | string | 是 | 是 | 模型返回后 trim |
| `prompts[].video_prompt` | string | 是 | 是 | 模型返回后 trim |

### 8.3 `reference_bindings[]` 子结构

| 字段路径 | 类型 | 必填 | 可空 | 来源 |
| --- | --- | --- | --- | --- |
| `reference_bindings[].reference_index` | integer | 是 | 否 | 由 shot 的角色顺序生成或模型返回 |
| `reference_bindings[].ref_id` | string (`Ref_n`) | 是 | 是（无匹配时可能空串） | 角色映射 |

---

## 9. 非 Stage 契约但常用产物

### 9.1 `index.json`

用于汇总最终路径与统计，不直接参与下一阶段推理。

| 字段路径 | 类型 | 必填 | 可空 | 来源 |
| --- | --- | --- | --- | --- |
| `project_id` | string | 是 | 否 | `task.task_id` |
| `contracts` | object | 是 | 否 | `01~05` 绝对路径 |
| `artifacts` | object | 是 | 否 | `final_prompts.md`、`index.json` 路径 |
| `stats.scene_count` | integer | 是 | 否 | `01.scenes` 长度 |
| `stats.character_count` | integer | 是 | 否 | `03.characters` 长度 |
| `stats.prompt_count` | integer | 是 | 否 | `05.prompts` 长度 |

### 9.2 可选调试文件：`stage05_context_{task_id}.json`

仅在请求参数提供 `stage5_dump_path` 时生成，包含提示词模板、provider 请求/响应、最终表快照。

---

## 10. 字段规则约定（读表说明）

1. “必填=是”表示该 key 会被流水线显式写出（即使值为空串/空数组）。
2. “可空=是”表示值可能为空串、空数组或空对象（依字段定义）。
3. “来源”优先指明“哪个 Stage 的哪个输入/中间结果”决定该字段值，便于定位问题。

