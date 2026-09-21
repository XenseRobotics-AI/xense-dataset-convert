# DAS-Ego / DAS-UMI 转 LeRobot 使用指南

本文面向第一次使用 `mcap2lerobot` 的用户，说明如何把 GenRobot DAS
设备产生的 MCAP 转换为 LeRobot v3 数据集，以及 DAS-UMI semantic schema
v3 的字段、对齐、动作和审计规则。

相关实现和配置：

- 默认 DAS-UMI 话题及质量阈值：[`../configs/das_umi.yaml`](../configs/das_umi.yaml)
- 通用 MCAP 文档：[`../docs/mcap2lerobot.md`](../docs/mcap2lerobot.md)
- 统一 Python 入口：[`../xense_dataset_convert/__init__.py`](../xense_dataset_convert/__init__.py)
- MCAP adapter：[`../xense_dataset_convert/adapters/mcap_to_lerobot.py`](../xense_dataset_convert/adapters/mcap_to_lerobot.py)

## 1. 输出位置规则：与 main 保持一致

已对比本仓库 `main`、`origin/main` 和当前 feat 分支，三者遵循同一条规则：

1. Python 统一入口的 `output=` 和 CLI 的 `--output/-o` 都是必填项。
2. `output` 经 `expanduser()` 和 `resolve()` 后，直接成为 LeRobot 数据集根目录。
3. 转换器不会自动追加 `output/`、任务 ID、时间戳、repo ID 或输入文件名。
4. `repo_id` 只影响 LeRobot 数据集标识；未提供时使用输出目录名，不改变磁盘位置。
5. 已存在的输出目录不会被覆盖，转换会直接失败。

例如：

```text
输入：/data/episode_001/merged.mcap
输出参数：/data/episode_001/merged_lerobot_v3
```

最终数据集就在：

```text
/data/episode_001/merged_lerobot_v3/
```

DAS-UMI 的预检和转换报告默认放在数据集同级，而不是数据集内部：

```text
/data/episode_001/merged_lerobot_v3_reports/
├── preflight.md
└── conversion.md
```

实际写入时先在输出目录同级创建隐藏临时目录，全部 Parquet、视频、LeRobot
加载和 schema 校验通过后，再原子发布到指定 `output`。失败不会覆盖已有数据；
若写出后校验失败，保留的临时目录会记录在 `conversion.md`。

## 2. DAS-Ego 和 DAS-UMI 的区别

### DAS-Ego

只有 `robot0` Ego 设备：

- 6 路相机：`robot0/sensor/camera0` … `camera5`
- 1 路 6 轴 IMU：`robot0/sensor/imu`
- 输出为 30 Hz observation-only 数据集
- 不生成 `observation.state` 和 `action`
- 保留标准 LeRobot 时间轴和兼容的 `observation.source_timestamp`
- 不使用 DAS-UMI strict preflight；`--check-only` 只支持 DAS-UMI

### DAS-UMI

一个 MCAP 中合并 Ego、左 Finger 和右 Finger。默认要求 15 路连续流：

| 类型 | 数量 | 输出或用途 |
|---|---:|---|
| Ego 相机 | 6 | `observation.images.camera0` … `camera5` |
| 左右腕相机 | 2 | `observation.images.left_wrist/right_wrist` |
| Ego/左腕/右腕 IMU | 3 | 三个 observation-only IMU feature |
| 左右磁编码器 | 2 | 生成归一化夹爪 state/action |
| Head world pose | 1 | 生成 canonical head state/action |
| Head relative pose | 1 | observation-only 辅助位姿 |

只要 MCAP 中出现 `robot1/` 或 `robot2/` 话题，自动检测就优先认定为
DAS-UMI。Finger 数据不完整时会按 DAS-UMI 报错，不会静默降级成 DAS-Ego。

## 3. 环境准备

推荐在仓库根目录安装：

```bash
cd /home/xense/src/DAS/xense-dataset-convert
pip install -e '.[dev]'
```

这台机器已验证可用的解释器是：

```text
/home/xense/miniforge3/envs/lerobot/bin/python
```

快速检查：

```bash
/home/xense/miniforge3/envs/lerobot/bin/python -c \
  'import lerobot, mcap, mcap_protobuf, pyarrow; print(lerobot.__version__)'
```

如果使用其他环境，需要同时安装 LeRobot、PyAV、MCAP protobuf 支持、NumPy、
PyArrow 和 PyYAML。MCAP 到 LeRobot 的视频写出使用 PyAV，不要求调用外部
`ffmpeg` 命令。

下面统一使用 `python -m mcap2lerobot.cli`，这样不依赖 console script 是否已加入
`PATH`。执行 `pip install -e '.[dev]'` 后也可将它替换为等价的 `mcap2lerobot`。

## 4. 推荐转换流程

### 4.1 先运行 DAS-UMI 预检

`--check-only` 仍要求提供 `--output`，但不会创建数据集目录；该路径用于确定
默认报告目录名。

```bash
python -m mcap2lerobot.cli /data/merged.mcap \
  --output /data/merged_lerobot_v3 \
  --profile das-umi \
  --schema-version 3 \
  --config configs/das_umi.yaml \
  --check-only
```

查看：

```text
/data/merged_lerobot_v3_reports/preflight.md
/data/merged_lerobot_v3_reports/conversion.md
```

只有 `preflight.md` 为 PASS 才执行正式转换。

### 4.2 正式转换 DAS-UMI

```bash
python -m mcap2lerobot.cli /data/merged.mcap \
  --output /data/merged_lerobot_v3 \
  --repo-id local/merged_lerobot_v3 \
  --profile das-umi \
  --schema-version 3 \
  --config configs/das_umi.yaml \
  --report-dir /data/merged_lerobot_v3_reports
```

`action` 默认开启。若只需要 observation/state，不需要下一帧动作目标：

```bash
python -m mcap2lerobot.cli /data/merged.mcap \
  --output /data/merged_observation_only \
  --profile das-umi \
  --schema-version 3 \
  --no-export-action
```

关闭 action 后不会丢弃 episode 最后一行。

### 4.3 转换 DAS-Ego

```bash
python -m mcap2lerobot.cli /data/das_ego.mcap \
  --output /data/das_ego_lerobot \
  --repo-id local/das_ego_lerobot \
  --profile das-ego \
  --fps 30
```

也可以用默认的 `--profile auto` 自动识别；生产任务建议显式指定 profile，方便
命令审计和复现。

### 4.4 使用主框架统一 Python API

这是与 main 最一致的调用方式：

```python
from xense_dataset_convert import Format, convert

result = convert(
    Format.MCAP,
    Format.LEROBOT,
    inputs="/data/merged.mcap",
    output="/data/merged_lerobot_v3",
    repo_id="local/merged_lerobot_v3",
    profile="das-umi",
    schema_version=3,
    config="/home/xense/src/DAS/xense-dataset-convert/configs/das_umi.yaml",
    report_dir="/data/merged_lerobot_v3_reports",
)

print(result.output_path)
print(result.details)
```

`result.output_path` 就是传入的 `output` 解析后的绝对路径。

## 5. 转换后的目录结构

典型 DAS-UMI v3 数据集：

```text
<output>/
├── data/
│   └── chunk-000/file-000.parquet
├── videos/
│   ├── observation.images.camera0/chunk-000/file-000.mp4
│   ├── ...
│   └── observation.images.right_wrist/chunk-000/file-000.mp4
└── meta/
    ├── episodes/chunk-000/file-000.parquet
    ├── info.json
    ├── stats.json
    ├── tasks.parquet
    ├── das_umi_schema.json
    ├── das_umi_topic_mapping.json
    ├── das_umi_alignment.parquet
    └── das_umi_calibration.json
```

其中：

- `meta/info.json`：LeRobot v3 feature schema、fps、robot type 和路径模板。
- `meta/das_umi_schema.json`：DAS-UMI semantic schema、action、夹爪及坐标语义。
- `meta/das_umi_topic_mapping.json`：源 MCAP topic 到输出 feature/component 的映射。
- `meta/das_umi_alignment.parquet`：逐帧原始时钟、源消息序号及同步误差。
- `meta/das_umi_calibration.json`：MCAP 中识别到的标定消息。

不要只复制 `data/` 或 `videos/`；一个可加载的 LeRobot 数据集需要整个
`<output>` 目录。

## 6. DAS-UMI schema v3 细则

这里的 `DAS-UMI schema v3` 是本转换器的语义版本。它与
`meta/info.json` 中 LeRobot 的 `codebase_version: v3.0` 不是同一个版本号。

### 6.1 固定时间轴和同步

所有连续流先取共同可用区间，再生成精确的整数纳秒 30 Hz 网格：

```text
T[k] = start_ns + floor(k × 1,000,000,000 / 30)
```

视频、IMU、编码器和 pose 都用有界最近邻匹配到该网格。平局取时间较后的
样本。每一路使用自身采样率对应的半周期上限，再增加配置中的少量 slack。

预检会拒绝：

- 必需流缺失或没有共同区间；
- 时间戳重复、倒退或部分消息混用 header/log clock；
- 序号丢失、采样周期异常、端点或同步误差超限；
- 非有限数值、无效 quaternion；
- 视频中段损坏、分辨率改变；
- 多设备相机跨度超过配置阈值。

### 6.2 state 和 action

默认 `observation.state` 与 `action` 都是 float32、11 维且名称完全一致：

| 索引 | 名称 | 语义 |
|---:|---|---|
| 0 | `left_gripper.pos` | 左夹爪，0 闭合 / 1 张开 |
| 1 | `right_gripper.pos` | 右夹爪，0 闭合 / 1 张开 |
| 2–4 | `head_camera.x/y/z` | Head world position |
| 5–10 | `head_camera.r1` … `r6` | world rotation 的前两列 |

rotation 6D 的顺序是：

```text
[R00, R10, R20, R01, R11, R21]
```

转换器不执行到 TacCap/XTac 相机坐标系的变换。

完整对齐状态为 `S[0…N-1]` 时：

```text
observation.state = S[0…N-2]
action            = S[1…N-1]
```

所以 `action[t]` 是下一帧绝对状态代理，不是 delta，也不是设备记录的原始控制器
command。启用 action 时，每个 episode 丢弃最后一个没有下一帧目标的 observation。

### 6.3 v3 夹爪归一化

DAS Fingers 官方标称最大开口为 220 mm。默认配置：

```yaml
gripper_normalization:
  min_opening_m: 0.000
  max_opening_m: 0.220
  decimals: 2
```

处理顺序：

```text
clipped_m = clip(distance_m, 0.000, 0.220)
normalized = clipped_m / 0.220
output = round_half_up(normalized, 2)
```

约定：

- 完全闭合为 `0.00`；
- 完全张开为 `1.00`；
- 小于 0 m 的输入截断为 `0.00`；
- 大于 0.220 m 的输入截断为 `1.00`；
- state 和 action 都使用同一份归一化矩阵；
- 写出后再次校验范围和两位小数量化。

注意 float32 在打印时可能显示为 `0.01999999955`，这不表示量化失败；校验按
float32 容差确认其对应 `0.02`。

原始开口距离不作为训练 feature 重复写入。每侧对齐后的原始最小/最大值、低端
和高端截断数量记录在 `meta/das_umi_schema.json`。

官方规格参考：<https://cn.genrobot.com/products/finger>

### 6.4 observation-only 辅助数据

以下数据是传感器观测，不生成同名 action：

| Feature | Shape | 说明 |
|---|---:|---|
| `observation.imu` | 6 | Ego 角速度 XYZ + 线加速度 XYZ |
| `observation.imu.left_wrist` | 6 | 左 Finger IMU |
| `observation.imu.right_wrist` | 6 | 右 Finger IMU |
| `observation.pose.head_relative` | 9 | relative XYZ + rotation 6D |

`head_world` 已经用于生成 canonical head state/action，v3 不再重复导出
`observation.pose.head_world`。IMU 和 relative pose 不属于机器人可执行命令，不能
为了可视化中出现实线/虚线而伪造成 action。

### 6.5 时间戳和审计数据

主训练 Parquet 只保留 LeRobot 标准时间字段：

```text
timestamp
frame_index
episode_index
index
task_index
```

源设备纳秒时钟不作为普通 observation feature，因此不会污染统计、策略输入或
LeRobot 曲线列表。它们保存在 `meta/das_umi_alignment.parquet`：

```text
episode_index
frame_index
target_grid_timestamp_ns
<stream>.source_timestamp_ns
<stream>.source_index
<stream>.sync_error_ns
```

其中：

```text
sync_error_ns = source_timestamp_ns - target_grid_timestamp_ns
```

全部源时间戳保持 int64 纳秒精度。

### 6.6 schema 兼容选项

| 参数 | 行为 |
|---|---|
| `--schema-version 3` | 默认；v2 sidecar 设计 + 0–220 mm 夹爪归一化 |
| `--schema-version 2` | 保留 sidecar，但夹爪维持旧版原始 float32 |
| `--schema-version 1` | 旧夹爪值、内联 `observation.source_*`、重复 world pose |

新数据应使用 v3。v1/v2 只用于复现历史输出，不应作为新训练集默认格式。

## 7. 与 `xtac-umi-g1-block-to-box` 的区别

两者的 LeRobot 存储格式相同，均为 `codebase_version: v3.0`、30 Hz，且逐 episode
满足：

```text
action[t] == observation.state[t + 1]
```

但硬件 feature schema 不相同：

| 项目 | DAS-UMI schema v3 | XTac 示例集 |
|---|---|---|
| `robot_type` | `das_umi` | `xtac_umi_g1` |
| state/action | 11 维 | 29 维 |
| 左右手 TCP | 默认没有 | 左右各 9 维 |
| Head 名称 | `head_camera.*` | `head.*` |
| 夹爪 | 0–220 mm 映射到 0–1，保留两位小数 | 0–1 连续 float32，不做两位小数量化 |
| 辅助数值 | 3 组 IMU + head relative pose | 2 维 tracker confidence |
| 视频 | 6 路 Ego + 2 路腕部 | 2 路头部 + 2 路腕部 + 4 路触觉 |
| 分辨率 | 默认 1600×1300 | 头/腕 640×480，触觉 700×400 |
| 审计元数据 | alignment/schema/topic mapping/calibration | hardware/xumi devices/TacCap extrinsics |

XTac 的 29 维顺序是：

```text
left_tcp(9), left_gripper(1),
right_tcp(9), right_gripper(1),
head(9)
```

DAS 默认只有：

```text
left_gripper(1), right_gripper(1), head_camera(9)
```

DAS 源文件没有左右手 TCP 时，转换器不会填零或伪造，因此不能仅通过重命名得到
XTac 的 29 维 schema。依赖 XTac 29 维输入或相机 key 的模型不能直接加载 DAS v3；
需要相应 policy 配置或真实手部 pose 数据。

夹爪方面，两者的目标语义一致：`0 = closed`、`1 = open`。但 DAS v3 按要求只
保留两位小数，而 XTac 示例保留高精度连续 float32，因此二者不是逐位相同的数值
编码。

参考数据集：

```text
/home/xense/.cache/huggingface/lerobot/TacVerse/xtac-umi-g1-block-to-box
```

## 8. 转换后检查与可视化

先确认报告：

```bash
sed -n '1,80p' /data/merged_lerobot_v3_reports/conversion.md
```

必须看到：

```text
Status: PASS
Gripper range [0, 1] and two-decimal quantization: PASS
Shifted action equality ...: PASS
```

使用 LeRobot 本地可视化：

```bash
lerobot-dataset-viz \
  --repo-id local/merged_lerobot_v3 \
  --root /data/merged_lerobot_v3 \
  --episode-index 0
```

预期表现：

- `observation.state` 与 `action` 的同名控制维度形成 current/next-frame 曲线；
- 三组 IMU 和 `head_relative` 只有 observation 曲线；
- 不应再看到 `observation.source_timestamp.*`；
- 不应再看到重复的 `observation.pose.head_world`。

## 9. 常见问题

### `lerobot is required to write the dataset`

当前 Python 环境没有 LeRobot。切换到已安装完整依赖的环境，或执行
`pip install -e '.[dev]'`。

### `Output path already exists`

转换器遵循 main 的非覆盖规则。更换新的 `--output`，或由用户确认并自行处理旧
目录；转换器不会自动删除它。

### 出现任意 Finger topic 后缺少其他 Finger 流

这是 partial DAS-UMI。自动检测会进入 strict DAS-UMI 并在预检报告中列出缺失流，
不会当成 DAS-Ego 转换。

### 夹爪归一化后数值范围没有覆盖 0–1

归一化不会强制一段演示必须经历完全闭合和完全张开。若原始操作只覆盖部分行程，
输出自然只覆盖 `[0, 1]` 的子区间。检查 `meta/das_umi_schema.json` 中每侧的
`raw_min_m`、`raw_max_m` 和截断计数。

### IMU 没有 action 虚线

这是正确行为。IMU 是 sensor observation，不是可执行 command。
