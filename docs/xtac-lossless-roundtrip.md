# XTac 双向转换与无损往返

`lerobot2mcap` 和 `mcap2lerobot` 现在都默认使用 `xtac`。
新导出的 MCAP 使用 `train/state`、`train/action` 的 JSON 消息，以及
`foxglove.CompressedVideo` protobuf 视频消息；不再需要为新导出结果指定 CDR 方言。

```bash
lerobot2mcap convert /datasets/source-lerobot -o /datasets/source-xtac
mcap2lerobot /datasets/source-xtac -o /datasets/restored-lerobot
```

输出必须是不存在的新目录，并且位于源数据目录之外。CLI 与统一 Python facade
的 `dialect`、`preserve_source` 默认行为相同。
源快照要求普通文件和目录，遇到符号链接或特殊文件会明确拒绝，避免伪称完整恢复。

## 无损的范围

两种格式的表达能力不同：LeRobot 的逐帧训练表不能表示原 XTac MCAP 中所有异步传感器、
channel schema、publish timestamp、metadata record 等内容；反过来，普通 XTac
topic 也不能表达任意 parquet 列、自定义文件和 LeRobot 原文件布局。

因此默认同时保存可读取的数据流和带校验的源内容：

| 链路 | 常规数据流或训练投影 | 完整返回所需内容 |
| --- | --- | --- |
| LeRobot → XTac → LeRobot | state/action、额外数值特征、任务文本、相机视频 | MCAP 附件中的源文件块和完整清单，恢复原 LeRobot 文件内容和目录结构 |
| 原始 XTac → LeRobot → XTac | 同步后的 state/action、tracker confidence、相机视频及采集元数据 | `meta/source_mcap/` 中的原始 MCAP，返回时逐字节恢复，保留所有未映射流 |

“完整无损”指完整返回后源文件集合、内容和空目录一致，包括原始 parquet、MP4、
校准文件、统计、时间戳、索引和自定义文件。不承诺操作系统文件时间、权限或 inode 相同。
原始 MCAP 返回时文件名和文件内容保持一致。
多个输入目录中的文件以共同父目录为基准保留相对路径，避免同名 MCAP 相互覆盖。

**训练投影仍然采用 state 为基准的最近邻同步。** 它不把全部异步传感器样本塞进逐帧表，
也不声称原始 MCAP 在 LeRobot 中变成了相同的普通特征。
完整源内容依靠保留载荷返回，不能删除这些载荷后继续要求完整恢复。

## 视频与特征

- 新 XTac 导出对原视频解码得到的 RGB 使用 `libx264rgb`、`CRF=0`、`rgb24`
  编码，不进行 YUV420 色度降采样；读取后 RGB 像素保持一致。
- XTac 写入 LeRobot 的普通训练投影也使用 RGB 无损编码。该路径要求
  **LeRobot >=0.6.1,<0.7**，项目依赖下限同步更新。0.5.x fork 不再满足默认路径要求。
- 原视频压缩字节并不等于新编码字节；原文件通过保留载荷原样恢复。
- 导出额外数值特征到 `lerobot/features/<feature>` JSON topic，包括
  `observation.tracker_confidence`；在 `xense_lerobot` metadata 中声明名称、dtype、shape、
  相机 topic 映射和任务索引。额外相机使用 `camera/lerobot/<feature>/h264`。
- 原采集器的 8 个标准相机 topic 保持不变。原 XTac confidence 仍支持从 `/pico/pose` 提取。
- 原 XTac 没有 `fps` 时优先使用 `nominal_encoder_fps`，再走时间戳推断回退。
- 源文件载荷覆盖普通训练表无法表示的字符串、图像对象等内容；禁用源保留后不保证这些内容往返。

RGB 无损、全关键帧编码通常比原来的有损帧间视频大得多。源文件载荷还会额外占用
约一份原始数据的空间。不要将该变化误认为重复帧；这是默认保真策略的存储成本。

## 校验与异常处理

源文件按最多 8 MiB 的附件块分配到不同 episode，避免把整个数据集或大视频读入内存。
每个 MCAP 都携带同一源清单及本文件 episode 身份。恢复时检查：

1. 输入来自相同源快照，且全部导出 episode 齐全，没有重复或混入其他 MCAP。
2. XTac topic/schema/消息时间、内容和 metadata 的投影摘要没有变化。
3. 附件块完整，路径安全，源文件 SHA-256 与清单一致。
4. 返回原始 MCAP 时，LeRobot 投影和保存的源 MCAP 校验值都没有变化。

缺集、篡改、过期源快照或路径穿越均报错。源文件恢复和 XTac 导出在临时目录完成，
通过检查后才发布到目标目录，避免留下看似完整的部分恢复结果。
普通原始 MCAP→LeRobot 的 writer 失败仍按原有转换器错误处理，不应把失败目录当作已完成。

## 修改、筛选与兼容

原始 XTac 也可以作为完整往返的起点，两步都使用默认参数：

```bash
mcap2lerobot /datasets/raw-xtac -o /datasets/projected-lerobot
lerobot2mcap convert /datasets/projected-lerobot -o /datasets/returned-xtac
```

返回的是原始 MCAP 文件，文件名和 SHA-256 应与起点一致；原始 H.264 压缩字节、
未映射传感器 topic、schema、消息时间和采集 metadata 均随原文件保留。
`meta/source_mcap/` 是该往返所需的数据，不能作为可删除的转换缓存处理。
中间 LeRobot 的普通训练表和 RGB 视频也应单独验证，不能仅凭原文件恢复成功
推断训练投影正确。若修改中间数据，默认返回会拒绝过期源快照；此时需选择下面的
显式转换模式。

需要筛选部分 episode、修改任务/FPS、或将编辑后的训练投影重新导出时，显式使用
`--no-preserve-source`（Python 为 `preserve_source=False`）。这表示按当前数据重新转换，
不承诺原始文件完整恢复。默认完整源恢复不允许在恢复时应用任务/FPS/robot type 覆盖。
原始 MCAP 首次转换的任务等覆盖只改变训练投影；保留的原 MCAP 仍然是转换前内容。

```bash
lerobot2mcap convert /datasets/source-lerobot -o /datasets/subset-xtac \
  --episodes 0 2 --no-preserve-source
mcap2lerobot /datasets/subset-xtac -o /datasets/subset-lerobot \
  --no-preserve-source --task "新的任务说明"
```

旧 CDR 导出仍可显式选择，但不提供新的完整无损保证：

```bash
lerobot2mcap convert /datasets/source-lerobot -o /datasets/legacy-cdr \
  --dialect lerobot_roundtrip
mcap2lerobot /datasets/legacy-cdr -o /datasets/legacy-restored \
  --dialect lerobot_roundtrip
```

已经导出的旧 CDR MCAP 不会因为更新程序而自动补回 confidence 或校准数据；
需要从原 LeRobot 重新导出 XTac。`--video-crf`、`--video-preset` 与 converter YAML
只影响旧 CDR 路径，新 XTac 路径固定采用 RGB 无损视频策略。

## 验证入口

```bash
pytest -q tests/test_xtac_roundtrip.py
pytest -q
ruff check --select E9,F63,F7,F82 .
```

新测试使用真实 MCAP/protobuf、随机 RGB 视频、两集数据、多任务、confidence、uint64
额外特征和校准文件。安装完整依赖时，还运行真实 LeRobot writer 和统一 facade 的双向
往返测试。轻量 CI 未安装 LeRobot 时明确跳过相应集成测试，不将跳过计为通过。

## 真实数据全量验证记录

2026-10-20 使用 `xtac-umi-g1-remove-wire-harness-260923` 的两份现有源数据验证：

| 起点及链路 | 覆盖范围 | 结果 |
| --- | --- | --- |
| `…-lerobot`：LeRobot → XTac → LeRobot | 10 集、13,348 帧、8 路相机 | state/action/confidence 一致，全部视频解码检查通过，返回后的完整源文件清单和 SHA-256 一致 |
| `…-mcap`：XTac → LeRobot → XTac | 10 集、13,341 帧、8 路相机，共 106,728 张 RGB 图像 | 全部训练数值、索引、时间戳检查通过，全部 RGB 像素一致，返回的 10 个原始 MCAP 文件名和 SHA-256 一致 |

两行使用不同的现有源数据，帧数差异来自起点，不是往返丢帧。
原始 XTac 的逐集帧数为
`[1451, 1313, 1196, 1435, 1254, 1256, 1314, 1278, 1348, 1496]`，
10 个源 MCAP 共 595,459,191 字节。验证时先将 SMB 原文件复制到本地临时目录，
对从 SMB 读取的字节和本地副本分别计算 SHA-256 并核对；转换使用默认 XTac 方言
和源保留策略、真实 LeRobot writer，并完成 `finalize()`。

训练表检查按声明 dtype 比较 state/action/confidence，检查逐集长度、索引及
`frame_index / fps` 时间戳；视频检查按相机拼接全部帧的 RGB 字节并计算摘要，
与源 MCAP 按同步规则选出的 RGB 字节比较。最终原文件 SHA-256 一致还覆盖
未投影 topic、schema、消息时间、metadata 和原始压缩视频字节。
这些结果不表示 LeRobot 训练表能独立表达全部异步传感器记录，完整返回仍需保留源载荷。

本次补强的自动回归还同时验证 XTac 起点的中间训练投影与原文件返回，防止仅有
源快照恢复成功掩盖训练数据错误。完整本地套件为 158 通过、8 跳过；
Python 3.12 轻量依赖套件为 153 通过、13 跳过。
