# mcap2lerobot 流式读取设计

本文记录 `mcap2lerobot` 读取 MCAP 的方式为什么从"整集解码进内存"改成"两遍读取、按需解码"，
以及实现必须遵守的约束。用户文档在 [`docs/mcap2lerobot.md`](../docs/mcap2lerobot.md)。

## 1. 背景与问题

0.3.2 及之前，`reader.py` 的 `read()` 一次遍历全部消息：遇到 `CompressedVideo` 包就立刻
H.264 解码成 RGB 数组，追加到 `episode.videos[相机]` 列表；`converter.py` 再对每个 state
的时间戳在列表里找最近邻，交给 lerobot 的 `add_frame`。整集所有相机的所有帧同时驻留在内存。

实测（7 路相机、640×480、100 帧）：

| 场景 | 解码帧驻留 |
|---|---|
| 7 路 640×480，100 帧，实测 | 615 MiB，进程 RSS 增长约 750 MiB |
| 外推 72 秒 @ 20 fps（1440 帧），640×480 | 约 8.7 GiB |
| 外推 72 秒 @ 20 fps，1280×720 | 约 26 GiB |

解码本身只需 0.4 秒，瓶颈不是速度而是驻留。除内存之外还有三个附带问题：

- 每个包都被解码，哪怕最近邻配对最终没有用到它。
- `nearest_value` 每次调用都重建一遍时间戳数组，整集是 O(N²)。
- 同步超差（`max_sync_gap_frames`）在已经写了一部分帧之后才报错，留下半个数据集。

lerobot 一侧不是瓶颈：`add_frame` 把图像交给它的 image writer 异步落盘，队列有界。

## 2. 目标与约束

- 内存与集长无关：任一时刻驻留的解码帧不超过"相机数 × 几帧"。
- 输出与 0.3.2 逐字节一致：配对规则不变，包括平局时取时间上靠后那个包。
- MCAP 格式、CLI 参数、`xense_dataset_convert` 统一接口都不变；没有内嵌元数据的旧 MCAP 照常可转。
- `fps` 缺失时从 state 时间戳推断这条回退路径必须保留。
- 同步超差要在创建数据集之前失败。

## 3. 备选方案

| 方案 | 内存 | 读文件 | 说明 |
|---|---|---|---|
| A 缓存压缩包，按需解码 | 约等于 MCAP 文件大小 | 1 遍 | 改动最小，但 72 秒仍需 0.5 到 1.5 GB |
| **B 两遍读取 + 装配器** | 每路相机约 3 帧 | 2 遍，第一遍不解码 | 采用 |
| C 单遍滑动窗口 | 同 B | 1 遍 | 窗口宽度依赖 fps，而 fps 的推断需要全部 state，逻辑绕 |

最近邻配对需要知道一个 state 前后各一个包，所以单遍必须靠窗口。B 的第一遍本来就要读全部 state，
顺手把所有时间戳收齐，多付的代价只是再读一遍文件，而不解码的一遍不到一秒。

## 4. 设计

```
第一遍  reader.read(path) -> EpisodeData
  遍历全部消息（log_time 排序，只对需要的消息做 CDR 解码）：
    state / action 话题  -> TimedArray（向量很小，全部保留，含 task_index）
    视频话题             -> 只记 log_time 到该相机的时间戳数组；
                            每路相机的第一个包解码一次，得到 (H, W, C) 和编码格式
    附件                 -> info / episode_meta

配对  converter.plan_pairing(episode, fps) -> Plan
  每路相机、以及 action：k[i] = state i 的最近邻包序号，searchsorted 一次算完
  任一 gap 超过 max_sync_gap_frames 立刻抛 ValueError；此时数据集尚未创建

第二遍  reader.iter_frames(episode, video_index) -> 逐 state 产出 {相机: RGB}
  只遍历视频话题：
    该相机第 k 个包，仅当 k 在 plan 里被某个 state 需要时才解码，否则跳过
    只要当前 state 需要的每路相机都已解码，就产出一帧，state 前进一位，
    并丢掉序号小于下一个 state 所需的缓存帧

写入  converter._write_episode
  for state, images in zip(episode.states, reader.iter_frames(...)):
      dataset.add_frame({observation.state, action[k_action[i]], task, **images})
```

### 数据结构

| 类型 | 位置 | 内容 |
|---|---|---|
| `TimedArray` | reader | `timestamp_ns`, `value`, `task_index` |
| `VideoStreamIndex` | reader | `topic`（文件里的原样话题名）, `timestamps_ns`（每个包一个）, `shape`, `format` |
| `EpisodeData` | reader | `states`, `actions`, `videos: {feature: VideoStreamIndex}`, `info`, `episode_meta` |
| `Plan` | converter | `action_index (N,)`, `video_index: {feature: (N,)}` |

### 不变量

- 两遍都以 `log_time_order=True` 遍历，同一话题内"第 k 个包"在两遍里指同一个包。
  lerobot2mcap 写的每路时间戳严格递增，第一遍已校验单调，不存在同话题同时刻两个包。
- 配对序号 `k[i]` 随 i 单调不减，所以装配器可以只向前丢弃缓存。
- 缓存上界：某路相机已解码但尚未产出的帧，只能是它相对最慢那路的到达超前量，
  受 `max_sync_gap_frames` 约束，通常两三帧。
- 平局规则：state 恰好落在两个包中间时取 `searchsorted` 返回的那个，即时间靠后的包，
  与 0.3.2 的 `nearest_value` 一致。

### 复杂度

配对 O(N log N)；第二遍解码次数等于被用到的包数，不多于 state 数 × 相机数；
内存 O(相机数 × 缓存帧)。

## 5. 错误处理

| 情况 | 行为 |
|---|---|
| 某路相机或 action 与 state 的 gap 超限 | `plan_pairing` 抛 `ValueError`，写入尚未开始 |
| 第二遍中某路相机的包提前用完 | `iter_frames` 抛 `ValueError`，指出停在第几帧 |
| 两个 state 落到同一个包 | 同一个解码帧被产出两次，不重复解码 |
| 第二集起 feature 布局变化 | 与之前一样在 `_write_episode` 开头拒绝 |
| 无内嵌元数据 | 名称、fps、robot_type、任务走原有回退，行为不变 |

## 6. 与 lerobot 的交互

- `build_features` 从第一遍记下的 `shape` 和内嵌的 `info.json` 建 feature 表，不再需要解码帧。
- 产出的数组交给 `add_frame` 后不再被本模块修改；lerobot 的 image writer 可能异步持有引用，
  这是安全的，因为装配器只是从缓存字典里删除键，不改数组内容。

## 7. 验证

- 单测：随机时间戳下向量化配对与 0.3.2 的 `nearest_value` 逐项相同；缓存帧数不超过上界；
  gap 超限时 dataset factory 从未被调用；两个 state 共用一个包；某路相机提前结束。
- 端到端：`hdf52lerobot -> lerobot2mcap -> mcap2lerobot` 在 lerobot 0.5.1 fork 与 0.6.1 上重跑，
  输出与源、与 0.3.2 逐值一致。
- 内存：7 路 640×480、100 帧，整段 `convert()`（帧写入一个丢弃型 sink，只计本模块开销）的 RSS 增长，
  改造前 668 MiB，改造后 74 MiB，用时也从 0.4 秒降到 0.2 秒，因为没用到的包不再解码。

## 8. 实施

0.3.3 起生效。reader 与 converter 的切换互相依赖，连同测试放在一个提交里，保证每个提交都能通过测试；
设计文档和发布各自一个提交。若将来需要把读文件从两遍降到一遍，
再在 B 的基础上做方案 C 的窗口化，第一遍收集时间戳的接口不必改。

## 9. XTac dialect（默认）

`ConverterConfig.dialect` 选择源格式，默认 `xtac`（XTacData），`lerobot_roundtrip`
保留上文的 ROS2 CDR 往返链路。两者共用 converter 的 `plan_pairing`、`build_features`、
`_write_episode`；差异全部落在 [`xtac.py`](xtac.py) 的 `XTacMcapReader` 里，它满足与
`McapEpisodeReader` 相同的两遍契约（`read` 后 `iter_frames`）。

XTac 与往返格式的关键区别，以及对应的处理：

- **混合编码**：`/train/*` 是 JSON，`/camera/*` 是 protobuf（`foxglove.CompressedVideo`）。
  `_XTacMessageDecoder` 按 `channel.message_encoding` 分派，不再是单一 CDR 工厂。
- **帧间 h264**：包不可独立解码（首帧要喂两个包，之后 1:1，flush 时出最后一帧）。
  `SequentialVideoDecoder` 顺序喂包、按产出顺序把第 n 帧配到第 n 个包，`flush()` 用
  `parse(b"")` 排空流水线延迟。因为 state:包 = 1:1，`iter_frames` 顺序解全部包（跳包对帧间流不安全）。
- **元数据在 metadata records 而非附件**：`read()` 读 `xense_train_mcap` / `episode` 两条记录，
  合成出与往返格式同形的 `info` / `episode_meta`（fps、29 维名字、`robot_type`、视频 codec、task_id），
  于是 converter 的 `resolve_fps` / `build_features` / `frame_task` 无需改动即可复用。
- **派生特征 tracker_confidence**：从 `/pico/pose` 的 `state.Motion.joints[].confidence` 取（按 sn 对应左右），
  放进 `EpisodeData.extra_vectors`，由 `plan_pairing` 用与 action 相同的最近邻规则配到每个 state。
  `extra_vectors` 是通用机制，往返格式为空。
- **额外 meta 文件**：`convert()` 在 `finalize()` 之后调用 `write_xtac_metadata`，从各集的 `episode`
  记录复原 `hardware.json`、`xumi_collection_devices.json`、`taccap_extrinsics.json`，并把 tactile 相机的
  `taccap.rectify_config` 合并进 `meta/info.json`。`tracker_to_ee` 留空——落盘位姿已是 EE/TCP。

参考数据集 `xtac-umi-g1-block-to-box` 属于另一批采集（`source_episode_id` 为 `EP-1788*`、序列号不同），
只作**结构模板**；本转换器产出的设备序列号来自各自的 mcap。已核对：feature 的 dtype/shape/names、
data parquet 列、五个 meta json 文件集合、以及三个 provenance 文件与 `rectify_config` 的键结构均与参考一致。
