# RLDS to LeRobot

`rlds2lerobot` converts generated RLDS/TFDS datasets such as DROID into a
LeRobot v3 dataset. Install the optional reader dependencies with:

```bash
pip install -e '.[rlds]'
rlds2lerobot /data/.../droid/1.0.0 -o /data/droid_lerobot \
  --config configs/rlds_droid.yaml --max-episodes 10
```

The reader streams episodes from TFRecord shards. DROID actions are preserved
as the source 7-vector (six joint velocities plus gripper position). The
default `observation.state` concatenates joint position and gripper position;
configured image observations are written as LeRobot video features.

LeRobot encodes multiple camera videos concurrently. When the installed LeRobot
API exposes `encoder_threads`, the converter divides the CPUs available to this
process across those camera encoders; use `--encoder-threads` to set the
per-camera budget manually. Older LeRobot APIs keep their codec default.
Optional `--image-writer-threads` enables background writes for temporary image
frames and defaults to `0` because its queue can retain frames when storage is
slower than the input.

```bash
rlds2lerobot /data/droid/1.0.0 -o /data/droid_lerobot \
  --encoder-threads 2 --image-writer-threads 8
```
