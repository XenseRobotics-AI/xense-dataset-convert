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
