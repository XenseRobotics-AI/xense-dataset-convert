# The TacVerse `meta/` schema, and how RDT conforms to it

Status: a record of what the schema **is**, and how this repo's RDT output
matches it. Not a design document — the design belongs to the
chain described below, and most of what an earlier draft of this file proposed
turned out to be already built there.

## Who defines it

One chain: **xense-taccap-lerobot writes, TacFlow-Engine post-processes.**
Between them they produce every TacVerse dataset and every operation performed
on one, so what they do *is* the schema.

| stage | repo | writes |
|---|---|---|
| record | xense-taccap-lerobot | `meta/hardware.json` |
| merge | TacFlow-Engine | `meta/sessions.json`, rewrites `hardware.json` epochs |
| prune / dedup | TacFlow-Engine | rewrites both |

This repo is downstream of neither. It converts third-party RDT recordings and
emits `robot_type: bi_rdt_gripper`, and its job here is to **conform**, not to
negotiate. Where its output differs the difference has to earn itself.

Out of scope, deliberately: `xumi_collection_devices.json`, written by
TacCap-Collector for `xtac_umi_g1`. Different rig, different chain. Mixing it
into this document is what made the earlier draft unreadable.

## `hardware.json`, as the chain defines it

```jsonc
{
  "robot_type": "bi_taccap_gripper",
  "robot_id": "taccap_0",        // dataset-level: nothing may resume this
                                 // dataset from another station
  "epochs": [
    {
      "from_episode": 0,
      "to_episode": 14,          // exclusive; null on the epoch still recording
      "recorded_at": "2026-08-31T14:02:11+08:00",
      "robot_id": "taccap_0",
      "role": "leader",
      "units": [
        {"side": "left", "gripper_sn": "TCGU01A28Z0025m",
         "tactile_sensors": [
           {"finger": "left", "observation_key": "left_tactile_left",
            "serial": "GSPS01A29Z0001"}]}
      ]
    }
  ]
}
```

Facts about it that are load-bearing, each from the code rather than inferred:

**`robot_id` is written at the dataset level, always, and never left out.**
The schema states it twice — once per epoch as a fact, once at the top as an
invariant: nothing may resume this dataset from another station. A dataset with
one station writes it; a dataset gathering several has no single value and
writes `<robot_type>_unknown`.

Stated rather than omitted, because the two cases are not the same and a
missing key cannot tell them apart. `bi_rdt_gripper_unknown` says "many
stations, do not resume here"; an absent key says that *or* "written by
something that did not know to". A reader can act on the first and cannot act
on the second. Same reasoning as `role: null` and `gripper_sn: null` elsewhere
in this file: a rig that has no such thing says so.

This is the rule for every `robot_type`, not a convenience for one:
`bi_taccap_gripper_unknown`, `xtac_umi_g1_unknown`, `bi_rdt_gripper_unknown`.

It has real instances rather than hypothetical ones. Twelve of the 122
`bi_taccap_gripper` datasets name more than two gripper serials — one names
nine — and a bimanual rig has two, so those span four or five stations. They are
cross-rig merges assembled after the fact, which is what
`check_dataset_station` exists to prevent at record time.
`bi_taccap_gripper_unknown` is the honest value for them, and it also means a
backfill cannot simply take "the" station: for those twelve there is not one.

*Current state, measured across 137 datasets on this machine: none of them has
a dataset-level `robot_id` at all.*

| family | datasets | dataset-level `robot_id` | epochs with empty `robot_id` |
|---|---|---|---|
| `bi_taccap_gripper` | 122 | absent in all 122 | 746 / 5830 |
| `xtac_umi_g1` | 5 | absent in all 5 | — (`epochs: []`) |
| `bi_rdt_gripper` | 10 | absent in all 10 (before the rewrite below) | 0 / 303 |

Cosmetic, though, next to what sits beside it. The same 746 epochs are exactly
the 746 whose `units` is also empty, and `units` is the only reason this file
exists — the map from an `observation_key` to the sensor that produced it.
Counted by episode rather than by dataset:

    episodes with a tactile serial        11445   64.7%
    episodes no epoch can name             6245   35.3%
    episodes no epoch covers at all           0

A third of the released corpus cannot say which sensor fed which stream. Do not
quote this as "100 of 122 datasets" (82%) without saying it is dataset-weighted:
the 22 that carry serials are the large ones, and the episode-weighted number is
the one a consumer feels. Backfilling `robot_id` while `units` stays empty would
make these look better described than they are, and would turn
`check_dataset_station` from vacuously passing into actively refusing on
datasets that still cannot answer the question anyone asks them.

The 122 were not written by any writer we can account for: not by
xense-taccap-lerobot's (which always emits `recorded_at` and leaves the last
epoch open — 0 of 5830 have it, and all 122 are closed), and not by TacFlow's
merge (whose placeholder is `null` and always carries `source` — no epoch here
has a `source` key). Their shape splits in two, which is a sharper fingerprint
than the values:

| shape | datasets | what it looks like |
|---|---|---|
| one epoch `[0, total)`, `units: []` | 100 | a placeholder stamped per dataset |
| many short epochs, `units` mixed | 17 | `epochs` used as a per-episode hardware column |
| one epoch, `units` filled | 5 | as intended |

The 17 hold 646 of the 746 empty epochs, scattered per episode rather than per
dataset — `arrange-tabletop-items` has 854 epochs over 1284 episodes, 623 of
them one episode long. Whoever wrote those was recording hardware per episode,
not per configuration.

**`units` is the identity; `robot_id` and `role` are context.** From
`_epoch_payload`: "the label names a *station*, and the devices are the parts
you swap in and out of it … a consumer picking a per-sensor calibration keys on
`units` alone."

**An epoch boundary means any recorded field changed, not just the hardware.**
Settled between the two owners while this document was being written. TacCap's
writer compares `_epoch_payload` (`robot_id`, `role`, `units`) because within a
single recording nothing else can differ. TacFlow's constraint is the binding
one: `registry.capture_dates` ranks `epoch.recorded_at` as the strongest
evidence of a capture date, above a source name and above a directory name, so
coalescing two same-rig days into one epoch drops a `recorded_at` and demotes
that evidence silently. Full-field comparison makes coalescing a no-op wherever
anything distinguishes two runs, and a no-op for TacFlow's merge output
outright, since every epoch there carries `source` and `recorded_at`.

**A source with no manifest still contributes an epoch, with `units: null`.**
TacFlow's `write_hardware_epochs`: "没有清单的源，`units` 记 `null`，不拿别的源
去补." An honest unknown beats an inherited value. Its output therefore has two
forms only — epochs non-empty, or no file at all.

**Rewrites are verified per episode, not per range.** TacFlow's D109
(`_remap_hardware_epochs`): every retained episode must land in an epoch whose
hardware description is field-for-field what it landed in before, or the whole
prune raises. Contiguous ranges can still hand an episode another unit's
serials; this catches it. It raises rather than warns because the failure is
silent — "只会让某一集对到别的序列号上". Two rounds of it cost 26 datasets each,
caught by hand before upload.

## `sessions.json`

TacFlow's, end to end. `raw/` has none by design — it is a read-only input
bucket — and from `merged/` on every dataset has one, including single-day
tasks, which go through `copy_single()` precisely so downstream need not treat
one-input and many-input datasets differently (D78 / D91).

```jsonc
{"sessions": [{"source": "taccap-g1-...-0724", "from_episode": 0,
               "to_episode": 214, "episodes": 214, "frames": 373294}],
 "total_episodes": 507}
```

So "file absent" is meaningful rather than a gap: it means the dataset has not
been through the pipeline.

## Where RDT conforms, and where it cannot

`tools/rdt_hardware.py` builds the same document from the recordings' file
names — every camera, tactile sensor and tracker is named by its serial, so
nothing has to be asked for.

Matching: `robot_type` / `epochs[]`, `from_episode` / `to_episode` exclusive,
`robot_id`, `role`, `units[].side` / `gripper_sn` / `tactile_sensors[]`, and
each sensor's `finger` / `observation_key` / `serial`. Verified by calling the
chain's own `manifest_epochs` on our output (`tests/test_rdt_hardware.py`),
rather than by asserting we chose the keys we chose.

Two values are `null` rather than absent, which is how this rig says "no such
thing": `gripper_sn` (the jaw encoder has no firmware serial; its calibration
range is recorded instead) and `role` (a handheld rig has no leader/follower).

Three fields exist here that the schema has no place for, and they are kept
because the information is real and nothing else holds it: the jaw encoder's
`gripper_calibration`, the `tracker_serial` and `wrist_camera_serial` per unit,
and an epoch-level `cameras[]` for the fixed side camera. A reader that knows
only the TacVerse shape ignores them.

`recorded_at` is derived from the episode directory name (`ep_20260810_120057`
becomes `2026-08-10T12:00:57+08:00`), which is the recorder's own naming. The
offset is measured, not assumed: that directory's first sample is a second
later in +08:00.

Epochs coalesce on the hardware fields *and the day*, which is where RDT has to
depart from full-field comparison. TacCap's `recorded_at` is per recording
session, so every episode of a session shares it; RDT's is per episode, so
comparing it whole would make every episode an epoch. An epoch's `recorded_at`
is its first episode's, and a new day always opens a new epoch, so the evidence
`registry.capture_dates` needs (the day) is never coalesced away.

The episodes are the ones the dataset holds, read from `meta/sources.json`, not
the ones the conversion was given. Conversion skips recordings that fail its
checks — a fifth of them across the ten datasets — and a manifest built from
the inputs, as the first version was, had every range after a skip pointing at
the wrong episodes and ending past the dataset: `turn-rotary-knob-0804` holds
4770 episodes and its manifest ran to 6022. `python tools/rdt_hardware.py
--dataset <dir>` rewrites a converted dataset's manifest from what it holds.

### Still not matching

Nothing, as of the `robot_id` change: this repo now writes the dataset-level
value, `bi_rdt_gripper_unknown` when a dataset gathers several stations, which
is the usual case for RDT.

That makes this the only one of the three families that follows the rule today.
`xtac_umi_g1` additionally writes `{"robot_type": "xtac_umi_g1", "epochs": []}`
— the empty-manifest shape this document spent a while attributing to merges.
It comes from the Collector path, not from a merge, and an empty `epochs` with
a `robot_type` beside it is exactly the form TacFlow's own writer will not
produce (it writes epochs or no file).

## What this repo should do

Nothing in the chain. Its two owners have built every operation this document
once proposed — merge-time epoch concatenation (`write_hardware_epochs`, D56),
prune-time remapping (D109), single-session `sessions.json` before a merge
(D78 / D91) — and the one outstanding defect is in taccap's own
`dataset_tools.py`, where `delete_episodes` and `split_dataset` renumber
episodes and then copy the manifest verbatim. They found it themselves and are
fixing it.

Here: nothing further. The RDT manifest conforms, and its dataset-level
`robot_id` follows the rule above.
