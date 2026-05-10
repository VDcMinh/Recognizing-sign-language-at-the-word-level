# Dataset Folder Overview

This folder contains a local working copy of the WLASL dataset metadata plus a partial set of downloaded sign videos. The main annotation source is `WLASL_v0.3.json`; the `nslt_*.json` files are classification-ready split manifests derived from it.

The notes below describe the folder as it exists locally on 2026-03-30.

## At a glance

- Master vocabulary: 2,000 glosses
- Master instances: 21,083 labeled video instances
- Local MP4 files present: 11,980
- Master-manifest video IDs missing locally: 9,103
- Unique signers in master manifest: 119
- Source sites in master manifest: 19
- Local size of `videos/`: about 5.02 GB

## Folder contents

| Path | Purpose | Local details |
| --- | --- | --- |
| `WLASL_v0.3.json` | Master metadata manifest               | 2,000 gloss entries, 21,083 instances                    |
| `videos/`         | Flat directory of downloaded MP4 files | 11,980 files, all matched to master-manifest `video_id`s |
| `missing.txt`     | Video IDs referenced by the master manifest but not present locally | 9,103 lines |
| `wlasl_class_list.txt` | Class index to gloss mapping used by the NSLT split files | 2,000 lines |
| `nslt_100.json` | 100-class split manifest | 2,038 entries |
| `nslt_300.json` | 300-class split manifest | 5,118 entries |
| `nslt_1000.json` | 1,000-class split manifest | 13,174 entries |
| `nslt_2000.json` | 2,000-class split manifest | 21,095 entries |

## Master manifest: `WLASL_v0.3.json`

This is the canonical metadata file in the folder. Its structure is a list of gloss entries; each gloss contains a list of video instances.

Example shape:

```json
[
  {
    "gloss": "book",
    "instances": [
      {
        "bbox": [385, 37, 885, 720],
        "fps": 25,
        "frame_end": -1,
        "frame_start": 1,
        "instance_id": 0,
        "signer_id": 118,
        "source": "aslbrick",
        "split": "train",
        "url": "http://...",
        "variation_id": 0,
        "video_id": "69241"
      }
    ]
  }
]
```

### What the master manifest contains

- 2,000 glosses
- 21,083 total instances
- 10.54 instances per gloss on average
- Minimum 6 instances per gloss
- Maximum 40 instances per gloss
- 119 unique signers
- 19 unique source collections/sites

### Split distribution in the master manifest

| Split   | Instances |
| ---     | ---:      |
| Train   | 14,289    |
| Val     | 3,916     |
| Test    | 2,878     |

### Useful annotation characteristics

- Every instance is marked as `fps: 25`
- `frame_start` is usually `1` and acts like a clip start index
- `frame_end` is `-1` for 18,958 instances, which likely means "use the full available clip" or "no explicit end frame"
- Only 2,125 instances have a non-negative `frame_end`
- `variation_id` is mostly `0`:
  - `0`: 20,043 instances
  - `1`: 1,017 instances
  - `2`: 23 instances

### Most common data sources

The master manifest aggregates clips from many sign-language resources. The largest contributors are:

| Source | Instances |
| --- | ---: |
| signingsavvy | 2,668 |
| handspeak | 2,211 |
| signschool | 1,968 |
| aslsearch | 1,875 |
| asldeafined | 1,833 |
| aslu | 1,827 |
| aslpro | 1,736 |
| spreadthesign | 1,584 |
| asl5200 | 1,561 |
| aslsignbank | 1,071 |

## Local video inventory: `videos/`

The `videos/` directory is a flat collection of MP4 files named by `video_id`, for example `69241.mp4`.

### Local coverage

- 11,980 MP4 files are present locally
- Every local MP4 maps to a `video_id` in `WLASL_v0.3.json`
- 9,103 `video_id`s from the master manifest are not present locally
- `missing.txt` matches those missing IDs exactly

### Local coverage by split

| Split | Present locally | Missing locally | Local coverage |
| --- | ---: | ---: | ---: |
| Train | 8,313 | 5,976 | 58.2% |
| Val | 2,253 | 1,663 | 57.5% |
| Test | 1,414 | 1,464 | 49.1% |

### File-size summary

- Total local video storage: about 5.02 GB
- Smallest file: about 0.013 MB
- Median file size: about 0.367 MB
- Mean file size: about 0.429 MB
- Largest file: about 7.344 MB

Largest files in the current folder snapshot:

| File | Size |
| --- | ---: |
| `69206.mp4` | 7.344 MB |
| `69412.mp4` | 7.334 MB |
| `69255.mp4` | 7.280 MB |
| `69225.mp4` | 7.146 MB |
| `69212.mp4` | 7.092 MB |

## NSLT split manifests

The `nslt_*.json` files do not use the same schema as `WLASL_v0.3.json`.

Instead, they are dictionaries keyed by `video_id`:

```json
{
  "05237": {
    "subset": "train",
    "action": [77, 1, 55]
  }
}
```

Interpretation of `action`:

- `action[0]`: class ID
- `action[1]`: start frame
- `action[2]`: end frame

`wlasl_class_list.txt` maps each class ID to a gloss. Its ordering matches the ordering of glosses in `WLASL_v0.3.json`. For example:

- Class `0` -> `book`
- Class `99` -> `thursday`
- Class `299` -> `money`
- Class `999` -> `suggest`
- Class `1999` -> `whistle`

### NSLT subset sizes

| File | Entries | Distinct class IDs | Split counts |
| --- | ---: | ---: | --- |
| `nslt_100.json` | 2,038 | 100 | train 1,442, val 338, test 258 |
| `nslt_300.json` | 5,118 | 300 | train 3,549, val 901, test 668 |
| `nslt_1000.json` | 13,174 | 1,000 | train 8,978, val 2,320, test 1,876 |
| `nslt_2000.json` | 21,095 | 2,000 | train 14,296, val 3,920, test 2,879 |

### Relationship between subset files

- `nslt_100.json` is a strict subset of `nslt_300.json`
- `nslt_300.json` is a strict subset of `nslt_1000.json`
- `nslt_1000.json` is a strict subset of `nslt_2000.json`
- Each file uses a contiguous class-id range starting at `0`

In practice, these behave like progressive prefixes of the 2,000-class setup.

## Consistency notes and quirks

### 1. The local folder is incomplete by design or by download state

The metadata describes 21,083 instances, but only 11,980 MP4 files are present locally. The remaining 9,103 IDs are listed in `missing.txt`.

### 2. `missing.txt` is trustworthy for the current snapshot

The IDs in `missing.txt` match the exact set difference:

`video_id`s in `WLASL_v0.3.json` minus MP4 filenames in `videos/`

### 3. `nslt_2000.json` contains 12 IDs that do not appear in `WLASL_v0.3.json`

These extra IDs are:

`09500`, `12209`, `13422`, `16096`, `20065`, `20138`, `39347`, `47639`, `48251`, `51153`, `57839`, `60721`

Implications:

- `nslt_2000.json` has 21,095 entries, which is 12 more than the 21,083 instances in the master manifest
- `nslt_300.json` includes 1 of these extra IDs
- `nslt_1000.json` includes 6 of these extra IDs
- `nslt_100.json` does not include any of them

If you are building training pipelines, `WLASL_v0.3.json` should be treated as the cleaner source of truth for metadata, while `nslt_*.json` is better viewed as a task-specific split/index layer.

## Practical takeaway

This folder is best understood as two related but not identical dataset views:

1. `WLASL_v0.3.json` + `videos/` + `missing.txt`
   This is the local master dataset snapshot and download state.
2. `nslt_*.json` + `wlasl_class_list.txt`
   This is a ready-to-use classification split/indexing layer over 100, 300, 1,000, or 2,000 classes.

If you need one entry point for analysis, start with `WLASL_v0.3.json`. If you need one entry point for model training, start with the appropriate `nslt_*.json` file and use `wlasl_class_list.txt` to decode class IDs back to gloss labels.
