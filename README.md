# footy-cvision

A standalone, detector-agnostic Python package for football tracking primitives. It does not alter or import from the sibling `football-yolo_1` and `roboflow_way` projects, and does not download models.

## Audit findings and scope

- The workspace's `football-yolo_1` project describes a YOLO/BoT-SORT-oriented workflow with custom identity association, team allocation, ball assignment, and camera-motion compensation. Its README identifies identity switches, occlusion, single-camera assumptions, ball occlusion, and event understanding as limitations.
- `football-yolo_1/main.py` contains project-specific absolute Windows input/model paths and invokes video/model-specific behavior. This package avoids those assumptions and does not depend on that project.
- `roboflow_way` contains two notebooks, rather than a reusable package/API evident from the workspace-level inventory. This package accepts detector callbacks or precomputed detections instead of assuming notebook/model setup.
- This is an architectural audit, not an accuracy benchmark: a representative local clip was used only for an end-to-end smoke test, without labels or enough footage to support accuracy claims.

## Reference inventory and adoption decisions

This inventory records named assets and workflow components found in the two sibling references. It is not a claim that the assets were independently re-trained, benchmarked, or licensed for redistribution.

| Reference and item | Decision | Rationale and reliability caveat |
| --- | --- | --- |
| `football-yolo_1`: `best.pt`, `new_best.pt`, `ball_referee.pt`, `yolo26n-cls.pt`, and `yolov8x.pt` | **Do not use directly** | These are project-local weights/checkpoints, not package dependencies. Weight provenance, label compatibility, licensing, and validation on the target camera must be checked before use. |
| `football-yolo_1`: BoT-SORT with ReID/custom identity association | **Validation** | The package's existing tracker remains a lightweight alternative. BoT-SORT/ReID behavior and its identity-switch/occlusion limitations must be benchmarked on target footage before adoption. |
| `football-yolo_1`: optical-flow camera translation compensation | **Optional** | Camera motion estimation is optional and does not itself establish pitch calibration or improve identity correctness. |
| `roboflow_way/football_ai.ipynb`: player model `football-players-detection-3zvbc-7rrnz/1` | **Optional** | The lazy `RoboflowBackend` can call a selected Roboflow model; credentials and the external inference service are caller-provided. |
| `roboflow_way/football_ai.ipynb`: field-keypoint model `football-field-detection-f07vi/14` | **Optional** | Its named keypoints can be mapped to `SoccerPitchConfig.landmarks`; keypoint ordering/labels and the correct pitch orientation must be verified for the model version. |
| `roboflow_way/football_ai.ipynb`: `google/siglip-base-patch16-224`, UMAP, and KMeans team classification | **Validation** | This embedding/clustering workflow is not silently reproduced by the color-based classifier. Validate appearance embeddings, cluster stability, and goalkeeper/referee handling on labeled target video. |
| `roboflow_way/football_ai.ipynb`: ByteTrack | **Validation** | It is an alternative tracker, not a behavior-preserving drop-in for this package's tracker. Compare ID metrics and occlusion recovery on the intended footage. |
| `roboflow_way/football_ai.ipynb`: goalkeeper assignment by centroid distance | **Validation** | Geometric proximity alone can assign incorrectly under camera perspective, overlap, or goalkeeper movement; use only after measuring assignment quality. |
| `roboflow_way/football_ai.ipynb`: `ViewTransformer` homography, pitch projection, and pitch-space outlier filtering | **Use directly (as an optional capability)** | This package now offers named-landmark homography estimation, image/pitch transforms, and pitch-space ball outlier flags. It uses explicit metric dimensions and confidence/reprojection checks; it does not guarantee calibration quality when landmarks are sparse, mislabeled, nearly collinear, or from a different camera view. |
| `roboflow_way` training notebook: `yolov8x.pt` initialization and custom dataset v10 | **Validation** | This is a training recipe/reference, not a bundled model or assertion that the resulting weights are compatible or accurate. Confirm the dataset version, label schema, class mapping, training provenance, and holdout performance before deployment. |

The `UltralyticsBackend` and `RoboflowBackend` adapters only normalize inference results to `Detection`; they do not download/train models, choose class labels, manage credentials, or promise the inference service/model is available. Optional packages are imported only when their adapter is constructed. The detection filter performs configurable class-aware IoU suppression so outputs from a player model and a ball/referee model can be fused without retaining duplicate player/referee boxes. Pitch homography uses OpenCV RANSAC when available and a NumPy DLT fallback; NumPy is required when estimating or inverting a homography. Estimation reports quality/status, reuses the last valid transform for at most the configured number of failed frames, and marks reuse as stale rather than valid. Ball trajectory output keeps rejected raw pitch coordinates while setting `pitch_position` to `null`, so downstream consumers can interpolate gaps without mistaking outliers for valid samples.

For the checked-in reference weights, the full-video runner assigns people to
`best.pt`, selects `ball_referee.pt` ball detections with a per-frame fallback
to `best.pt`, and treats secondary player/referee outputs as validation data.
This avoids blindly merging duplicate people. It does not claim that either
detector is more accurate; ground-truth labels are needed to prove that.

## Local video evaluation

The evaluation runner samples every tenth frame by default to limit CPU load.
It requires local video and model files; neither videos nor model weights are
included in this repository. Put your private clips in `input/`, for example:

```text
input/vid_for_data.mp4
input/vid_for_data_30s.mp4
```

The 30-second clip is a local test input and is not currently present in this
checkout. Create it locally from the longer video if needed. All video formats
and generated evaluation files are ignored by Git.

The runner's default model paths reference the sibling `football-yolo_1`
project. For a standalone checkout, place compatible weights locally, for
example `models/player/best.pt` and `models/ball_referee.pt`, and pass their
paths explicitly. Model files are ignored and must be obtained separately.
Install the optional video evaluation dependencies:

```powershell
python -m pip install numpy opencv-python ultralytics
```

Run the 30-second local test from the project root (adjust the checkpoint paths
to match your local files):

```powershell
python scripts/evaluate_reference_video.py `
  --video input/vid_for_data_30s.mp4 `
  --primary-model models/player/best.pt `
  --ball-model models/ball_referee.pt `
  --output outputs/vid_for_data_30s `
  --frame-stride 10 `
  --cpu-threads 2
```

For lower CPU use, increase `--frame-stride`; use `--frame-stride 1` only when
the machine can handle inference on every frame. The script's defaults instead
use `football-yolo_1/data/videos/vid002.mp4` and its reference weights, without
modifying the reference project. Outputs include annotated videos, JSONL
tracking records, detection/tracking reports, and debug reports under the
specified `outputs/` folder. Automatically flagged reappearances, jumps,
overlaps, and team reclassifications are suspicious events, not ground-truth
ID switches.

The annotated video is encoded at source FPS divided by the sampling stride
so its playback duration stays close to the source duration; overlays update
only at sampled frames. JSONL frame indices remain source-frame indices.
The clean annotation uses the reference football-ellipse geometry, centered ID
badges, red/white markers only for sufficiently stable generic team
assignments, gray `UNCLASSIFIED` markers for uncertain or missing assignments,
separate referee ellipses, and a compact ball marker. Bounding boxes,
confidence, and proposed team labels are reserved for
`debug/annotated_video.mp4`. Team names are within-match identifiers, not
home/away or real-world team names. To rerender from existing records without
running the detectors again:

```powershell
python scripts/render_tracking_video.py `
  --video input\vid_for_data_30s.mp4 `
  --tracking-data outputs\vid_for_data_30s\tracking_data.jsonl `
  --output outputs\vid_for_data_30s `
  --frame-stride 10
```

The clean video avoids a misleading "camera motion: offline" banner. The
debug video displays the recorded camera estimator state and shows pitch as
unavailable when keypoint inference was not configured. Same-class boxes are
suppressed by IoU or high containment; person containment additionally
requires compatible footpoint/center geometry, height ratio, and (when both
features exist) appearance similarity. A sampled frame previously showed two
IDs on one visible player, but close interactions can still be ambiguous.
Per-class suppression totals are recorded for review. Jersey color features
use a central torso crop, discard green pixels, and select the dominant BGR
color from two deterministic pixel clusters, following the useful torso-color
technique in `football-yolo_1`. The dominant colors are clustered into two
generic kits. Per-track features and confidence-weighted team votes use bounded
histories; a confident raw-color change resets stale appearance history, and
conflicting recent votes render as gray until the assignment stabilizes. The
result remains sensitive to lighting, occlusion, and tracker ID switches.
OpenCV reports 1,051 frames in this file's container metadata, but only 901
frames were returned before decoding stopped. The cause (file truncation versus
decoder limitation) has not been established. Reports preserve both counts
instead of claiming the extra 150 frames were processed.

### `vid002.mp4` CPU sampling run

The latest local run used stride 10 and two CPU threads. The decoder returned
901 frames before stopping, yielding 91 inference samples across the returned
timeline. Runtime was 188.45 seconds (0.48 sampled frames/s, 4.83
source-frame/s equivalent). Both annotated MP4s were readable at 3 FPS and
contained 91 frames.

Observed model outputs (not accuracy metrics): primary `best.pt` returned
1,107 player, 114 referee, 64 ball, and 1 goalkeeper detections; secondary
`ball_referee.pt` returned 796 player, 86 referee, and 104 ball detections.
The selected pipeline retained 854 player, 100 referee, 103 ball, and 1
goalkeeper detections after same-class suppression. It suppressed 253 player,
14 referee, and 1 ball candidate (none of the goalkeeper candidates). The
person-specific duplicate rule also checks aligned footpoints, center
distance, relative height, and available appearance descriptors. At IoU 0.5,
625 primary player candidates and 67 primary referee
candidates had overlapping secondary candidates. These overlaps support
treating the models as overlapping sources, but do not identify which
detection is correct; consequently only `best.pt`
supplies people, while the specialist model supplies ball detections with
primary-model fallback.

The ball was detected in 80/91 sampled frames (87.9% of sampled frames), with
6 intervals containing one or more missed ball samples; the longest gap was
4 sample intervals. Camera motion was estimated in 90/91 sampled frames and
compensated 1,038 prior-track boxes over 90 association updates. The tracker
reported 121 IDs and 128 heuristic fragmentation segments, versus 121 IDs and
141 segments for the no-appearance ablation. This difference is not proof of
identity improvement: there are no ground-truth IDs or manually verified
switch labels. The report leaves verified and ground-truth ID-switch counts
null.

The dominant-torso-color classifier produced 40 TEAM_A, 44 TEAM_B, and 5
unclassified player-track assignments out of 89 player IDs; 38 assignments
were marked uncertain at their final observation. Fourteen tracks changed
team label during the run. These counts and visual samples are not labeled
accuracy measurements. At source frame 390, clearly red and white players
were separated more consistently; two conflicting white-player tracks were
left gray instead of retaining a confident stale red assignment. Errors remain
possible, including missed ID switches and ambiguous crops.

The possession heuristic emitted 7 player-control candidate sequences with 3
candidate team changes; the sampled records included 13 TEAM_A and 4 TEAM_B
confirmed possession samples, while 74 samples were unclassified or
unobserved. These are still heuristics, not ground truth. Passes, carries, and
shots remain `null`/unsupported, so the data is not yet suitable as verified
possession analytics.

No field-keypoint weights are stored locally in either reference project, and
the benchmark deliberately does not upload video to an external inference
service. The reference notebook identifies Roboflow's hosted
`football-field-detection-f07vi/14` model but does not identify its underlying
architecture or provide a checkpoint URL. It expects 32 ordered landmarks;
`PitchGeometry` currently defaults to five named landmarks and only consumes
keypoints supplied by the caller. Pitch coordinates, outside-pitch detection
rates, and homography quality are therefore reported as unavailable rather
than fabricated. Model-weight extensions under `models/keypoints/` are
excluded by `.gitignore` so downloaded checkpoints are not accidentally
committed. The container reports 1,051 frames, but decoding stops after 901;
the cause is unresolved. The comparison to the original end-to-end `football-yolo_1`
entrypoint is also not run because its hard-coded external video/model paths
do not match this workspace. The same video and local weights are exercised
in the new runner, but that is not represented as a head-to-head tracker
benchmark.

## Previous real-footage smoke validation

Using the local reference assets read-only, the two lazy Ultralytics adapters were exercised on the first frame of `football-yolo_1/data/videos/vid001.mp4`:

- `football-yolo_1/models/best.pt`: 15 detections (`12 player`, `3 referee`)
- `football-yolo_1/models/ball_referee.pt`: 15 detections (`11 Player`, `3 Referee`, `1 Ball`)
- After class normalization and class-aware IoU suppression, the combined pipeline produced 13, 11, and 12 detections across three frames, with persistent structured tracks and no temporal violations.

This confirms model loading, result normalization, multi-model fusion, duplicate suppression, tracking, and JSON-serializable pipeline output. It does **not** measure precision, recall, ID switches, fragmentation, or pitch-keypoint accuracy; those require labeled target footage.

## Architecture

```text
main.py                    JSON/JSONL CLI and optional module:callable backend
footy_cvision/
  core.py                  Typed, validated frame/detection/track records
  detection/filter.py      Confidence, class, NMS, and optional pitch-boundary filters
  tracking/                Deterministic greedy association and track lifecycle
  camera/motion.py         OpenCV affine estimate used for association compensation
  football/                Team classification/stabilization, possession, pitch geometry, trajectories
  models/backends.py       Optional lazy Ultralytics and Roboflow adapters
  evaluation/metrics.py    Detection and tracking summary metrics
  pipeline/pipeline.py     Detector callback orchestration + JSON records
configs/default.yaml       Starter configuration
tests/                     unittest-compatible focused tests
```

Tracking combines IoU, center distance, optional cosine appearance distance,
and affine compensation for tracks seen in the immediately previous sample.
Greedy matching is deterministic for equal-cost pairs; it is not globally
optimal Hungarian assignment. A missed track remains eligible for association
through `max_age`, with a widening center-distance gate for recovery after
brief occlusions. Ball assignment uses distance, confidence gating, and a
previous-player hysteresis margin. Possession requires repeated assignment
evidence and emits generic control events; it does not infer passes, carries,
or shots.

## Limitations

- There is no bundled detector, pretrained model, video decoder, or model download. Video/image decoding and inference belong in the injected detector callback.
- Association is a lightweight greedy approximation, not a full motion-filter/ReID system; long occlusions, crossings, camera cuts, and visually similar players can still cause switches. Camera compensation currently applies only to tracks seen in the immediately previous sample.
- Team classification uses a dominant torso color only and does not account for lighting, shadows, or kit changes. It adapts to confident appearance changes, but a persistent tracker ID switch can still delay or corrupt assignment. Confidence gating leaves uncertain players unclassified in the clean output; TEAM_A/B are deterministic within-match conventions, not home/away labels.
- The default motion estimator needs OpenCV and suitable array frames; it returns an explicit invalid/no-op result when OpenCV, frames, or features are unavailable. Valid image motion is not pitch calibration.
- Possession remains a nearest-player heuristic with temporal confirmation. It can be wrong around duels, occlusions, loose balls, and sparse stride sampling; downstream events remain unsupported.
- The dead-ball rule only checks whether a supplied ball center is outside a supplied pitch polygon. It does not infer whistles or game state.
- Metrics are compact summaries, not a full MOTChallenge/COCO implementation. ID-switch/fragmentation calculations require ground-truth IDs in track metadata; speed is reported only when supplied in track metadata.

## Phases

1. **Reusable base (implemented):** data models, configurable multi-signal filtering, deterministic association/tracking, camera-compensated association, confidence-aware team classification, conservative possession output, ball assignment, validation, metrics, and CLI.
2. **Backend integration:** write and test a detector callback for the target model and frame source; keep model-specific dependencies outside the core package.
3. **Dataset validation:** provide labeled match clips and ground-truth IDs, measure metrics, tune thresholds per camera/resolution, and record failure cases.
4. **Advanced analytics:** validate robust appearance embeddings, event/dead-ball state models, and multi-camera support only when target-data evaluation supports them. Optional pitch calibration is available but requires suitable per-frame named landmarks.

## Install and use

Python 3.10+ is recommended. Core tracking and centroid classification use the
standard library. The video torso-color extractor needs NumPy and OpenCV; YAML
configuration uses PyYAML. Install the base dependency with:

```powershell
python -m pip install -r requirements.txt
```

No model packages are required unless your detector callback uses them.

Install only the optional packages needed by a chosen integration:

```powershell
python -m pip install numpy opencv-python
python -m pip install ultralytics
python -m pip install inference-sdk
```

From this folder, process JSON or JSONL input where each frame has a `detections` list:

```json
[
  {"detections": [
    {"bbox": [10, 20, 40, 100], "confidence": 0.93, "class_name": "player"},
    {"bbox": [50, 60, 55, 65], "confidence": 0.81, "class_name": "ball"}
  ]}
]
```

```powershell
python main.py --input frames.json --output results.json
python main.py --input frames.jsonl --output results.json --config configs/default.yaml
```

A custom detector is specified as an importable `module:callable`; it receives an input frame record and returns `Detection` objects or mappings with `bbox`, `confidence`, and `class_name` keys:

```powershell
python main.py --input frames.json --output results.json --detector my_backend:detect
```

Programmatic use:

```python
from footy_cvision.pipeline import Pipeline

pipeline = Pipeline(detector=my_detector)
frame_results = pipeline.run(frame_records)
json_records = [result.to_dict() for result in frame_results]
```

For optional metric pitch coordinates, construct a `PitchGeometry` and provide
named image keypoints on each frame. Names must match the configured pitch
landmarks; values are image-pixel `(x, y)` or `(x, y, confidence)` tuples:

```python
from footy_cvision.football.pitch_geometry import PitchGeometry, SoccerPitchConfig
from footy_cvision.pipeline import Pipeline, PipelineConfig

geometry = PitchGeometry(SoccerPitchConfig())
pipeline = Pipeline(config=PipelineConfig(pitch_geometry=geometry))
result = pipeline.process_frame(
    {
        "pitch_keypoints": {
            "top_left": (120, 80, 0.98),
            "top_right": (900, 95, 0.96),
            "bottom_right": (1100, 650, 0.94),
            "bottom_left": (30, 620, 0.92),
            "center": (530, 360, 0.88),
        },
        "detections": [...],
    },
    frame_index=0,
)
```

Active track metadata receives the projected footpoint and homography status.
The frame result includes a homography quality/status record and an optional
ball-trajectory point; rejected ball samples have `pitch_position: null` and
retain `raw_pitch_position` for downstream interpolation or inspection. Tune
the configured metric speed threshold to the video frame rate before using
outlier flags as a hard rejection decision.

Run focused tests from this folder:

```powershell
python -m unittest discover -s tests -v
```
