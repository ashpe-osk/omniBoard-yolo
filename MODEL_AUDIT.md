# Reference model and workflow audit

This audit is based on the files and notebook cells present in the sibling
projects. The reference projects remain unchanged. Counts below describe
available assets and notebook behavior, not accuracy.

## `football-yolo_1`

| Component | What it does | Input and output | Reliability and decision |
| --- | --- | --- | --- |
| `models/best.pt` and `models/new_best.pt` | Player-oriented YOLO detection/tracking models used by the custom tracker | A video frame; player boxes, confidence, class, and tracker IDs | Useful on the local clip, but label schema and provenance are project-specific. Use through the optional `UltralyticsBackend`, not as a package dependency. |
| `models/ball_referee.pt` | Detects football objects separately from player tracking | A video frame; player, referee, and ball boxes | Useful because the ball is small and needs a dedicated detector. Fuse optionally and apply class-aware NMS. |
| BoT-SORT/ReID configuration | Associates player detections over time and supports appearance matching | Per-frame player detections; short-lived tracker IDs | Stronger than naive association but must be benchmarked for ID switches and occlusions. Keep as a validation backend rather than silently depending on it. |
| Custom global-ID association | Attempts to preserve IDs when tracker IDs change using position, appearance, box size, and camera motion | Tracker detections plus frames; global track IDs | Addresses tracker-ID resets but is tightly coupled to the reference implementation. The new tracker exposes replaceable association instead. |
| Optical-flow camera compensation | Estimates broadcast translation/affine motion between adjacent frames | Two images; affine transform and motion magnitude | Useful for motion-aware association, but not pitch calibration. The reusable pipeline keeps it optional; the `vid002` evaluator enables it before track association. |
| Team allocator | Extracts torso color, removes green pixels, and clusters two teams with KMeans | Player crop and frame; team cluster ID | Useful baseline, sensitive to lighting and kits. The new benchmark adopts this torso-color approach with a central upper-body crop, deterministic two-cluster pixel color extraction, match-level dominant-color clustering, bounded per-track feature/vote histories, and confidence-gated adaptation to appearance changes; embedding classification remains optional validation work. |
| Player-ball assigner | Assigns the ball to the nearest player foot corner under a pixel threshold | Player boxes and ball box; player ID or no assignment | Useful for possession candidates but pixel distance is camera-dependent. The new assignment adds confidence and hysteresis, while temporal possession remains explicitly heuristic. |

Other files such as `yolov8x.pt` and `yolo26n-cls.pt` are checkpoints or
training assets. They are not imported automatically because class mappings,
weight provenance, and deployment compatibility are not established.

## `roboflow_way`

### `football_ai.ipynb`

| Component | What it does | Input and output | Reliability and decision |
| --- | --- | --- | --- |
| `football-players-detection-3zvbc-7rrnz/1` | Roboflow model for ball, goalkeeper, player, and referee detection | Video frame through `inference`; `supervision.Detections` | Useful as an optional cloud/backend model. Requires credentials and external service availability, so it is not a core dependency. |
| `ByteTrack` | Tracks non-ball detections | Per-frame `supervision.Detections`; tracker IDs | Useful alternative to compare against the package tracker. It should be selected by evaluation, not assumed superior. |
| SigLIP `google/siglip-base-patch16-224` | Embeds player crops for appearance-based team clustering | Player image crops; image embeddings | More expressive than raw RGB, but computationally expensive and sensitive to crop quality. Keep as an optional team-classification backend. |
| UMAP plus KMeans / `sports.common.team.TeamClassifier` | Reduces embeddings and assigns two team clusters | Embeddings or crops; team IDs | Useful for team separation. Cluster IDs are arbitrary and require temporal stabilization and goalkeeper handling. |
| Goalkeeper centroid assignment | Assigns goalkeeper to the nearest team centroid | Player and goalkeeper footpoints; team ID | Heuristic that can fail under perspective or sparse teams. Use only as validation or an explicit fallback. |
| `football-field-detection-f07vi/14` | Roboflow-hosted field-keypoint model used by the notebook | A frame passed to `get_model(...).infer`; ordered keypoint coordinates and per-point confidence | The notebook expects 32 ordered landmarks and warns that low-confidence points can be wrong. The notebook identifies the hosted model ID, not its underlying network architecture or a downloadable checkpoint. |
| `SoccerPitchConfiguration` | Defines metric pitch vertices and edges | Static pitch geometry; metric landmark coordinates | Useful target coordinate system. The new `SoccerPitchConfig` uses standard 105 x 68 m defaults and accepts custom landmarks. |
| `ViewTransformer` / homography | Maps image points to pitch coordinates | At least four matched image/pitch landmarks; homography and projected points | Directly useful for radars, heatmaps, distances, and tactical geometry. The new implementation adds RANSAC/DLT fallback, smoothing, reuse limits, and quality status. |
| Pitch-space filtering | Removes players near/outside touchlines after projection | Projected player coordinates; filtered detections | Useful false-positive control, but margins must be calibrated for the camera and pitch model. |
| Ball trajectory cleanup | Keeps the highest-confidence ball and removes implausible pitch-space jumps | Per-frame ball detections plus homographies; valid/raw trajectory points | Directly integrated as an optional trajectory validator; rejected raw points remain available for interpolation. |
| Voronoi/radar drawing | Visualizes team control and player positions on the pitch | Projected team/player points; rendered pitch image | Downstream visualization only; not part of the core tracking contract. |

### `omniboard_train_player_detector.ipynb`

This notebook downloads Roboflow football-player dataset version 10, trains
Ultralytics `yolov8x.pt` for 50 epochs at 1280 resolution, validates the
resulting `best.pt`, and deploys it to Roboflow. It is a training recipe, not
a separate inference model bundled with the reference folder. Its dataset
version, class schema, and training result must be validated before deployment.

## Integration boundary

The reference notebook obtains `football-field-detection-f07vi/14` through
Roboflow's `inference.get_model` using an API key; it does not load a local
checkpoint. Its homography code expects the model's 32 ordered keypoints to
align index-for-index with `SoccerPitchConfiguration.vertices`. This is not a
detector currently included in `footy-cvision`: `PitchGeometry` only estimates
transforms from caller-provided landmarks, defaults to five named landmarks,
and `Pipeline` reads `pitch_keypoints` supplied in each frame. No local
keypoint weights, architecture metadata for model version 14, or checkpoint
download URL were found in the workspace. Do not treat the five-landmark
default as a compatible adapter for the notebook's 32-point output.

`footy-cvision` integrates the reusable ideas, not notebook state or
credentials:

1. Normalize optional detector outputs into `Detection`.
2. Fuse model outputs with class-aware IoU suppression.
3. Track normalized player/referee detections with persistent IDs.
4. Estimate field homography only when enough confident landmarks exist.
5. Project footpoints and ball centers into metric pitch coordinates.
6. Reject implausible pitch-space ball jumps without discarding raw evidence.
7. Preserve explicit quality/status fields so downstream analytics can exclude
   stale or invalid geometry.
8. Apply a valid affine camera transform to immediately previous track boxes
   during association, and report how many track candidates used it.
9. Use generic TEAM_A/TEAM_B labels only as confidence-bearing within-match
   identities; uncertain assignments render as unclassified rather than as a
   confirmed team color.
10. Emit conservative possession sequences only after repeated ball/player
    assignment evidence. Passes, carries, and shots stay unsupported without
    adequate pitch and ball-trajectory evidence.
