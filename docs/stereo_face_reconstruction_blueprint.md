# Stereo Face Reconstruction Blueprint

Goal:

```text
Two synchronized videos
  -> matched faces across fixed cameras
  -> expression-preserved 3D face mesh
  -> frontal face images
  -> synthetic multi-angle emotion dataset
```

This document maps the math to the current implementation under
`src/multiview_face`.

## Production Architecture

Pipeline stages:

```text
VideoReader
  -> FrameSynchronizer
  -> FaceDetector
  -> DenseLandmarkExtractor
  -> FaceQualityFilter
  -> CrossCameraMatcher
  -> ExpressionAwareReconstructor
  -> FaceRenderer
  -> EmotionConsistencyValidator
  -> DatasetExporter
```

Core data structures:

```text
FrameMetadata      camera id, frame index, PTS, OCR timestamp
VideoFrame         image + metadata
StereoFramePair    synchronized frames
FaceObservation    bbox, crop, embedding, landmarks, quality
StereoFacePair     matched left/right observation
StereoCameraModel  K1, K2, D1, D2, R, T, E, F
FaceMesh3D         vertices, faces, textures, expression coeffs, metadata
```

Recommended libraries:

```text
OpenCV     video IO, calibration, epipolar geometry, triangulation, warping
MediaPipe  dense 468 face landmarks and optional blendshapes
NumPy      matrix math and data structures
PyTorch    emotion model, identity model, optional neural validation
PyTorch3D  higher-quality rendering when OpenCV triangle warping is not enough
```

## Phase 1: Stereo Vision Fundamentals

Intuition: each camera maps a 3D point to a 2D pixel. With two cameras, the
same face landmark appears in two images. The difference between those 2D
positions constrains the 3D position.

Pinhole projection:

```text
X_world = [X, Y, Z, 1]^T
x_pixel ~ K [R | T] X_world

K = [ fx  0  cx
       0 fy  cy
       0  0   1 ]
```

`K` is intrinsics: focal length and principal point. `R,T` are extrinsics:
rotation and translation from one camera coordinate system to another.

Stereo geometry:

```text
cam1 center C1                 cam2 center C2
      o------------------------------o
       \                            /
        \                          /
         \                        /
          \                      /
             3D face landmark X
```

A point in camera 1 must lie on an epipolar line in camera 2:

```text
x2^T F x1 = 0
E = [T]_x R
F = K2^-T E K1^-1
```

Why this matters: epipolar geometry rejects wrong cross-camera identity matches.
If person A in camera 1 is matched to person B in camera 2, `x2^T F x1` will be
large for face centers and landmarks.

Depth from rectified stereo:

```text
Z = f * B / d

f = focal length
B = baseline distance between cameras
d = disparity = x_left - x_right
```

For non-rectified cameras, use triangulation instead of raw disparity.

OpenCV implementation in this repo:

```python
from src.multiview_face.geometry import projection_matrices, triangulate_landmarks

P1, P2 = projection_matrices(camera_model)
points_3d = triangulate_landmarks(left_landmarks, right_landmarks, camera_model)
```

What can go wrong:

```text
Bad K/R/T              -> warped or inverted 3D face
Unsynchronized frames  -> landmarks do not correspond to the same expression
Lens distortion        -> triangulation has high reprojection error
Tiny baseline          -> depth is numerically unstable
Wrong identity match   -> impossible 3D geometry
```

Debugging:

```text
1. Print median reprojection error. Good calibration should be low.
2. Draw epipolar lines over both videos.
3. Check that triangulated Z is finite and mostly consistent.
4. Run with a single clear face before using a full classroom scene.
```

## Phase 2: Multi-View Face Detection

Why this stage exists: stereo reconstruction requires corresponding face
observations from both videos at the same time.

Timestamp synchronization:

```text
camera_clock = video_pts + camera_start_clock
offset = start_clock_cam2 - start_clock_cam1
target_cam2_pts = cam1_pts - offset
```

Current implementation:

```python
from src.multiview_face.sync import FrameSynchronizer

synchronizer = FrameSynchronizer(config)
for pair in synchronizer.iter_pairs("cam1.mov", "cam2.mov"):
    left_frame = pair.left.image
    right_frame = pair.right.image
```

Face matching score:

```text
score = 0.72 * cosine(face_embedding_left, face_embedding_right)
      + 0.28 * epipolar_geometry_score
```

Why both terms are needed:

```text
Embedding  handles identity.
Epipolar   handles camera geometry and reduces false matches in crowds.
Tracking   keeps stable person ids across time.
```

Output:

```text
StereoFacePair(
  left=FaceObservation(...),
  right=FaceObservation(...),
  identity_score=...,
  geometry_score=...,
  match_score=...
)
```

Debugging:

```text
Draw bbox ids in both views.
Draw epipolar line of cam1 face center on cam2.
Log match_score and identity_score.
Reject pairs with large vertical mismatch in equal-height camera setups.
```

## Phase 3: 3D Landmark Reconstruction

MediaPipe FaceMesh returns dense corresponding landmarks. Landmark index 61 is
the same semantic point in both cameras, so it can be triangulated with index 61
from the other camera.

Coordinate systems:

```text
Image pixels:      x right, y down
Camera 3D:         x right, y down/up by convention, z forward
Canonical mesh:    centered and normalized for rendering
```

Minimal complete triangulation example:

```python
import cv2
import numpy as np


def triangulate_points(left_xy, right_xy, K1, K2, R, T):
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K2 @ np.hstack([R, T.reshape(3, 1)])

    pts4 = cv2.triangulatePoints(
        P1.astype(np.float32),
        P2.astype(np.float32),
        left_xy.T.astype(np.float32),
        right_xy.T.astype(np.float32),
    )

    w = pts4[3:4]
    valid = np.abs(w) > 1e-8
    if valid.sum() < 4:
        raise ValueError("Not enough valid homogeneous points")

    pts3 = (pts4[:3, valid.reshape(-1)] / w[:, valid.reshape(-1)]).T
    return pts3.astype(np.float32)
```

Expected input:

```text
left_xy:  shape (468, 2), pixel landmarks from cam1
right_xy: shape (468, 2), pixel landmarks from cam2
K1,K2,R,T: stereo calibration
```

Expected output:

```text
points_3d: shape (468, 3), one 3D vertex per landmark
```

Time complexity: `O(N)` for `N` landmarks.

Debugging:

```text
Project 3D landmarks back to both images.
Compute per-point reprojection error in pixels.
Reject pairs when median reprojection error is high.
```

## Phase 4: 3D Face Mesh Construction

A mesh contains:

```text
vertices: shape (N, 3)
faces:    shape (M, 3), triangle vertex indices
uv:       shape (N, 2), texture coordinates
texture:  source image crop or fused texture atlas
```

This repo currently builds triangles from landmarks using Delaunay fallback in
`src/multiview_face/reconstruction.py`. For production, prefer the fixed
MediaPipe FaceMesh tessellation because triangle topology is stable across
frames.

Why stable topology matters:

```text
Frame 100 vertex 61 must connect to the same neighbors as frame 101.
Otherwise textures flicker and synthetic pose generation drifts.
```

Output:

```text
FaceMesh3D(
  vertices=(468, 3),
  faces=(M, 3),
  source_landmarks={"cam1": ..., "cam2": ...},
  source_images={"cam1": crop1, "cam2": crop2},
  expression_coefficients={...}
)
```

## Phase 5: Expression Preservation

Expression is geometry deformation. A smile moves mouth corners, changes cheek
shape, narrows eyes, and changes local surface normals.

A common model:

```text
V = V_identity + sum_i beta_i * B_expression_i
```

`V_identity` is the neutral person-specific face. `B_expression_i` are
blendshape deformation bases. `beta_i` are expression coefficients.

Head pose is different:

```text
V_camera = R_head V_expression + t_head
```

To frontalize while preserving expression:

```text
1. Estimate expression geometry in 3D.
2. Do not neutralize mouth/eye/brow landmarks.
3. Remove or replace only the head/camera pose.
4. Render from yaw=0, pitch=0, roll=0.
```

The current repo stores compact expression proxies:

```text
mouth_open, smile, brow_raise, eye_open
```

These are useful for validation and metadata. For higher quality, use MediaPipe
Tasks blendshape outputs or a 3DMM/FLAME model.

Debugging:

```text
Compare mouth_open before and after render.
Run emotion classifier on original and rendered samples.
Reject samples where emotion probability drifts too much.
```

## Phase 6: Face Frontalization

Yaw, pitch, roll:

```text
yaw   turns left/right around the vertical axis
pitch nods up/down around the horizontal axis
roll  tilts around the viewing axis
```

Rotation:

```text
V_render = R_virtual V_expression
```

Rotating a 2D image only moves pixels in the image plane. It cannot reveal the
hidden side of the nose or cheek. Rotating a 3D mesh changes the virtual camera
view while preserving the 3D mouth, eyes, and brow geometry.

Current implementation:

```python
from src.multiview_face.rendering import FaceRenderer

renderer = FaceRenderer(config)
frontal_image, depth = renderer.render(mesh, yaw=0.0, pitch=0.0, roll=0.0)
```

What can go wrong:

```text
Bad depth ordering      -> nose/cheek triangles overwrite each other
Sparse landmarks        -> holes around eyes/mouth
Poor texture visibility -> one side of the face is missing
```

Debugging:

```text
Export depth maps.
Render wireframe overlays.
Render yaw=-30,0,30 and inspect whether expression stays fixed.
```

## Phase 7: Rendering

Rendering maps 3D triangles to a 2D image.

Virtual camera projection:

```text
x_virtual ~ K_virtual [R_virtual | T_virtual] X_mesh
```

Rasterization decides which pixels are covered by each triangle and which
triangle wins in depth.

Recommended renderers:

```text
OpenCV triangle warping: simple, fast, good for prototype and CPU pipeline.
PyTorch3D: batch rendering, differentiable optimization, GPU validation.
OpenGL: production real-time rendering.
Blender: offline high-quality synthetic dataset generation.
```

The repo uses OpenCV triangle warping in `FaceRenderer`. It fuses texture from
both cameras with camera-dependent weights and fills holes with conservative
inpainting.

## Phase 8: Synthetic Dataset Generation

Required pose grid:

```text
yaw:   -30, -20, -10, 0, 10, 20, 30
pitch: -15, -10, 0, 10, 15
roll:  0
```

The updated config renders the yaw-pitch cross product and skips duplicate
`0,0,0` unless `include_frontal_synthetic=True`.

Run example:

```bash
python main.py multiview ^
  --cam1 data/raw_videos/cam1.mov ^
  --cam2 data/raw_videos/cam2.mov ^
  --calibration_file data/calibration/stereo.json ^
  --output_dir data/multiview_dataset ^
  --device cuda
```

Metadata should include:

```json
{
  "person_id": "001",
  "frame_index": {"cam1": 120, "cam2": 118},
  "timestamp": {"cam1_seconds": 33022.4, "cam2_seconds": 33022.4},
  "emotion": "happy",
  "yaw": -30,
  "pitch": 10,
  "roll": 0,
  "synthetic": true,
  "identity_score": 0.81,
  "geometry_score": 0.92,
  "expression_coefficients": {"smile": 0.47, "mouth_open": 0.12}
}
```

What can go wrong:

```text
Too much yaw creates texture holes.
Texture from wrong identity contaminates the face.
Inpainting changes expression around mouth or eyes.
Synthetic set becomes class-imbalanced.
```

Debugging:

```text
Group samples by person_id/emotion/view.
Check identity embedding similarity against original.
Check emotion classifier probability drift.
Keep original and synthetic metadata side by side.
```

## Phase 9: Dataset Validation

Reject bad samples using three checks.

Emotion drift:

```text
KL(original_emotion_probs, rendered_emotion_probs) < threshold
or same top-1 label with confidence margin
```

Identity drift:

```text
cosine(embedding_original, embedding_rendered) > threshold
```

Geometry distortion:

```text
median_reprojection_error < threshold
landmark_ratio_mouth_open within tolerance
landmark_ratio_eye_open within tolerance
finite depth ratio > threshold
```

Validation output:

```text
accepted_samples.jsonl
rejected_samples.jsonl
debug_overlays/
  epipolar_lines/
  landmarks/
  depth/
  wireframe/
```

Production rule: never trust a rendered image without metadata and validation
scores. A realistic-looking bad synthetic sample can poison emotion training.

## Calibration Workflow

Capture 15-30 synchronized chessboard image pairs from both fixed cameras.
The board should cover different image positions and depths.

Python example:

```python
from pathlib import Path

from src.multiview_face.calibration import (
    calibrate_stereo_from_chessboards,
    save_camera_model,
)

left = sorted(Path("data/calibration/left").glob("*.jpg"))
right = sorted(Path("data/calibration/right").glob("*.jpg"))

model = calibrate_stereo_from_chessboards(
    left,
    right,
    board_size=(9, 6),
    square_size=0.024,  # meters if each square is 24 mm
)
save_camera_model(model, "data/calibration/stereo.json")
```

Good calibration signs:

```text
RMS reprojection error is low.
Epipolar lines pass through corresponding chessboard corners.
Triangulated face depth is stable over adjacent frames.
The baseline T magnitude matches the physical camera distance.
```

Bad calibration signs:

```text
RMS error is high.
T has unrealistic scale or direction.
Faces look inside-out after frontalization.
Median landmark reprojection error is high.
```
