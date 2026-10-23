# Multi-View Face Frontalization System

This project now includes an end-to-end stereo face pipeline under
`src/multiview_face`.

## Run

```bash
python main.py multiview \
  --cam1 data/raw_videos/cam1.mp4 \
  --cam2 data/raw_videos/cam2.mp4 \
  --output_dir data/multiview_dataset \
  --device cuda
```

Useful development flags:

```bash
python main.py multiview \
  --cam1 data/raw_videos/cam1.mp4 \
  --cam2 data/raw_videos/cam2.mp4 \
  --max_pairs 100 \
  --frame_stride 5 \
  --no_timestamp_ocr
```

## Implemented Stages

1. Video ingestion: streaming OpenCV readers with frame index, PTS, FPS, and dimensions.
2. Timestamp OCR and synchronization: optional EasyOCR/pytesseract overlay parsing, with PTS fallback.
3. Face detection: InsightFace when available, OpenCV Haar fallback.
4. Cross-camera matching: embedding cosine similarity plus epipolar/equal-height geometry constraints.
5. Tracking: temporal identity IDs with embedding smoothing and occlusion tolerance.
6. Quality filtering: size, blur, detection confidence, and landmark availability.
7. Dense landmarks: MediaPipe classic FaceMesh, MediaPipe Tasks FaceLandmarker, or sparse fallback when allowed.
8. Camera geometry: calibration JSON loader, landmark fundamental matrix updates, approximate fixed-camera fallback.
9. 3D reconstruction: stereo triangulation when stable, expression-preserving pseudo-depth fallback otherwise.
10. Expression preservation: expression coefficients are measured and then frozen; only head/camera pose changes.
11. Frontalization: mesh-based rendering to yaw=0, pitch=0, roll=0.
12. Texture fusion: both camera crops are warped through the mesh and weighted per virtual view.
13. Restoration: conservative OpenCV enhancement by default; GFPGAN/CodeFormer can be added behind the adapter.
14. Emotion validation: original, frontal, and synthetic images are compared before export.
15. Dataset export: images, OBJ mesh, depth map, and JSON metadata.
16-24. Synthetic generation: virtual camera yaw/pitch/roll grid renders from the same expression-frozen mesh.

## Calibration JSON

`--calibration_file` accepts:

```json
{
  "K_left": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]],
  "K_right": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]],
  "D_left": [0, 0, 0, 0, 0],
  "D_right": [0, 0, 0, 0, 0],
  "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  "T": [[0.08], [0.0], [0.0]],
  "E": null,
  "F": null,
  "image_size": [1920, 1080]
}
```

For best geometry, calibrate fixed cameras with OpenCV `stereoCalibrate()` and pass the resulting matrices.

## MediaPipe Tasks Model

If the installed MediaPipe package only exposes the Tasks API, pass a local
`face_landmarker.task` file:

```bash
python main.py multiview \
  --cam1 cam1.mp4 \
  --cam2 cam2.mp4 \
  --mediapipe_task_model models/face_landmarker.task
```

Without classic FaceMesh or a Tasks model, use `--allow_sparse_landmarks` for
debug runs. Production runs should keep dense landmarks enabled.

## Output

The exporter writes:

```text
data/multiview_dataset/
  person_001/
    happy/
      frontal/
        f0000123_c0000123_frontal.jpg
        f0000123_c0000123_frontal.obj
        f0000123_c0000123_frontal_depth.png
        f0000123_c0000123_frontal.json
      yaw_left_15/
      yaw_left_30/
      yaw_right_15/
      yaw_right_30/
      pitch_up_10/
      pitch_down_10/
```

Each JSON file includes identity, timestamps, frame indices, emotion, original
head pose, rendered pose, expression coefficients, quality metrics, and source
camera metadata.
