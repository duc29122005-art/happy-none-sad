# Emotion AI Pipeline

Real-time face emotion detection (happy / sad / none) using InsightFace + DETR + EfficientNet.

## Project Structure

```
emoji_v2/
├── data/
│   ├── raw_videos/          ← Put source videos here
│   ├── frames/              ← Auto-extracted frames
│   ├── faces/               ← Auto-cropped face images
│   ├── dataset/
│   │   ├── train/{happy,sad,none}/
│   │   ├── val/{happy,sad,none}/
│   │   └── test/{happy,sad,none}/
│   └── outputs/             ← Annotated output videos
├── models/
│   ├── checkpoints/         ← best_model.pth saved here
│   └── exports/             ← TorchScript exports
├── src/
│   ├── config.py            ← All paths + hyperparameters
│   ├── utils.py             ← Shared helper functions
│   ├── emotion_model.py     ← EfficientNet-B0 classifier
│   ├── segmentation_model.py← DETR panoptic person segmentation
│   ├── dataset.py           ← PyTorch Dataset + DataLoaders
│   ├── build_dataset.py     ← Frame extraction + face detection
│   ├── train_model.py       ← Training loop
│   └── inference_video.py   ← Real-time video pipeline
└── main.py                  ← Unified CLI entry point
```

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Install PyTorch (CUDA 11.8 example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install remaining dependencies
pip install -r requirements.txt
```

## Usage

### Step 1 — Build dataset from raw videos
```bash
# Put .mp4 / .avi / .mov files in data/raw_videos/
# With 30fps videos, this captures 1 frame per second.
python main.py build --capture_every_n_frames 30
```
Then manually sort faces in `data/faces/` into:
```
data/dataset/train/happy/   data/dataset/train/sad/   data/dataset/train/none/
data/dataset/val/happy/     data/dataset/val/sad/     data/dataset/val/none/
data/dataset/test/happy/    data/dataset/test/sad/    data/dataset/test/none/
```

### Step 2 — Train emotion model
```bash
python main.py train --epochs 30 --batch_size 32 --device cuda
```

### Step 3 — Run inference on video
```bash
python main.py --input data/raw_videos/test.mp4 --output data/outputs/test_out.mp4 --device cuda
python main.py infer --input data/raw_videos/test.mp4 --device cuda
# Webcam:
python main.py --input 0
# Headless (no display window):
python main.py --input test.mp4 --no_display
# Skip DETR for max speed:
python main.py --input test.mp4 --no_segmentation
```

### Export TorchScript
```bash
python main.py export
```

## Key Design Decisions

| Component | Choice | Reason |
|-----------|--------|--------|
| Face detector | InsightFace buffalo_l | Best accuracy/speed tradeoff |
| Segmentation | DETR resnet-50-panoptic | Person mask from panoptic output |
| Emotion backbone | EfficientNet-B0 (timm) | Lightweight + pretrained |
| Inference | AMP (fp16) | 2× faster on CUDA |
| Smoothing | Sliding-window majority vote | Stable labels on video |
| DETR skip-frames | Every 5 frames | Segmentation is expensive |

## Configuration

All settings are in `src/config.py`:

```python
IMAGE_SIZE              = 224   # Emotion model input size
DEFAULT_VIDEO_FPS       = 30.0  # Fallback FPS when metadata is missing
CAPTURE_EVERY_N_FRAMES  = 30    # Capture 1 frame after every 30 frames
MIN_FACE_SIZE           = 60    # Pixels — filter small faces
BLUR_THRESHOLD          = 100.0 # Laplacian variance cutoff
EMOTION_SMOOTH_WINDOW = 7    # Smoothing history window
DETR_SKIP_FRAMES  = 5        # Re-run DETR every N frames
```

## Class Index Mapping

```
0 → happy
1 → none
2 → sad
```
(Alphabetical order — matches `torchvision.datasets.ImageFolder`)
