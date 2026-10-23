from pathlib import Path

import torch


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'

RAW_VIDEO_DIR = DATA_DIR / 'raw_videos'
FRAME_DIR = DATA_DIR / 'frames'
FACE_DIR = DATA_DIR / 'faces'
DATASET_DIR = DATA_DIR / 'dataset'
OUTPUT_DIR = DATA_DIR / 'outputs'

CHECKPOINT_DIR = MODELS_DIR / 'checkpoints'
EXPORT_DIR = MODELS_DIR / 'exports'

IMAGE_SIZE = 112

CHANNELS = 3

BATCH_SIZE = 32

LEARNING_RATE = 1e-3

EPOCHS = 20

FPS = 2

NUM_CLASSES = 3

CLASS_NAMES = [
    'happy',
    'sad',
    'none'
]

MIN_FACE_SIZE = 80

MIN_FACE_AREA = 80 * 80

BLUR_THRESHOLD = 100

EMOTION_CONFIDENCE_THRESHOLD = 0.8

DETECTION_CONFIDENCE = 0.7

SMOOTHING_WINDOW = 10

DETR_MODEL_NAME = 'facebook/detr-resnet-50'

INSIGHTFACE_MODEL = 'buffalo_l'
DET_SIZE = (640, 640)
FACE_MODEL = INSIGHTFACE_MODEL
EMOTION_CLASSES = CLASS_NAMES

DEVICE = (
    'cuda'
    if torch.cuda.is_available()
    else 'cpu'
)

DIRS = [
    RAW_VIDEO_DIR,
    FRAME_DIR,
    FACE_DIR,
    DATASET_DIR,
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    EXPORT_DIR,
]

for directory in DIRS:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )