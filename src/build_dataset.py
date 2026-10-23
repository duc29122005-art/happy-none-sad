import os
import cv2
import random
import shutil

from insightface.app import FaceAnalysis

from utils import ensure_dir
from config import *


RAW_VIDEO_DIR = 'data/raw_videos'

FRAME_DIR = 'data/frames'

FACE_DIR = 'data/faces'

DATASET_DIR = 'data/dataset'


# -----------------------------
# INSIGHTFACE
# -----------------------------

providers = (
    ['CUDAExecutionProvider']
    if torch.cuda.is_available()
    else ['CPUExecutionProvider']
)

app = FaceAnalysis(
    name=INSIGHTFACE_MODEL,
    providers=providers
)

app.prepare(
    ctx_id=0,
    det_size=DET_SIZE
)


# -----------------------------
# EXTRACT FRAME
# -----------------------------

def extract_frames(video_path, output_dir, fps=1):

    ensure_dir(output_dir)

    cap = cv2.VideoCapture(video_path)

    video_fps = cap.get(cv2.CAP_PROP_FPS)

    frame_interval = max(int(video_fps / fps), 1)

    frame_id = 0
    saved_id = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % frame_interval == 0:

            save_path = os.path.join(
                output_dir,
                f'frame_{saved_id}.jpg'
            )

            cv2.imwrite(save_path, frame)

            saved_id += 1

        frame_id += 1

    cap.release()


# -----------------------------
# EXTRACT FACE
# -----------------------------

def extract_faces(frame_dir, face_dir):

    ensure_dir(face_dir)

    frame_files = os.listdir(frame_dir)

    face_id = 0

    for frame_name in frame_files:

        frame_path = os.path.join(
            frame_dir,
            frame_name
        )

        frame = cv2.imread(frame_path)

        if frame is None:
            continue

        faces = app.get(frame)

        for face in faces:

            x1, y1, x2, y2 = face.bbox.astype(int)

            x1 = max(0, x1)
            y1 = max(0, y1)

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            save_path = os.path.join(
                face_dir,
                f'face_{face_id}.jpg'
            )

            cv2.imwrite(save_path, face_crop)

            face_id += 1


# -----------------------------
# SPLIT DATASET
# -----------------------------

def split_dataset():

    emotions = EMOTION_CLASSES

    for emotion in emotions:

        source_dir = os.path.join(
            FACE_DIR,
            emotion
        )

        if not os.path.exists(source_dir):
            continue

        images = os.listdir(source_dir)

        random.shuffle(images)

        total = len(images)

        train_end = int(0.7 * total)
        val_end = int(0.85 * total)

        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]

        splits = {
            'train': train_imgs,
            'val': val_imgs,
            'test': test_imgs
        }

        for split_name, split_imgs in splits.items():

            target_dir = os.path.join(
                DATASET_DIR,
                split_name,
                emotion
            )

            ensure_dir(target_dir)

            for img_name in split_imgs:

                src = os.path.join(
                    source_dir,
                    img_name
                )

                dst = os.path.join(
                    target_dir,
                    img_name
                )

                shutil.copy(src, dst)


# -----------------------------
# MAIN
# -----------------------------

def build_dataset():

    videos = os.listdir(RAW_VIDEO_DIR)

    for video_name in videos:

        video_path = os.path.join(
            RAW_VIDEO_DIR,
            video_name
        )

        video_id = os.path.splitext(video_name)[0]

        frame_output = os.path.join(
            FRAME_DIR,
            video_id
        )

        face_output = os.path.join(
            FACE_DIR,
            video_id
        )

        print(f'Extract frames: {video_name}')

        extract_frames(
            video_path,
            frame_output
        )

        print(f'Extract faces: {video_name}')

        extract_faces(
            frame_output,
            face_output
        )

    print('DONE')


if __name__ == '__main__':
    build_dataset()