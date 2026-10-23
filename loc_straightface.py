import os
import cv2
import numpy as np
import shutil
import onnxruntime as ort
from insightface.app import FaceAnalysis

# =====================
# CONFIG
# =====================
INPUT_DIR = r"D:\congviec\emoji_v2\data\dataset\train\happy_filtered"
OUTPUT_DIR = r"D:\congviec\emoji_v2\data\dataset\train\happy_filtered_straight"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================
# FORCE GPU PROVIDERS
# =====================
providers = [
    ('CUDAExecutionProvider', {'device_id': 0}),
    'CPUExecutionProvider'
]

print("ONNX Providers:", ort.get_available_providers())

# =====================
# INIT INSIGHTFACE (GPU)
# =====================
app = FaceAnalysis(name="buffalo_l")

app.prepare(
    ctx_id=0,
    det_size=(640, 640),
    providers=providers
)

# =====================
# HEAD POSE (SolvePnP)
# =====================
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # nose
    (-30.0, -30.0, -30.0),    # left eye
    (30.0, -30.0, -30.0),     # right eye
    (-25.0, 30.0, -30.0),     # left mouth
    (25.0, 30.0, -30.0),      # right mouth
    (0.0, 60.0, -40.0)        # chin
], dtype=np.float64)


def get_head_pose(kps, img_shape):

    image_points = np.array([
        kps[2],
        kps[0],
        kps[1],
        kps[3],
        kps[4],
        (kps[3] + kps[4]) / 2
    ], dtype=np.float64)

    h, w = img_shape[:2]
    focal_length = w
    center = (w / 2, h / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        dist,
        flags=cv2.SOLVEPNP_EPNP
    )

    if not success:
        return None, None, None

    rmat, _ = cv2.Rodrigues(rvec)

    sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)

    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
        yaw   = np.degrees(np.arctan2(-rmat[2, 0], sy))
        roll  = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
    else:
        pitch = yaw = roll = 0

    return yaw, pitch, roll


def is_straight(yaw, pitch):
    return abs(yaw) < 12 and abs(pitch) < 12


# =====================
# PROCESS DATASET
# =====================
total = 0
kept = 0

for file in os.listdir(INPUT_DIR):

    if not file.lower().endswith(".jpg"):
        continue

    path = os.path.join(INPUT_DIR, file)
    img = cv2.imread(path)

    if img is None:
        continue

    faces = app.get(img)

    if len(faces) == 0:
        continue

    face = max(faces, key=lambda f: f.det_score)

    yaw, pitch, roll = get_head_pose(face.kps, img.shape)

    if yaw is None:
        continue

    total += 1

    if is_straight(yaw, pitch):
        shutil.copy(path, os.path.join(OUTPUT_DIR, file))
        kept += 1

print("TOTAL:", total)
print("STRAIGHT:", kept)
print("DONE (GPU MODE)")