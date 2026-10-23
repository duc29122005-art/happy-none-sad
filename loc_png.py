# fast_remove_duplicates.py

from pathlib import Path
from PIL import Image
import imagehash
import shutil

# =========================
# CONFIG
# =========================

INPUT_DIR = r"D:\congviec\emoji_v2\data\dataset\train\none"
OUTPUT_DIR = r"D:\congviec\emoji_v2\data\dataset\train\happy_filtered"

# càng nhỏ càng nghiêm
HASH_THRESHOLD = 14

# =========================
# PATH
# =========================

input_path = Path(INPUT_DIR)
output_path = Path(OUTPUT_DIR)

output_path.mkdir(parents=True, exist_ok=True)

# =========================
# LOAD FILES
# =========================

image_paths = sorted([
    p for p in input_path.glob("*")
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
])

print(f"Found {len(image_paths)} images")

# =========================
# HASH STORAGE
# =========================

saved_hashes = []

kept = 0
removed = 0

# =========================
# MAIN
# =========================

for idx, image_path in enumerate(image_paths):

    try:
        img = Image.open(image_path)

        # pHash
        current_hash = imagehash.phash(img)

    except Exception as e:
        print(f"ERROR: {image_path.name}")
        continue

    is_duplicate = False

    # giống unordered_map lookup
    for old_hash in saved_hashes:

        # khoảng cách hash
        diff = current_hash - old_hash

        if diff <= HASH_THRESHOLD:
            is_duplicate = True
            removed += 1

            print(
                f"[REMOVE] {image_path.name} "
                f"| hash diff = {diff}"
            )

            break

    if not is_duplicate:

        saved_hashes.append(current_hash)

        shutil.copy(
            image_path,
            output_path / image_path.name
        )

        kept += 1

        print(f"[KEEP] {image_path.name}")

# =========================
# RESULT
# =========================

print("\n===== DONE =====")
print(f"Kept   : {kept}")
print(f"Removed: {removed}")