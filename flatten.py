import os
import shutil
from pathlib import Path
from PIL import Image


def flatten_dataset(src_root: str, dst_root: str, label_from_study=True):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    for study_path in src_root.glob("*/*/*"):  # study folders
        if not study_path.is_dir():
            continue

        # Extract label from folder name
        if label_from_study:
            label = study_path.name.split("_")[-1]  # 'negative' from 'study1_negative'
        else:
            label = study_path.name  # entire folder name

        label_dir = dst_root / label
        label_dir.mkdir(parents=True, exist_ok=True)

        for img_path in study_path.glob("*.png"):
            dst_path = label_dir / f"{study_path.parent.name}_{study_path.name}_{img_path.name}"
            shutil.copy(img_path, dst_path)

    print(f"Flattened dataset saved to: {dst_root}")

def find_broken_images(root_dir):
    root = Path(root_dir)
    bad_files = []

    for path in root.rglob("*.png"):  # or *.jpg, *.jpeg, etc
        try:
            with Image.open(path) as img:
                img.verify()  # Just verify header
        except Exception as e:
            bad_files.append((path, str(e)))

    print(f"Found {len(bad_files)} bad files.")
    for path, err in bad_files:
        print(f"Corrupted: {path} — {err}")
    
    return bad_files


if __name__ == "__main__":
    # flatten_dataset("./data/train", "flattened/train")
    # flatten_dataset("./data/valid", "flattened/valid")
    find_broken_images("flattened/train")
    find_broken_images("flattened/valid")
