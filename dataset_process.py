import os
import shutil
import random
from pathlib import Path
from PIL import Image

def flatten_dataset(src_root: str, dst_root: str, label_from_study=True, max_images_per_label=None, seed=42):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    image_map = {}

    for study_path in src_root.glob("*/*/*"):
        if not study_path.is_dir():
            continue

        study_type = study_path.parent.parent.name  # e.g., XR_ELBOW
        if label_from_study:
            label = study_path.name.split("_")[-1]  # 'negative' or 'positive'
        else:
            label = "unknown"

        combined_label = f"{study_type}_{label}"  # e.g., XR_ELBOW_negative
        label_dir = dst_root / combined_label
        label_dir.mkdir(parents=True, exist_ok=True)

        image_map.setdefault(combined_label, [])
        for img_path in study_path.glob("*.png"):
            dst_path = label_dir / f"{study_path.parent.name}_{study_path.name}_{img_path.name}"
            image_map[combined_label].append((img_path, dst_path))

    # Random sampling
    random.seed(seed)
    for label, image_pairs in image_map.items():
        sampled = (
            random.sample(image_pairs, min(max_images_per_label, len(image_pairs)))
            if max_images_per_label is not None
            else image_pairs
        )
        for src, dst in sampled:
            shutil.copy(src, dst)

    print(f"Flattened dataset saved to: {dst_root}")

def find_broken_images(root_dir):
    """
    Scan PNG files under a directory and report broken/corrupted ones.

    Args:
        root_dir (str): Root folder to scan
    """
    root = Path(root_dir)
    bad_files = []

    for path in root.rglob("*.png"):
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as e:
            bad_files.append((path, str(e)))

    print(f"\n🧹 Found {len(bad_files)} bad files.")
    for path, err in bad_files:
        print(f"   ❌ Corrupted: {path} — {err}")



if __name__ == "__main__":
    # You can change the limits here for balancing or quick tests
    # flatten_dataset("./data/train", "flattened-limited/train", max_images_per_label=3000)
    # flatten_dataset("./data/valid", "flattened-limited/valid", max_images_per_label=700)

    find_broken_images("./data/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/train")
    find_broken_images("./data/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/test")
    find_broken_images("./data/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/val")

