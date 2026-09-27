"""Image-only HAM10000 loading and permanent lesion-level partitions."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/HAM10000"
SPLIT = ROOT / "data/splits/split_assignments.csv"
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
LABELS = {name: i for i, name in enumerate(CLASSES)}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def image_paths(raw=RAW):
    paths = {}
    duplicates = []
    for folder in ("HAM10000_images_part_1", "HAM10000_images_part_2"):
        directory = Path(raw) / folder
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in directory.glob("*.jpg"):
            if path.stem in paths:
                duplicates.append(path.stem)
            paths[path.stem] = path
    return paths, duplicates


def metadata(raw=RAW):
    frame = pd.read_csv(Path(raw) / "HAM10000_metadata.csv")
    required = {"image_id", "lesion_id", "dx"}
    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError("Missing required metadata or null values")
    if frame.image_id.duplicated().any():
        raise ValueError("Duplicate image IDs in metadata")
    if set(frame.dx) != set(CLASSES):
        raise ValueError(f"Unexpected classes: {set(frame.dx) ^ set(CLASSES)}")
    if (frame.groupby("lesion_id").dx.nunique() > 1).any():
        raise ValueError("Conflicting lesion labels")
    paths, duplicates = image_paths(raw)
    missing = sorted(set(frame.image_id) - set(paths))
    if missing or duplicates:
        raise ValueError(f"Missing images: {missing[:10]} ({len(missing)}); duplicate mappings: {duplicates[:10]} ({len(duplicates)})")
    return frame, paths


def create_or_load_split(raw=RAW, split_path=SPLIT):
    frame, paths = metadata(raw)
    split_path = Path(split_path)
    if split_path.exists():
        result = pd.read_csv(split_path)
        if set(result.image_id) != set(frame.image_id) or len(result) != len(frame):
            raise ValueError("Saved split does not match metadata")
        expected = frame.set_index("image_id")[["lesion_id", "dx"]]
        found = result.set_index("image_id")[["lesion_id", "diagnosis"]].rename(columns={"diagnosis": "dx"})
        if not expected.sort_index().equals(found.sort_index()):
            raise ValueError("Saved split labels/lesions differ from raw metadata")
    else:
        result = frame[["image_id", "lesion_id", "dx"]].rename(columns={"dx": "diagnosis"}).copy()
        result["label"] = result.diagnosis.map(LABELS).astype(int)
        result["split"] = "train"
        splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)
        for fold, (_, held) in enumerate(splitter.split(frame, frame.dx, frame.lesion_id)):
            if fold < 3:
                result.loc[held, "split"] = "test"
            elif fold < 6:
                result.loc[held, "split"] = "val"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(split_path, index=False)
    verify_split(result)
    return result, paths


def verify_split(frame):
    if set(frame.split) != {"train", "val", "test"} or frame.image_id.duplicated().any():
        raise ValueError("Invalid split membership")
    groups = {name: set(frame.loc[frame.split == name, "lesion_id"]) for name in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        count = len(groups[a] & groups[b])
        print(f"{a} <-> {b} lesion overlap = {count}", flush=True)
        if count:
            raise ValueError("Lesion leakage")
    if (frame.label != frame.diagnosis.map(LABELS)).any():
        raise ValueError("Label mapping mismatch")
    print(pd.crosstab(frame.split, frame.diagnosis).reindex(["train", "val", "test"]).to_string(), flush=True)


def transform(training=False, size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        *([transforms.RandomHorizontalFlip(p=0.5)] if training else []),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class LesionDataset(Dataset):
    def __init__(self, rows, paths, training=False, size=224):
        self.rows = rows.reset_index(drop=True)
        self.paths = paths
        self.transform = transform(training, size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(self.paths[row.image_id]) as image:
            pixels = self.transform(image.convert("RGB"))
        return pixels, int(row.label)


def class_counts(rows):
    return np.bincount(rows.label.to_numpy(), minlength=len(CLASSES))
