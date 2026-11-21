from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


FOLDERS = ("im1", "im2", "label1", "label2")
PALETTE = np.array([
    (255, 255, 255),  # 0: No change
    (0, 0, 255),      # 1: Water
    (128, 128, 128),  # 2: Non-vegetated ground
    (0, 128, 0),      # 3: Low vegetation
    (0, 255, 0),      # 4: Trees
    (128, 0, 0),      # 5: Buildings
    (255, 0, 0),      # 6: Playgrounds
], dtype=np.uint8)


def encode_mask(rgb):
    mask = np.full(rgb.shape[:2], -1, dtype=np.int64)
    for class_id, color in enumerate(PALETTE):
        mask[np.all(rgb == color, axis=-1)] = class_id

    if (mask == -1).any():
        raise ValueError("Found an RGB color outside the SECOND palette.")
    return mask


class SecondPatches(Dataset):
    def __init__(self, root, pair_names):
        self.root = Path(root)
        self.pair_names = pair_names

    def __len__(self):
        return len(self.pair_names) * 4

    def __getitem__(self, index):
        pair_index, tile = divmod(index, 4)
        row, col = divmod(tile, 2)
        left, top = col * 256, row * 256
        box = (left, top, left + 256, top + 256)
        name = self.pair_names[pair_index]

        arrays = []
        for folder in FOLDERS:
            path = self.root / folder / name
            with Image.open(path) as image:
                if image.size != (512, 512):
                    raise ValueError(f"Expected a 512 × 512 image: {path}")
                arrays.append(np.array(image.crop(box).convert("RGB")))

        images = [
            torch.from_numpy(a).permute(2, 0, 1).float().div(255)
            for a in arrays[:2]
        ]
        masks = [torch.from_numpy(encode_mask(a)) for a in arrays[2:]]
        return (*images, *masks)


def make_val_loader(data_dir, batch_size=4, seed=42):
    data_dir = Path(data_dir)
    names_by_folder = [
        {p.name for p in (data_dir / folder).glob("*.png")}
        for folder in FOLDERS
    ]
    names = names_by_folder[0]
    if not names or any(group != names for group in names_by_folder[1:]):
        raise ValueError(f"Expected complete SECOND image and mask pairs in {data_dir}")

    # Match the split used for model comparison.
    shuffled = np.random.default_rng(seed).permutation(sorted(names)).tolist()
    val_names = shuffled[int(0.8 * len(shuffled)):]

    return DataLoader(
        SecondPatches(data_dir, val_names),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )