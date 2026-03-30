import csv
import os
import torch
import numpy as np
from typing import Any, Callable, Optional, Tuple

from PIL import Image
import random
from decord import VideoReader, cpu


class CustomVideoDataset:
    def __init__(
        self,
        root: str = "/data2/ynshah/Kinetics400/k400/train.csv",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self._entries = []
        self._load_csv(root)
        self.transform = transform
        self.target_transform = target_transform

    def _load_csv(self, csv_path: str) -> None:
        if not os.path.exists(csv_path):
            raise RuntimeError(f"CSV file not found: {csv_path}")

        with open(csv_path, "r") as f:
            reader = csv.reader(f, delimiter=" ")
            for row in reader:
                if len(row) != 2:
                    continue
                path, label = row
                self._entries.append((path, int(label)))

    def _sample_uniform_frames(self, vr):
        total_frames = len(vr)

        if total_frames == 0:
            return None

        if total_frames < 8:
            indices = np.linspace(0, total_frames - 1, total_frames).astype(int)
            pad = np.ones(8 - total_frames, dtype=int) * (total_frames - 1)
            indices = np.concatenate([indices, pad])
        else:
            indices = np.linspace(
                0,
                total_frames - 1,
                8
            ).astype(int)

        return vr.get_batch(indices).asnumpy()

    def get_video_data(self, index: int) -> bytes:
        path, _ = self._entries[index]
        
        for _ in range(20):
            if not os.path.exists(path):
                print(f"Video path does not exist: {path}")
                path, _ = random.choice(self._entries)
            else:
                try:
                    vr = VideoReader(path, ctx=cpu(0))
                except:
                    print(f"Error with video: {path}")
                    path, _ = random.choice(self._entries)
                    continue
                frames = self._sample_uniform_frames(vr)
                if frames is not None:
                    frames = [Image.fromarray(frames[i]) for i in range(len(frames))]
                    return frames
                else:
                    print(f"Empty video: {path}")
                    path, _ = random.choice(self._entries)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image = self.get_video_data(index)
        except Exception as e:
            raise RuntimeError(
                f"Cannot read image or video for sample {index}"
            ) from e
    
        target = self._entries[index][1]

        if self.transform is not None:
            video_tensor = torch.stack([self.transform(im) for im in image])
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (video_tensor, target)

    def __len__(self) -> int:
        return len(self._entries)
