import os
import h5py
import torch
import torchvision
from PIL import Image

TVSD_TRANSFORMS = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

DUMMY_LABEL = 0

class TVSDImages(torch.utils.data.Dataset):
    """
    THINGS Ventral Spiking Dataset
    """

    def __init__(self, stim_path, transform=None):
        super().__init__()
        self.stim_path = stim_path
        self.transform = transform
        self.image_paths = self._load_image_paths()

    def _get_image_paths_for_monkey(self, metadata):
        path = 'train_imgs'
        refs = metadata[path]['things_path'][:].flatten()
        decoded_strings = []
        for ref in refs:
            arr = metadata[path][ref][()]
            arr = arr.flatten()
            s = ''.join(map(chr, arr))
            decoded_strings.append(
                os.path.join(self.stim_path, "images", s.replace('\\', '/'))
            )
        return decoded_strings

    def _load_image_paths(self):
        metadata_path_F = os.path.join(
            self.stim_path, 'things_imgs_monkeyF.mat'
        )
        metadataF = h5py.File(metadata_path_F, 'r')
        return self._get_image_paths_for_monkey(metadataF)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path = self.image_paths[idx]
        x = Image.open(image_path).convert("RGB")
        
        if self.transform:
            x = self.transform(x)

        return x, DUMMY_LABEL
