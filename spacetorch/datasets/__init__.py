from .imagenet import ImageNetData, IMAGENET_TRANSFORMS, IMAGENET_TRANSFORMS_UNNORMALIZED, IMAGENET_TRANSFORMS_192X192
from .sine_gratings import SineGrating2019
from .floc import fLocData
from .retinal_waves import RetinalWaveData, DEFAULT_RWAVE_DIRS
from .noise import NoiseImages, NOISE_TRANSFORMS
from .nsd import NSDImages, NSD_TRANSFORMS
from .tvsd import TVSDImages, TVSD_TRANSFORMS

import torchvision
from spacetorch import paths

DEFAULT_TRANSFORMS = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

DEFAULT_TRANSFORMS_192x192 = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(192),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

_DATASETS = {
    "SineGrating2019": (
        SineGrating2019,
        (paths.SINE_GRATING_2019_DIR, DEFAULT_TRANSFORMS),
    ),
    "SineGrating2019_192x192": (
        SineGrating2019,
        (paths.SINE_GRATING_2019_DIR, DEFAULT_TRANSFORMS_192x192),
    ),
    "fLoc": (fLocData, (paths.FLOC_DIR, DEFAULT_TRANSFORMS)),
    "fLoc192x192": (fLocData, (paths.FLOC_DIR, DEFAULT_TRANSFORMS_192x192)),
    "NSD": (NSDImages, (paths.NSD_PATH, NSD_TRANSFORMS)),
    "TVSD": (TVSDImages, (paths.TVSD_PATH, TVSD_TRANSFORMS)),
    "ImageNet": (
        ImageNetData,
        (paths.DEFAULT_IMAGENET_VAL_DIR, IMAGENET_TRANSFORMS),
    ),
    "ImageNet_Unnormalized": (
        ImageNetData,
        (paths.DEFAULT_IMAGENET_VAL_DIR, IMAGENET_TRANSFORMS_UNNORMALIZED),
    ),
    "ImageNet192x192": (
        ImageNetData,
        (paths.DEFAULT_IMAGENET_VAL_DIR, IMAGENET_TRANSFORMS_192X192),
    ),
    "ImageNet_train": (
        ImageNetData,
        (paths.DEFAULT_IMAGENET_TRAIN_DIR, IMAGENET_TRANSFORMS),
    ),
    "RetinalWaves": (RetinalWaveData, (DEFAULT_RWAVE_DIRS, DEFAULT_TRANSFORMS)),
    "noise": (NoiseImages, (640, NOISE_TRANSFORMS)),
}


class DatasetRegistry:
    def __init__(self):
        self._DATASETS = _DATASETS

    @staticmethod
    def get(dataset_name: str):
        dataset = _DATASETS.get(dataset_name)
        if dataset is None:
            raise ValueError(
                f"Sorry, {dataset_name} not in registry. Try one of {_DATASETS.keys()}"
            )
        dataset_cls, dataset_args = dataset
        return dataset_cls(*dataset_args)

    @staticmethod
    def list() -> None:
        """
        Print all datasets in the registry
        """
        print("Available datasets:")
        for dataset in _DATASETS:
            print(f"\t {dataset}")
