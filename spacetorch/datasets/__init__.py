from .imagenet import ImageNetData, IMAGENET_TRANSFORMS, IMAGENET_TRANSFORMS_UNNORMALIZED
from .sine_gratings import SineGrating2019
from .expanding_sine_gratings import ExpandingSineGrating2019
from .tiled_sine_gratings import TiledSineGrating2019
from .floc import fLocData
from .retinal_waves import RetinalWaveData, DEFAULT_RWAVE_DIRS
from .noise import NoiseImages, NOISE_TRANSFORMS
from .nsd import NSDImages, NSD_TRANSFORMS
from .tvsd import TVSDImages, TVSD_TRANSFORMS, TVSD_TRANSFORMS_UNNORMALIZED
from .discrimination import DiscriminationData, DISCRIMINATION_TRANSFORMS

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

DEFAULT_TRANSFORMS_UNNORMALIZED = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
    ]
)


_DATASETS = {
    "SineGrating2019": (
        SineGrating2019,
        (paths.SINE_GRATING_2019_DIR, DEFAULT_TRANSFORMS),
    ),
    "SineGrating2019_Unnormalized": (
        SineGrating2019,
        (paths.SINE_GRATING_2019_DIR, DEFAULT_TRANSFORMS_UNNORMALIZED),
    ),
    "ExpandingSineGrating2019": (
        ExpandingSineGrating2019,
        (paths.SINE_GRATING_2019_DIR, DEFAULT_TRANSFORMS_UNNORMALIZED),
    ),
    "ExpandingSineGratingLC2026": (
        ExpandingSineGrating2019,
        (paths.SINE_GRATING_LC_2026_DIR, DEFAULT_TRANSFORMS_UNNORMALIZED),
    ),
    "TiledSineGrating2019": (
        TiledSineGrating2019,
        (DEFAULT_TRANSFORMS_UNNORMALIZED,),
    ),
    "fLoc": (fLocData, (paths.FLOC_DIR, DEFAULT_TRANSFORMS)),
    "fLoc_Unnormalized": (fLocData, (paths.FLOC_DIR, DEFAULT_TRANSFORMS_UNNORMALIZED)),
    "NSD": (NSDImages, (paths.NSD_PATH, NSD_TRANSFORMS)),
    "TVSD": (TVSDImages, (paths.TVSD_PATH, TVSD_TRANSFORMS)),
    "TVSD_Unnormalized": (TVSDImages, (paths.TVSD_PATH, TVSD_TRANSFORMS_UNNORMALIZED)),
    "ImageNet": (
        ImageNetData,
        (paths.DEFAULT_IMAGENET_VAL_DIR, IMAGENET_TRANSFORMS),
    ),
    "ImageNet_Unnormalized": (
        ImageNetData,
        (paths.DEFAULT_IMAGENET_VAL_DIR, IMAGENET_TRANSFORMS_UNNORMALIZED),
    ),
    "ImageNet_train": (
        ImageNetData,
        (paths.DEFAULT_IMAGENET_TRAIN_DIR, IMAGENET_TRANSFORMS),
    ),
    "Discrimination_train": (
        DiscriminationData,
        (paths.DEFAULT_DISCRIMINATION_TRAIN_DIR, DISCRIMINATION_TRANSFORMS)
    ),
    "Discrimination_val": (
        DiscriminationData,
        (paths.DEFAULT_DISCRIMINATION_VAL_DIR, DISCRIMINATION_TRANSFORMS)
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
