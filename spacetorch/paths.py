from pathlib import Path


# standard locations for things:
PROJ_DIR = Path('/ccn2/u/ynshah/spacetorchv2/')

git_root = PROJ_DIR

DS_DIR = PROJ_DIR / "datasets"
NSD_PATH = DS_DIR / "nsd_stimuli.hdf5"
SINE_GRATING_2019_DIR = DS_DIR / "sine_grating_images_20190507"
FLOC_DIR = DS_DIR / "fLoc_stimuli"
IMAGENET_DIR = Path("/data/ynshah/imagenet")
DEFAULT_IMAGENET_TRAIN_DIR = IMAGENET_DIR / "train"
DEFAULT_IMAGENET_VAL_DIR = IMAGENET_DIR / "val"
RWAVE_CONTAINER_PATH = DS_DIR / "rwave_python_images"
