import seaborn as sns


MACAQUE_MODEL = ("macaque", "#00AB33")
TDANN_MODEL = ("tdann", "#7402E5")
UNOPTIMIZED_MODEL = ("vitb14_dinov2_imagenet_unoptimized", "darkgray")
TASK_ONLY_MODEL = ("vitb14_dinov2_imagenet_noswapopt_lw0", "black")
NOSWAPOPT_MODELS = zip([
    f"vitb14_dinov2_imagenet_noswapopt_lw{i}" for i in ["01", "x1", "x2", "x5", "x10"]
], sns.color_palette("Reds", n_colors=5))
SWAPOPT_MODELS = zip([
    f"vitb14_dinov2_imagenet_swapopt_lw{i}" for i in ["x1", "x1x1", "x2x1", "x5x1", "x10x1", "x10x2", "x10x5", "x10x10"]
], sns.color_palette("Blues", n_colors=8))

ALL_MODELS = [
    MACAQUE_MODEL,
    TDANN_MODEL,
    UNOPTIMIZED_MODEL,
    TASK_ONLY_MODEL,
    *NOSWAPOPT_MODELS,
    *SWAPOPT_MODELS,
]
