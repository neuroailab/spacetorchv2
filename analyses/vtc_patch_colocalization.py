import argparse
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from pathlib import Path

from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant
from spacetorch.variants.positions import get_positions
from spacetorch.datasets.floc import DOMAIN_CONTRASTS
from spacetorch.maps.floc_tissue import get_floc_tissue
from spacetorch.utils import spatial_utils


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--layer", type=str)
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


args = get_parser().parse_args()
contrasts = DOMAIN_CONTRASTS
contrast_dict = {c.name: c for c in contrasts}


def get_sel(tissue, contrast):
    """Helper that gets selectivity from tissue, whether it's a ITMap or an ITN"""
    if hasattr(tissue, "responses"):
        return tissue.responses.selectivity(on_categories=contrast.on_categories)

    return tissue.maps[contrast.name]


def model_colocalization(
    tissue, con1, con2, t, window_width: float = 1,
):
    T_THRESH = t

    # split the tissue into non-overlapping 1mm x 1mm windows, reject windows with
    # less than 6 units
    windows = [
        window
        for window in spatial_utils.get_adjacent_windows(tissue.positions, width=7)
        if len(window.indices) > 5
    ]

    # get the selectivity of all units to each contrast
    sel1 = get_sel(tissue, con1)
    sel2 = get_sel(tissue, con2)

    # compute the fraction of units in each window that clear the selectivity threshold
    frac1_list = []
    frac2_list = []
    for window in windows:
        frac1 = np.mean(sel1[window.indices] > T_THRESH)
        frac2 = np.mean(sel2[window.indices] > T_THRESH)

        frac1_list.append(frac1)
        frac2_list.append(frac2)

    # check the correlation of the counts across windows
    r, p = stats.spearmanr(frac1_list, frac2_list)
    distance = 1 - r
    return 1 - distance / 2


def main():
    cfg = get_cfg(args)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)

    model = variant.eval_model
    is_tdann = "tdann" in cfg.name
    positions = get_positions(cfg, rescale=is_tdann)[args.layer]

    is_swinv2 = ("swinv2" in cfg.name)

    save_dir = Path(cfg.output_dir) / args.layer
    save_dir.mkdir(parents=True, exist_ok=True)

    counter = 0
    for _ in [2, 3, 4, 6, 8, 10, 12]:
        if (save_dir / f"vtc_patch_coloc_t{_}.npz").exists():
            print(f"Found existing results in {save_dir}, skipping...")
            counter += 1
        else:
            break
        if counter == 7:
            return
        
    _t = [2, 3, 4, 6, 8, 10, 12][counter:]

    floc_tissue = get_floc_tissue(
        cfg.name,
        model,
        positions,
        layer=args.layer,
        output_dir=save_dir,
        is_swinv2=is_swinv2
    )

    for t in _t:
        floc_tissue.patches = []
        for contrast in contrasts:
            floc_tissue.find_patches(
                contrast,
                t=t,
                output_dir=save_dir,
                # skip_cache=True
            )
        
        coloc_res = {"Pair": [], "Coloc": []}
        fp = model_colocalization(
            floc_tissue, contrast_dict["Faces"], contrast_dict["Places"], t=t
        )
        fb = model_colocalization(
            floc_tissue, contrast_dict["Faces"], contrast_dict["Bodies"], t=t
        )
        coloc_res["Pair"].append("Face-Body")
        coloc_res["Coloc"].append(fb)
        coloc_res["Pair"].append("Face-Place")
        coloc_res["Coloc"].append(fp)

        np.savez(save_dir / f"vtc_patch_coloc_t{t}.npz", **coloc_res)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(1, 1))
        ax.scatter(fb, fp, c="black", s=10)
        ax.axhline(0.5, c="k", alpha=0.8, lw=0.75)
        ax.axvline(0.5, c="k", alpha=0.8, lw=0.75)
        ax.set_xlabel("Face-Body\nOverlap")
        ax.set_ylabel("Face-Place\nOverlap")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xticks([0.5])
        ax.set_yticks([0.5])
        sns.despine(fig, ax)
        fig.savefig(save_dir / f"vtc_patch_coloc_t{t}.png", dpi=300, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    """
    Example usage:
    python3 visualize/vtc_patch_colocalization.py --config configs/analysis_configs/vitb14_dinov2_imagenet_unoptimized.yaml --layer blocks.10
    """
    main()