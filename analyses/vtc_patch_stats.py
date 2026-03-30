import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant
from spacetorch.variants.positions import get_positions
from spacetorch.datasets.floc import DOMAIN_CONTRASTS
from spacetorch.maps.floc_tissue import get_floc_tissue


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


def plot_helper(ax, df, key):
    """We use the same function to plot patch count and patch area"""

    sns.barplot(
        data=df, y=key, color="black", ax=ax
    )

    ax.set_xticks([])


def main():
    cfg = get_cfg(args)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)

    model = variant.eval_model
    is_tdann = "tdann" in cfg.name and not "tdann_logpolar" in cfg.name
    positions = get_positions(cfg, rescale=is_tdann)[args.layer]

    is_swinv2 = ("swinv2" in cfg.name)

    save_dir = Path(cfg.output_dir) / args.layer
    save_dir.mkdir(parents=True, exist_ok=True)

    # counter = 0
    # for _ in [2, 3, 4, 6, 8, 10, 12]:
    #     if (save_dir / f"vtc_patch_area_t{_}.npz").exists():
    #         print(f"Found existing results in {save_dir}, skipping...")
    #         counter += 1
    #     else:
    #         break
    #     if counter == 7:
    #         return
        
    _t = [2, 3, 4, 6, 8, 10, 12][:]

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
        
        patch_area_results = {
            "Contrast": [],
            "Area": [],
        }
        if len(floc_tissue.patches) == 0:
            for contrast in contrasts:
                patch_area_results["Contrast"].append(contrast)
                patch_area_results["Area"].append(0)
        for patch in floc_tissue.patches:
            patch_area_results["Contrast"].append(patch.contrast.name)
            patch_area_results["Area"].append(np.mean(patch.area))

        patch_count_results = {
            "Contrast": [],
            "Count": [],
        }
        for contrast in contrasts:
            matching_patches = list(
                filter(lambda p: p.contrast == contrast, floc_tissue.patches)
            )
            patch_count_results["Contrast"].append(contrast.name)
            patch_count_results["Count"].append(len(matching_patches))

        np.savez(save_dir / f"vtc_patch_area_t{t}.npz", **patch_area_results)
        np.savez(save_dir / f"vtc_patch_count_t{t}.npz", **patch_count_results)

        patch_area_df = pd.DataFrame(patch_area_results)
        patch_count_df = pd.DataFrame(patch_count_results)

        fig, axes = plt.subplots(figsize=(2.5, 0.75), ncols=2, gridspec_kw={"wspace": 1.2})

        plot_helper(axes[0], patch_count_df, "Count")
        plot_helper(axes[1], patch_area_df, "Area")

        # pretty
        axes[0].set_ylabel("Number of Patches")
        axes[1].set_ylabel(r"Patch Area [$mm^2$]")
        axes[0].yaxis.set_major_formatter("{x:.1f}")
        for ax in axes:
            ax.set_xlabel("")
            ax.set_xticks([])
            sns.despine(fig, ax)
            
        fig.savefig(save_dir / f"vtc_patch_stats_t{t}.png", dpi=300, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    """
    Example usage:
    python3 visualize/vtc_patch_stats.py --config configs/analysis_configs/vitb14_dinov2_imagenet_unoptimized.yaml --layer blocks.10
    """
    main()