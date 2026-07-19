import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from spacetorch.receptive_fields.rf_helper import *
from spacetorch.receptive_fields.visualize_helper import *
from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant
from spacetorch.maps.expanding_sine_tissue import get_expanding_sine_responses
from spacetorch.variants.positions import get_positions
from spacetorch.colormaps import nauhaus_colormaps, nauhaus_raw_colormaps


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


def main():
    cfg = get_cfg(args)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)
    model = variant.eval_model
    is_tdann = "tdann" in cfg.name
    positions = get_positions(cfg, rescale=is_tdann)[args.layer].coordinates

    save_dir = Path(cfg.output_dir) / args.layer
    save_dir.mkdir(parents=True, exist_ok=True)

    is_lcnn = ("lcnn" in cfg.name)
    is_llcnn = ("llcnn" in cfg.name)

    responses = get_expanding_sine_responses(
        model,
        layers=[args.layer],
        normalize_to_ringach_firing_rates=("tdann" in cfg.name) or ("resnet" in cfg.name),
        is_lcnn=is_lcnn and not is_llcnn,
        is_llcnn=is_llcnn,
    )

    cv, pref_angles, modulation = responses.get_peripheral_tuning_to_central_grating()

    np.savez(save_dir / "axial_specificity.npz", **{"cv": cv, "pref_angles": pref_angles, "modulation": modulation})

    angle_vals = np.linspace(0, 180, 8, endpoint=False)
    angle_vals_remapped = [90, 67.5, 45, 22.5, 0, 157.5, 135, 112.5]
    cv_mask = cv < 0.6

    for t in [0.3]:
        for angle in range(8):
            modulation_mask = modulation[angle] > t * np.max(modulation[angle])

            fig, axs = plt.subplots(1, 2, figsize=(5, 2))
            axs[0].scatter(positions[cv_mask & modulation_mask, 0], positions[cv_mask & modulation_mask, 1], s=3, c=pref_angles[cv_mask & modulation_mask], cmap=nauhaus_colormaps["angles"])
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            axs[0].set_xlim(0, 36.75)
            axs[0].set_ylim(0, 36.75)
            pref_angles_remapped = np.array([angle_vals_remapped[np.where(angle_vals == pref_angle)[0][0]] for pref_angle in pref_angles])
            axs[1].hist(pref_angles_remapped[cv_mask & modulation_mask], bins=8)
            plt.subplots_adjust(wspace=0.5)
            plt.savefig(save_dir / f"bouton_distribution_{angle}.png", dpi=300, bbox_inches='tight')
            plt.close()

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5, 5))

        for angle in range(8):
            mask = (cv < 0.6) & (modulation[angle] > t * np.max(modulation[angle]))
            angles_deg = pref_angles[mask]

            # mirror to full circle for density estimation
            angles_rad = np.deg2rad(angles_deg)
            mirrored   = angles_rad + np.pi
            all_angles = np.concatenate([angles_rad, mirrored])

            # KDE over the full circle
            theta_grid = np.linspace(0, 2 * np.pi, 8, endpoint=False)
            kde = gaussian_kde(all_angles, bw_method=0.2)
            density = kde(theta_grid)
            density = density / density.max()

            colorbar_index = [4, 3, 2, 1, 0, 7, 6, 5]
            color = nauhaus_raw_colormaps["angles"][colorbar_index[angle]]
            ax.fill(theta_grid, density, alpha=0.2, color=color)
            ax.plot(theta_grid, density, color=color, linewidth=1.5)

            # close the polygon
            ax.plot([theta_grid[-1], theta_grid[0]], [density[-1], density[0]],
                    color=color, linewidth=1.5)

        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"])
        ax.set_yticks([])
        ax.set_title(
            f"Orientation preference distribution",
            pad=15,
        )
        ax.spines["polar"].set_visible(False)

        plt.savefig(save_dir / f"axial_specificity.png", dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    main()
