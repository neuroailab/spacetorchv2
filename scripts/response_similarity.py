import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from spacetorch.datasets import DatasetRegistry
from spacetorch.feature_extractor import get_features_from_layer
from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant
from spacetorch.maps.imagenet_tissue import create_imnet_tissue
from spacetorch.variants.positions import get_positions
from spacetorch.utils import spatial_utils


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--layers", nargs="+")
    return parser


args = get_parser().parse_args()


def main():
    cfg = get_cfg(args)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)

    model = variant.eval_model
    dataset = DatasetRegistry.get("ImageNet")

    layers = args.layers
    widths = [5, 5, 5, 5, 40, 40]

    for i, layer in enumerate(layers):
        features = get_features_from_layer(
            model,
            dataset,
            verbose=True,
            max_batches=32,
            batch_size=32,
            model_layer_strings=[layer],
        )

        position_dict = get_positions(cfg)

        tissue = create_imnet_tissue(
            features, position_dict[layer].coordinates
        )

        save_path = Path(cfg.output_dir) / "figures"
        save_path.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(5, 5))
        colors = sns.color_palette("rainbow", n_colors=40)

        for width in list(range(widths[i], 201, 5)):
            tissue.reset_unit_mask()

            window_params = spatial_utils.WindowParams(
                width=width / 2,
                window_number_limit=1,
                edge_buffer=0,
                unit_number_limit=1000,
            )

            try:
                distances, curves = tissue.correlation_over_distance(
                    window_params=window_params,
                    normalize_x_axis=True,
                )

                plt.plot(distances, np.mean(curves, axis=0), color=colors[width // 5], label=width)

                plt.legend(fontsize=5)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            except:
                break

            # np.savez(save_path / f"response_similarity_{layer}.png", distances=distances, curves=curves)


if __name__ == "__main__":
    """
    Example usage:
    python3 scripts/response_similarity.py --config configs/analysis_configs/vitb14_dinov2_imagenet_unoptimized.yaml --layers blocks.1 blocks.2 blocks.4 blocks.5 blocks.6 blocks.7 blocks.8 blocks.9 blocks.10 blocks.11
    """
    main()
