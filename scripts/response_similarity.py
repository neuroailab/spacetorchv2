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
    return parser


args = get_parser().parse_args()


def main():
    cfg = get_cfg(args)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)

    model = variant.eval_model
    dataset = DatasetRegistry.get("ImageNet")

    layers = [f"blocks.{i}" for i in range(40)]
    widths = [5, 5, 5, 5, 5, 5, 5, 40, 40, 40, 40]

    save_path = Path(cfg.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 5))
    colors = sns.color_palette("Greys", n_colors=11)

    for i, (layer, width) in enumerate(zip(layers, widths)):
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

        tissue.reset_unit_mask()

        window_params = spatial_utils.WindowParams(
            width=width / 2,
            window_number_limit=1,
            edge_buffer=0,
            unit_number_limit=1000,
        )

        distances, curves = tissue.correlation_over_distance(
            window_params=window_params,
            normalize_x_axis=True,
        )

        plt.plot(distances, np.mean(curves, axis=0), color=colors[i], label=layer)

    plt.legend(fontsize=5)
    plt.savefig(save_path / "response_similarity.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
