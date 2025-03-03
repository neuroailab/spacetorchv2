import argparse
from spacetorch.losses.losses_torch import spatial_loss_batch, spatial_correlation_loss
from spacetorch.datasets import DatasetRegistry
from spacetorch.feature_saver import FeatureSaver
import numpy as np
import torch
from pathlib import Path
from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant
from spacetorch.variants.positions import get_positions


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--layers", nargs="+")
    return parser


args = get_parser().parse_args()


def main():
    cfg = get_cfg(args)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)

    model = variant.eval_model
    positions = get_positions(cfg, rescale=False)

    dataset = DatasetRegistry.get(args.dataset_name)

    save_path: Path = ""
    feature_saver: FeatureSaver = FeatureSaver(
        model, args.layers, dataset, save_path, max_images=1000
    )

    feature_saver.compute_features(batch_size=args.batch_size)

    features = feature_saver._features

    spatial_losses = []
    for layer, feature in features.items():
        print(layer)
        feature = feature.reshape(len(feature), -1)
        pos = positions[layer]

        loss = spatial_loss_batch(
            spatial_correlation_loss,
            torch.from_numpy(feature).float().cuda(),
            torch.from_numpy(pos.coordinates).float().cuda(),
            torch.from_numpy(pos.neighborhood_indices).cuda(),
            n_runs=1000,
        )
        
        spatial_losses.append(loss.item())
    
    with open("imagenet_spatial_losses.txt", "a") as f:
        f.write(cfg.name + ",")
        f.write(",".join(str(item) for item in spatial_losses))
        f.write("\n")


if __name__ == "__main__":
    main()
