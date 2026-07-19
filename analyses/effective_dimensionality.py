import re
import argparse
import torch
import torch.nn as nn
from pathlib import Path

from spacetorch.datasets import DatasetRegistry
from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant
from spacetorch.feature_extractor import get_features_from_layer
from spacetorch.utils.dimensionality import effective_dim, compute_eigvals


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--dataset_name", type=str)
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

    save_dir = Path(cfg.output_dir) / args.layer
    save_dir.mkdir(parents=True, exist_ok=True)

    # if (save_dir / "effective_dim.txt").exists():
    #     print(f"Found existing results in {save_dir}, skipping...")
    #     return

    is_lcnn = ("lcnn" in cfg.name)
    is_llcnn = ("llcnn" in cfg.name)

    dataset_name = args.dataset_name or ("TVSD_Unnormalized" if (is_lcnn or is_llcnn) else "TVSD")
    dataset = DatasetRegistry.get(dataset_name)
    
    features = get_features_from_layer(
        model=model,
        dataset=dataset,
        verbose=True,
        max_batches=8,
        batch_size=128,
        model_layer_strings=[args.layer]
    )
    
    features = torch.tensor(features)
    pool = nn.AdaptiveAvgPool2d(1)
    pooled_features = pool(features).squeeze()

    eigvals = compute_eigvals(pooled_features)
    eff_dim = effective_dim(eigvals=eigvals)

    with open(save_dir / f"effective_dim_{dataset_name}.txt", "w") as f:
        f.write(f"{eff_dim.item()}")


if __name__ == "__main__":
    main()
