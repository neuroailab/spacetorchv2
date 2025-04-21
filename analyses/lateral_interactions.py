import argparse
import torch
import numpy as np
from pathlib import Path

from spacetorch.datasets import DatasetRegistry
from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant
from spacetorch.feature_extractor import get_features_from_layer
from spacetorch.variants.positions import get_positions


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
    positions = get_positions(cfg, rescale=False)[args.layer]

    save_dir = Path(cfg.output_dir) / args.layer
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = DatasetRegistry.get(args.dataset_name or "ImageNet")
    
    out = get_features_from_layer(
        model=model,
        dataset=dataset,
        verbose=True,
        max_batches=79,
        batch_size=128,
        model_layer_strings=[args.layer]
    )

    qkv = torch.tensor(out).reshape(-1, 196, 3, 12, 64).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    layer_idx = int(args.layer.split(".")[-1])
    q, k = model.blocks[layer_idx].attn.q_norm(q), model.blocks[layer_idx].attn.k_norm(k)
    q = q * model.blocks[layer_idx].attn.scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    
    B, H, Q, K = attn.shape
    
    weighted_dists = torch.zeros(B, H, Q, device=attn.device)
    for b in range(B):
        coords = torch.tensor(positions.coordinates).reshape(768, 196, 2).mean(0)
        distances = torch.cdist(coords, coords)
        for h in range(H):
            attn = attn[b, h]
            avg_dist = (attn * distances).sum(dim=-1)
            weighted_dists[b, h] = avg_dist

    np.save(save_dir / "attn_weighted_distances.npy", weighted_dists.cpu().numpy())


if __name__ == "__main__":
    main()
