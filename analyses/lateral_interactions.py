import math
import argparse
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from spacetorch.datasets import DatasetRegistry
from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant
from spacetorch.feature_extractor import get_features_from_layer
from spacetorch.variants.positions import get_positions
from spacetorch.variants import OUTPUT_DIMS_FOR_224_INPUTS


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
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

mapping = {
    "dinov2": "vitb14",
    "mocov3": "vitb16",
    "simclr": "vitb16",
    "supervised": "vitb16",
    "mae": "vitb16",
    "swinv2_simmim": "swinv2_base",
    "swin_moby": "swin_tiny",
}


def main():
    cfg = get_cfg(args)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)
    model = variant.eval_model
    positions = get_positions(cfg, rescale=False)

    is_swinv2 = ("swinv2" in cfg.name)

    dataset = DatasetRegistry.get("ImageNet" if not is_swinv2 else "ImageNet192x192")

    out = get_features_from_layer(
        model=model,
        dataset=dataset,
        verbose=True,
        shuffle=False,
        max_batches=8,
        batch_size=128,
        model_layer_strings=[x + ".attn.attn_matrix" for x in OUTPUT_DIMS_FOR_224_INPUTS[mapping[cfg.variant.name]]],
        reshape=False,
    )

    is_attn = (cfg.variant.architecture[-1] == "a")
    is_swin = ("swin" in cfg.name)
    
    for l in tqdm(out.keys()):
        layer = (l[:-17] if not is_attn else l[:-12]) + ("_output" if is_swin else "")
        save_dir = Path(cfg.output_dir) / layer
        save_dir.mkdir(parents=True, exist_ok=True)

        attn = torch.tensor(out[l])

        B, H, Q, K = attn.shape

        sK = int(math.sqrt(K))
        is_square = sK * sK == K

        if not is_square:
            attn = attn[:, :, 1:, 1:]
            B, H, Q, K = attn.shape

        # weighted_dists = torch.zeros(B, H, Q, device=attn.device)
        # for b in range(B):
        #     coords = torch.tensor(positions[layer].coordinates).reshape(-1, Q, 2).mean(0)
        #     distances = torch.cdist(coords, coords)
        #     for h in range(H):
        #         _attn = attn[b, h]
        #         avg_dist = (_attn * distances).sum(dim=-1)
        #         weighted_dists[b, h] = avg_dist
    
        np.save(save_dir / "attn_matrix.npy", attn)


if __name__ == "__main__":
    main()
