import math
import argparse
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.spatial.distance import cdist

from spacetorch.datasets import DatasetRegistry
from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant
from spacetorch.feature_extractor import get_features_from_layer
from spacetorch.variants.positions import get_positions
from spacetorch.variants import OUTPUT_DIMS_FOR_224_INPUTS
from spacetorch.utils.torch_utils import resolve_sequential_module_from_str


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


def recover_swin_window_token_coords(block, device=None):
    """
    Recover token indices and (h, w) coordinates for each attention window in a SwinTransformerBlock.

    Returns
    -------
    windows : list of dicts, length = nW
        Each dict contains:
        {
            'token_indices': (N,) LongTensor, global indices into H*W
            'coords': (N, 2) LongTensor, (h, w) per token
        }
    """
    H, W = block.input_resolution
    ws = block.window_size
    ss = block.shift_size
    device = device or block.attn.relative_position_index.device

    # global grid
    h = torch.arange(H, device=device)
    w = torch.arange(W, device=device)
    grid = torch.stack(torch.meshgrid(h, w, indexing="ij"), dim=-1)  # (H, W, 2)

    # apply cyclic shift
    if ss > 0:
        grid = torch.roll(grid, shifts=(-ss, -ss), dims=(0, 1))

    # partition into windows
    grid = grid.view(
        H // ws, ws,
        W // ws, ws,
        2
    )
    grid = grid.permute(0, 2, 1, 3, 4).contiguous()
    grid = grid.view(-1, ws * ws, 2)  # (nW, N, 2)

    # undo shift for final coordinates
    if ss > 0:
        grid = (grid + ss) % torch.tensor([H, W], device=device)

    # convert to global token indices
    token_indices = grid[..., 0] * W + grid[..., 1]  # (nW, N)

    windows = []
    for i in range(grid.shape[0]):
        windows.append({
            "token_indices": token_indices[i],
            "coords": grid[i]
        })

    return windows


def top_mass_mask(attn, alpha=0.9):
    """
    attn: np.ndarray of shape (B, H, Q, K)
    Returns: boolean mask of same shape
    """
    # sort descending along keys
    idx = np.argsort(attn, axis=-1)[..., ::-1]
    sorted_attn = np.take_along_axis(attn, idx, axis=-1)

    cumsum = np.cumsum(sorted_attn, axis=-1)

    keep = cumsum <= alpha
    keep[..., 0] = True  # always keep max

    # unsort back to original order
    inv_idx = np.argsort(idx, axis=-1)
    keep = np.take_along_axis(keep, inv_idx, axis=-1)

    return keep


def compute_wiring_for_swin(block, attn, coords):
    windows = recover_swin_window_token_coords(block, device="cpu")
    
    B, H, Q, K = attn.shape
    nW = len(windows)
    batch_size = B // nW

    attn = attn.reshape(batch_size, nW, H, Q, K)
    # coords: (T, 2)
    global_distances = cdist(coords, coords)

    max_dist = global_distances.max()
    n_bins = 10
    bin_edges = np.linspace(0, max_dist, n_bins + 1)

    sparseness = {str(ran): [] for ran in bin_edges[:-1]}

    res = []

    for window_idx, window in enumerate(tqdm(windows, desc="Computing intra-layer functional wiring cost")):
        token_indices = window['token_indices']
        dists = global_distances[np.ix_(token_indices, token_indices)]

        weighted = attn[:, window_idx] * dists[None, None, :, :]
        mean_window = weighted.sum(axis=(-1)).mean(axis=0)

        res.append(mean_window)

        for i in range(len(bin_edges) - 1):
            d_min = bin_edges[i]
            d_max = bin_edges[i + 1]

            bin_mask = (dists >= d_min) & (dists < d_max)
            bin_mask = bin_mask & (dists > 0)

            mass = (attn[:, window_idx] * bin_mask[None, None]).sum(axis=-1)
            total_mass = (attn[:, window_idx]).sum(axis=-1)

            sparseness[str(d_min)].append((mass / total_mass).mean(axis=0))

    sparseness = {ran: np.mean(vals, axis=0) for ran, vals in sparseness.items()}
    res = np.concatenate(res, axis=0)

    return res, sparseness


def compute_wiring_for_vit(attn, coords):
    B, H, Q, K = attn.shape

    sK = int(math.sqrt(K))
    is_square = sK * sK == K

    if not is_square:
        attn = attn[:, :, 1:, 1:]
        B, H, Q, K = attn.shape

    distances = cdist(coords, coords)
    max_dist = distances.max()
    n_bins = 10
    bin_edges = np.linspace(0, max_dist, n_bins + 1)

    weighted = (attn * distances[None, None, :, :])

    sparseness = {}

    for i in range(len(bin_edges) - 1):
        d_min = bin_edges[i]
        d_max = bin_edges[i + 1]

        bin_mask = (distances >= d_min) & (distances < d_max)
        bin_mask = bin_mask & (distances > 0)

        mass = (attn * bin_mask[None, None]).sum(axis=-1)
        total_mass = (attn).sum(axis=-1)

        sparseness[str(d_min)] = (mass / total_mass).mean(axis=0)

    return weighted.sum(axis=(-1)).mean(axis=0), sparseness


def main():
    cfg = get_cfg(args)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)
    model = variant.eval_model
    positions = get_positions(cfg, rescale=False)

    is_swinv2 = ("swinv2" in cfg.name)
    is_attn = (cfg.variant.architecture[-1] == "a")
    is_swin = ("swin" in cfg.name)

    dataset = DatasetRegistry.get("ImageNet" if not is_swinv2 else "ImageNet192x192")

    attn = get_features_from_layer(
        model=model,
        dataset=dataset,
        verbose=True,
        shuffle=False,
        max_batches=8,
        batch_size=128,
        model_layer_strings=args.layer + ".attn.attn_matrix",
        reshape=False,
    )
    
    layer = args.layer if not is_attn else args.layer + (".attn_output" if is_swin else ".attn")

    L, W, _ = OUTPUT_DIMS_FOR_224_INPUTS[cfg.variant.architecture][layer]

    coords = positions[layer].coordinates
    coords = coords.reshape(L, W * W, 2).mean(0)
    
    save_dir = Path(cfg.output_dir) / layer
    save_dir.mkdir(parents=True, exist_ok=True)

    if is_swin:
        res, sparseness = compute_wiring_for_swin(
            block=resolve_sequential_module_from_str(model, args.layer),
            attn=attn,
            coords=coords,
        )
    else:
        res, sparseness = compute_wiring_for_vit(
            attn=attn,
            coords=coords,
        )

    print(res.shape)

    np.save(save_dir / "lateral_wiring.npy", res)
    np.savez(save_dir / "sparseness.npz", **sparseness)


if __name__ == "__main__":
    main()
