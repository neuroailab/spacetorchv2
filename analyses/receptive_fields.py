import argparse
import math
from pathlib import Path
from collections import OrderedDict
from torch.utils.data import DataLoader

from spacetorch.receptive_fields.rf_helper import *
from spacetorch.receptive_fields.visualize_helper import *
from spacetorch.datasets import DatasetRegistry
from spacetorch.configs import get_cfg
from spacetorch.utils.torch_utils import resolve_sequential_module_from_str
from spacetorch.variants import get_variant, SCALE_FACTOR


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

    save_dir = Path(cfg.output_dir) / args.layer
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = DatasetRegistry.get("ImageNet_Unnormalized")

    data_loader: DataLoader = DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=1, pin_memory=True
    )

    transform = None
    what = None

    hook_dict = OrderedDict()

    def get_hook(name):
        """Returns a hook function that saves the output of the layer to the outputs dictionary."""
        def hook(module, input, output):
            if "attn_output" in name:
                output = module.attn_output
            spatial_features = output.float()
            
            if len(spatial_features.shape) < 4:
                N, L, H = spatial_features.shape

                # Check if L is a perfect square
                sL = int(math.sqrt(L))
                is_square = sL * sL == L

                # Remove CLS token only if L is not already a square
                if not is_square:
                    spatial_features = spatial_features[:, 1:]
                    L = spatial_features.shape[1]
                    sL = int(math.sqrt(L))

                spatial_features = spatial_features.reshape(N, sL, sL, H).permute(0, 3, 1, 2)
            
            hook_dict[name] = spatial_features
        return hook

    for run in range(5):
        if "attn_output" in args.layer or "attn_matrix" in args.layer:
            hook_layer = args.layer[:-12]
        else:
            hook_layer = args.layer
        
        layer = resolve_sequential_module_from_str(model, hook_layer)

        handle = layer.register_forward_hook(get_hook(args.layer))

        analysis_output = analysis_single_layer(
            model=model,
            layer_name=args.layer,
            scale_factor=SCALE_FACTOR[cfg.variant.architecture][args.layer],
            data_loader=data_loader,
            hook_dict=hook_dict, 
            max_image_num=1024,
            device="cuda",
        )

        handle.remove()

        distance_matrix, point_dict = analysis_output

        rfs = create_plots(
            distance_matrix=distance_matrix,
            point_dict=point_dict,
            path=save_dir,
            extra=f"{run}",
            visualize=True,
        )
        np.savez(save_dir / f"rfs_{run}.npz", **rfs)


if __name__ == "__main__":
    main()
