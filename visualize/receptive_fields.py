import argparse
from collections import OrderedDict
from pathlib import Path
import math
import re
from torch.utils.data import DataLoader
from spacetorch.receptive_fields.rf_helper import *
from spacetorch.receptive_fields.visualize_helper import *
from spacetorch.datasets import DatasetRegistry
from spacetorch.configs import get_cfg
from spacetorch.utils.torch_utils import resolve_sequential_module_from_str
from spacetorch.variants import get_variant


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--layer", type=str)
    parser.add_argument("--checkpoint", type=str)
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


def extract_parts(variable):
    # Extract the base part (blocks.[0-11])
    base_match = re.match(r"(blocks\.\d+)", variable)
    base = base_match.group(0) if base_match else None

    # Extract whether .attn or .mlp is present
    attn = ".attn" in variable
    mlp = ".mlp" in variable

    return base, "attn" if attn else ("mlp" if mlp else None)


def main():
    cfg = get_cfg(args)
    extra = args.checkpoint
    cfg.variant.params.pretrained_weights = cfg.variant.params.pretrained_weights.replace("124999", extra)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)
    model = variant.eval_model

    save_dir = Path(cfg.output_dir) / args.layer
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = DatasetRegistry.get(args.dataset_name)
    data_loader: DataLoader = DataLoader(
        dataset, batch_size=128, shuffle=True, num_workers=1, pin_memory=True
    )

    hook_dict = OrderedDict()

    def get_hook(name):
        """Returns a hook function that saves the output of the layer to the outputs dictionary."""
        def hook(module, input, output):
            spatial_features = output[:, 1:].float()
            N, L, H = spatial_features.shape
            sL = int(math.sqrt(L))
            spatial_features = spatial_features.reshape(N, sL, sL, H).permute(0, 3, 1, 2)
            hook_dict[name] = spatial_features
        return hook

    layer_name = args.layer
    layer = resolve_sequential_module_from_str(model, layer_name)

    # for i in range(100):
    handle = layer.register_forward_hook(get_hook(layer_name))

    analysis_output = analysis_single_layer(
        model=model,
        layer_name=layer_name,
        data_loader=data_loader,
        hook_dict=hook_dict, 
        max_image_num=1000,
        device="cuda"
    )

    handle.remove()

    average_image_dict_average, hook_dict = analysis_output

    rfs = create_plots(
        average_image_dict=average_image_dict_average,
        hook_dict=hook_dict,
        path=save_dir,
        extra=extra + f"_{args.dataset_name}"  # + f"_{i}"
    )

    with open(save_dir.parent / f"receptive_field_sizes_checkpoints_{args.dataset_name}.csv", "a") as f:
        extracted = extract_parts(layer_name)
        f.write(f"{extracted[0]},{extracted[1]},{extra},{rfs[0]}\n")


if __name__ == "__main__":
    main()
