import argparse
import timm
import torch
import math
from pathlib import Path
from collections import OrderedDict
from torch.utils.data import DataLoader
from spacetorch.receptive_fields.rf_helper import *
from spacetorch.receptive_fields.visualize_helper import *
from spacetorch.datasets import DatasetRegistry
from spacetorch.utils.torch_utils import resolve_sequential_module_from_str


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timm_model", type=str)
    return parser


args = get_parser().parse_args()


def main():
    for is_pretrained in [True, False]:
        model = timm.create_model(args.timm_model, pretrained=False)
        if is_pretrained:
            checkpoint = torch.load("checkpoints/other_models/vit-b-300ep.pth.tar")
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % "head"):
                    # remove prefix
                    state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            model.load_state_dict(state_dict, strict=False)
        print(model)
        extra = "optimized" if is_pretrained else "unoptimized"
        model_name = args.timm_model

        for dataset_name in ["SineGrating2019", "ImageNet"]:
            dataset = DatasetRegistry.get(dataset_name)
            data_loader: DataLoader = DataLoader(
                dataset, batch_size=128, shuffle=True, num_workers=1, pin_memory=True
            )

            for layer in [f"blocks.{i}.attn" for i in range(12)]:
                save_dir = Path("checkpoints/other_models")
                save_dir.mkdir(parents=True, exist_ok=True)

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

                layer_name = layer
                layer = resolve_sequential_module_from_str(model, layer_name)

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
                    extra=extra + f"_{dataset_name}",
                    visualize=False
                )

                with open(save_dir / f"receptive_field_sizes.csv", "a") as f:
                    f.write(f"{dataset_name},{model_name},{extra},{layer_name},{rfs[0]}\n")


if __name__ == "__main__":
    main()
