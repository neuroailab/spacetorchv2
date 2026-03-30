import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant, OUTPUT_DIMS_FOR_224_INPUTS
from spacetorch.osa import new_init, Optimize
from spacetorch.osa.augmentations import Clip, Tile, Jitter, RepeatBatch, ColorJitter, GaussianNoise
from spacetorch.osa.hooks import ViTBlockHook
from spacetorch.osa.loss import LossArray, TotalVariation, ViTEnsFeatHook


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--layer", type=str)
    return parser


args = get_parser().parse_args()

IMAGE_SIZE = 224


def main():
    args = get_parser().parse_args()
    cfg = get_cfg(args)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)

    model = variant.eval_model
    print([m for m in model.named_modules()])

    _, axs = plt.subplots(1, 10, figsize=(10, 1))

    save_dir = Path(cfg.output_dir) / args.layer
    save_dir.mkdir(parents=True, exist_ok=True)
        
    max_features = OUTPUT_DIMS_FOR_224_INPUTS[cfg.variant.architecture][args.layer][0]
    random_features = np.random.choice(np.arange(1, max_features), size=10)

    for fi, feature in enumerate(random_features):
        print(f"\tfeature {feature}")
    
        loss = LossArray()
        layer_name = args.layer
        if "attn_output" in args.layer:
            layer_name = args.layer[:-12]
        loss += ViTEnsFeatHook(ViTBlockHook(model, block_name=layer_name, attn_output="attn_output" in args.layer), key='high', feat=feature, coefficient=1)
        loss += TotalVariation(2, IMAGE_SIZE, coefficient=0.00005)

        pre, post = torch.nn.Sequential(RepeatBatch(8), ColorJitter(8, shuffle_every=True),
                                        GaussianNoise(8, True, 0.5, 400), Tile(IMAGE_SIZE // IMAGE_SIZE), Jitter()), Clip()

        image = new_init(IMAGE_SIZE, 1)
        visualizer = Optimize(loss, pre, post, lr=0.1, steps=400)
        image.data = visualizer(image)

        img = image.squeeze(0)          # (3, 224, 224)
        img = img.permute(1, 2, 0)      # (224, 224, 3)
        axs[fi].imshow(img.cpu().detach().numpy())
        axs[fi].axis("off")
        axs[fi].set_axis_off()
    
    plt.savefig(save_dir / ("optimal_stimuli.png"), dpi=300, transparent=True, bbox_inches='tight')


if __name__ == '__main__':
    main()
