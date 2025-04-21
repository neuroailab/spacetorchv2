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
    parser.add_argument("--layers", nargs="+")
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
    L = len(args.layers)

    save_dir = Path(cfg.output_dir) / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    if L == 1:
        _, axs = plt.subplots(1, 10, figsize=(10, 1))
    else:
        _, axs = plt.subplots(10, L, figsize=(L, 10))

    for li, layer in enumerate(args.layers):
        print(f"Computing most exciting stimuli for layer {layer}")
        
        max_features = OUTPUT_DIMS_FOR_224_INPUTS[cfg.variant.name][layer][0]
        random_features = np.random.choice(np.arange(1, max_features), size=10)

        for fi, feature in enumerate(random_features):
            print(f"\tfeature {feature}")
        
            loss = LossArray()
            loss += ViTEnsFeatHook(ViTBlockHook(model, block_name=layer), key='high', feat=feature, coefficient=1)
            loss += TotalVariation(2, IMAGE_SIZE, coefficient=0.00005)

            pre, post = torch.nn.Sequential(RepeatBatch(8), ColorJitter(8, shuffle_every=True),
                                            GaussianNoise(8, True, 0.5, 400), Tile(IMAGE_SIZE // IMAGE_SIZE), Jitter()), Clip()

            image = new_init(IMAGE_SIZE, 1)
            visualizer = Optimize(loss, pre, post, lr=0.1, steps=400)
            image.data = visualizer(image)

            if L == 1:
                axs[fi].imshow(image)
                axs[fi].set_axis_off()
            else:
                axs[fi][li].imshow(image)
                axs[fi][li].set_axis_off()
    
    plt.savefig(save_dir / ("optimal_stimuli.png"), dpi=300, transparent=True, bbox_inches='tight')


if __name__ == '__main__':
    main()
