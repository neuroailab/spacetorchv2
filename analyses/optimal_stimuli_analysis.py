import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from lucent.modelzoo import util
from lucent.optvis import render, param, objectives, transform
from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant, OUTPUT_DIMS_FOR_224_INPUTS


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--layer", type=str)
    return parser


args = get_parser().parse_args()


def main():
    args = get_parser().parse_args()
    cfg = get_cfg(args)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)

    model = variant.eval_model
    print(util.get_model_layers(model))

    fig, axs = plt.subplots(1, 10, figsize=(10, 1))

    save_dir = Path(cfg.output_dir) / args.layer
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Computing most exciting stimuli for layer {args.layer}")
    
    max_features = OUTPUT_DIMS_FOR_224_INPUTS[cfg.variant.architecture][args.layer][0]
    random_features = np.random.choice(np.arange(1, max_features), size=10)
    
    batch_param_f = lambda: param.image(
        w=224,
        batch=len(random_features),
        fft=True,
        decorrelate=True
    )
    obj = sum([objectives.channel(args.layer.replace(".", "_"), n, b) for b, n in enumerate(random_features)])
    transforms = [
        transform.pad(12, mode="constant", constant_value=0.5),
        # transform.jitter(8),
        transform.random_scale([1 + (i - 5) / 50.0 for i in range(11)]),
        transform.random_rotate(list(range(-10, 11)) + 5 * [0]),
        # transform.jitter(4),
    ]

    optimized_images = render.render_vis(model, obj, batch_param_f, transforms=transforms, show_image=False)

    for i, img in enumerate(optimized_images[0]):
        axs[i].imshow(img)
        axs[i].set_axis_off()
    
    plt.savefig(save_dir / ("optimal_stimuli.png"), dpi=300, transparent=True, bbox_inches='tight')


if __name__ == "__main__":
    """
    Example usage:
    python3 spacetorch/scripts/optimal_stimuli_analysis.py --config spacetorch/configs/analysis_configs/tdann.yaml --layer blocks.1
    """
    main()
