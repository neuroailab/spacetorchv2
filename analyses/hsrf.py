import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from spacetorch.receptive_fields.rf_helper import *
from spacetorch.receptive_fields.visualize_helper import *
from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant
from spacetorch.maps.expanding_sine_tissue import get_expanding_sine_responses


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

    is_lcnn = ("lcnn" in cfg.name)
    is_llcnn = ("llcnn" in cfg.name)

    responses = get_expanding_sine_responses(
        model,
        layers=[args.layer],
        normalize_to_ringach_firing_rates=("tdann" in cfg.name) or ("resnet" in cfg.name),
        is_lcnn=is_lcnn and not is_llcnn,
        is_llcnn=is_llcnn,
    )

    # get distribution of curves
    curves, _, _, unique_diams = responses.get_response_at_diameter(architecture=cfg.variant.architecture, layer=args.layer)

    np.save(save_dir / "hsrf_curves.npy", curves)

    plt.figure()
    for i in range(len(curves[0])):
        sns.lineplot(x=unique_diams, y=curves[:, i], c="gray", linewidth=0.8)
    sns.lineplot(x=unique_diams, y=curves.mean(axis=1), c="k", linewidth=2)
    plt.xlabel("Stimulus Diameter " r"($^\circ$)")
    plt.ylabel("Response Amplitude (a.u.)")
    plt.savefig(save_dir / "hsrf_curves.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
