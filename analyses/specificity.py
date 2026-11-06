import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from spacetorch.receptive_fields.rf_helper import *
from spacetorch.receptive_fields.visualize_helper import *
from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant
from spacetorch.maps.expanding_sine_tissue import get_expanding_sine_responses
from spacetorch.variants.positions import get_positions
from spacetorch.colormaps import nauhaus_colormaps, nauhaus_raw_colormaps


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
        normalize_to_ringach_firing_rates=False,
        is_lcnn=is_lcnn and not is_llcnn,
        is_llcnn=is_llcnn,
    )

    cv, pref_angles, modulation = responses.get_peripheral_tuning_to_central_grating()

    np.savez(save_dir / "axial_specificity.npz", **{"cv": cv, "pref_angles": pref_angles, "modulation": modulation})


if __name__ == "__main__":
    main()
