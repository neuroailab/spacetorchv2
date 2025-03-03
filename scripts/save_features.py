import argparse
import torch
from pathlib import Path

from spacetorch.datasets import DatasetRegistry
from spacetorch.feature_saver import FeatureSaver
from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--max_images", type=int)
    parser.add_argument("--batch_size", type=int, default=128)
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


def log(to_print: str):
    print(f"\nLOG: {to_print}")


def main():
    args = get_parser().parse_args()
    cfg = get_cfg(args)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)

    model = variant.eval_model
    layer_names = variant.get_model_layernames(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    dataset = DatasetRegistry.get(args.dataset_name)

    log("Constructing feature saver")
    output_dir = Path(cfg.output_dir)
    save_path: Path = output_dir / "features" / f"{args.dataset_name}.h5"
    feature_saver: FeatureSaver = FeatureSaver(
        model, layer_names, dataset, save_path, max_images=args.max_images
    )

    log("Extracting features")
    feature_saver.compute_features(batch_size=args.batch_size)

    log("Saving features")
    feature_saver.save_features()

    log("All done!")


if __name__ == "__main__":
    """
    Example usage:
    python3 scripts/save_features.py --config configs/vitb14_dinov2_imagenet.yaml --dataset_name SineGrating2019 output_dir=checkpoints/vitb14_dinov2_imagenet
    """
    main()
