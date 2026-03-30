import argparse

from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("SpaceTorch Evaluation", add_help=add_help)
    parser.add_argument("--config", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--local-rank", default=0, type=int, help="Variable for distributed computing.")
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


def main(args):
    cfg = get_cfg(args)

    variant = get_variant(cfg.variant.name)
    variant.kinetics_init(cfg)

    variant.start_kinetics_protocol()


if __name__ == "__main__":
    """
    Example usage:
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    
    python -m torch.distributed.launch --nproc_per_node=8 spacetorch/run/eval.py --config spacetorch/configs/analysis_configs/vitb14_dinov2_imagenet_swapopt_lwx1.yaml output_dir=checkpoints/vitb14_dinov2_imagenet_swapopt_lwx1/eval/training_124999/
    """
    args = get_args_parser(add_help=True).parse_args()
    main(args)
