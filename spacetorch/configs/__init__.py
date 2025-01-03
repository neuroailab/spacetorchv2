import pathlib

from omegaconf import OmegaConf


def load_cfg(cfg_name: str):
    cfg_fname = cfg_name + ".yaml"
    return OmegaConf.load(pathlib.Path(__file__).parent.resolve() / cfg_fname)


spacetorch_default_cfg = load_cfg("default")


def get_cfg(args):
    default_cfg = OmegaConf.create(spacetorch_default_cfg)
    cfg = OmegaConf.load(args.config)
    try:
        if args.opts:
            cfg = OmegaConf.merge(default_cfg, cfg, OmegaConf.from_cli(args.opts))
    except AttributeError:
        print('No opts to merge with the config')
    return cfg
