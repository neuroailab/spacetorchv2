import importlib


def extract_module(dotpath):
    return dotpath.rsplit(".", 1)[0]


def extract_fn(dotpath):
    return dotpath.rsplit(".", 1)[-1]


def get_fn(dotpath):
    module_name = extract_module(dotpath)
    fn_name = extract_fn(dotpath)
    variant_cfg_setup_module = importlib.import_module(module_name)
    return getattr(variant_cfg_setup_module, fn_name)
