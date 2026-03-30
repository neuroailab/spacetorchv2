import torch
import torch.nn as nn


class BasicHook:
    def __init__(self, module: nn.Module):
        self.hook = module.register_forward_hook(self.base_hook_fn)
        self.activations = None

    def close(self):
        self.hook.remove()

    def base_hook_fn(self, model: nn.Module, input_t: torch.tensor, output_t: torch.tensor):
        x = input_t
        x = x[0][0] if isinstance(x[0], tuple) else x[0]
        return self.hook_fn(model, x)

    def hook_fn(self, model: nn.Module, x: torch.tensor):
        raise NotImplementedError


class ViTHook(BasicHook):
    def __init__(self, module: nn.Module, return_output: bool, name: str, attn_output: bool = False):
        super().__init__(module)
        self.mode = return_output
        self.name = name
        self.attn_output = attn_output

    def base_hook_fn(self, model: nn.Module, input_t: torch.tensor, output_t: torch.tensor):
        if self.attn_output:
            x = model.attn_output
        else:
            x = input_t if not self.mode else output_t
        x = x[0] if isinstance(x, tuple) else x
        return self.hook_fn(model, x)

    def hook_fn(self, model: nn.Module, x: torch.tensor):
        self.activations = x


class ViTBlockHook(nn.Module):
    def __init__(self, classifier: nn.Module, block_name: str = "blocks.1", attn_output: bool = False):
        super().__init__()
        self.cl = classifier
        module = dict(self.cl.named_modules()).get(block_name, None)
        if module is None:
            raise ValueError(f"Block with name {block_name} not found in the classifier.")
        hook = ViTHook(module, True, 'high', attn_output=attn_output)
        self.high = hook

    def forward(self, x: torch.tensor) -> tuple[dict, torch.tensor]:
        out = self.cl(x)
        options = [self.high]
        options = [l.activations if l is not None else None for l in options]
        names = ['high']
        return {n: o for n, o in zip(names, options) if o is not None}, out