import argparse
import numpy as np
import torch
import torch.nn as nn
import pickle as pkl
import matplotlib.pyplot as plt
from pathlib import Path

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget

from spacetorch.configs import get_cfg
from spacetorch.variants import get_variant
from spacetorch.datasets import DatasetRegistry
from spacetorch.feature_extractor import get_features_from_layer


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--layer", type=str)
    parser.add_argument("--probe", type=str, choices=["linear", "mlp"], default="linear")
    parser.add_argument("--n_gradcam_images", type=int, default=20)
    parser.add_argument("--gradcam_seed", type=int, default=0)
    return parser


args = get_parser().parse_args()


class ProbedModel(nn.Module):
    """
    Wraps a backbone + a TRAINED sklearn probe (LogisticRegression or
    MLPClassifier, fit on flattened whole-image features from `hook_layer`)
    into a single differentiable module whose forward pass reproduces the
    probe's decision score end-to-end, given only a raw image. This lets
    GradCAM hook into the backbone and backprop from the actual decision
    value the probe uses -- not an ImageNet class logit (there is none
    here) and not an approximate stand-in direction.
    """

    def __init__(self, backbone, hook_layer, sk_probe, probe_type):
        super().__init__()
        self.backbone = backbone
        self.hook_layer = hook_layer
        self.probe_type = probe_type

        self._activation = {}
        self.hook_layer.register_forward_hook(self._hook_fn)

        if probe_type == "linear":
            # binary LogisticRegression stores a single weight vector;
            # its output IS the logit for the positive class, so no
            # multi-class index or zero-row workaround is needed here
            n_features = sk_probe.coef_.shape[1]
            self.linear = nn.Linear(n_features, 1)
            self.linear.weight.data = torch.tensor(sk_probe.coef_, dtype=torch.float32)
            self.linear.bias.data = torch.tensor(sk_probe.intercept_, dtype=torch.float32)
        elif probe_type == "mlp":
            # port sklearn's MLPClassifier weights (coefs_, intercepts_)
            # into an equivalent stack of nn.Linear + ReLU layers. For a
            # binary problem, sklearn's final layer has a single output
            # neuron (logistic activation), matching the linear case above.
            layers = []
            n_layers = len(sk_probe.coefs_)
            for i, (W, b) in enumerate(zip(sk_probe.coefs_, sk_probe.intercepts_)):
                in_dim, out_dim = W.shape
                lin = nn.Linear(in_dim, out_dim)
                lin.weight.data = torch.tensor(W.T, dtype=torch.float32)
                lin.bias.data = torch.tensor(b, dtype=torch.float32)
                layers.append(lin)
                if i < n_layers - 1:
                    layers.append(nn.ReLU())
            self.mlp = nn.Sequential(*layers)

    def _hook_fn(self, module, inp, out):
        self._activation["feat"] = out

    def forward(self, x):
        _ = self.backbone(x)
        feat = self._activation["feat"]
        feat_flat = feat.reshape(feat.shape[0], -1)

        if self.probe_type == "linear":
            return self.linear(feat_flat)  # (B, 1) logit
        else:
            return self.mlp(feat_flat)  # (B, 1) logit


def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def run_gradcam_visualization(model, hook_layer, sk_probe, probe_type,
                               val_inputs, n_images, seed, save_dir):
    probed_model = ProbedModel(model, hook_layer, sk_probe, probe_type).cuda().eval()
    cam = GradCAM(model=probed_model, target_layers=[hook_layer], reshape_transform=reshape_transform)

    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(val_inputs), size=n_images, replace=False)

    fig, axs = plt.subplots(1, n_images, figsize=(3 * n_images, 3))
    if n_images == 1:
        axs = [axs]

    for ax, idx in zip(axs, chosen_idx):
        img_tensor = torch.tensor(val_inputs[idx], dtype=torch.float32).unsqueeze(0)

        targets = [RawScoresOutputTarget()]  # backprop from the single decision logit directly
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]
        ax.imshow(grayscale_cam, cmap="magma_r")
        ax.axis("off")

    plt.subplots_adjust(wspace=0.02, hspace=0)
    save_path = save_dir / "gradcam_examples.svg"
    plt.savefig(save_path, format="svg", dpi=300, bbox_inches="tight", pad_inches=0)
    print(f"Saved GradCAM visualization to {save_path}")
    plt.close(fig)

    save_path_images = Path("/ccn2/u/ynshah/spacetorchv2/paper/discrimination_images.svg")

    fig, axs = plt.subplots(1, 20, figsize=(20, 1))
    for i, idx in enumerate(chosen_idx):
        axs[i].imshow(val_inputs[idx].transpose(1, 2, 0))
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].axis("off")
    plt.savefig(
        save_path_images,
        bbox_inches="tight", format="svg", transparent=True, dpi=300,
    )
    plt.close(fig)


def main():
    cfg = get_cfg(args)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)
    model = variant.eval_model

    save_dir = Path(cfg.output_dir) / args.layer / "discrimination"

    # --- load the ALREADY-TRAINED probe from disk, rather than retraining ---
    probe_path = save_dir / f"{args.probe}_probe.pkl"
    if not probe_path.exists():
        raise FileNotFoundError(
            f"No trained probe found at {probe_path}. Run the training script "
            f"for --config {args.config} --layer {args.layer} --probe {args.probe} first."
        )
    with open(probe_path, "rb") as f:
        probe = pkl.load(f)
    print(f"Loaded trained probe from {probe_path}")

    # --- extract validation inputs only; no features/labels needed here
    #     since we're not fitting or scoring anything, just visualizing ---
    val_dataset = DatasetRegistry.get("Discrimination_val")
    _, val_inputs, _ = get_features_from_layer(
        model=model,
        dataset=val_dataset,
        verbose=True,
        batch_size=128,
        model_layer_strings=[args.layer],
        return_inputs_and_labels=True,
    )
    val_inputs = np.asarray(val_inputs)

    # --- GradCAM visualization using the loaded probe ---
    hook_layer = model.get_submodule(args.layer)
    run_gradcam_visualization(
        model, hook_layer, probe, args.probe,
        val_inputs, args.n_gradcam_images, args.gradcam_seed, save_dir,
    )


if __name__ == "__main__":
    main()
