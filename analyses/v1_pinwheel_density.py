import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

from spacetorch.configs import get_cfg
from spacetorch.maps.pinwheel_detector import PinwheelDetector
from spacetorch.variants import get_variant
from spacetorch.variants.positions import get_positions
from spacetorch.maps import TissueMap
from spacetorch.maps.sine_tissue import get_sine_tissue


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--layer", type=str)
    parser.add_argument("--e", type=int)
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


def fit_gaussian(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def get_pinwheel_density(tissue: TissueMap):
    tissue.reset_unit_mask()
    lims = [0, 100]
    tissue.set_mask_by_pct_limits([lims, lims])

    try:

        pindet = PinwheelDetector(tissue, width=1.5, stride=0.15, verbose=True)

        zmap = np.exp(2j * np.deg2rad(pindet.smoothed))

        fft_map = np.fft.fft2(zmap)
        fft_shifted = np.fft.fftshift(fft_map)
        power_spectrum = np.abs(fft_shifted) ** 2

        H, W = power_spectrum.shape
        freq_y = np.fft.fftshift(np.fft.fftfreq(H))
        freq_x = np.fft.fftshift(np.fft.fftfreq(W))
        fx, fy = np.meshgrid(freq_x, freq_y)
        radial_freq = np.sqrt(fx**2 + fy**2)

        r = radial_freq.ravel()
        p = power_spectrum.ravel()

        # Remove DC component (radial_freq == 0)
        nonzero = r > 0
        x = r[nonzero]
        y = p[nonzero]

        # Fit Gaussian
        p0 = [y.max(), x[np.argmax(y)], 0.1]
        popt, _ = curve_fit(fit_gaussian, x, y, p0=p0)
        _, peak_freq, _ = popt

        column_spacing_in_pixels = 1 / peak_freq
        total_px = pindet.smoothed.shape[0]
        total_mm = np.ptp(tissue._positions)
        px_per_mm = total_px / total_mm
        column_spacing_in_mm = column_spacing_in_pixels / px_per_mm

        tissue.reset_unit_mask()
        pindet = PinwheelDetector(tissue, width=1.5, stride=0.45, verbose=False)

        pos, neg = pindet.count_pinwheels()
        total = pos + neg

        edge_size = np.ptp(tissue._positions)
        num_hcol = edge_size / column_spacing_in_mm

        res = {
            "column_spacing_in_pixels": column_spacing_in_pixels,
            "column_spacing_in_mm": column_spacing_in_mm,
            "pinwheel_density": total / (num_hcol**2),
        }

    except Exception as e:
        print(f"Error computing pinwheel density: {e}")
        res = {
            "column_spacing_in_pixels": np.nan,
            "column_spacing_in_mm": np.nan,
            "pinwheel_density": np.nan,
        }

    return res


def main():
    cfg = get_cfg(args)
    variant = get_variant(cfg.variant.name)
    variant.set_eval_cfg(cfg)
    variant.set_eval_model(cfg)

    model = variant.eval_model
    is_tdann = "tdann" in cfg.name and not "tdann_logpolar" in cfg.name
    positions = get_positions(cfg, rescale=is_tdann)[args.layer]

    save_dir = Path(cfg.output_dir) / args.layer
    save_dir.mkdir(parents=True, exist_ok=True)

    # if (save_dir / "pinwheel_density.npz").exists():
    #     print(f"Found existing results in {save_dir}, skipping...")
    #     return

    v1_tissue = get_sine_tissue(
        cfg.name,
        model,
        positions,
        layer=args.layer,
        output_dir=save_dir,
        smooth_orientation_tuning_curves=False,
        skip_cache=True,
    )

    density = get_pinwheel_density(v1_tissue)
    print(density)
    np.savez(save_dir / "pinwheel_density.npz", **density)

    plt.figure(figsize=(1.5, 1.5))
    b = plt.bar(["Model"], [density["pinwheel_density"]], color="black")
    for rect in b:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.3f}', ha='center', va='bottom', fontsize=7)
    plt.ylabel(("Pinwheels /" "\n" r"Column Spacing$^2$"))
    plt.yticks([0, 1, 2, 3, 4])
    plt.savefig(save_dir / "v1_pinwheel_density.png", bbox_inches="tight", dpi=300)
    

if __name__ == "__main__":
    main()
