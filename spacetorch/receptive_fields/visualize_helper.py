from matplotlib import pyplot as plt
import torch
import numpy as np
import seaborn as sns


def visualize_weighted_distances(distance_matrix, vmin=0, vmax=1, path=".", extra=""):
    """
    Visualize a weighted distance matrix using a heatmap.

    Parameters:
    - distance_matrix: The matrix containing weighted distances to be visualized.
    - title: Title for the plot.
    - ax: Matplotlib axis object for plotting. Useful for subplots.
    - show: Whether to show the plot. Set to False when plotting subplots.
    """
    plt.figure()
    plt.imshow(distance_matrix, vmin=0, vmax=1, cmap="binary")
    plt.axis("off")
    plt.savefig(path / f"rf_{extra}.png", bbox_inches="tight", dpi=300)
    plt.close()

    plt.figure()
    H, W = distance_matrix.shape
    yy, xx = np.mgrid[0:H, 0:W]
    x = xx.ravel()
    y = yy.ravel()
    weights = np.abs(distance_matrix).ravel()
    sns.kdeplot(
        x=x,
        y=y,
        weights=weights,
        fill=False,
        cmap="binary",
    )
    plt.ylim(0, 224)
    plt.xlim(0, 224)
    plt.axis("off")
    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal")
    plt.savefig(path / f"rf_kde_{extra}.svg", format="svg", transparent=True, bbox_inches="tight", dpi=300)
    plt.close()


def radial_effective_rf(grad_map, center=None, n_bins=None):
    """
    Compute the radial effective receptive field as in "Are ViTs as global as we think?"

    Parameters
    ----------
    grad_map : np.ndarray, shape (H, W)
        Gradient map of the output with respect to the input pixels.
    center : tuple (cy, cx), optional
        Center coordinates of the receptive field. Defaults to image center.
    n_bins : int, optional
        Number of radial bins for integration. Defaults to max(H, W)//2
    normalize : bool
        If True, normalize by the maximum gradient value.

    Returns
    -------
    scalar : float
        Effective receptive field size (larger => more global).
    radial_vals : np.ndarray
        Mean gradient values per radial bin.
    """
    H, W = grad_map.shape
    if center is None:
        cy, cx = (H / 2, W / 2)
    else:
        cy, cx = center

    if n_bins is None:
        n_bins = max(H, W) // 2

    # compute distance of each pixel from the center
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    r = np.sqrt((y - cy)**2 + (x - cx)**2)

    r_max = r.max()
    bin_edges = np.linspace(0, r_max, n_bins + 1)

    grad = grad_map.copy()

    radial_vals = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i+1])
        if np.any(mask):
            radial_vals[i] = grad[mask].mean()
        else:
            radial_vals[i] = 0.0

    # integrate over radius
    dr = bin_edges[1] - bin_edges[0]
    effective_rf = (radial_vals * dr).sum() / r_max

    return effective_rf, radial_vals, r_max


def gaussian_rf_fit_quality(rf_map, center=None):
    """
    Fit a Gaussian to a 2D RF map via moment matching and compute variance explained.

    Parameters
    ----------
    rf_map : np.ndarray, shape (H, W)
        Receptive field or gradient-based sensitivity map.
    center : tuple (cy, cx), optional
        If None, uses center of mass.

    Returns
    -------
    sigma : float
        Gaussian-equivalent pRF sigma (pixels).
    r2 : float
        Variance explained by the Gaussian fit.
    gaussian_fit : np.ndarray
        Best-fit Gaussian RF (same shape as rf_map).
    """

    # rectify
    M = np.abs(rf_map)
    H, W = M.shape
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    # normalize for shape comparison
    P = M / (M.sum() + 1e-8)

    # center
    if center is None:
        cy = (P * y).sum()
        cx = (P * x).sum()
    else:
        cy, cx = center

    # second moment → sigma
    r2 = (y - cy)**2 + (x - cx)**2
    sigma = np.sqrt((P * r2).sum() / 2.0)

    # best-fit isotropic Gaussian (unnormalized shape)
    G = np.exp(-r2 / (2 * sigma**2))
    G /= G.sum() + 1e-8

    # variance explained (R^2)
    M_flat = P.ravel()
    G_flat = G.ravel()

    ss_res = np.sum((M_flat - G_flat)**2)
    ss_tot = np.sum((M_flat - M_flat.mean())**2)

    r2_explained = 1.0 - ss_res / (ss_tot + 1e-8)

    return sigma, r2_explained, G


def create_plots(distance_matrix, point_dict, path, extra="", visualize=True):
    res = {
        "eff_rf_radial_avg": np.nan,
        "radial_vals": np.nan,
        "r_max": np.nan,
        "eff_rf_avg": np.nan,
        "sigma": np.nan,
        "r2": np.nan,
        "shape": point_dict["shape"],
        "points": point_dict["points"],
    }

    distance_matrix = np.concatenate(distance_matrix, axis=0)
    distance_matrix = distance_matrix.mean(axis=(0, 1))

    sigma, r2, _ = gaussian_rf_fit_quality(distance_matrix)
    res["sigma"] = sigma
    res["r2"] = r2
    print("r^2:", r2)
    
    m = distance_matrix.max()
    
    if m > 0:
        distance_matrix = distance_matrix / m

        eff_rf_radial_avg, radial_vals, r_max = radial_effective_rf(distance_matrix)
        
        eff_rf_avg = np.mean(distance_matrix)
        
        res["eff_rf_radial_avg"] = eff_rf_radial_avg
        res["radial_vals"] = radial_vals
        res["r_max"] = r_max
        res["eff_rf_avg"] = eff_rf_avg

        if visualize:
            visualize_weighted_distances(distance_matrix, vmin=0, vmax=1, path=path, extra=extra)
        
    return res


# Function to convert a tensor to a numpy image
def tensor_to_image(tensor):
    # The normalization mean and std are for the pretrained models
    # We need to reverse the normalization before plotting
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Convert to numpy and reshape
    image = tensor.permute(1, 2, 0).cpu().numpy()
    
    # Reverse the normalization
    image = (image * std + mean) * 255
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def investigate_tensor(tensor):
    print("tensor shape:", tensor.shape)
    print("max min", tensor.max(), tensor.min())
