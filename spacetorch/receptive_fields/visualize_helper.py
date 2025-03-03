from matplotlib import pyplot as plt
import torch
import numpy as np


def visualize_weighted_distances(distance_matrix, vmin=0, vmax=100, path=".", extra=""):
    """
    Visualize a weighted distance matrix using a heatmap.

    Parameters:
    - distance_matrix: The matrix containing weighted distances to be visualized.
    - title: Title for the plot.
    - ax: Matplotlib axis object for plotting. Useful for subplots.
    - show: Whether to show the plot. Set to False when plotting subplots.
    """
    plt.figure()
    plt.imshow(distance_matrix, interpolation="nearest", vmin=vmin, vmax=vmax, cmap="binary")
    plt.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    plt.gca().spines["top"].set_visible(True)
    plt.gca().spines["bottom"].set_visible(True)
    plt.gca().spines["right"].set_visible(True)
    plt.gca().spines["left"].set_visible(True)
    plt.savefig(path / f"rf_{extra}.png", bbox_inches="tight", dpi=300)
    plt.clf()
    plt.close()


def create_plots(average_image_dict, hook_dict, path, extra="", visualize=True):
    receptive_fields = []
    for _, (k, _) in enumerate(hook_dict.items()):

        key = k

        distance_matrix = torch.tensor(np.array(average_image_dict[key]))

        distance_matrix = distance_matrix.mean(dim=0).mean(dim=0)
        
        m = distance_matrix.max()
        if m > 0:
            receptive_field = (distance_matrix / m).mean()
            receptive_fields.append(receptive_field.item())
            
            distance_matrix = distance_matrix / distance_matrix.sum()
            
            if visualize:
                visualize_weighted_distances(distance_matrix, vmin=0, vmax=distance_matrix.max(), path=path, extra=extra)
        else:
            receptive_fields.append(0)
        
    return receptive_fields


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
