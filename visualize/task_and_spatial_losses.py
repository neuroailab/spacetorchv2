import json
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_loss_file", type=str)
    parser.add_argument("--spatial_loss_file", type=str)
    parser.add_argument("--output_dir", type=str)
    return parser


args = get_parser().parse_args()


def main():
    task_loss = []
    block_values = {f"blocks.{i}": [] for i in range(12)}

    # load spatial_loss_file
    spatial_loss_file_path = args.spatial_loss_file
    with open(spatial_loss_file_path, "r") as file:
        for line in file:
            data = json.loads(line.strip())
            for key in block_values.keys():
                if key in data:
                    block_values[key].append(data[key])

    # load task_loss_file
    task_loss_file_path = args.task_loss_file
    with open(task_loss_file_path, "r") as file:
        for line in file:
            data = json.loads(line.strip())
            task_loss.append(data["total_loss"])

    # Plot results
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig)
    colors = sns.color_palette('rainbow', n_colors=12)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(task_loss, lw=2, color='black')
    ax1.set_title("Task Loss")

    ax2 = fig.add_subplot(gs[0, 1:])
    for i, (block, values) in enumerate(block_values.items()):
        ax2.plot(values[::8], label=block, lw=2, color=colors[i])
    ax2.set_title("Spatial Loss")

    ax1.set_xlabel("Iteration")
    ax2.set_xlabel("Iteration")
    ax2.legend()

    ax1.grid(True)
    ax2.grid(True)

    save_dir = Path(args.output_dir) / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_dir / "losses.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    """
    Example usage:
    python3 visualize/task_and_spatial_losses.py --task_loss_file checkpoints/vitb14_dinov2_imagenet_noswapopt_lwx1/training_metrics.json --task_loss_file checkpoints/vitb14_dinov2_imagenet_noswapopt/spatial_losses.json positions --swapopt_positions_dir --output_dir=checkpoints/vitb14_dinov2_imagenet_noswapopt_lwx1
    """
    main()