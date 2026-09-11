"""Plotting helpers for the workshop notebook."""

from io import BytesIO

import matplotlib.pyplot as plt
from IPython.display import display


def plot_backend_benchmark(results) -> None:
    """Plot inference throughput measured in the notebook."""
    throughput = results.pivot(index="Batch", columns="Backend", values="Throughput (images/s)")
    ax = throughput[["PyTorch", "MIGraphX"]].plot.bar(
        color=["#666666", "#ED1C24"], figsize=(9.5, 4.5), width=0.72,
    )
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.0f", padding=3)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Images/s — higher is better")
    ax.set_title("YOLO26n-OBB FP16 throughput on W7900D")
    ax.legend(title=None, frameon=False)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.margins(y=0.12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    image = BytesIO()
    ax.figure.savefig(image, format="png", dpi=100, bbox_inches="tight")
    plt.close(ax.figure)
    display({"image/png": image.getvalue()}, raw=True)
