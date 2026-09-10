"""Plotting helpers for the workshop notebook."""

import matplotlib.pyplot as plt


def plot_backend_benchmark(results) -> None:
    """Plot inference throughput measured in the notebook."""
    labels = results.apply(
        lambda row: f"{row['Backend']}\n{row['Precision']} · batch {row['Batch']}", axis=1
    )
    ax = results.assign(Configuration=labels).plot.bar(
        x="Configuration", y="Throughput (images/s)", legend=False,
        color="#777777", figsize=(11, 4.5),
    )
    colors = ["#666666", "#AAAAAA", "#666666", "#AAAAAA", "#B5121B", "#ED1C24"]
    for bar, color in zip(ax.patches, colors):
        bar.set_color(color)
    ax.axvline(1.5, color="#DDDDDD", linewidth=1)
    ax.axvline(3.5, color="#DDDDDD", linewidth=1)
    ax.set_ylabel("Throughput (images/s) — higher is better")
    ax.set_title("YOLO26n-OBB inference throughput on W7900D")
    ax.bar_label(ax.containers[0], fmt="%.1f")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    plt.close(ax.figure)
