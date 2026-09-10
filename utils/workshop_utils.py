"""Small visualization helpers for the workshop notebook."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_obb_train_samples(dataset: Path, filenames: list[str]) -> None:
    """Show labeled OBB training samples side by side."""
    fig, axes = plt.subplots(1, len(filenames), figsize=(16, 6))

    for axis, filename in zip(axes, filenames):
        image_path = dataset / "images/train" / filename
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]

        label_path = dataset / "labels/train" / f"{image_path.stem}.txt"
        for label in label_path.read_text().splitlines():
            box = np.array(label.split()[1:], dtype=float).reshape(4, 2)
            box = (box * [width, height]).astype(int)
            cv2.polylines(image, [box], True, (0, 165, 255), 2)

        video = filename.split("_")[1].removeprefix("v")
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(f"Training video {video}")
        axis.axis("off")

    plt.tight_layout()
    plt.show()


def show_predictions(results, titles: list[str]) -> None:
    """Show Ultralytics prediction results side by side."""
    fig, axes = plt.subplots(1, len(results), figsize=(16, 6))

    for axis, result, title in zip(axes, results, titles):
        image = result.plot(labels=True, conf=True, line_width=2)
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(title)
        axis.axis("off")

    plt.tight_layout()
    plt.show()
