"""Small visualization helpers for the workshop notebook."""

from functools import lru_cache
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.preview_widgets import image_slider_panel, side_by_side_previews


def show_obb_train_samples(dataset: str | Path, filenames: list[str]) -> None:
    """Show labeled OBB training samples side by side."""
    dataset = Path(dataset)
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

        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(
            f"AUTH-Sheep video {filename.split('_')[1].removeprefix('v')}\n{filename}"
            if filename.startswith("auth_")
            else f"SheepCounter\n{filename}"
        )
        axis.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close(fig)


@lru_cache(maxsize=64)
def _render_obb_sample(dataset: str, filename: str, canvas_size: tuple[int, int]) -> bytes:
    """Render one labeled sample into a fixed-size JPEG canvas."""
    dataset = Path(dataset)
    image_path = dataset / "images/train" / filename
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]

    for label in (dataset / "labels/train" / f"{image_path.stem}.txt").read_text().splitlines():
        box = np.array(label.split()[1:], dtype=float).reshape(4, 2)
        box = (box * [width, height]).astype(int)
        cv2.polylines(image, [box], True, (0, 165, 255), 2)

    return _encode_preview(image, canvas_size)


def _encode_preview(image: np.ndarray, canvas_size: tuple[int, int]) -> bytes:
    """Fit an image into a fixed canvas and encode it as JPEG."""
    height, width = image.shape[:2]
    canvas_width, canvas_height = canvas_size
    scale = min(canvas_width / width, canvas_height / height)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)))
    canvas = np.full((canvas_height, canvas_width, 3), 32, dtype=np.uint8)
    top = (canvas_height - resized.shape[0]) // 2
    left = (canvas_width - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized

    success, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise RuntimeError("Failed to encode preview image")
    return encoded.tobytes()


def browse_obb_train_samples(dataset: str | Path):
    """Browse AUTH-Sheep and SheepCounter in two fixed preview panels."""
    dataset = Path(dataset)
    auth_samples = sorted(path.name for path in (dataset / "images/train").glob("auth_*.jpg"))
    sheepcounter_samples = sorted(path.name for path in (dataset / "images/train").glob("sc_*.jpg"))

    return side_by_side_previews(
        image_slider_panel(
            "AUTH-Sheep",
            auth_samples,
            "auth_v1_00330.jpg",
            lambda filename: _render_obb_sample(str(dataset), filename, (640, 360)),
        ),
        image_slider_panel(
            "SheepCounter",
            sheepcounter_samples,
            "sc_train_00388.jpg",
            lambda filename: _render_obb_sample(str(dataset), filename, (640, 360)),
        ),
    )


def browse_pretrained_coco_predictions(model):
    """Browse cached COCO sheep predictions on COCO and AUTH-Sheep images."""
    coco_samples = sorted(path.name for path in Path("/datasets/coco-sheep-samples").glob("*.jpg"))
    auth_samples = sorted(path.name for path in Path("/datasets/sheep-datasets/images/train").glob("auth_*.jpg"))

    @lru_cache(maxsize=64)
    def render(image: str) -> tuple[bytes, str]:
        result = model.predict(image, classes=[18], end2end=False, verbose=False)[0]
        preview = _encode_preview(result.plot(labels=True, conf=True, line_width=2), (640, 360))
        return preview, f"{len(result.boxes)} sheep detected"

    return side_by_side_previews(
        image_slider_panel(
            "COCO val2017",
            coco_samples,
            "coco_val2017_000000459437.jpg",
            lambda filename: render(f"/datasets/coco-sheep-samples/{filename}"),
        ),
        image_slider_panel(
            "AUTH-Sheep",
            auth_samples,
            "auth_v1_00330.jpg",
            lambda filename: render(f"/datasets/sheep-datasets/images/train/{filename}"),
        ),
    )


def browse_finetuning_comparison(coco_model, tuned_model):
    """Compare COCO HBB and fine-tuned OBB predictions on the same test image."""
    samples = sorted(path.name for path in Path("/datasets/sheep-datasets/images/test").glob("*.jpg"))
    sheep_id = next(index for index, name in coco_model.names.items() if name == "sheep")

    @lru_cache(maxsize=64)
    def render(filename: str):
        image = f"/datasets/sheep-datasets/images/test/{filename}"
        before = coco_model.predict(
            image, classes=[sheep_id], imgsz=640, quantize="fp16", end2end=False, verbose=False
        )[0]
        after = tuned_model.predict(
            image, imgsz=640, quantize="fp16", end2end=False, verbose=False
        )[0]
        return (
            (_encode_preview(before.plot(labels=False, conf=False, line_width=2), (640, 360)),
             f"{len(before.boxes)} sheep detected"),
            (_encode_preview(after.plot(labels=False, conf=False, line_width=2), (640, 360)),
             f"{len(after.obb)} sheep detected"),
        )

    return side_by_side_previews(
        image_slider_panel("COCO HBB", samples, "auth_v3_00330.jpg", lambda filename: render(filename)[0]),
        image_slider_panel("Fine-tuned Sheep OBB", samples, "auth_v3_00330.jpg", lambda filename: render(filename)[1]),
        linked=True,
    )


def show_dataset_tree(dataset: str | Path) -> None:
    """Show the training-ready image and label layout with file counts."""
    dataset = Path(dataset)
    splits = ("train", "sheepcounter_valid", "test")

    print(f"{dataset.name}/")
    for folder_index, folder in enumerate(("images", "labels")):
        last_folder = folder_index == 1
        print(f"{'└──' if last_folder else '├──'} {folder}/")
        for split_index, split in enumerate(splits):
            count = sum(path.is_file() for path in (dataset / folder / split).iterdir())
            stem = "    " if last_folder else "│   "
            branch = "└──" if split_index == len(splits) - 1 else "├──"
            print(f"{stem}{branch} {split}/  ({count:,} files)")


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
