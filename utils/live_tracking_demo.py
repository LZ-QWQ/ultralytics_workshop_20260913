#!/usr/bin/env python3
"""A live Jupyter widget for frames produced inside a notebook loop."""

import time

import cv2
import numpy as np
from IPython.display import display
from ipywidgets import HTML, Image, Layout, VBox


_ACTIVE_CANVAS = None


class LiveCanvas:
    """A remote-Jupyter alternative to cv2.imshow()."""

    def __init__(self, fps=30.0, size=(960, 540), jpeg_quality=60):
        global _ACTIVE_CANVAS
        if _ACTIVE_CANVAS is not None:
            _ACTIVE_CANVAS.stop()

        self.fps = float(fps or 30.0)
        self.size = tuple(size)
        self.jpeg_quality = jpeg_quality
        self.period = 1 / self.fps
        self.deadline = None
        self.frame_number = 0
        self.finished = False
        self.status_updated = time.perf_counter()

        width, height = self.size
        ok, blank = cv2.imencode(".jpg", np.zeros((height, width, 3), dtype=np.uint8))
        self.image = Image(
            value=blank.tobytes() if ok else b"", format="jpeg", width=width, height=height,
            layout=Layout(max_width="100%"),
        )
        self.status = HTML(value="<span style='font:13px monospace;color:#666'>Waiting for video…</span>")
        self.widget = VBox([self.image, self.status], layout=Layout(width=f"{width}px", max_width="100%"))

        _ACTIVE_CANVAS = self
        display(self.widget)

    def _draw(self, result):
        frame = result.orig_img.copy()
        active = 0
        if result.obb is not None and result.obb.id is not None:
            polygons = result.obb.xyxyxyxy.cpu().numpy()
            ids = result.obb.id.int().cpu().tolist()
            active = len(ids)

            for polygon, track_id in zip(polygons, ids):
                points = polygon.astype(np.int32)
                color = (
                    (37 * track_id) % 200 + 35,
                    (17 * track_id) % 200 + 35,
                    (97 * track_id) % 200 + 35,
                )
                cv2.polylines(frame, [points], True, color, 2, cv2.LINE_AA)

        cv2.rectangle(frame, (12, 12), (360, 48), (20, 20, 20), -1)
        cv2.putText(
            frame, f"Sheep OBB + ByteTrack | {active} tracks | LIVE {self.fps:.0f} FPS",
            (22, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 1, cv2.LINE_AA,
        )
        return frame

    def write(self, result):
        """Render and publish one Ultralytics result."""
        rendered = self._draw(result)
        now = time.perf_counter()
        ok, jpeg = cv2.imencode(
            ".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if ok:
            self.image.value = jpeg.tobytes()
            self.frame_number += 1
            if now - self.status_updated >= 1:
                self.status.value = (
                    f"<span style='font:13px monospace;color:#666'>"
                    f"Live stream · frame {self.frame_number} · {self.fps:.0f} FPS</span>"
                )
                self.status_updated = now

        self.deadline = now if self.deadline is None else self.deadline
        self.deadline += self.period
        delay = self.deadline - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        elif delay < -self.period:
            self.deadline = time.perf_counter()

    def finish(self):
        """Mark the current stream as complete."""
        self.finished = True
        self.status.value = (
            f"<span style='font:13px monospace;color:#666'>"
            f"Video complete · {self.frame_number} frames</span>"
        )

    def stop(self):
        if self.finished:
            return
        self.finished = True
        self.widget.close()
