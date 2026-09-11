#!/usr/bin/env python3
"""A responsive Jupyter image stream for notebook tracking loops."""

import time

import cv2
import numpy as np
from IPython.display import display
from ipywidgets import HTML, Image, Layout, VBox


_ACTIVE_CANVAS = None


class LiveCanvas:
    """Render every tracking result while rate-limiting browser updates."""

    def __init__(self, fps=30.0, size=(960, 540), jpeg_quality=50, display_fps=30.0):
        global _ACTIVE_CANVAS
        if _ACTIVE_CANVAS is not None:
            _ACTIVE_CANVAS.stop()

        self.fps = float(fps or 30.0)
        self.display_fps = min(max(float(display_fps), 1.0), self.fps)
        self.size = tuple(size)
        self.jpeg_quality = jpeg_quality
        self.period = 1 / self.fps
        self.deadline = None
        self.display_credit = self.fps - self.display_fps
        self.status_updated = time.perf_counter()
        self.frame_number = 0
        self.displayed_frames = 0
        self.displayed_source_frame = 0
        self.latest_result = None
        self.finished = False

        width, height = self.size
        blank = np.full((height, width, 3), 24, dtype=np.uint8)
        message = "Preparing tracking model..."
        (text_width, text_height), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        cv2.putText(
            blank, message, ((width - text_width) // 2, (height + text_height) // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (210, 210, 210), 1, cv2.LINE_AA,
        )
        ok, blank = cv2.imencode(".jpg", blank)
        self.image = Image(
            value=blank.tobytes() if ok else b"", format="jpeg",
            layout=Layout(width="100%", height="auto", object_fit="contain", overflow="hidden"),
        )
        self.status = HTML(value="<span style='font:13px monospace;color:#666'>Preparing tracking model…</span>")
        self.widget = VBox(
            [self.image, self.status],
            layout=Layout(width="100%", max_width=f"{width}px", overflow="hidden"),
        )

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
            frame, f"Sheep OBB + ByteTrack | {active} tracks | LIVE",
            (22, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 1, cv2.LINE_AA,
        )
        return frame

    def _publish(self, result):
        ok, jpeg = cv2.imencode(
            ".jpg", self._draw(result), [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if ok:
            self.image.value = jpeg.tobytes()
            self.displayed_frames += 1
            self.displayed_source_frame = self.frame_number

    def write(self, result):
        """Track every source frame and publish at a browser-safe rate."""
        now = time.perf_counter()
        self.frame_number += 1
        self.latest_result = result

        self.display_credit += self.display_fps
        if self.display_credit >= self.fps:
            self._publish(result)
            self.display_credit -= self.fps

        if now - self.status_updated >= 1:
            self.status.value = (
                f"<span style='font:13px monospace;color:#666'>"
                f"Tracking {self.fps:.0f} FPS · displaying {self.display_fps:.0f} FPS</span>"
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
        """Keep the final tracked frame visible after playback."""
        if self.latest_result is not None and self.displayed_source_frame != self.frame_number:
            self._publish(self.latest_result)
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
