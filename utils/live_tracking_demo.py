#!/usr/bin/env python3
"""A low-latency Jupyter canvas for frames produced inside a notebook loop."""

import asyncio
import json
import threading
import time
import cv2
import numpy as np
from aiohttp import web


_ACTIVE_CANVAS = None


class LiveCanvas:
    """A remote-Jupyter alternative to cv2.imshow() with latest-frame delivery."""

    def __init__(
        self, fps=30.0, size=(960, 540), jpeg_quality=60, port=8765,
    ):
        global _ACTIVE_CANVAS
        if _ACTIVE_CANVAS is not None:
            _ACTIVE_CANVAS.stop()

        self.fps = float(fps or 30.0)
        self.size = tuple(size)
        self.jpeg_quality = jpeg_quality
        self.port = port
        self.period = 1 / self.fps
        self.deadline = None
        self.frame_number = 0

        self.lock = threading.Lock()
        self.jpeg = None
        self.sequence = 0
        self.finished = False
        self.stats = {}
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.loop = asyncio.new_event_loop()

        threading.Thread(target=self._serve, daemon=True).start()
        if not self.ready.wait(5):
            raise RuntimeError("The live WebSocket server did not start")

        _ACTIVE_CANVAS = self
        self._show()

    async def _websocket(self, request):
        socket = web.WebSocketResponse(heartbeat=20)
        await socket.prepare(request)
        last = -1
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    jpeg, sequence, finished = self.jpeg, self.sequence, self.finished

                if jpeg is not None and sequence != last:
                    transport = request.transport
                    if transport is not None and transport.get_write_buffer_size() < 256 * 1024:
                        await socket.send_bytes(jpeg)
                        last = sequence

                if finished and jpeg is not None and last == sequence:
                    break
                await asyncio.sleep(0.002)
        except (ConnectionResetError, RuntimeError):
            pass
        await socket.close()
        return socket

    async def _status(self, _):
        return web.json_response(self.stats)

    def _serve(self):
        asyncio.set_event_loop(self.loop)
        app = web.Application()
        app.router.add_get("/ws", self._websocket)
        app.router.add_get("/status.json", self._status)
        self.runner = web.AppRunner(app)
        self.loop.run_until_complete(self.runner.setup())
        self.site = web.TCPSite(self.runner, "127.0.0.1", self.port)
        self.loop.run_until_complete(self.site.start())
        self.ready.set()
        self.loop.run_forever()

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
        """Render and publish one Ultralytics result without queueing stale frames."""
        rendered = self._draw(result)
        ok, jpeg = cv2.imencode(
            ".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if ok:
            with self.lock:
                self.jpeg = jpeg.tobytes()
                self.sequence += 1
                self.frame_number += 1
                self.stats = {"frame": self.frame_number, "source_fps": self.fps}

        now = time.perf_counter()
        self.deadline = now if self.deadline is None else self.deadline
        self.deadline += self.period
        delay = self.deadline - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        elif delay < -self.period:
            self.deadline = time.perf_counter()

    def finish(self):
        """End the current stream after the newest frame reaches the browser."""
        with self.lock:
            self.finished = True

    def _show(self):
        from IPython.display import HTML, display

        width, height = self.size
        display(HTML(f"""
        <div style="max-width:{width}px">
          <canvas id="live-canvas-{self.port}" width="{width}" height="{height}"
                  style="width:100%;background:#111"></canvas>
          <div id="live-status-{self.port}" style="font:13px monospace;color:#666"></div>
        </div>
        <script>
        (() => {{
          const config = JSON.parse(document.getElementById('jupyter-config-data').textContent);
          const base = config.baseUrl || '/';
          const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
          const canvas = document.getElementById('live-canvas-{self.port}');
          const status = document.getElementById('live-status-{self.port}');
          let frames = 0, tick = performance.now();
          const socket = new WebSocket(`${{protocol}}//${{location.host}}${{base}}proxy/{self.port}/ws`);
          socket.binaryType = 'blob';
          socket.onopen = () => status.textContent = 'Live stream connected';
          socket.onmessage = async event => {{
            const image = await createImageBitmap(event.data);
            canvas.getContext('2d').drawImage(image, 0, 0, canvas.width, canvas.height);
            image.close();
            frames++;
            const now = performance.now();
            if (now - tick >= 1000) {{
              status.textContent = `Live stream | ${{(frames * 1000 / (now - tick)).toFixed(1)}} FPS`;
              frames = 0;
              tick = now;
            }}
          }};
          socket.onerror = () => status.textContent = 'Live stream connection failed';
          socket.onclose = () => status.textContent = 'Live stream ended';
        }})();
        </script>
        """))

    def stop(self):
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        if hasattr(self, "runner") and self.loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self.runner.cleanup(), self.loop)
            try:
                future.result(timeout=3)
            except Exception:
                pass
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
