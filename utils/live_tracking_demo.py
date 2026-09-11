#!/usr/bin/env python3
"""A low-latency Jupyter canvas for notebook tracking loops."""

import asyncio
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from aiohttp import web
from IPython.display import HTML, Javascript, display
from jupyter_server.serverapp import list_running_servers


_ACTIVE_CANVAS = None


def _jupyter_base_url():
    """Return the active server base URL without browser-side discovery."""
    try:
        servers = list(list_running_servers())
    except Exception:
        servers = []

    cwd = Path.cwd().resolve()
    for server in servers:
        root = server.get("root_dir")
        if root and cwd.is_relative_to(Path(root).resolve()):
            return server.get("base_url") or "/"
    return servers[0].get("base_url", "/") if servers else "/"


class LiveCanvas:
    """Render tracking results with latest-frame WebSocket delivery."""

    def __init__(self, fps=30.0, size=(960, 540), jpeg_quality=60, port=8765):
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
        except (ConnectionResetError, RuntimeError, asyncio.CancelledError):
            pass
        await socket.close()
        return socket

    async def _status(self, _):
        return web.json_response({"frame": self.frame_number, "source_fps": self.fps})

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
        """Render and publish the newest tracking result."""
        rendered = self._draw(result)
        ok, jpeg = cv2.imencode(
            ".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if ok:
            with self.lock:
                self.jpeg = jpeg.tobytes()
                self.sequence += 1
                self.frame_number += 1

        now = time.perf_counter()
        self.deadline = now if self.deadline is None else self.deadline
        self.deadline += self.period
        delay = self.deadline - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        elif delay < -self.period:
            self.deadline = time.perf_counter()

    def finish(self):
        """End the stream after its newest frame is delivered."""
        with self.lock:
            self.finished = True

    def _show(self):
        width, height = self.size
        suffix = time.time_ns()
        canvas_id = f"live-canvas-{suffix}"
        status_id = f"live-status-{suffix}"
        websocket_path = f"{_jupyter_base_url()}proxy/{self.port}/ws"

        display(HTML(
            f'<div style="max-width:{width}px">'
            f'<canvas id="{canvas_id}" width="{width}" height="{height}" '
            f'style="display:block;width:100%;background:#111"></canvas>'
            f'<div id="{status_id}" style="font:13px monospace;color:#666">Connecting…</div>'
            f'</div>'
        ))
        display(Javascript(f"""
        (() => {{
          const canvas = document.getElementById({json.dumps(canvas_id)});
          const status = document.getElementById({json.dumps(status_id)});
          const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
          const socket = new WebSocket(protocol + '//' + location.host + {json.dumps(websocket_path)});
          socket.binaryType = 'blob';

          let pending = null;
          let drawing = false;
          let frames = 0;
          let tick = performance.now();

          const drawLatest = async () => {{
            if (drawing || pending === null) return;
            drawing = true;
            const blob = pending;
            pending = null;
            try {{
              const image = await createImageBitmap(blob);
              canvas.getContext('2d').drawImage(image, 0, 0, canvas.width, canvas.height);
              image.close();
              frames++;
              const now = performance.now();
              if (now - tick >= 1000) {{
                status.textContent = `Live stream · ${{(frames * 1000 / (now - tick)).toFixed(1)}} FPS`;
                frames = 0;
                tick = now;
              }}
            }} catch (error) {{
              status.textContent = 'Frame decode failed';
            }} finally {{
              drawing = false;
              if (pending !== null) drawLatest();
            }}
          }};

          socket.onopen = () => status.textContent = 'Live stream connected';
          socket.onmessage = event => {{
            pending = event.data;
            drawLatest();
          }};
          socket.onerror = () => status.textContent = 'Live stream connection failed';
          socket.onclose = () => status.textContent = 'Live stream ended';
        }})();
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
