from __future__ import annotations

import argparse
import io
import json
import sys
import time
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pymmm import ND2Experiment  # noqa: E402


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PyMMM ND2 Viewer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101114;
      --panel: #17191e;
      --panel2: #20232a;
      --text: #eef1f5;
      --muted: #aab2bf;
      --accent: #66c2ff;
      --border: #333844;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.35 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      overflow: hidden;
    }
    header {
      display: grid;
      grid-template-columns: minmax(280px, 1fr) auto;
      gap: 14px;
      align-items: center;
      padding: 10px 12px;
      background: var(--panel);
      border-bottom: 1px solid var(--border);
    }
    h1 {
      margin: 0;
      font-size: 16px;
      font-weight: 650;
      letter-spacing: 0;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .sub {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: end;
      justify-content: end;
    }
    label {
      display: grid;
      gap: 3px;
      color: var(--muted);
      font-size: 11px;
    }
    select, input, button {
      height: 30px;
      border: 1px solid var(--border);
      border-radius: 5px;
      background: var(--panel2);
      color: var(--text);
      font: inherit;
    }
    select { min-width: 120px; padding: 0 8px; }
    input[type="range"] { width: 180px; }
    button {
      padding: 0 10px;
      cursor: pointer;
    }
    button:hover { border-color: var(--accent); }
    main {
      position: relative;
      min-height: 0;
      background: #050506;
    }
    canvas {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      display: block;
      cursor: grab;
      image-rendering: auto;
    }
    canvas.dragging { cursor: grabbing; }
    .hud {
      position: absolute;
      left: 12px;
      bottom: 12px;
      display: grid;
      gap: 4px;
      max-width: min(720px, calc(100vw - 24px));
      padding: 8px 10px;
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 6px;
      background: rgba(0,0,0,0.58);
      color: var(--text);
      pointer-events: none;
      font-size: 12px;
    }
    .status { color: var(--accent); }
    .error { color: #ff8a8a; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1 id="title">PyMMM ND2 Viewer</h1>
      <div class="sub" id="subtitle">Loading metadata...</div>
    </div>
    <div class="controls">
      <label>FOV<select id="fov"></select></label>
      <label>Time <span id="timeLabel">0</span><input id="time" type="range" min="0" max="0" value="0" /></label>
      <label>Channel<select id="channel"></select></label>
      <label>Image size<select id="maxDim">
        <option value="1024">1024 px</option>
        <option value="1536" selected>1536 px</option>
        <option value="2304">full frame</option>
      </select></label>
      <button id="fit">Fit</button>
      <button id="reset">Reset</button>
    </div>
  </header>
  <main id="stage">
    <canvas id="canvas"></canvas>
    <div class="hud">
      <div id="readout">Waiting for first frame...</div>
      <div id="status" class="status">Idle</div>
    </div>
  </main>
  <script>
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d", { alpha: false });
    const stage = document.getElementById("stage");
    const fov = document.getElementById("fov");
    const channel = document.getElementById("channel");
    const timeSlider = document.getElementById("time");
    const timeLabel = document.getElementById("timeLabel");
    const maxDim = document.getElementById("maxDim");
    const readout = document.getElementById("readout");
    const statusEl = document.getElementById("status");
    const title = document.getElementById("title");
    const subtitle = document.getElementById("subtitle");

    let meta = null;
    let image = null;
    let imageInfo = null;
    let scale = 1;
    let panX = 0;
    let panY = 0;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    let requestId = 0;

    function setStatus(text, error=false) {
      statusEl.textContent = text;
      statusEl.className = error ? "error" : "status";
    }

    function resizeCanvas() {
      const rect = stage.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }

    function fitImage() {
      if (!image) return;
      const rect = stage.getBoundingClientRect();
      scale = Math.min(rect.width / image.width, rect.height / image.height);
      panX = (rect.width - image.width * scale) / 2;
      panY = (rect.height - image.height * scale) / 2;
      draw();
    }

    function resetView() {
      scale = 1;
      panX = 0;
      panY = 0;
      fitImage();
    }

    function draw() {
      const rect = stage.getBoundingClientRect();
      ctx.save();
      ctx.setTransform(window.devicePixelRatio || 1, 0, 0, window.devicePixelRatio || 1, 0, 0);
      ctx.fillStyle = "#050506";
      ctx.fillRect(0, 0, rect.width, rect.height);
      if (image) {
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(image, panX, panY, image.width * scale, image.height * scale);
      }
      ctx.restore();
    }

    function populateControls() {
      title.textContent = `PyMMM ND2 Viewer`;
      subtitle.textContent = `${meta.path} | ${meta.dims.T} T x ${meta.dims.P} FOV x ${meta.dims.C} C x ${meta.dims.Y} x ${meta.dims.X}`;
      fov.replaceChildren(...meta.fovs.map((name, idx) => new Option(name, idx)));
      channel.replaceChildren(...meta.channels.map((name, idx) => new Option(name, idx)));
      timeSlider.max = String(meta.dims.T - 1);
      timeSlider.value = "0";
      timeLabel.textContent = `0/${meta.dims.T - 1}`;
    }

    async function loadFrame({ keepView = true } = {}) {
      if (!meta) return;
      const id = ++requestId;
      const p = Number(fov.value);
      const t = Number(timeSlider.value);
      const c = Number(channel.value);
      const max = Number(maxDim.value);
      timeLabel.textContent = `${t}/${meta.dims.T - 1}`;
      setStatus(`Loading ${meta.fovs[p]} T=${t} C=${meta.channels[c]}...`);
      const started = performance.now();
      const url = `/frame.jpg?p=${p}&t=${t}&c=${c}&max_dim=${max}&_=${Date.now()}`;
      try {
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(await response.text());
        }
        const blob = await response.blob();
        if (id !== requestId) return;
        const next = new Image();
        const objectUrl = URL.createObjectURL(blob);
        next.onload = () => {
          URL.revokeObjectURL(objectUrl);
          image = next;
          imageInfo = {
            p, t, c, max,
            serverMs: response.headers.get("x-render-ms"),
            sourceShape: response.headers.get("x-source-shape"),
            displayShape: response.headers.get("x-display-shape"),
          };
          if (!keepView) fitImage();
          else if (!imageInfo || !image) fitImage();
          draw();
          const elapsed = Math.round(performance.now() - started);
          readout.textContent = `${meta.fovs[p]} | T=${t} | ${meta.channels[c]} | source ${imageInfo.sourceShape} | displayed ${imageInfo.displayShape} | zoom ${scale.toFixed(2)}x`;
          setStatus(`Loaded in ${elapsed} ms, server ${imageInfo.serverMs} ms`);
        };
        next.src = objectUrl;
      } catch (error) {
        if (id !== requestId) return;
        setStatus(String(error), true);
      }
    }

    function scheduleFrame(keepView=true) {
      loadFrame({ keepView });
    }

    timeSlider.addEventListener("input", () => scheduleFrame(true));
    fov.addEventListener("change", () => scheduleFrame(false));
    channel.addEventListener("change", () => scheduleFrame(true));
    maxDim.addEventListener("change", () => scheduleFrame(false));
    document.getElementById("fit").addEventListener("click", fitImage);
    document.getElementById("reset").addEventListener("click", resetView);

    canvas.addEventListener("pointerdown", (event) => {
      dragging = true;
      canvas.classList.add("dragging");
      lastX = event.clientX;
      lastY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      panX += event.clientX - lastX;
      panY += event.clientY - lastY;
      lastX = event.clientX;
      lastY = event.clientY;
      draw();
    });
    canvas.addEventListener("pointerup", (event) => {
      dragging = false;
      canvas.classList.remove("dragging");
      canvas.releasePointerCapture(event.pointerId);
    });
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      if (!image) return;
      const rect = stage.getBoundingClientRect();
      const mx = event.clientX - rect.left;
      const my = event.clientY - rect.top;
      const beforeX = (mx - panX) / scale;
      const beforeY = (my - panY) / scale;
      const factor = Math.exp(-event.deltaY * 0.0012);
      scale = Math.min(24, Math.max(0.03, scale * factor));
      panX = mx - beforeX * scale;
      panY = my - beforeY * scale;
      draw();
    }, { passive: false });
    canvas.addEventListener("dblclick", fitImage);
    window.addEventListener("resize", resizeCanvas);
    window.addEventListener("keydown", (event) => {
      if (event.key === "ArrowRight") {
        timeSlider.value = String(Math.min(Number(timeSlider.max), Number(timeSlider.value) + 1));
        scheduleFrame(true);
      } else if (event.key === "ArrowLeft") {
        timeSlider.value = String(Math.max(0, Number(timeSlider.value) - 1));
        scheduleFrame(true);
      }
    });

    async function main() {
      resizeCanvas();
      const response = await fetch("/metadata");
      meta = await response.json();
      populateControls();
      await loadFrame({ keepView: false });
    }
    main().catch((error) => setStatus(String(error), true));
  </script>
</body>
</html>
"""


def make_viewer_handler(exp: ND2Experiment, nd2_path: Path):
    dims = {name: int(size) for name, size in exp.data.sizes.items()}
    metadata = {
        "path": str(nd2_path),
        "dims": dims,
        "channels": list(exp.channel_names),
        "fovs": list(exp.fov_names),
        "pixel_size_um": float(exp.pixel_size_um),
        "time_interval_ms": float(exp.time_interval_ms),
    }

    @lru_cache(maxsize=96)
    def render_frame(p: int, t: int, c: int, max_dim: int) -> tuple[bytes, dict[str, str]]:
        start = time.perf_counter()
        arr = exp.data.isel(P=p, T=t, C=c).compute().values
        arr = np.asarray(arr)
        if arr.ndim != 2:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D frame after indexing, got shape {arr.shape}")

        sample = arr[:: max(1, arr.shape[0] // 512), :: max(1, arr.shape[1] // 512)]
        low, high = np.percentile(sample, [0.2, 99.8])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(np.min(sample))
            high = float(np.max(sample))
        if high <= low:
            high = low + 1

        scaled = np.clip((arr.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
        image = Image.fromarray(scaled, mode="L")
        source_shape = image.size
        if max_dim > 0 and max(image.size) > max_dim:
            image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        rgb = Image.merge("RGB", (image, image, image))
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=88, optimize=True)
        elapsed_ms = (time.perf_counter() - start) * 1000
        headers = {
            "Content-Type": "image/jpeg",
            "Cache-Control": "no-store",
            "X-Render-ms": f"{elapsed_ms:.1f}",
            "X-Source-Shape": f"{source_shape[1]}x{source_shape[0]}",
            "X-Display-Shape": f"{image.size[1]}x{image.size[0]}",
            "X-Contrast": f"{low:.3f},{high:.3f}",
        }
        return buf.getvalue(), headers

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

        def send_bytes(self, body: bytes, status: HTTPStatus = HTTPStatus.OK, headers: dict[str, str] | None = None):
            self.send_response(status)
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, payload, status: HTTPStatus = HTTPStatus.OK):
            body = json.dumps(payload).encode("utf-8")
            self.send_bytes(body, status, {"Content-Type": "application/json", "Cache-Control": "no-store"})

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_bytes(HTML.encode("utf-8"), headers={"Content-Type": "text/html; charset=utf-8"})
                return
            if parsed.path == "/metadata":
                self.send_json(metadata)
                return
            if parsed.path == "/frame.jpg":
                query = parse_qs(parsed.query)
                try:
                    p = int(query.get("p", ["0"])[0])
                    t = int(query.get("t", ["0"])[0])
                    c = int(query.get("c", ["0"])[0])
                    max_dim = int(query.get("max_dim", ["1536"])[0])
                    if not (0 <= p < dims["P"] and 0 <= t < dims["T"] and 0 <= c < dims["C"]):
                        raise ValueError(f"Indices out of range: p={p}, t={t}, c={c}")
                    if max_dim not in {1024, 1536, 2304}:
                        raise ValueError(f"Unsupported max_dim={max_dim}")
                    body, headers = render_frame(p, t, c, max_dim)
                    self.send_bytes(body, headers=headers)
                except Exception as error:
                    self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nd2", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2730)
    args = parser.parse_args()

    if not args.nd2.exists():
        raise FileNotFoundError(args.nd2)

    print(f"Loading ND2 metadata from {args.nd2}", flush=True)
    exp = ND2Experiment(args.nd2)
    handler = make_viewer_handler(exp, args.nd2)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"PyMMM ND2 viewer: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
