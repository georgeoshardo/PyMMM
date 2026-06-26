import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    import time
    from pathlib import Path

    import marimo as mo
    import vizarr

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from pymmm import ND2Experiment
    from pymmm.lazy_zarr import LazyXarrayZarrStore

    _contained_vizarr_esm = r"""
    import * as vizarr from "https://hms-dbmi.github.io/vizarr/index.js";

    function send(model, payload, { timeout = 3000 } = {}) {
      let uuid = globalThis.crypto.randomUUID();
      return new Promise((resolve, reject) => {
        let timer = setTimeout(() => {
          reject(new Error(`Promise timed out after ${timeout} ms`));
          model.off("msg:custom", handler);
        }, timeout);

        function handler(msg, buffers) {
          if (!(msg.uuid === uuid)) return;
          clearTimeout(timer);
          resolve({ data: msg.payload, buffers });
          model.off("msg:custom", handler);
        }

        model.on("msg:custom", handler);
        model.send({ payload, uuid });
      });
    }

    function get_source(model, source) {
      if (typeof source === "string") {
        return source;
      }
      return {
        async has(key) {
          const { data } = await send(model, {
            method: "has",
            target: [source.id, key],
          });
          return data.success;
        },
        async get(key) {
          const { data, buffers } = await send(model, {
            method: "get",
            target: [source.id, key],
          });
          if (!data.success) {
            return undefined;
          }
          return new Uint8Array(buffers[0].buffer);
        },
      };
    }

    function mirrorMuiStyles(host) {
      const styleRoot = host.getRootNode() instanceof ShadowRoot ? host.getRootNode() : host;
      const mirrored = new WeakMap();

      function shouldMirror(style) {
        const text = style.textContent || "";
        const meta = style.getAttribute("data-meta") || "";
        return (
          style.hasAttribute("data-jss") ||
          meta.includes("Mui") ||
          text.includes("MuiSlider") ||
          text.includes("MuiGrid") ||
          text.includes("MuiAccordion") ||
          text.includes("MuiIconButton") ||
          text.includes("MuiTypography")
        );
      }

      function mirror(style) {
        if (!(style instanceof HTMLStyleElement) || !shouldMirror(style)) {
          return;
        }
        let clone = mirrored.get(style);
        if (!clone) {
          clone = document.createElement("style");
          clone.setAttribute("data-pymmm-mirrored-vizarr-style", "true");
          mirrored.set(style, clone);
          styleRoot.appendChild(clone);
        }
        clone.textContent = style.textContent;
      }

      function sync() {
        document.head.querySelectorAll("style").forEach(mirror);
      }

      const observer = new MutationObserver(sync);
      observer.observe(document.head, {
        childList: true,
        subtree: true,
        characterData: true,
      });
      sync();
      requestAnimationFrame(sync);
      setTimeout(sync, 0);
      setTimeout(sync, 250);
      setTimeout(sync, 1000);
      return observer;
    }

    export default {
      async render({ model, el }) {
        el.style.display = "block";
        el.style.width = "100%";
        el.style.minWidth = "0";
        el.style.maxWidth = "100%";
        el.style.overflow = "hidden";
        el.style.boxSizing = "border-box";

        let localStyle = document.createElement("style");
        localStyle.textContent = `
          .pymmm-vizarr-shell {
            position: relative !important;
            width: 100% !important;
            min-width: 0 !important;
            max-width: 100% !important;
            overflow: hidden !important;
            background: black !important;
            box-sizing: border-box !important;
            isolation: isolate !important;
          }
        `;

        let div = document.createElement("div");
        div.className = "pymmm-vizarr-shell";
        div.style.height = model.get("height");

        model.on("change:height", () => {
          div.style.height = model.get("height");
        });

        const styleObserver = mirrorMuiStyles(el);
        let viewer = await vizarr.createViewer(div, { menuOpen: true });

        // Keep viewport state browser-local. In marimo, syncing every pan/zoom
        // through Python can trigger a widget re-render that resets the view.

        for (const config of model.get("_configs")) {
          const source = get_source(model, config.source);
          viewer.addImage({ ...config, source });
        }
        model.on("change:_configs", () => {
          const last = model.get("_configs").at(-1);
          if (!last) return;
          const source = get_source(model, last.source);
          viewer.addImage({ ...last, source });
        });

        el.replaceChildren(localStyle, div);

        return () => {
          styleObserver.disconnect();
          viewer.destroy();
        };
      },
    };
    """

    class ContainedVizarrViewer(vizarr.Viewer):
        _esm = _contained_vizarr_esm


    return (
        ContainedVizarrViewer,
        LazyXarrayZarrStore,
        ND2Experiment,
        Path,
        mo,
        time,
    )


@app.cell
def _(mo):
    app_header = mo.md(r"""
    # PyMMM ND2 Browser

    Load an ND2 lazily and browse the full multidimensional array through vizarr without materializing the whole file.
    """)
    return (app_header,)


@app.cell
def _(mo):
    nd2_path = mo.ui.text(
        value="",
        label="ND2 path",
        placeholder="/data/20260331_SB5_6_7_8_16/20260331.nd2",
        full_width=True,
    )

    source_panel = mo.vstack(
        [
            mo.md("### Source"),
            nd2_path,
        ],
        gap=0.4,
    )
    return nd2_path, source_panel


@app.cell
def _(LazyXarrayZarrStore):
    def axis_order_for_viewer(data):
        preferred = ["P", "T", "Z", "C", "Y", "X"]
        ordered = [dim for dim in preferred if dim in data.dims]
        ordered.extend(dim for dim in data.dims if dim not in ordered)
        if "Y" in ordered and "X" in ordered:
            ordered = [dim for dim in ordered if dim not in {"Y", "X"}] + ["Y", "X"]
        return tuple(ordered)


    def chunks_for_viewer(data, axis_order):
        chunks = []
        for dim in axis_order:
            size = int(data.sizes[dim])
            if dim in {"Y", "X"}:
                chunks.append(min(512, size))
            else:
                chunks.append(1)
        return tuple(chunks)


    def build_lazy_vizarr_store(exp):
        axis_order = axis_order_for_viewer(exp.data)
        view_data = exp.data.transpose(*axis_order)
        chunks = chunks_for_viewer(view_data, axis_order)
        attrs = {
            "axis_order": list(axis_order),
            "axis_labels": [dim.lower() for dim in axis_order],
            "channel_names": list(exp.channel_names),
            "fov_names": list(exp.fov_names),
        }
        store = LazyXarrayZarrStore(view_data, chunks=chunks, attrs=attrs)
        return store, view_data, axis_order, chunks


    return (build_lazy_vizarr_store,)


@app.cell
def _(ND2Experiment, Path, nd2_path):
    _path_text = nd2_path.value.strip()
    path = Path(_path_text).expanduser() if _path_text else None
    exp = None
    load_error = None

    if not _path_text:
        load_error = "Enter an ND2 path to load an experiment."
    elif path is None or not path.exists():
        load_error = f"ND2 file not found: `{path}`"
    else:
        try:
            exp = ND2Experiment(path)
        except Exception as _error:
            load_error = f"Could not load ND2 file `{path}`: {_error}"

    experiment_loaded = exp is not None
    return exp, experiment_loaded, load_error, path


@app.cell
def _(exp, experiment_loaded, load_error, mo, path):
    if experiment_loaded:
        dims = dict(exp.data.sizes)
        dim_text = " x ".join(f"{name}={size}" for name, size in dims.items())
        channel_text = ", ".join(exp.channel_names)
        fov_preview = ", ".join(exp.fov_names[:5])
        if exp.n_fovs > 5:
            fov_preview += ", ..."

        experiment_stats = mo.hstack(
            [
                mo.stat(value=str(exp.n_fovs), label="FOVs"),
                mo.stat(value=str(exp.n_timepoints), label="Timepoints"),
                mo.stat(value=str(len(exp.channel_names)), label="Channels"),
                mo.stat(value=f"{exp.pixel_size_um:.4f} um", label="Pixel size"),
            ],
            widths="equal",
            gap=0.8,
        )

        experiment_summary = mo.md(
            f"""
    ### Loaded experiment

    - **Path:** `{path}`
    - **Dimensions:** `{dim_text}`
    - **Channels:** `{channel_text}`
    - **FOVs:** {exp.n_fovs} ({fov_preview})
    - **Time interval:** {exp.time_interval_ms:.1f} ms
    """
        )
    else:
        experiment_stats = mo.hstack(
            [
                mo.stat(value="-", label="FOVs"),
                mo.stat(value="-", label="Timepoints"),
                mo.stat(value="-", label="Channels"),
                mo.stat(value="-", label="Pixel size"),
            ],
            widths="equal",
            gap=0.8,
        )
        experiment_summary = mo.md(f"""
    ### Loaded experiment

    {load_error}
    """)
    return experiment_stats, experiment_summary


@app.cell
def _(experiment_loaded, mo):
    if experiment_loaded:
        build_preview = mo.ui.button(
            value=0,
            on_click=lambda count: count + 1,
            label="Build / refresh lazy viewer",
            full_width=True,
        )
        viewer_status = mo.md("""
    ### Vizarr preview

    The viewer receives the full ND2 as a lazy xarray-backed Zarr store. Chunks are computed from the ND2 only when the browser asks for them.
    """)
        viewer_controls = mo.vstack(
            [
                viewer_status,
                build_preview,
            ],
            gap=0.5,
        )
    else:
        build_preview = mo.ui.button(
            value=0,
            on_click=lambda count: count + 1,
            label="Build / refresh lazy viewer",
            disabled=True,
            full_width=True,
        )
        viewer_status = mo.md("### Vizarr preview\n\nLoad a valid ND2 file to enable the viewer.")
        viewer_controls = mo.vstack(
            [
                viewer_status,
                build_preview,
            ],
            gap=0.5,
        )

    return build_preview, viewer_controls


@app.cell
def _(
    ContainedVizarrViewer,
    build_lazy_vizarr_store,
    build_preview,
    exp,
    experiment_loaded,
    load_error,
    mo,
    time,
):
    if not experiment_loaded:
        lazy_store = None
        raw_browser = mo.md(load_error)
        browser_details = mo.md("No vizarr-backed image is loaded.")
    elif build_preview.value < 1:
        lazy_store = None
        raw_browser = mo.md("Click **Build / refresh lazy viewer** to expose the ND2 to vizarr.")
        browser_details = mo.md("No vizarr-backed image is loaded.")
    else:
        _start = time.perf_counter()
        with mo.status.spinner(
            title="Preparing lazy Zarr view",
            subtitle="No image data is materialized until vizarr requests chunks.",
        ):
            lazy_store, view_data, axis_order, chunks = build_lazy_vizarr_store(exp)
        _elapsed = time.perf_counter() - _start

        _channel_axis = axis_order.index("C") if "C" in axis_order else None
        _config = {
            "source": lazy_store,
            "name": exp.experiment_name,
            "axis_labels": [dim.lower() for dim in axis_order],
        }
        if _channel_axis is not None:
            _channel_names = list(exp.channel_names)
            _config.update(
                {
                    "channel_axis": _channel_axis,
                    "names": _channel_names,
                    "colors": ["#FFFFFF", "#00FFFF", "#FFFF00", "#FF0000"][: len(_channel_names)],
                    "visibilities": [True] + [False] * (len(_channel_names) - 1),
                }
            )

        raw_browser = ContainedVizarrViewer(height="680px")
        raw_browser.add_image(**_config)
        browser_details = mo.md(f"""
    Lazy Zarr view shape: `{tuple(view_data.shape)}`. Chunks: `{chunks}`. Axis order: `{axis_order}`. Setup time: `{_elapsed:.2f} s`. Image chunks are computed on demand from the ND2-backed dask array.
    """)

    return browser_details, raw_browser


@app.cell
def _(
    app_header,
    browser_details,
    experiment_stats,
    experiment_summary,
    mo,
    raw_browser,
    source_panel,
    viewer_controls,
):
    dashboard = mo.vstack(
        [
            app_header,
            experiment_stats,
            mo.hstack(
                [
                    mo.vstack(
                        [
                            source_panel,
                            experiment_summary,
                        ],
                        gap=1.0,
                    ),
                    mo.vstack(
                        [
                            viewer_controls,
                            raw_browser,
                            browser_details,
                        ],
                        gap=0.8,
                    ),
                ],
                widths=[1, 2.2],
                gap=1.2,
                align="start",
            ),
        ],
        gap=1.0,
    )

    dashboard
    return


if __name__ == "__main__":
    app.run()
