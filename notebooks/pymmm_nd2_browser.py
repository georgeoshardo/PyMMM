import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    import marimo as mo
    from pathlib import Path
    from dask.distributed import Client, LocalCluster
    import hvplot.xarray
    import hvplot

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from pymmm import ND2Experiment

    hvplot.extension("bokeh")

    cluster = LocalCluster(
        processes=True,
        threads_per_worker=1,
        n_workers=0,
        memory_limit="10GB",
    )
    client = Client(cluster)
    return ND2Experiment, Path, cluster, mo


@app.cell
def _(mo):
    app_header = mo.md(r"""
    # PyMMM ND2 Browser

    Load an ND2 lazily and browse the raw xarray-backed image data with the same hvPlot/Bokeh pattern used in the notebook pipeline.
    """)
    return (app_header,)


@app.cell
def _(mo):
    nd2_path = mo.ui.text(
        value="",
        label="ND2 path",
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
def _(cluster, experiment_loaded, mo):
    if experiment_loaded:
        cluster.adapt(minimum=0, maximum=1)
        browse_status = mo.md("### Raw data browser")
    else:
        browse_status = mo.md("### Raw data browser\n\nLoad a valid ND2 file to enable browsing.")
    return (browse_status,)


@app.cell
def _(exp, experiment_loaded, load_error, mo):
    if experiment_loaded:
        raw_browser = exp.data.hvplot.image(
            x="X", y="Y", cmap="Greys_r", dynamic=True,
            rasterize=True, widget_location="top", aspect="equal"
        )
    else:
        raw_browser = mo.md(load_error)
    return (raw_browser,)


@app.cell
def _(
    app_header,
    browse_status,
    experiment_stats,
    experiment_summary,
    mo,
    raw_browser,
    source_panel,
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
                            browse_status,
                            raw_browser,
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
