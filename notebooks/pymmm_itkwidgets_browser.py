import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    from pathlib import Path

    import marimo as mo

    DEFAULT_ND2_PATH = Path("/data/20260331_SB5_6_7_8_16/20260331.nd2")
    DEFAULT_VIEWER_URL = os.environ.get("PYMMM_ITKWIDGETS_VIEWER_URL", "")
    PREVIEW_STACK = Path("/tmp/pymmm-itkwidgets-preview/20260331_fov0_PC_tstack_256.ngff.zarr")
    return DEFAULT_ND2_PATH, DEFAULT_VIEWER_URL, PREVIEW_STACK, mo


@app.cell
def _(DEFAULT_ND2_PATH, DEFAULT_VIEWER_URL, PREVIEW_STACK, mo):
    header = mo.md(
        f"""
        # PyMMM itkwidgets Browser

        This is a thin marimo wrapper around the itkwidgets standalone viewer.
        It is currently pointed at a small real-ND2 preview stack:

        - Source ND2: `{DEFAULT_ND2_PATH}`
        - Preview store: `{PREVIEW_STACK}`
        - Preview contents: `FOV XYPos:0`, `PC`, `T=8`, centre crop `256 x 256`
        """
    )

    viewer_url = mo.ui.text(
        value=DEFAULT_VIEWER_URL,
        label="itkwidgets standalone viewer URL",
        placeholder="http://127.0.0.1:37480/itkwidgets/index.html?...",
        full_width=True,
    )
    return header, viewer_url


@app.cell
def _(mo, viewer_url):
    import html as _html
    import json as _json

    url = viewer_url.value.strip()
    if url:
        _safe_url = _json.dumps(url)
        _link_url = _html.escape(url, quote=True)
        viewer = mo.iframe(
            f"""
            <!doctype html>
            <html>
              <head>
                <meta charset="utf-8" />
                <style>
                  html, body {{
                    margin: 0;
                    width: 100%;
                    height: 100%;
                    background: #111;
                    color: #eee;
                    font-family: system-ui, sans-serif;
                  }}
                  .fallback {{
                    padding: 16px;
                  }}
                  a {{ color: #8ab4ff; }}
                </style>
              </head>
              <body>
                <div class="fallback">
                  Loading itkwidgets. If it does not appear, open
                  <a href="{_link_url}" target="_blank" rel="noreferrer">the standalone viewer</a>.
                </div>
                <script>
                  window.location.replace({_safe_url});
                </script>
              </body>
            </html>
            """,
            height="820px",
        )
        status = mo.md(f"Embedded viewer target: `{url}`")
    else:
        viewer = mo.md("Start the itkwidgets standalone server, then paste its URL above.")
        status = mo.md("No itkwidgets viewer URL is loaded.")
    return status, viewer


@app.cell
def _(header, mo, status, viewer, viewer_url):
    dashboard = mo.vstack(
        [
            header,
            viewer_url,
            status,
            viewer,
        ],
        gap=0.8,
    )
    dashboard
    return


if __name__ == "__main__":
    app.run()
