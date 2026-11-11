# PyMMM

PyMMM turns mother-machine time-lapses (Nikon ND2) into one OME-Zarr image per
trench: it registers each field of view, finds its lanes and trenches, and cuts every
trench out of every frame into an *experiment group*, `<experiment>.zarr`. Later steps
(segmentation, tracking, curation) write their results into each trench's image, beside
its pixels. [docs/layout.md](docs/layout.md) specifies the format.

PyMMM is driven from Jupyter notebooks, with a person in the loop: each detection step
has a widget for tuning its settings on real images before it runs on every FOV.

## Install

```bash
pip install -e .                        # read ND2s, run the steps, read and write the experiment group
pip install -e ".[notebook,view]"       # with the widgets and viewers
```

| extra | adds | for |
|---|---|---|
| `notebook` | ipywidgets, ndv | the steps' widgets and ndv viewers in Jupyter |
| `segment` | omnipose, torch | the Segmenter (Omnipose 2.0 is installed from git, not PyPI) |
| `view` | napari, napari-ome-zarr | `python -m pymmm.view`, experiments dropped onto napari, and single trench images |
| `tracks` | geff | writing tracking runs' GEFFs |

`pixi run test` and `pixi run notebook` use this repository's own environment (PyMMM
with the `notebook`, `view` and `tracks` extras). In phage-signatures, PyMMM is a
submodule and the workspace's environments install it: `default` with `notebook` and
`view`, `seg` with `notebook`, `segment` and `tracks`.

## The pipeline

```python
from pymmm import CompanionStore, Extractor, LaneDetector, ND2Experiment, Registrator, TrenchDetector

exp = ND2Experiment("raw_data/experiment.nd2").select_times(0, 260)
store = CompanionStore.for_experiment(exp)        # checkpoints beside the ND2

reg = Registrator(exp, store, registration_channel="PC", mode="first",
                  roi={"y": (250, 800)}, downsample=4)
reg.compute_tmats()                               # drift per FOV and frame
reg.save()

lanes = LaneDetector(exp, reg, store, detection_channel="PC")
lanes.interactive_detect_lanes()                  # tune, then "Apply to all FOVs"
lanes.save()
# or, for an image cropped to its trenches (exp.select_roi(y=...)), no lane detection:
# lanes.use_image_edge("down")                    # trenches open down into a lane below

trenches = TrenchDetector(exp, reg, lanes, store, detection_channel="PC")
trenches.interactive_detect_trenches()
trenches.save()

extractor = Extractor(exp, reg, trenches, output_path="experiment.zarr")
extractor.preview()                               # lazy: the trenches as they will be written
extractor.extract()                               # resumes an interrupted run
```

Each step saves its result to the companion store, `<experiment>.pymmm.zarr` beside
the ND2, and `Registrator.load`, `LaneDetector.load` and `TrenchDetector.load` pick it
up again. Without widgets, `detect_lanes(...)` and `detect_trenches(...)` take the
settings directly. With `use_image_edge`, each FOV's lane is the image's edge where the
trenches open, and trench detection takes the image's whole height by default.
Registration and extraction run on the active Dask client if there
is one, else in a process pool.

After extraction:

- `Segmenter(path, channel, ...)` labels every trench with Omnipose, one label image
  per *hypothesis*, the base first. A hypothesis is a model under some settings: a stock
  model under the S1–S4 presets, or a fine-tuned checkpoint (`checkpoint_hypotheses`).
  It runs on the active Dask client's workers if there is one, else in a process pool or
  in this process. A Dask worker on another machine needs only the environment and
  PyMMM: each task carries a trench's frames and returns its labels, the checkpoints go
  to each worker once, and the notebook alone writes the experiment group.
- A tracker writes each trench's tracks with `pymmm.store.geff.write_tracks`: the
  masks it chose and a GEFF graph.
- mm-track curates the tracks in a curated copy beside them.

`pymmm.config.Experiment` reads an experiment file (TOML) naming the ND2, the experiment
group and the segmentation settings, so every notebook of the pipeline starts from the
same file.

## Reading an experiment group

```python
from pymmm.store import experiment_group as eg

root = eg.open_experiment("experiment.zarr")
for trench in eg.trench_ids(root):
    pixels = eg.image(root, trench)                    # (t, c, y, x) zarr array
    names, labels = eg.segmentation(root, trench, "pc_affinity_ensemble")
```

`pymmm.store` needs only numpy and zarr. To browse every trench in napari:

```bash
python -m pymmm.view experiment.zarr                   # one slider over all trenches
napari --plugin napari-ome-zarr experiment.zarr/trench_0007
```

PyMMM is also a napari plugin: drop `experiment.zarr` onto napari and choose PyMMM to get the same
layers. napari offers every folder reader for any folder, so it asks each time; its "remember this
choice" box remembers that one folder.

## Modules

| module | |
|---|---|
| `source.nd2` | `ND2Experiment`: the ND2 as a lazy xarray, and its subsets |
| `source.metadata` | ND2 acquisition and per-frame metadata |
| `store.experiment_group` | the experiment group: open it, read trench images, write and read label images and segmentations |
| `store.geff` | tracking runs: write their masks and GEFF, tracklet and lineage ids, read a GEFF as arrays |
| `store.ome` | every OME-Zarr attribute PyMMM writes |
| `store.companion` | `CompanionStore`, the detection checkpoints beside the ND2 |
| `store.provenance` | the schema version, provenance records, and attributes as plain JSON |
| `steps.registration` | `Registrator`: drift transforms (stackreg or OpenCV) |
| `steps.lanes`, `steps.trenches` | `LaneDetector`, `TrenchDetector`: lane and trench geometry |
| `steps.extraction` | `Extractor`: writes the experiment group |
| `steps.segmentation` | `Segmenter`: Omnipose labels per hypothesis |
| `steps.omnipose` | loading Omnipose models (stock or checkpoint files) and building masks by affinity |
| `widgets` | the notebook UIs the steps open: lane, trench and segmentation tuning, the label viewer |
| `parallel` | where batches run: the active Dask client, a process pool, or this process; `local_cluster`, a notebook's Dask cluster, whose workers outlive the Interrupt button (it then drops the queued batches) |
| `config` | the experiment file |
| `plotting`, `view` | diagnostic figures; the napari viewer |

## Tests

```bash
python -m pytest -n 4 tests                    # all, on 4 workers (pytest-xdist)
python -m pytest -n 4 tests -m "not golden"    # without the golden tests (seconds)
python -m pytest tests -m segment              # the Omnipose tests (the seg environment)
```

Four workers is the fastest: each worker runs the golden pipeline's setup (about 12 s)
itself, so more workers repeat it more often, and `--dist loadfile` runs the golden
pipeline's file on one worker, end to end. In the main repo, run the default and seg
environments at the same time (about 45 s for both).

One test file per module, named after it (`test_steps_extraction.py` for
`pymmm/steps/extraction.py`, `test_widgets_lanes.py` for `pymmm/widgets/lanes.py`). Their
fakes are in `tests/fake_experiment.py`: an ND2 file, experiment groups as the Extractor
writes them, registered FOVs and an Omnipose model. A test that needs an install extra
is marked with it (`notebook`, `segment`, `tracks`, `view`) and skips without it.

`tests/golden` pins today's output on real data: registration, detection, extraction,
segmentation and the experiment-group writers on two FOVs and ten frames of the ty013
ND2, through every compute path. They skip without the ND2 (`PYMMM_GOLDEN_ND2`). After
an intended change of output, regenerate the references with `PYMMM_GOLDEN_UPDATE=1`
and review their diff.

The widget tests drive the controls without a browser and check what the widgets
compute and show (the `notebook` fixture in `tests/conftest.py`, whose ndv viewers are
fakes). `tests/test_widgets_viewers.py` makes real ndv viewers in a real Jupyter kernel
and sends them the events a browser sends (`tests/browser_events.py`): each must handle
them and draw. How the widgets and viewers look in JupyterLab, and napari's window, are
checked by hand.
