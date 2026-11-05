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
| `view` | napari, napari-ome-zarr | `python -m pymmm.view` and single trench images in napari |
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
settings directly. Registration and extraction run on the active Dask client if there
is one, else in a process pool.

After extraction:

- `Segmenter(path, channel, model)` labels every trench with Omnipose, one label image
  per *hypothesis* (alternative settings), the base first.
- A tracker writes each trench's tracks with `trench_images.write_tracks`: the masks it
  chose and a GEFF graph.
- mm-track curates the tracks in a curated copy beside them.

`pymmm.config.Experiment` reads an experiment file (TOML) naming the ND2, the experiment
group and the segmentation settings, so every notebook of the pipeline starts from the
same file.

## Reading an experiment group

```python
from pymmm import trench_images as ti

root = ti.open_experiment("experiment.zarr")
for trench in ti.trench_ids(root):
    pixels = ti.image(root, trench)                    # (t, c, y, x) zarr array
    names, labels = ti.segmentation(root, trench, "pc_affinity_ensemble")
```

A core install is enough to read and write it. To browse every trench in napari:

```bash
python -m pymmm.view experiment.zarr                   # one slider over all trenches
napari --plugin napari-ome-zarr experiment.zarr/trench_0007
```

## Modules

| module | |
|---|---|
| `experiment` | `ND2Experiment`: the ND2 as a lazy xarray, and its subsets |
| `metadata` | ND2 acquisition and per-frame metadata |
| `registrator` | `Registrator`: drift transforms (stackreg or OpenCV) |
| `lane_detector`, `trench_detector` | lane and trench geometry, with their widgets |
| `extractor` | `Extractor`: writes the experiment group |
| `segmenter` | `Segmenter`: Omnipose labels per hypothesis, with its widget |
| `trench_images` | the experiment group: open, read, and write labels and tracks |
| `tracks` | GEFF tracklet and lineage ids, and reading GEFFs as arrays |
| `checkpoint` | `CompanionStore`, the detection checkpoints |
| `store.ome` | every OME-Zarr attribute PyMMM writes |
| `parallel` | where batches run: the active Dask client, a process pool, or this process |
| `config` | the experiment file |
| `provenance` | the schema version, provenance records, and attributes as plain JSON |
| `plotting`, `view` | diagnostic figures; the napari viewer |

## Tests

```bash
python -m pytest tests                         # all
python -m pytest tests -m "not golden"         # without the golden tests (seconds)
```

`tests/golden` pins today's output on real data: registration, detection, extraction,
segmentation and the experiment-group writers on two FOVs and ten frames of the ty013
ND2, through every compute path. They skip without the ND2 (`PYMMM_GOLDEN_ND2`), and
the segmentation and GEFF tests skip without omnipose or geff. After an intended change
of output, regenerate the references with `PYMMM_GOLDEN_UPDATE=1` and review their
diff.
