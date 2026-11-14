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
with the `notebook`, `view` and `tracks` extras).

## The pipeline

```python
from pymmm import (CompanionStore, Extractor, LaneDetector, ND2Experiment, Registrator,
                   ShadingCorrector, TrenchDetector)

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

shading = ShadingCorrector(exp, reg, trenches, store, mode="subtract")  # or "divide"
shading.fit()                                     # each FOV's background, on the PDMS
shading.plot()                                    # how the cells follow it
shading.save()

extractor = Extractor(exp, reg, trenches, shading, output_path="experiment.zarr")
extractor.preview()                               # lazy: the trenches as they will be written
extractor.extract()                               # resumes an interrupted run
```

Each step saves its result to the companion store, `<experiment>.pymmm.zarr` beside the
ND2, and `Registrator.load`, `LaneDetector.load`, `TrenchDetector.load` and
`ShadingCorrector.load` pick it up again. Without widgets, `detect_lanes(...)` and
`detect_trenches(...)` take the settings directly. With `use_image_edge`, each FOV's
lane is the image's edge where the trenches open, and trench detection takes the image's
whole height by default. The PDMS between neighbouring trenches is the same material
everywhere, so `ShadingCorrector` takes a smooth surface through it as each FOV's
background and extraction takes that out of the detection channel, every frame at its
own level. Subtract it when the cells brighten one for one with it (light added, as
scattered by the chip); divide it out when they brighten in proportion (a gain).
`plot()` shows which. Registration and extraction run on the active Dask client if there
is one, else in a process pool.

The same steps up to the preview also run in napari:

```bash
python -m pymmm.gui experiments/experiment.toml    # an experiment file, or an ND2
```

The dock has one tab per step: Experiment, Register, Lanes, Trenches, Shading and
Preview. Each tab's settings act on the field of view under napari's P slider as you
change them. Apply saves to the same companion store the notebook uses, so either one
can pick up where the other stopped. Frames are read only when napari shows them. The
preview cuts the trench and frame on screen, and nothing is written. Extraction itself
is still `extractor.extract()` in the notebook. In napari, the dock is also under
Plugins → PyMMM → Extraction.

After extraction:

- `Segmenter(path, channel, ...)` labels every trench with Omnipose, one label image
  per *hypothesis*, the base first. A hypothesis is a model under some settings: a stock
  model under the S1–S4 presets, or a fine-tuned checkpoint (`checkpoint_hypotheses`).
  It runs on the active Dask client's workers if there is one, else in a process pool or
  in this process. A Dask worker on another machine needs only the environment and
  PyMMM: each task carries a trench's frames and returns its labels, the checkpoints go
  to each worker once, and the notebook alone writes the experiment group.
- `Tracker(path, segmentation, chaintrack=".../chain.py")` tracks every trench with
  [chaintrack](https://github.com/georgeoshardo/chaintrack), a program kept outside PyMMM
  (clone it and give the path to its `chain.py`): `fit(trenches, recipe)` measures the
  experiment's own growth, division length and noise once and records them in the group,
  and `track()` writes each trench's tracks, a GEFF graph and the masks it chose
  (`pymmm.store.geff.write_tracks`). `track()` resumes with the trenches not tracked
  yet, and `plot(trench)` draws a trench's kymograph with its tracks. chaintrack tracks
  several trenches at once in its own processes, not through a Dask cluster.
- mm-track curates the tracks in a curated copy beside them.

`pymmm.config.Experiment` reads an experiment file (TOML) naming the ND2, the experiment
group and the segmentation settings, so every notebook of the pipeline starts from the
same file. `experiment.segmenter()` and `experiment.tracker(chaintrack)` build the two
steps from it.

## Reading an experiment group

```python
from pymmm.store import experiment_group as eg

root = eg.open_experiment("experiment.zarr")
for trench in eg.trench_ids(root):
    pixels = eg.image(root, trench)                    # (t, c, y, x) zarr array
    names, labels = eg.segmentation(root, trench, "pc")
```

`pymmm.store` needs only numpy and zarr. To browse every trench in napari:

```bash
python -m pymmm.view experiment.zarr                   # one slider over all trenches
napari --plugin napari-ome-zarr experiment.zarr/trench_0007
```

PyMMM is also a napari plugin: drop `experiment.zarr` onto napari and choose PyMMM to get the same
layers. napari offers every folder reader for any folder, so it asks each time; its "remember this
choice" box remembers that one folder.
