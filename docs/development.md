# Modules and tests

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
| `steps.shading` | `ShadingCorrector`: each FOV's shading, measured on the PDMS between trenches, for extraction to take out of the detection channel |
| `steps.extraction` | `Extractor`: writes the experiment group |
| `steps.segmentation` | `Segmenter`: Omnipose labels per hypothesis |
| `steps.tracking` | `Tracker`: runs chaintrack on the experiment group (fit, track, resume, kymograph) |
| `steps.omnipose` | loading Omnipose models (stock or checkpoint files) and building masks by affinity |
| `widgets` | the notebook UIs the steps open: lane, trench and segmentation tuning, the label viewer |
| `gui.extraction` | the extraction dock in napari: the notebook's steps up to the trench preview (`python -m pymmm.gui`) |
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
