# The experiment group

PyMMM writes an experiment as one zarr group, `<experiment>.zarr`: one OME-Zarr image
per trench, with the experiment's metadata beside them. Later steps write their results
(segmentations, tracks, curated tracks) into each trench's image, beside its pixels.
This document specifies that layout: what every reader may rely on, and what every
writer must keep true.

It is Zarr v3 throughout; the trench images and their label images are OME-Zarr 0.5
(NGFF). Chunk keys use `.` as the separator (`0/c.0.0.0.0`), so a chunk is one file,
not nested directories. `schema_version` (now 5) records which version of this layout
wrote a group; nothing checks it.

```
<experiment>.zarr/                  plain group; attrs.pymmm (below)
    T                               (T,) int32: original ND2 frame index of each t
    C                               (C,) string: channel name of each c
    trenches/                       the trench table: one 1-D array per column
    acquisition/  detection/  events/  registration/  source_frames/
                                    experiment metadata (xarray datasets)
    trench_0000/                    OME-Zarr image; attrs.pymmm: its table row
        0                           (t, c, y, x) pixels
        labels/                     OME-Zarr labels group; ome.labels lists the finished ones
            <segmentation>.<hypothesis>     a segmentation's labels, one per hypothesis
            tracks.<run>            the masks a tracking run chose
            curation.<run>          the curated masks
        tracks/<run>.geff           a tracking run's graph (GEFF); regenerable
        curation/<run>.geff         the curated copy, made at the trench's first edit
    trench_0001/  ...
```

`pymmm.store.experiment_group` reads and writes this layout, `pymmm.store.geff` its
tracks (with the geff package), and `pymmm.store.ome` builds every `ome` attribute in it.

## The experiment group

### `attrs.pymmm`

| key | value |
|---|---|
| `layout` | `"experiment"`. A group without it is not an experiment group: the Extractor never overwrites one. |
| `schema_version` | `5` |
| `description` | prose |
| `image_path` | `"trench_{trench:04d}"`: a trench's image, by its id |
| `channels` | channel names, as `C` |
| `pixel_size_um` | µm per pixel of the trench images |
| `frame_interval_ms` | the median gap between the ND2's frame times |
| `experiment_name` | the ND2's file stem |
| `source_nd2`, `source_file_size_bytes`, `source_mtime_ns` | the ND2 the trenches were cut from |
| `source_subset_metadata` | the part of the ND2 extracted: `raw_fov_names`, `active_fov_names`, `discarded_fov_names`, `time_slice` (`start`, `stop`, `step`), `roi` (`y`, `x`), `raw_timepoint_count`, `active_timepoint_count` |
| `provenance` | the code that wrote it: `pymmm_version`, `pymmm_commit`, `pymmm_dirty`, `libraries`, `created` |
| `made_from` | fingerprints of the registration and trench table the trenches were cut with: `registration`, `trench_detection` (see [The companion store](#the-companion-store)). Groups extracted before PyMMM kept this record have none. |
| `completed_fovs` | FOVs whose trenches are fully written, in table order |
| `extraction_complete` | `true` once every FOV is written; until then the trench images may be partly empty |
| `segmentations` | `{<segmentation>: record}`, see [Segmentations](#segmentations) |
| `tracks` | `{<run>: record}`, see [Tracks](#tracks) |
| `fits` | `{<fit>: record}`, written by a tracker that fits itself to the experiment; see [Fits](#fits). Absent until there is one. |

The trench images have the same `t` and `c` axes as `T` and `C`. Frame times are in
`source_frames/` (per FOV, since each FOV is imaged at its own time).

### `trenches/`: the trench table

One row per trench, in trench-id order: 1-D arrays with dimension `trench`. The group's
attributes are `layout: "trench_table"` and a description.

| column | dtype | meaning |
|---|---|---|
| `trench` | int32 | trench id (unique; the table's coordinate) |
| `path` | string | the trench's image, relative to the experiment group (`trench_0007`) |
| `fov_name` | string | the FOV it was cut from |
| `fov_index` | int32 | index into the FOV dimension of `registration/` and `source_frames/` |
| `lane_index` | int32 | its lane within the FOV |
| `x_left`, `x_right`, `y_top`, `y_bottom` | int32 | its crop in the registered FOV, in pixels (half-open) |
| `orientation` | int8 | `1`: the trench opens downward into its lane; `-1`: upward |
| `needs_flip` | bool | the crop was flipped vertically (`orientation == -1`) |

Every trench has the same height and width.

### Metadata groups

xarray datasets, each with `layout` (its name) in its attributes. Variables carry
`long_name` and `units`.

- **`acquisition/`**: per channel (`C`): `color` (`#RRGGBB`), `emission_wavelength`,
  `excitation_wavelength` (nm), `exposure_time` (ms), `modality`; scalars `pixel_size`
  (µm) and `time_interval` (ms); `raw_json`, the ND2's whole metadata as JSON.
  Attributes name the objective (`objective_name`, `objective_magnification`,
  `objective_numerical_aperture`, `immersion_refractive_index`, `zoom_magnification`).
- **`source_frames/`**: per FOV and frame (`FOV`, `T`): `relative_time` (ms since the
  start), `julian_date`, `source_seq_index`, `pfs_offset`, `pfs_status`,
  `stage_position` (`Axis`: x, y, z, µm); per FOV: `original_p_index` and
  `frame_metadata_json`.
- **`events/`**: the ND2's acquisition events (`AcquisitionEvent`), if any.
- **`registration/`**: `tmats` (`FOV`, `T`, `Row`, `Col`): each frame's 3×3 transform,
  and `shift` (`FOV`, `T`, `ShiftAxis`): its translation (y, x) in pixels. Attributes:
  the registration settings (`channel`, `backend`, `mode`, `rotation`, `roi`,
  `downsample`) and `transform_convention`.
- **`detection/`**: attributes only: `lane_detection` and `trench_detection`, each
  `{channel, params}`, the settings that produced the trench table, the detection
  image among them (`mean_n_frames` registered frames, from `mean_from`).

### Transforms and crops

`transform_convention` `raw_translation_then_rotation_v1`: `tmats[fov, t]` maps a pixel
`(x, y, 1)` of the registered frame to the raw ND2 frame (the inverse map of
`skimage.transform.warp`). An output rotation of `rotation` degrees about the frame's
centre composes after it: a trench's pixels are the raw frame warped by
`tmats[fov, t] @ R` (bilinear), then rows `y_top:y_bottom` and columns
`x_left:x_right`, flipped vertically if `needs_flip`. After the flip every trench
opens downward: `y = 0` is the closed end.

## Trench images

`trench_NNNN/` is an OME-Zarr 0.5 image:

- `ome.multiscales`: one level, `0`, with axes `t` (time, minute), `c` (channel), `y`,
  `x` (space, micrometer) and a scale of `[frame_interval_ms / 60000, 1, pixel_size_um,
  pixel_size_um]`. Its `name` is the group's name; `type` is `"none"` (nothing is
  downsampled).
- `ome.omero.channels`: per channel `label`, `color` (`RRGGBB`, from the ND2),
  `active` (only the registration channel), and `window`: `min`/`max` the dtype's range,
  `start`/`end` the trench's 0.5th and 99.9th percentiles.
- `attrs.pymmm`: the trench's table row (`trench`, `fov_name`, `fov_index`,
  `lane_index`, `x_left`, `x_right`, `y_top`, `y_bottom`, `orientation`,
  `needs_flip`), `units` of the crop (pixel) and `experiment: ".."`.

`0` is `(t, c, y, x)`, the ND2's dtype (uint16), with `dimension_names` `t c y x`. It is
one shard of the whole image holding one inner chunk per channel (and time chunk, if
the extraction set one), Blosc zstd with byte shuffle, so reading one channel reads only
that channel's chunk.

## Label images

Every label image is an OME-Zarr image-label `(t, y, x)` in `trench_NNNN/labels/`:

- `ome.multiscales`: axes `t` (minute), `y`, `x` (micrometer), scale
  `[minutes per frame, µm per pixel, µm per pixel]`, one level `0`.
- `ome.image-label.source.image`: `"../../"`.
- `0`: `(t, y, x)`, integer (uint16 by default), one chunk, Blosc zstd with bit
  shuffle; fill value 0. 0 is background.
- `attrs.pymmm`: the writer's notes about these labels.

A label image is (t, y, x), like a trench channel once a viewer splits the channels, so
viewers line the labels up with the image by position.

**A label image counts once it is listed** in `labels/`'s `ome.labels`. Writers list a
label image only after its pixels are in, and all of a segmentation's hypotheses
together, so a reader never takes half a segmentation for a finished one. Removing a
segmentation unlists it before deleting any pixels.

Names: `<segmentation>.<hypothesis>`, `tracks.<run>`, `curation.<run>`.

## Segmentations

A segmentation is a fixed set of *hypotheses*: alternative labels of the same trench (a
model checkpoint each, or a model under different settings), the base first. Each
trench gets one label image per hypothesis, `labels/<segmentation>.<hypothesis>`. A
trench's segmentation is finished when all of them are listed.

The experiment group records it once, `attrs.pymmm.segmentations[<segmentation>]`:

| key | |
|---|---|
| `hypotheses` | required: the hypotheses' names, the base first |
| `channel` | the channel segmented |
| `description`, `made_by`, `created`, `provenance` | what made it, and when |
| `params` | its settings |

The Segmenter also records `model`, `settings` (each hypothesis's settings),
`schema_version` and `extraction_created`: the extraction it segmented. Its label values
are 1..n per frame from the closed end (1 nearest `y = 0`), and each label image's
`pymmm` attributes hold `segmentation`, `hypothesis`, `hypothesis_index`, `channel`,
`model`, the trench's intensity bounds `lower` and `upper`, and `failed_frames` (frames
Omnipose failed on, left empty).

## Tracks

A tracking run writes, for each trench it tracks:

- `labels/tracks.<run>`: the masks it chose, `(t, y, x)` uint16. A mask's value is its
  node's `seg_id`; values are unique within a frame. `attrs.pymmm.tracks` is
  `"../../tracks/<run>.geff"`.
- `tracks/<run>.geff`: its graph, a GEFF 1.3 (Zarr v3) group:
  - directed; axes `t` (frame), `y`, `x` (pixels in the trench, no scale: the trench
    image carries the calibration, and a reader that applied a scale would miss the
    masks);
  - node ids 1..n (0 is the masks' background to readers such as funtracks; GEFFs
    written before 2026-11-05 number from 0);
  - node props `t` (int32), `y`, `x` (float32, the mask's centroid), `seg_id` (int32),
    `track_id` and `lineage_id` (int64, GEFF tracklets and lineages: a node continues
    its parent's tracklet when it is the parent's only child and has no other parent;
    a lineage is a connected component; both numbered from 1, since mm-track takes 0
    for no track), named in `track_node_props`;
  - edge prop `score` (float32, may be NaN);
  - `related_objects`: the labels `../../labels/tracks.<run>` (by `seg_id`) and the
    image `../..`;
  - `extra`: the tracker's provenance, `trench`, `pixel_size_um`, `frame_interval_min`.

`tracks/` (attributes: a description) holds only regenerable output. A trench has a
run's tracks when its GEFF exists and `tracks.<run>` is listed.

The experiment group records each run, `attrs.pymmm.tracks[<run>]`: `tracker`,
`segmentation` (the one it tracked), `params`, `made_by`, `created` and, for
chaintrack, `alternatives` (which hypotheses it could choose from).

chaintrack's runs since 2026-11-11 also record:

| key | |
|---|---|
| `fit` | the fit the run used: `name`, `values` and `created`, copied from its [fit record](#fits) |
| `recipe` | that fit's recipe: the tracker's hand-set constants, in µm and minutes |
| `params` | the parameters as the tracker used them, in the trench images' pixels and frames |
| `pixel_size_um`, `frame_interval_min` | the calibration that converted one to the other |
| `chaintrack` | the tracker's code: `commit`, and `dirty` (tracked files had uncommitted changes) |
| `continued` | present once trenches were added to the run later: a list of `{at, chaintrack}` |

A run is continued only with the fit, segmentation and alternatives it was started
with, so its record holds for every trench of the run. Each trench's GEFF `extra` holds
the parameters used for that trench as well.

## Fits

A tracker may fit some of its parameters to the experiment's own cells before tracking
it. chaintrack records each fit, `attrs.pymmm.fits[<fit>]` (the default name is the
segmentation's):

| key | |
|---|---|
| `tracker`, `segmentation` | what was fitted, on which segmentation's masks |
| `values` | the fitted values, in µm and minutes |
| `recipe` | the tracker's hand-set constants as they were at the fit, in µm and minutes. This copy is frozen: runs that use the fit track with it, not with the recipe file it came from. |
| `fit_start` | the values the fit started from, when it started from the recipe |
| `trenches`, `frames`, `rounds`, `started_from` | what the fit was made on, and how |
| `pixel_size_um`, `frame_interval_min` | the calibration at the fit |
| `made_by`, `chaintrack`, `created`, `note` | what made it, and when |

A fit that a run has used is not overwritten: another fit of the same experiment gets
another name. A fit made before fits were recorded (2026-11-11) may lack `fit_start`
and the tracker's commit, and says so in its `note`.

## Curation

The tracker's output is never edited. mm-track's first edit of a trench copies its
tracks to `curation/<run>.geff` and `labels/curation.<run>` (same format, `related_objects`
pointing at `curation.<run>`), and every edit goes there. A trench opens as its curated
copy once it has one.

An edit does not make a trench curated: the curator marks it reviewed, a stamp in the
curated GEFF's `geff.extra`:

```json
"reviewed": {"at": "2026-11-05T14:02:11+04:00", "by": "georgeos", "revision": 12}
```

`revision` is the edit-history revision that was reviewed. Only reviewed trenches are
ground truth.

## Writing safely

- Extraction writes the group's metadata and every trench image's group first, then
  each FOV's pixels; `completed_fovs` grows as FOVs finish, and an interrupted
  extraction resumes from it if nothing it records has changed.
- A label image is listed only after its pixels are written; a segmentation's
  hypotheses are listed together.
- Each trench is written by one process at a time; processes write different trenches.
- `update_experiment` replaces one record of `segmentations`, `tracks` or `fits` and
  keeps the others.

## The companion store

Detection checkpoints are kept beside the ND2, in `<experiment_name>.pymmm.zarr`, so a
notebook can resume without redoing registration or detection:

```
<experiment_name>.pymmm.zarr/       attrs: layout "pymmm_companion", the source ND2, created, provenance
    registration/                   tmats (P, T, row, col)
    lane_detection/                 one row per lane: fov, lane_index, y_center, orientation
    trench_detection/               one row per trench: fov, lane_index, x_left, x_right,
                                    y_top, y_bottom, orientation, needs_flip
```

Each section is an xarray dataset. Its attributes hold the `params` that produced it,
the `source_subset_metadata` it was computed on, `provenance`, `made_from`, and
`complete`, set only once the whole section is written.

`made_from` holds the fingerprints of what the section was made from. Registration
records `source`, lane detection records `registration`, and trench detection records
`registration` and `lane_detection`. The experiment group's `made_from` (above) links
the extraction into the same chain. A fingerprint is a short SHA-256 of a result, not
of its settings, and `pymmm.store.lineage` computes it:

| fingerprint of | covers |
|---|---|
| `source` | the ND2's size in bytes and `source_subset_metadata` (not its path or modification time) |
| `registration` | `tmats` as float64 in `(P, T, row, col)` order, the FOV names, and `rotation` |
| `lane_detection`, `trench_detection` | the table's columns (`fov`, `lane_index`, `y_center`, `orientation`; the trench table's as in the experiment group), whatever their dtypes, rows sorted |

A section is current when every fingerprint it records matches what is saved now and
everything it was made from is current. `CompanionStore.lineage(experiment,
experiment_group)` gives each step's status: `current`, `stale`, `missing`, or
`unrecorded` for a section saved before these records were kept.
