import numpy as np

from pymmm.store import ome


def test_image_has_units_only_for_a_known_calibration():
    known = ome.image(
        "trench_0001",
        channels=["PC", "YFP"],
        colors=["FFFFFF", "FFFF00"],
        active="PC",
        dtype=np.dtype("uint16"),
        frame_interval_ms=120_000.0,
        pixel_size_um=0.1,
    )
    (multiscale,) = known["multiscales"]
    assert [a.get("unit") for a in multiscale["axes"]] == [
        "minute",
        None,
        "micrometer",
        "micrometer",
    ]
    assert multiscale["datasets"][0]["coordinateTransformations"][0]["scale"] == [
        2.0,
        1.0,
        0.1,
        0.1,
    ]
    assert [c["active"] for c in known["omero"]["channels"]] == [True, False]
    assert known["omero"]["channels"][1]["window"] == {
        "min": 0.0,
        "max": 65535.0,
        "start": 0.0,
        "end": 65535.0,
    }

    unknown = ome.image(
        "trench_0001",
        channels=["PC"],
        colors=["FFFFFF"],
        active="PC",
        dtype=np.dtype("float32"),
        frame_interval_ms=float("nan"),
        pixel_size_um=0.0,
    )
    (multiscale,) = unknown["multiscales"]
    assert all("unit" not in a for a in multiscale["axes"])
    assert (
        multiscale["datasets"][0]["coordinateTransformations"][0]["scale"] == [1.0] * 4
    )
    assert unknown["omero"]["channels"][0]["window"]["max"] == 1.0


def test_with_windows_leaves_its_input_alone():
    attrs = ome.image(
        "t",
        channels=["PC"],
        colors=["FFFFFF"],
        active="PC",
        dtype=np.dtype("uint16"),
        frame_interval_ms=60_000.0,
        pixel_size_um=0.1,
    )
    pixels = np.arange(2 * 1 * 4 * 4, dtype=np.uint16).reshape(2, 1, 4, 4)
    windowed = ome.with_windows(attrs, ome.display_windows(pixels))
    assert attrs["omero"]["channels"][0]["window"]["end"] == 65535.0
    window = windowed["omero"]["channels"][0]["window"]
    assert (window["start"], window["end"]) == tuple(np.percentile(pixels, [0.5, 99.9]))


def test_labels_lists_names_in_order():
    assert ome.labels(["b", "a"]) == {"version": "0.5", "labels": ["b", "a"]}
    label = ome.label_image("seg.S1", minutes_per_frame=1.0, um_per_pixel=0.1)
    assert label["image-label"] == {"source": {"image": "../../"}}
    assert label["multiscales"][0]["name"] == "seg.S1"
