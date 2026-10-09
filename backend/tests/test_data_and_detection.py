import numpy as np

from app.services import dataset_loader
from app.services.detection import boxes_to_grid, decode_predictions, score_detections
from tests.conftest import make_png


def _dm():
    from app.services.shared_state import data_manager

    return data_manager


def test_relabel_and_empty_labels_are_not_trained():
    dm = _dm()
    ds = dm.create_dataset("relabel", "IMAGE_CLASSIFICATION")["id"]
    for fn in ["cat_1.png", "cat_2.png", "dog_1.png", "dog_2.png"]:
        dm.add_sample(ds, "unsorted", "IMAGE_CLASSIFICATION", make_png(), fn)
    assert dm.bulk_relabel_by_regex(ds, r"^([a-z]+)_") == 4
    dm.auto_split_dataset(ds, 100, 0, 0)
    kept, dropped = dataset_loader.classification_labels(ds)
    assert kept == ["cat", "dog"]
    assert "unsorted" in dropped


def test_rename_label_keeps_indices_consistent():
    dm = _dm()
    ds = dm.create_dataset("rename", "IMAGE_CLASSIFICATION")["id"]
    for lbl in ["a", "a", "b"]:
        dm.add_sample(ds, lbl, "IMAGE_CLASSIFICATION", make_png(), f"{lbl}.png")
    assert dm.rename_label(ds, "a", "z") == 2
    assert dm.get_dataset_labels(ds) == ["b", "z"]
    assert dm.delete_label(ds, "a") == 0
    assert len(dm.get_samples(ds)) == 3


def test_fomo_targets_decode_and_score():
    grid = boxes_to_grid([{"class_name": "cup", "cx": 0.55, "cy": 0.3, "w": .1, "h": .1}], ["cup", "pen"], (8, 8))
    assert grid[2, 4] == 1 and grid.sum() == 1

    probs = np.zeros((8, 8, 3), np.float32)
    probs[..., 0] = 1.0
    probs[2, 4] = [0.1, 0.9, 0.0]   # correct
    probs[6, 6] = [0.2, 0.0, 0.8]   # false positive (pen)
    dets = decode_predictions(probs)
    assert {d["class_index"] for d in dets} == {0, 1}
    score = score_detections([grid], [probs], 2)
    assert score["tp"] == 1 and score["fp"] == 1 and score["fn"] == 0
    assert score["per_class"][0]["f1"] == 1.0


def test_quality_report_finds_leakage_and_removes_duplicates(client):
    from tests.conftest import make_png

    dm = _dm()
    ds = dm.create_dataset("quality", "IMAGE_CLASSIFICATION")["id"]
    same = make_png((10, 200, 10))
    ids = [dm.add_sample(ds, "a", "IMAGE_CLASSIFICATION", same, f"dup{i}.png") for i in range(2)]
    dm.add_sample(ds, "b", "IMAGE_CLASSIFICATION", make_png((200, 10, 10)), "b.png")
    dm.add_sample(ds, "b", "IMAGE_CLASSIFICATION", b"not an image", "broken.png")
    dm.set_sample_split(ids[0], "train")
    dm.set_sample_split(ids[1], "test")

    report = client.get(f"/api/datasets/{ds}/quality").json()["report"]
    codes = {i["code"] for i in report["issues"]}
    assert {"leakage", "unreadable", "small_classes"} <= codes
    assert report["leakage_group_count"] == 1 and report["unreadable_count"] == 1

    assert client.post(f"/api/datasets/{ds}/remove_duplicates").json()["removed"] == 1
    remaining = {s["id"]: s["split"] for s in dm.get_samples(ds)}
    assert ids[0] in remaining and ids[1] not in remaining  # the train copy is kept
