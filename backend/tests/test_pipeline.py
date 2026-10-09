"""End-to-end: dataset -> training -> INT8 optimization -> evaluation ->
Arduino export -> live inference, on tiny synthetic data (CPU, ~1 min)."""

import io
import zipfile

import pytest

from tests.conftest import make_png, make_wav


def _train(task, base_model, input_shape, samples, epochs=4):
    from app.services.shared_state import data_manager as dm, trainer

    ds = dm.create_dataset(f"{task}-pipeline", task)["id"]
    for label, raw, name, anns in samples:
        dm.add_sample(ds, label, task, raw, name, annotations=anns)
    dm.auto_split_dataset(ds, 70, 15, 15)
    tid = trainer.create_training_session(task, ds, epochs, 8, 0.003, base_model, input_shape, seed=0)
    trainer.train(tid)  # synchronous here; the API runs it on the job queue
    return tid, trainer.training_sessions[tid]


def _optimize(tid, method="int8"):
    from app.services import optimizer

    oid = optimizer.create_optimization_session(tid, method, quantization="int8", fine_tune_epochs=1)
    optimizer.optimize(oid)
    return oid, optimizer.get_session(oid)


@pytest.mark.slow
def test_image_classification_pipeline(client):
    samples = [(c, make_png(color), f"{c}{i}.png", None)
               for i in range(24) for c, color in (("red", (220, 20, 20)), ("blue", (20, 20, 220)))]
    tid, session = _train("IMAGE_CLASSIFICATION", "Custom3LayerCNN", [32, 32, 3], samples)
    assert session["status"] == "completed", session.get("error")
    assert session["evaluation"]["accuracy"] >= 0.9
    assert len(session["evaluation"]["confusion_matrix"]) == 2

    oid, opt = _optimize(tid)
    assert opt["status"] == "completed", opt.get("error")
    comparison = opt["comparison"]
    assert comparison["optimized"]["accuracy"] >= 0.9
    assert opt["optimized_size_bytes"] < opt["original_size_bytes"]
    assert "CONV_2D" in opt["metrics"]["ops"]

    from app.services import exporter

    pkg = zipfile.ZipFile(io.BytesIO(exporter.generate_export_package(oid, "ESP32_S3_N16R8")))
    sketch = pkg.read("sketch.ino").decode()
    assert "MicroMutableOpResolver" in sketch and "AddConv2D" in sketch
    assert "heap_caps_aligned_alloc" in sketch

    r = client.post("/api/inference/run", data={"training_id": tid, "optimization_id": oid},
                    files={"file": ("x.png", make_png((220, 20, 20)), "image/png")})
    assert r.status_code == 200, r.text
    assert r.json()["result"]["top_class"] == "red"


@pytest.mark.slow
def test_keyword_spotting_pipeline(client):
    samples = [(lbl, make_wav(freq), f"{lbl}{i}.wav", None)
               for i in range(20) for lbl, freq in (("low", 300.0), ("high", 2500.0))]
    tid, session = _train("KEYWORD_SPOTTING", "MFCC_CNN", None, samples, epochs=6)
    assert session["status"] == "completed", session.get("error")
    assert session["input_shape"] == [40, 49, 1]

    oid, opt = _optimize(tid)
    assert opt["status"] == "completed", opt.get("error")

    from app.services import exporter

    pkg = zipfile.ZipFile(io.BytesIO(exporter.generate_export_package(oid, "ESP32_S3_N16R8")))
    assert {"mfcc_frontend.h", "send_wav.py"} <= set(pkg.namelist())
    assert "i2s_read" in pkg.read("sketch.ino").decode()


@pytest.mark.slow
def test_object_detection_pipeline():
    import numpy as np

    rng = np.random.default_rng(0)
    samples = []
    for i in range(40):
        x, y = int(rng.integers(6, 26)), int(rng.integers(6, 26))
        samples.append(("obj", make_png((30, 30, 30), box=(x, y)), f"o{i}.png",
                        [{"class_name": "chip", "cx": x / 32, "cy": y / 32, "w": .25, "h": .25}]))
    tid, session = _train("OBJECT_DETECTION", "FOMO_Tiny", [32, 32, 3], samples, epochs=3)
    assert session["status"] == "completed", session.get("error")
    assert session["labels"] == ["chip"]
    assert "f1" in session["evaluation"]
    oid, opt = _optimize(tid)
    assert opt["status"] == "completed", opt.get("error")
