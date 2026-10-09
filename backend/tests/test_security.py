import io
import zipfile


def test_upload_id_traversal_rejected(client):
    from app import config

    marker = config.UPLOAD_DIR.parent / "keep.txt"
    marker.write_text("x")
    r = client.delete("/api/datasets/upload_zip/%2E%2E")
    assert r.status_code == 400
    assert marker.exists()


def test_chunk_upload_requires_known_session(client):
    r = client.put("/api/datasets/upload_zip/chunk/00000000-0000-0000-0000-000000000000/0", content=b"x")
    assert r.status_code == 404


def test_api_token(client, monkeypatch):
    from app import config

    monkeypatch.setattr(config, "API_TOKEN", "s3cret")
    assert client.get("/api/datasets/list_datasets").status_code == 401
    assert client.get("/api/datasets/list_datasets", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/api/datasets/list_datasets?api_key=s3cret").status_code == 200
    assert client.get("/api/health").status_code == 200


def test_export_zip_paths_are_sanitized(client):
    from app.services.shared_state import data_manager as dm

    ds = dm.create_dataset("evil", "IMAGE_CLASSIFICATION")["id"]
    dm.add_sample(ds, "../../etc", "IMAGE_CLASSIFICATION", b"data", "../x.png")
    r = client.get(f"/api/datasets/export/full/{ds}")
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert names and all(".." not in n.split("/") and not n.startswith("/") for n in names)


def test_exporter_escapes_labels_and_validates_pins():
    import pytest

    from app.services import exporter

    arr = exporter._labels_array_cpp(['say "hi"', "back\\slash", "caffè"])
    assert '\\"hi\\"' in arr and "back\\\\slash" in arr and "\\x" in arr
    with pytest.raises(ValueError):
        exporter._validate_pins({"cs": "5; system()"})
    with pytest.raises(ValueError):
        exporter._validate_pins({"cs": 999})


def test_resolver_method_names():
    from app.services.exporter import _resolver_method

    assert _resolver_method("CONV_2D") == "AddConv2D"
    assert _resolver_method("DEPTHWISE_CONV_2D") == "AddDepthwiseConv2D"
    assert _resolver_method("FULLY_CONNECTED") == "AddFullyConnected"
    assert _resolver_method("HARD_SWISH") == "AddHardSwish"
    assert _resolver_method("UNIDIRECTIONAL_SEQUENCE_LSTM") == "AddUnidirectionalSequenceLSTM"
