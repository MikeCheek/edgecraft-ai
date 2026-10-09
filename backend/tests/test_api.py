class TestHealth:
    def test_health_check(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert "auth_required" in body

    def test_backend_info(self, client):
        data = client.get("/api/info").json()
        assert len(data["tasks"]) == 5
        assert "DS_CNN" in data["models"]["KEYWORD_SPOTTING"]
        assert data["default_input_shapes"]["KEYWORD_SPOTTING"] == [40, 49, 1]


class TestDatasets:
    def test_list_datasets(self, client):
        response = client.get("/api/datasets/list_datasets")
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_dataset_statistics(self, client):
        data = client.get("/api/datasets/stats").json()
        assert "total_samples" in data


class TestTraining:
    def test_list_models(self, client):
        response = client.get("/api/training/models")
        assert response.json()["status"] == "success"

    def test_queue_endpoint(self, client):
        assert client.get("/api/training/queue").json()["status"] == "success"

    def test_rejects_unsplit_dataset(self, client):
        ds = client.post("/api/datasets/create", json={"name": "unsplit", "task": "IMAGE_CLASSIFICATION"}).json()
        dataset_id = ds.get("dataset", ds).get("id") if isinstance(ds.get("dataset", ds), dict) else None
        if dataset_id is None:
            dataset_id = client.get("/api/datasets/list_datasets").json()["datasets"][-1]["id"]
        r = client.post("/api/training/start", json={"task": "IMAGE_CLASSIFICATION", "dataset_id": dataset_id})
        assert r.json()["status"] == "error"


class TestOptimization:
    def test_get_supported_boards(self, client):
        data = client.get("/api/optimization/boards").json()
        assert len(data["boards"]) == 4

    def test_unknown_training_id(self, client):
        r = client.post("/api/optimization/quantize", json={"training_id": "nope", "method": "INT8_QUANTIZATION"})
        assert r.status_code == 404
