import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestHealth:
    def test_health_check(self):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_backend_info(self):
        response = client.get("/api/info")
        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data
        assert "boards" in data
        assert len(data["tasks"]) > 0

class TestDatasets:
    def test_list_empty_datasets(self):
        response = client.get("/api/datasets/list")
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_dataset_statistics(self):
        response = client.get("/api/datasets/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_samples" in data

class TestTraining:
    def test_list_models(self):
        response = client.get("/api/training/models")
        assert response.status_code == 200
        assert response.json()["status"] == "success"

class TestOptimization:
    def test_get_supported_boards(self):
        response = client.get("/api/optimization/boards")
        assert response.status_code == 200
        data = response.json()
        assert "boards" in data
        assert len(data["boards"]) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
