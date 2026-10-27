from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict():
    response = client.post(
        "/predict",
        json={
            "text": "I forgot my password"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "intent" in data
    assert "category" in data
    assert "confidence" in data

    assert 0 <= data["confidence"] <= 1


def test_empty_text():
    response = client.post(
        "/predict",
        json={
            "text": ""
        },
    )

    assert response.status_code == 422


def test_missing_text():
    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 422