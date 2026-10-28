from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def payload(vibration=40):
    return {
        "machine_model": "model3",
        "age": 12,
        "readings": [{"volt": 170, "rotate": 446, "pressure": 100, "vibration": vibration}
                     for _ in range(24)],
        "errors_last_24h": {"error1": 1},
        "days_since_maintenance": {"comp1": 30, "comp2": 60, "comp3": 20, "comp4": 90},
    }


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict():
    r = client.post("/predict", json=payload())
    assert r.status_code == 200
    assert 0 <= r.json()["failure_probability"] <= 1


def test_bad_input():
    bad = payload()
    bad["readings"] = bad["readings"][:10]
    assert client.post("/predict", json=bad).status_code == 422