from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

art = joblib.load(Path(__file__).with_name("model.joblib"))
pipe, THRESHOLD, FEATURES = art["model"], art["threshold"], art["features"]

SENSORS = ["volt", "rotate", "pressure", "vibration"]
app = FastAPI(title="Machine Failure Prediction")


class Reading(BaseModel):
    volt: float
    rotate: float
    pressure: float
    vibration: float


class Machine(BaseModel):
    machine_model: Literal["model1", "model2", "model3", "model4"]
    age: int
    readings: list[Reading] = Field(min_length=24, max_length=24)  # last 24 ghante, purana pehle
    errors_last_24h: dict[str, int] = {}          # e.g. {"error1": 2}
    days_since_maintenance: dict[str, float] = {}  # e.g. {"comp1": 30}


def make_features(m: Machine) -> pd.DataFrame:
    df = pd.DataFrame([r.model_dump() for r in m.readings])
    row = {}
    for w in (3, 24):
        tail = df.tail(w)
        for s in SENSORS:
            row[f"{s}_mean_{w}h"] = tail[s].mean()
            row[f"{s}_std_{w}h"] = tail[s].std()
    for i in range(1, 6):
        row[f"error{i}_24h"] = m.errors_last_24h.get(f"error{i}", 0)
    for i in range(1, 5):
        row[f"days_since_comp{i}"] = m.days_since_maintenance.get(f"comp{i}", 365.0)
    row["age"] = m.age
    row["model"] = m.machine_model
    return pd.DataFrame([row])[FEATURES]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(m: Machine):
    prob = float(pipe.predict_proba(make_features(m))[0, 1])
    return {"failure_probability": round(prob, 4), "alert": prob >= THRESHOLD}