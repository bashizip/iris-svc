import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

app = FastAPI(title="Iris Classifier", version="0.1.0")
bundle = joblib.load("model.joblib")
model = bundle["model"]
target_names = bundle["target_names"]

PREDICTIONS = Counter("predictions_total", "Number of predictions served")

class Features(BaseModel):
    # 4 floats: sepal_length, sepal_width, petal_length, petal_width
    x: list[float] = Field(..., min_items=4, max_items=4)

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/predict")
def predict(f: Features):
    pred = model.predict([f.x])[0]
    PREDICTIONS.inc()
    return {"class_index": int(pred), "class_name": target_names[int(pred)]}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
