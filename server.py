from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import uvicorn

app = FastAPI()

# In-memory metric storage
metrics_store = []


class Metric(BaseModel):
    run_id: int
    type: str
    step: int
    timestamp: float

    loss: float | None = None
    val_loss: float | None = None
    lr: float | None = None
    grad_norm: float | None = None
    tokens_per_sec: int | None = None
    gpu_mem_gb: float | None = None


# Receive training metrics
@app.post("/train_metrics")
async def receive_metrics(metric: Metric):

    metrics_store.append(metric.dict())

    # keep only last 1000 metrics
    if len(metrics_store) > 1000:
        metrics_store.pop(0)

    return {"status": "ok"}


# Endpoint to retrieve metrics
@app.get("/metrics")
async def get_metrics():

    return metrics_store


# Serve dashboard HTML
@app.get("/", response_class=HTMLResponse)
async def dashboard():

    with open("static/index.html") as f:
        return f.read()


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
