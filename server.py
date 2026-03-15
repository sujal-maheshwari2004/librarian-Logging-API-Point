from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import time

app = FastAPI()

metrics_store = {}

MAX_POINTS = 5000
MAX_RUNS = 20


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


def downsample(data, target=800):
    if len(data) <= target:
        return data
    step = len(data) // target
    return data[::step]


@app.post("/train_metrics")
async def receive_metrics(metric: Metric):

    run = metrics_store.setdefault(metric.run_id, [])

    run.append(metric.dict())

    if len(run) > MAX_POINTS:
        run.pop(0)

    if len(metrics_store) > MAX_RUNS:
        oldest = list(metrics_store.keys())[0]
        del metrics_store[oldest]

    return {"status": "ok"}


@app.get("/metrics")
async def get_metrics(run_id: int = 1):

    run = metrics_store.get(run_id, [])

    return downsample(run)


@app.post("/clear")
async def clear(run_id: int | None = None):

    if run_id is None:
        metrics_store.clear()
    else:
        metrics_store.pop(run_id, None)

    return {"status": "cleared"}


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    with open("static/index.html") as f:
        return f.read()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "training-dashboard",
        "timestamp": time.time()
    }


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
