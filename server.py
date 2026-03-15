from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

app = FastAPI()

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


@app.post("/train_metrics")
async def receive_metrics(metric: Metric):

    metrics_store.append(metric.dict())

    if len(metrics_store) > 2000:
        metrics_store.pop(0)

    return {"status": "ok"}


@app.get("/metrics")
async def get_metrics():
    return metrics_store


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    with open("static/index.html") as f:
        return f.read()

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "training-dashboard",
        "timestamp": __import__("time").time()
    }


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
