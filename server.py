from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn
import time
import os
import math
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# ------------------------------------------------
# Auth
# ------------------------------------------------
DASHBOARD_KEY = os.environ.get("DASHBOARD_KEY", "changeme")
security = HTTPBearer(auto_error=False)


def verify_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials is None or credentials.credentials != DASHBOARD_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing key")
    return credentials.credentials


# ------------------------------------------------
# Storage
# ------------------------------------------------
metrics_store: list[dict] = []
MAX_POINTS = 10000


# ------------------------------------------------
# Models
# ------------------------------------------------
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


class AuthRequest(BaseModel):
    key: str


# ------------------------------------------------
# Auth endpoint — browser calls this first
# ------------------------------------------------
@app.post("/auth")
async def auth(req: AuthRequest):
    if req.key != DASHBOARD_KEY:
        raise HTTPException(status_code=401, detail="Wrong passkey")
    return {"status": "ok", "token": DASHBOARD_KEY}


# ------------------------------------------------
# Ingest — open, training loop posts here freely
# ------------------------------------------------
@app.post("/train_metrics")
async def receive_metrics(metric: Metric):
    metrics_store.append(metric.model_dump())
    if len(metrics_store) > MAX_POINTS:
        metrics_store.pop(0)
    return {"status": "ok"}


# ------------------------------------------------
# Protected read endpoints
# ------------------------------------------------
@app.get("/metrics")
async def get_metrics(_: str = Depends(verify_key)):
    return metrics_store


@app.get("/summary")
async def get_summary(_: str = Depends(verify_key)):
    """Pre-computed summary stats so the frontend doesn't have to crunch everything."""
    if not metrics_store:
        return {}

    train = [m for m in metrics_store if m["type"] == "train"]
    evals = [m for m in metrics_store if m["type"] in ("eval", "checkpoint")]
    ckpts = [m for m in metrics_store if m["type"] == "checkpoint"]

    # Run IDs
    run_ids = sorted(set(m["run_id"] for m in metrics_store))

    # Latest train
    last_train = train[-1] if train else {}
    last_eval = evals[-1] if evals else {}

    # Best val loss
    val_losses = [m["val_loss"] for m in evals if m.get("val_loss") is not None]
    best_val = min(val_losses) if val_losses else None

    # Loss delta (last 10 train steps)
    recent_losses = [m["loss"] for m in train[-10:] if m.get("loss") is not None]
    loss_delta = None
    if len(recent_losses) >= 2:
        loss_delta = round(recent_losses[-1] - recent_losses[0], 5)

    # Perplexity from last eval
    last_val_loss = last_eval.get("val_loss")
    perplexity = round(math.exp(last_val_loss), 3) if last_val_loss is not None else None

    # Tokens seen (total)
    total_tokens = sum(
        m.get("tokens_per_sec", 0) or 0
        for m in train
    )

    # ETA estimate (steps remaining × avg time per step)
    eta_seconds = None
    if len(train) >= 2:
        times = [m["timestamp"] for m in train]
        steps = [m["step"] for m in train]
        elapsed = times[-1] - times[0]
        step_span = steps[-1] - steps[0]
        if step_span > 0:
            secs_per_step = elapsed / step_span
            # We don't know total_steps from here, but expose secs_per_step
            eta_seconds = round(secs_per_step, 4)

    # Throughput stats
    tok_rates = [m["tokens_per_sec"] for m in train if m.get("tokens_per_sec")]
    avg_tok = round(sum(tok_rates) / len(tok_rates)) if tok_rates else None
    peak_tok = max(tok_rates) if tok_rates else None

    # Grad norm stats
    grads = [m["grad_norm"] for m in train if m.get("grad_norm") is not None]
    avg_grad = round(sum(grads) / len(grads), 4) if grads else None
    max_grad = round(max(grads), 4) if grads else None

    # GPU peak
    gpus = [m["gpu_mem_gb"] for m in train if m.get("gpu_mem_gb") is not None]
    peak_gpu = round(max(gpus), 3) if gpus else None

    return {
        "run_ids": run_ids,
        "current_run_id": run_ids[-1] if run_ids else None,
        "total_steps_logged": len(train),
        "total_eval_steps": len(evals),
        "checkpoint_count": len(ckpts),
        "current_step": last_train.get("step"),
        "current_loss": last_train.get("loss"),
        "current_lr": last_train.get("lr"),
        "current_grad_norm": last_train.get("grad_norm"),
        "current_gpu_gb": last_train.get("gpu_mem_gb"),
        "current_tok_per_sec": last_train.get("tokens_per_sec"),
        "last_val_loss": last_eval.get("val_loss"),
        "best_val_loss": best_val,
        "perplexity": perplexity,
        "loss_delta_last10": loss_delta,
        "secs_per_step": eta_seconds,
        "avg_tok_per_sec": avg_tok,
        "peak_tok_per_sec": peak_tok,
        "avg_grad_norm": avg_grad,
        "max_grad_norm": max_grad,
        "peak_gpu_gb": peak_gpu,
        "total_tokens_seen": total_tokens,
        "run_start_timestamp": metrics_store[0]["timestamp"] if metrics_store else None,
        "last_timestamp": metrics_store[-1]["timestamp"] if metrics_store else None,
    }


@app.post("/clear")
async def clear(_: str = Depends(verify_key)):
    metrics_store.clear()
    return {"status": "cleared"}


# ------------------------------------------------
# Dashboard
# ------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    with open("static/index.html") as f:
        return f.read()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "librarian-dashboard",
        "timestamp": time.time(),
        "metrics_count": len(metrics_store),
    }


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)