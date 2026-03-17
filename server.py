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
stage_store:   list[dict] = []

MAX_POINTS = 10000
MAX_STAGE_EVENTS = 2000

# Ordered pipeline stages for display
PIPELINE_STAGES = [
    "download",
    "clean",
    "train_tokenizer",
    "tokenize",
    "pack",
    "train",
]

STAGE_LABELS = {
    "download":        "Data Download",
    "clean":           "Data Cleaning",
    "train_tokenizer": "Tokenizer Training",
    "tokenize":        "Tokenization",
    "pack":            "Token Packing",
    "train":           "Model Training",
}


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


class StageEvent(BaseModel):
    run_id: int
    stage: str
    event: str          # "start" | "progress" | "end" | "error"
    timestamp: float
    elapsed_s: float | None = None
    metrics: dict = {}


class AuthRequest(BaseModel):
    key: str


# ------------------------------------------------
# Auth endpoint
# ------------------------------------------------
@app.post("/auth")
async def auth(req: AuthRequest):
    if req.key != DASHBOARD_KEY:
        raise HTTPException(status_code=401, detail="Wrong passkey")
    return {"status": "ok", "token": DASHBOARD_KEY}


# ------------------------------------------------
# Ingest — training metrics (open)
# ------------------------------------------------
@app.post("/train_metrics")
async def receive_metrics(metric: Metric):
    metrics_store.append(metric.model_dump())
    if len(metrics_store) > MAX_POINTS:
        metrics_store.pop(0)
    return {"status": "ok"}


# ------------------------------------------------
# Ingest — stage events (open)
# ------------------------------------------------
@app.post("/stage_metrics")
async def receive_stage(event: StageEvent):
    stage_store.append(event.model_dump())
    if len(stage_store) > MAX_STAGE_EVENTS:
        stage_store.pop(0)
    return {"status": "ok"}


# ------------------------------------------------
# Protected read endpoints
# ------------------------------------------------
@app.get("/metrics")
async def get_metrics(_: str = Depends(verify_key)):
    return metrics_store


@app.get("/stages")
async def get_stages(_: str = Depends(verify_key)):
    return stage_store


@app.get("/stage_summary")
async def get_stage_summary(_: str = Depends(verify_key)):
    """
    Returns a per-stage summary keyed by stage name.
    Each entry has: status, start_time, end_time, elapsed_s, latest_metrics, error.
    Stages are returned in pipeline order.
    """
    if not stage_store:
        return {"stages": [], "current_stage": None, "run_id": None}

    # Group by run_id — use the latest run
    run_ids = sorted(set(e["run_id"] for e in stage_store))
    latest_run = run_ids[-1]
    events = [e for e in stage_store if e["run_id"] == latest_run]

    stage_data: dict[str, dict] = {}

    for ev in events:
        s = ev["stage"]
        if s not in stage_data:
            stage_data[s] = {
                "stage":           s,
                "label":           STAGE_LABELS.get(s, s),
                "status":          "pending",
                "start_time":      None,
                "end_time":        None,
                "elapsed_s":       None,
                "latest_metrics":  {},
                "progress_events": [],
                "error":           None,
            }

        if ev["event"] == "start":
            stage_data[s]["status"]     = "running"
            stage_data[s]["start_time"] = ev["timestamp"]

        elif ev["event"] == "progress":
            stage_data[s]["status"] = "running"
            stage_data[s]["latest_metrics"].update(ev.get("metrics", {}))
            stage_data[s]["progress_events"].append({
                "timestamp": ev["timestamp"],
                "metrics":   ev.get("metrics", {}),
            })

        elif ev["event"] == "end":
            stage_data[s]["status"]          = "done"
            stage_data[s]["end_time"]        = ev["timestamp"]
            stage_data[s]["elapsed_s"]       = ev.get("elapsed_s")
            stage_data[s]["latest_metrics"].update(ev.get("metrics", {}))

        elif ev["event"] == "error":
            stage_data[s]["status"] = "error"
            stage_data[s]["error"]  = ev.get("metrics", {}).get("error", "unknown")

    # Determine current active stage
    current_stage = None
    for s, d in stage_data.items():
        if d["status"] == "running":
            current_stage = s
            break

    # Build ordered list — include all known pipeline stages, plus any extras
    ordered = []
    seen = set()
    for s in PIPELINE_STAGES:
        if s in stage_data:
            ordered.append(stage_data[s])
            seen.add(s)
    for s, d in stage_data.items():
        if s not in seen:
            ordered.append(d)

    return {
        "run_id":        latest_run,
        "current_stage": current_stage,
        "stages":        ordered,
    }


@app.get("/summary")
async def get_summary(_: str = Depends(verify_key)):
    """Pre-computed summary stats for the training section."""
    if not metrics_store:
        return {}

    train = [m for m in metrics_store if m["type"] == "train"]
    evals = [m for m in metrics_store if m["type"] in ("eval", "checkpoint")]
    ckpts = [m for m in metrics_store if m["type"] == "checkpoint"]

    run_ids    = sorted(set(m["run_id"] for m in metrics_store))
    last_train = train[-1] if train else {}
    last_eval  = evals[-1] if evals else {}

    val_losses = [m["val_loss"] for m in evals if m.get("val_loss") is not None]
    best_val   = min(val_losses) if val_losses else None

    recent_losses = [m["loss"] for m in train[-10:] if m.get("loss") is not None]
    loss_delta    = None
    if len(recent_losses) >= 2:
        loss_delta = round(recent_losses[-1] - recent_losses[0], 5)

    last_val_loss = last_eval.get("val_loss")
    perplexity    = round(math.exp(last_val_loss), 3) if last_val_loss is not None else None

    total_tokens = sum(m.get("tokens_per_sec", 0) or 0 for m in train)

    eta_seconds = None
    if len(train) >= 2:
        times     = [m["timestamp"] for m in train]
        steps     = [m["step"]      for m in train]
        elapsed   = times[-1] - times[0]
        step_span = steps[-1] - steps[0]
        if step_span > 0:
            eta_seconds = round(elapsed / step_span, 4)

    tok_rates = [m["tokens_per_sec"] for m in train if m.get("tokens_per_sec")]
    avg_tok   = round(sum(tok_rates) / len(tok_rates)) if tok_rates else None
    peak_tok  = max(tok_rates) if tok_rates else None

    grads    = [m["grad_norm"] for m in train if m.get("grad_norm") is not None]
    avg_grad = round(sum(grads) / len(grads), 4) if grads else None
    max_grad = round(max(grads), 4)              if grads else None

    gpus     = [m["gpu_mem_gb"] for m in train if m.get("gpu_mem_gb") is not None]
    peak_gpu = round(max(gpus), 3) if gpus else None

    return {
        "run_ids":              run_ids,
        "current_run_id":       run_ids[-1] if run_ids else None,
        "total_steps_logged":   len(train),
        "total_eval_steps":     len(evals),
        "checkpoint_count":     len(ckpts),
        "current_step":         last_train.get("step"),
        "current_loss":         last_train.get("loss"),
        "current_lr":           last_train.get("lr"),
        "current_grad_norm":    last_train.get("grad_norm"),
        "current_gpu_gb":       last_train.get("gpu_mem_gb"),
        "current_tok_per_sec":  last_train.get("tokens_per_sec"),
        "last_val_loss":        last_eval.get("val_loss"),
        "best_val_loss":        best_val,
        "perplexity":           perplexity,
        "loss_delta_last10":    loss_delta,
        "secs_per_step":        eta_seconds,
        "avg_tok_per_sec":      avg_tok,
        "peak_tok_per_sec":     peak_tok,
        "avg_grad_norm":        avg_grad,
        "max_grad_norm":        max_grad,
        "peak_gpu_gb":          peak_gpu,
        "total_tokens_seen":    total_tokens,
        "run_start_timestamp":  metrics_store[0]["timestamp"] if metrics_store else None,
        "last_timestamp":       metrics_store[-1]["timestamp"] if metrics_store else None,
    }


@app.post("/clear")
async def clear(_: str = Depends(verify_key)):
    metrics_store.clear()
    stage_store.clear()
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
        "status":        "ok",
        "service":       "librarian-dashboard",
        "timestamp":     time.time(),
        "metrics_count": len(metrics_store),
        "stage_count":   len(stage_store),
    }


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
