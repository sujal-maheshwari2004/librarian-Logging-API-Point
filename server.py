from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn
import time
import os
import math
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# ────────────────────────────────────────────────────────────
# Auth
# ────────────────────────────────────────────────────────────
DASHBOARD_KEY = os.environ.get("DASHBOARD_KEY", "changeme")
security = HTTPBearer(auto_error=False)


def verify_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials is None or credentials.credentials != DASHBOARD_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing key")
    return credentials.credentials


# ────────────────────────────────────────────────────────────
# Storage — per-run, so all runs are preserved in memory
# ────────────────────────────────────────────────────────────
metrics_by_run: dict[int, list[dict]] = defaultdict(list)
stage_by_run:   dict[int, list[dict]] = defaultdict(list)

MAX_POINTS_PER_RUN = 10_000
MAX_STAGE_PER_RUN  = 2_000

# Pipeline stage definitions
PIPELINE_STAGES = [
    "download", "clean", "train_tokenizer",
    "tokenize", "pack", "train", "evaluate",
]

STAGE_LABELS = {
    "download":        "Data Download",
    "clean":           "Data Cleaning",
    "train_tokenizer": "Tokenizer Training",
    "tokenize":        "Tokenization",
    "pack":            "Token Packing",
    "train":           "Model Training",
    "evaluate":        "Evaluation",
}

# ────────────────────────────────────────────────────────────
# Alert thresholds (tuneable via env)
# ────────────────────────────────────────────────────────────
STALL_SECONDS     = int(os.environ.get("STALL_SECONDS",     "300"))   # 5 min
LOSS_SPIKE_FACTOR = float(os.environ.get("LOSS_SPIKE_FACTOR","2.0"))  # 2× rolling mean
GRAD_WARN         = float(os.environ.get("GRAD_WARN",        "5.0"))
GRAD_EXPLODE      = float(os.environ.get("GRAD_EXPLODE",     "20.0"))

# Total training token target — used for progress + ETA
# Override with TOKEN_TARGET env var if you change total_steps or batch
TOKEN_TARGET = int(os.environ.get("TOKEN_TARGET", str(32 * 16 * 512 * 100_000)))


# ────────────────────────────────────────────────────────────
# Models
# ────────────────────────────────────────────────────────────
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
    event: str          # start | progress | end | error
    timestamp: float
    elapsed_s: float | None = None
    metrics: dict = {}


class AuthRequest(BaseModel):
    key: str


# ────────────────────────────────────────────────────────────
# Auth endpoint
# ────────────────────────────────────────────────────────────
@app.post("/auth")
async def auth(req: AuthRequest):
    if req.key != DASHBOARD_KEY:
        raise HTTPException(status_code=401, detail="Wrong passkey")
    return {"status": "ok", "token": DASHBOARD_KEY}


# ────────────────────────────────────────────────────────────
# Ingest — open endpoints (called from training server)
# ────────────────────────────────────────────────────────────
@app.post("/train_metrics")
async def receive_metrics(metric: Metric):
    store = metrics_by_run[metric.run_id]
    store.append(metric.model_dump())
    if len(store) > MAX_POINTS_PER_RUN:
        store.pop(0)
    return {"status": "ok"}


@app.post("/stage_metrics")
async def receive_stage(event: StageEvent):
    store = stage_by_run[event.run_id]
    store.append(event.model_dump())
    if len(store) > MAX_STAGE_PER_RUN:
        store.pop(0)
    return {"status": "ok"}


# ────────────────────────────────────────────────────────────
# Protected read endpoints
# ────────────────────────────────────────────────────────────

@app.get("/runs")
async def list_runs(_: str = Depends(verify_key)):
    """List all known run IDs with basic stats."""
    all_run_ids = sorted(
        set(list(metrics_by_run.keys()) + list(stage_by_run.keys())),
        reverse=True,
    )
    result = []
    for rid in all_run_ids:
        train = [m for m in metrics_by_run.get(rid, []) if m["type"] == "train"]
        evals = [m for m in metrics_by_run.get(rid, []) if m["type"] in ("eval", "checkpoint")]
        val_losses = [m["val_loss"] for m in evals if m.get("val_loss") is not None]
        result.append({
            "run_id":        rid,
            "steps_logged":  len(train),
            "current_step":  train[-1]["step"] if train else None,
            "best_val_loss": min(val_losses) if val_losses else None,
            "start_time":    metrics_by_run[rid][0]["timestamp"] if metrics_by_run.get(rid) else None,
            "last_time":     metrics_by_run[rid][-1]["timestamp"] if metrics_by_run.get(rid) else None,
        })
    return result


@app.get("/metrics")
async def get_metrics(
    run_id: int | None = Query(default=None),
    _: str = Depends(verify_key),
):
    if run_id is not None:
        return metrics_by_run.get(run_id, [])
    # default: latest run
    if not metrics_by_run:
        return []
    latest = max(metrics_by_run.keys())
    return metrics_by_run[latest]


@app.get("/stages")
async def get_stages(
    run_id: int | None = Query(default=None),
    _: str = Depends(verify_key),
):
    if run_id is not None:
        return stage_by_run.get(run_id, [])
    if not stage_by_run:
        return []
    latest = max(stage_by_run.keys())
    return stage_by_run[latest]


@app.get("/stage_summary")
async def get_stage_summary(
    run_id: int | None = Query(default=None),
    _: str = Depends(verify_key),
):
    all_ids = sorted(
        set(list(metrics_by_run.keys()) + list(stage_by_run.keys()))
    )
    if not all_ids:
        return {"stages": [], "current_stage": None, "run_id": None}

    target_run = run_id if run_id is not None else all_ids[-1]
    events = stage_by_run.get(target_run, [])

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
            stage_data[s]["status"]    = "done"
            stage_data[s]["end_time"]  = ev["timestamp"]
            stage_data[s]["elapsed_s"] = ev.get("elapsed_s")
            stage_data[s]["latest_metrics"].update(ev.get("metrics", {}))
        elif ev["event"] == "error":
            stage_data[s]["status"] = "error"
            stage_data[s]["error"]  = ev.get("metrics", {}).get("error", "unknown")

    current_stage = next(
        (s for s, d in stage_data.items() if d["status"] == "running"), None
    )

    ordered, seen = [], set()
    for s in PIPELINE_STAGES:
        if s in stage_data:
            ordered.append(stage_data[s])
            seen.add(s)
    for s, d in stage_data.items():
        if s not in seen:
            ordered.append(d)

    return {
        "run_id":        target_run,
        "current_stage": current_stage,
        "stages":        ordered,
    }


@app.get("/summary")
async def get_summary(
    run_id: int | None = Query(default=None),
    _: str = Depends(verify_key),
):
    all_ids = sorted(set(list(metrics_by_run.keys()) + list(stage_by_run.keys())))
    if not all_ids:
        return {}

    target_run  = run_id if run_id is not None else all_ids[-1]
    all_metrics = metrics_by_run.get(target_run, [])
    if not all_metrics:
        return {}

    train  = [m for m in all_metrics if m["type"] == "train"]
    evals  = [m for m in all_metrics if m["type"] in ("eval", "checkpoint")]
    ckpts  = [m for m in all_metrics if m["type"] == "checkpoint"]

    last_train = train[-1] if train else {}
    last_eval  = evals[-1] if evals else {}

    val_losses   = [m["val_loss"] for m in evals if m.get("val_loss") is not None]
    best_val     = min(val_losses) if val_losses else None
    last_val     = last_eval.get("val_loss")
    perplexity   = round(math.exp(last_val), 3) if last_val is not None else None

    recent_losses = [m["loss"] for m in train[-20:] if m.get("loss") is not None]
    loss_delta    = None
    if len(recent_losses) >= 2:
        loss_delta = round(recent_losses[-1] - recent_losses[0], 5)

    # ── Token budget tracking (correct: count tokens per step, not sum rates) ──
    # Each train entry covers batch_size * seq_len tokens. We use tokens_per_sec
    # and the time between consecutive steps to integrate actual token count.
    total_tokens = 0
    if len(train) >= 2:
        for i in range(1, len(train)):
            dt   = train[i]["timestamp"] - train[i-1]["timestamp"]
            rate = train[i].get("tokens_per_sec") or train[i-1].get("tokens_per_sec") or 0
            total_tokens += int(dt * rate)
    elif train and train[0].get("tokens_per_sec"):
        # Single point — can't integrate, use step count heuristic
        # 32 batch * 16 grad_accum * 512 seq = 262144 tokens per optimizer step
        total_tokens = train[-1].get("step", 0) * 262_144

    token_pct  = round(100 * total_tokens / TOKEN_TARGET, 2) if TOKEN_TARGET > 0 else None
    token_eta  = None
    if total_tokens > 0 and TOKEN_TARGET > total_tokens:
        tok_rates   = [m["tokens_per_sec"] for m in train[-50:] if m.get("tokens_per_sec")]
        if tok_rates:
            avg_rate   = sum(tok_rates) / len(tok_rates)
            remaining  = TOKEN_TARGET - total_tokens
            token_eta  = round(remaining / avg_rate)   # seconds

    # ── Stall detection ──
    last_ts   = all_metrics[-1]["timestamp"] if all_metrics else None
    stalled   = (last_ts is not None) and ((time.time() - last_ts) > STALL_SECONDS)
    stall_for = round(time.time() - last_ts) if last_ts else None

    # ── Gradient health ──
    recent_grads = [m["grad_norm"] for m in train[-20:] if m.get("grad_norm") is not None]
    grad_status  = "ok"
    if recent_grads:
        max_recent = max(recent_grads)
        if max_recent >= GRAD_EXPLODE:
            grad_status = "exploding"
        elif max_recent >= GRAD_WARN:
            grad_status = "warning"

    # ── Loss spike detection ──
    loss_spike = False
    if len(recent_losses) >= 10:
        rolling_mean = sum(recent_losses[:-1]) / len(recent_losses[:-1])
        if rolling_mean > 0 and recent_losses[-1] > rolling_mean * LOSS_SPIKE_FACTOR:
            loss_spike = True

    # ── Throughput ──
    tok_rates    = [m["tokens_per_sec"] for m in train if m.get("tokens_per_sec")]
    avg_tok      = round(sum(tok_rates) / len(tok_rates)) if tok_rates else None
    peak_tok     = max(tok_rates) if tok_rates else None

    # ── ETA to end of training (step-based) ──
    eta_step_s = None
    if len(train) >= 2:
        times     = [m["timestamp"] for m in train[-50:]]
        steps     = [m["step"]      for m in train[-50:]]
        elapsed   = times[-1] - times[0]
        step_span = steps[-1] - steps[0]
        if step_span > 0:
            eta_step_s = round(elapsed / step_span, 4)

    grads    = [m["grad_norm"] for m in train if m.get("grad_norm") is not None]
    avg_grad = round(sum(grads) / len(grads), 4) if grads else None
    max_grad = round(max(grads), 4)              if grads else None

    gpus     = [m["gpu_mem_gb"] for m in train if m.get("gpu_mem_gb") is not None]
    peak_gpu = round(max(gpus), 3) if gpus else None

    # ── Checkpoint list ──
    checkpoint_list = [
        {
            "step":      m["step"],
            "val_loss":  m.get("val_loss"),
            "timestamp": m["timestamp"],
        }
        for m in ckpts
    ]

    return {
        # run
        "run_id":               target_run,
        "all_run_ids":          all_ids,
        "run_start_timestamp":  all_metrics[0]["timestamp"],
        "last_timestamp":       last_ts,
        "total_steps_logged":   len(train),
        "total_eval_steps":     len(evals),
        "checkpoint_count":     len(ckpts),
        "checkpoint_list":      checkpoint_list,
        # live
        "current_step":         last_train.get("step"),
        "current_loss":         last_train.get("loss"),
        "current_lr":           last_train.get("lr"),
        "current_grad_norm":    last_train.get("grad_norm"),
        "current_gpu_gb":       last_train.get("gpu_mem_gb"),
        "current_tok_per_sec":  last_train.get("tokens_per_sec"),
        # val
        "last_val_loss":        last_val,
        "best_val_loss":        best_val,
        "perplexity":           perplexity,
        "loss_delta_last20":    loss_delta,
        # token budget
        "total_tokens_seen":    total_tokens,
        "token_target":         TOKEN_TARGET,
        "token_pct":            token_pct,
        "token_eta_s":          token_eta,
        # throughput
        "secs_per_step":        eta_step_s,
        "avg_tok_per_sec":      avg_tok,
        "peak_tok_per_sec":     peak_tok,
        # gradient
        "avg_grad_norm":        avg_grad,
        "max_grad_norm":        max_grad,
        "grad_status":          grad_status,
        # alerts
        "stalled":              stalled,
        "stall_seconds":        stall_for,
        "loss_spike":           loss_spike,
        "peak_gpu_gb":          peak_gpu,
    }


# ────────────────────────────────────────────────────────────
# Download — export a full run as JSON
# ────────────────────────────────────────────────────────────
@app.get("/download/{run_id}")
async def download_run(run_id: int, _: str = Depends(verify_key)):
    metrics = metrics_by_run.get(run_id, [])
    stages  = stage_by_run.get(run_id, [])
    if not metrics and not stages:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    payload = {
        "run_id":    run_id,
        "exported":  time.time(),
        "metrics":   metrics,
        "stages":    stages,
    }
    return JSONResponse(
        content=payload,
        headers={
            "Content-Disposition": f'attachment; filename="librarian_run_{run_id}.json"'
        },
    )


# ────────────────────────────────────────────────────────────
# Clear
# ────────────────────────────────────────────────────────────
@app.post("/clear")
async def clear(
    run_id: int | None = Query(default=None),
    _: str = Depends(verify_key),
):
    if run_id is not None:
        removed_m = len(metrics_by_run.pop(run_id, []))
        removed_s = len(stage_by_run.pop(run_id, []))
        return {"status": "cleared", "run_id": run_id,
                "metrics_removed": removed_m, "stages_removed": removed_s}
    # clear all
    total_m = sum(len(v) for v in metrics_by_run.values())
    total_s = sum(len(v) for v in stage_by_run.values())
    metrics_by_run.clear()
    stage_by_run.clear()
    return {"status": "all_cleared", "metrics_removed": total_m, "stages_removed": total_s}


# ────────────────────────────────────────────────────────────
# Dashboard + health
# ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    with open("static/index.html") as f:
        return f.read()


@app.get("/health")
async def health():
    all_run_ids = sorted(
        set(list(metrics_by_run.keys()) + list(stage_by_run.keys()))
    )
    return {
        "status":        "ok",
        "service":       "librarian-dashboard",
        "timestamp":     time.time(),
        "runs":          len(all_run_ids),
        "latest_run":    all_run_ids[-1] if all_run_ids else None,
        "metrics_count": sum(len(v) for v in metrics_by_run.values()),
        "stage_count":   sum(len(v) for v in stage_by_run.values()),
    }


app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
