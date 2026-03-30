from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.responses import HTMLResponse
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
DASHBOARD_KEY = os.environ.get("DASHBOARD_KEY", "lib-450M-large")
security = HTTPBearer(auto_error=False)


def verify_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials is None or credentials.credentials != DASHBOARD_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing key")
    return credentials.credentials


# ────────────────────────────────────────────────────────────
# Storage — per run_id so history is preserved across runs
# ────────────────────────────────────────────────────────────
metrics_by_run: dict[int, list[dict]] = defaultdict(list)
stage_by_run:   dict[int, list[dict]] = defaultdict(list)

MAX_POINTS_PER_RUN = 10_000
MAX_STAGE_PER_RUN  = 2_000
MAX_RUNS_KEPT      = 20      # evict oldest when exceeded

# ── alert thresholds (override via env) ──────────────────────
ALERT_GRAD_EXPLODE  = float(os.environ.get("ALERT_GRAD_EXPLODE",  "10.0"))
ALERT_LOSS_SPIKE    = float(os.environ.get("ALERT_LOSS_SPIKE",    "0.5"))
ALERT_STALL_SECS    = float(os.environ.get("ALERT_STALL_SECS",    "300"))
TOTAL_TARGET_TOKENS = int(os.environ.get("TOTAL_TARGET_TOKENS",   "26000000000"))  # 26B

# ── pipeline stage metadata ──────────────────────────────────
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
# Pydantic models
# ────────────────────────────────────────────────────────────
class Metric(BaseModel):
    run_id:         int
    type:           str
    step:           int
    timestamp:      float
    loss:           float | None = None
    val_loss:       float | None = None
    lr:             float | None = None
    grad_norm:      float | None = None
    tokens_per_sec: int   | None = None
    gpu_mem_gb:     float | None = None


class StageEvent(BaseModel):
    run_id:    int
    stage:     str
    event:     str        # start | progress | end | error
    timestamp: float
    elapsed_s: float | None = None
    metrics:   dict = {}


class AuthRequest(BaseModel):
    key: str


# ────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────
def _evict_old_runs(store: dict):
    while len(store) > MAX_RUNS_KEPT:
        del store[min(store.keys())]


def _resolve_run(run_id: int | None, store: dict) -> int | None:
    if not store:
        return None
    return run_id if (run_id is not None and run_id in store) else max(store.keys())


def _compute_alerts(run_metrics: list[dict]) -> list[dict]:
    alerts = []
    train = [m for m in run_metrics if m["type"] == "train"]
    if not train:
        return alerts

    # Stall
    stale = time.time() - train[-1]["timestamp"]
    if stale > ALERT_STALL_SECS:
        alerts.append({
            "type":    "stall",
            "level":   "warning",
            "message": f"No training metrics for {stale/60:.1f} min — training may have stalled",
        })

    # Gradient explosion
    recent_grads = [m["grad_norm"] for m in train[-5:] if m.get("grad_norm") is not None]
    if recent_grads and max(recent_grads) > ALERT_GRAD_EXPLODE:
        alerts.append({
            "type":    "grad_explode",
            "level":   "danger",
            "message": f"Gradient norm {max(recent_grads):.2f} exceeds threshold {ALERT_GRAD_EXPLODE} — check LR",
        })

    # Loss spike
    recent_losses = [m["loss"] for m in train[-10:] if m.get("loss") is not None]
    if len(recent_losses) >= 2:
        spike = recent_losses[-1] - recent_losses[-2]
        if spike > ALERT_LOSS_SPIKE:
            alerts.append({
                "type":    "loss_spike",
                "level":   "warning",
                "message": f"Loss jumped +{spike:.4f} in one log interval",
            })

    # NaN loss
    nan_steps = [m["step"] for m in train[-20:] if m.get("loss") is not None and math.isnan(m["loss"])]
    if nan_steps:
        alerts.append({
            "type":    "nan_loss",
            "level":   "danger",
            "message": f"NaN loss at step(s): {nan_steps} — training is broken",
        })

    return alerts


def _compute_token_budget(run_metrics: list[dict]) -> dict:
    train = [m for m in run_metrics if m["type"] == "train"]
    if not train:
        return {}

    # Accumulate tokens as area under tok/s curve
    total_tokens = 0
    for i in range(1, len(train)):
        dt  = train[i]["timestamp"] - train[i - 1]["timestamp"]
        tps = train[i].get("tokens_per_sec") or 0
        total_tokens += tps * dt

    pct = total_tokens / TOTAL_TARGET_TOKENS * 100 if TOTAL_TARGET_TOKENS else 0

    # Average tok/s over last 50 steps for ETA
    recent  = train[-50:]
    tps_vals = [m["tokens_per_sec"] for m in recent if m.get("tokens_per_sec")]
    avg_tps  = sum(tps_vals) / len(tps_vals) if tps_vals else 0

    tokens_remaining = max(0, TOTAL_TARGET_TOKENS - total_tokens)
    eta_seconds = round(tokens_remaining / avg_tps) if avg_tps > 0 else None

    return {
        "total_tokens_trained": int(total_tokens),
        "target_tokens":        TOTAL_TARGET_TOKENS,
        "pct_complete":         round(pct, 3),
        "avg_tok_per_sec":      round(avg_tps),
        "eta_seconds":          eta_seconds,
    }


def _build_summary(run_metrics: list[dict]) -> dict:
    if not run_metrics:
        return {}

    train  = [m for m in run_metrics if m["type"] == "train"]
    evals  = [m for m in run_metrics if m["type"] in ("eval", "checkpoint")]
    ckpts  = [m for m in run_metrics if m["type"] == "checkpoint"]

    run_ids     = sorted(set(m["run_id"] for m in run_metrics))
    last_train  = train[-1] if train else {}
    last_eval   = evals[-1] if evals else {}

    val_losses  = [m["val_loss"] for m in evals if m.get("val_loss") is not None]
    best_val    = min(val_losses) if val_losses else None

    recent_losses = [m["loss"] for m in train[-10:] if m.get("loss") is not None]
    loss_delta    = round(recent_losses[-1] - recent_losses[0], 5) if len(recent_losses) >= 2 else None

    last_val_loss = last_eval.get("val_loss")
    perplexity    = round(math.exp(last_val_loss), 3) if last_val_loss is not None else None

    # ETA per step (seconds between consecutive logged steps)
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
        "run_ids":             run_ids,
        "current_run_id":      run_ids[-1] if run_ids else None,
        "total_steps_logged":  len(train),
        "total_eval_steps":    len(evals),
        "checkpoint_count":    len(ckpts),
        "current_step":        last_train.get("step"),
        "current_loss":        last_train.get("loss"),
        "current_lr":          last_train.get("lr"),
        "current_grad_norm":   last_train.get("grad_norm"),
        "current_gpu_gb":      last_train.get("gpu_mem_gb"),
        "current_tok_per_sec": last_train.get("tokens_per_sec"),
        "last_val_loss":       last_eval.get("val_loss"),
        "best_val_loss":       best_val,
        "perplexity":          perplexity,
        "loss_delta_last10":   loss_delta,
        "secs_per_step":       eta_seconds,
        "avg_tok_per_sec":     avg_tok,
        "peak_tok_per_sec":    peak_tok,
        "avg_grad_norm":       avg_grad,
        "max_grad_norm":       max_grad,
        "peak_gpu_gb":         peak_gpu,
        "total_tokens_seen":   sum(m.get("tokens_per_sec", 0) or 0 for m in train),
        "run_start_timestamp": run_metrics[0]["timestamp"] if run_metrics else None,
        "last_timestamp":      run_metrics[-1]["timestamp"] if run_metrics else None,
    }


def _build_stage_summary(events: list[dict], run_id: int) -> dict:
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
            stage_data[s]["progress_events"].append({"timestamp": ev["timestamp"], "metrics": ev.get("metrics", {})})
        elif ev["event"] == "end":
            stage_data[s]["status"]    = "done"
            stage_data[s]["end_time"]  = ev["timestamp"]
            stage_data[s]["elapsed_s"] = ev.get("elapsed_s")
            stage_data[s]["latest_metrics"].update(ev.get("metrics", {}))
        elif ev["event"] == "error":
            stage_data[s]["status"] = "error"
            stage_data[s]["error"]  = ev.get("metrics", {}).get("error", "unknown")

    current_stage = next((s for s, d in stage_data.items() if d["status"] == "running"), None)

    ordered = []
    seen = set()
    for s in PIPELINE_STAGES:
        if s in stage_data:
            ordered.append(stage_data[s])
            seen.add(s)
    for s, d in stage_data.items():
        if s not in seen:
            ordered.append(d)

    return {"run_id": run_id, "current_stage": current_stage, "stages": ordered}


# ────────────────────────────────────────────────────────────
# Auth endpoint
# ────────────────────────────────────────────────────────────
@app.post("/auth")
async def auth(req: AuthRequest):
    if req.key != DASHBOARD_KEY:
        raise HTTPException(status_code=401, detail="Wrong passkey")
    return {"status": "ok", "token": DASHBOARD_KEY}


# ────────────────────────────────────────────────────────────
# Ingest — open so trainer can always post
# ────────────────────────────────────────────────────────────
@app.post("/train_metrics")
async def receive_metrics(metric: Metric):
    store = metrics_by_run[metric.run_id]
    store.append(metric.model_dump())
    if len(store) > MAX_POINTS_PER_RUN:
        store.pop(0)
    _evict_old_runs(metrics_by_run)
    return {"status": "ok"}


@app.post("/stage_metrics")
async def receive_stage(event: StageEvent):
    store = stage_by_run[event.run_id]
    store.append(event.model_dump())
    if len(store) > MAX_STAGE_PER_RUN:
        store.pop(0)
    _evict_old_runs(stage_by_run)
    return {"status": "ok"}


# ────────────────────────────────────────────────────────────
# Protected read endpoints
# ────────────────────────────────────────────────────────────
@app.get("/runs")
async def get_runs(_: str = Depends(verify_key)):
    """All known runs with basic stats, newest first."""
    all_ids = sorted(
        set(list(metrics_by_run.keys()) + list(stage_by_run.keys())),
        reverse=True,
    )
    result = []
    for rid in all_ids:
        train = [m for m in metrics_by_run.get(rid, []) if m["type"] == "train"]
        evals = [m for m in metrics_by_run.get(rid, []) if m["type"] in ("eval", "checkpoint")]
        val_losses = [m["val_loss"] for m in evals if m.get("val_loss") is not None]
        result.append({
            "run_id":     rid,
            "steps":      len(train),
            "last_step":  train[-1]["step"]      if train else None,
            "last_loss":  train[-1]["loss"]      if train else None,
            "best_val":   min(val_losses)         if val_losses else None,
            "start_time": train[0]["timestamp"]  if train else None,
            "last_time":  train[-1]["timestamp"] if train else None,
        })
    return result


@app.get("/metrics")
async def get_metrics(
    run_id: int | None = Query(default=None),
    _: str = Depends(verify_key),
):
    rid = _resolve_run(run_id, metrics_by_run)
    return metrics_by_run.get(rid, []) if rid is not None else []


@app.get("/stages")
async def get_stages(
    run_id: int | None = Query(default=None),
    _: str = Depends(verify_key),
):
    rid = _resolve_run(run_id, stage_by_run)
    return stage_by_run.get(rid, []) if rid is not None else []


@app.get("/stage_summary")
async def get_stage_summary(
    run_id: int | None = Query(default=None),
    _: str = Depends(verify_key),
):
    all_ids = set(stage_by_run.keys()) | set(metrics_by_run.keys())
    if not all_ids:
        return {"stages": [], "current_stage": None, "run_id": None}
    target = run_id if (run_id is not None and run_id in all_ids) else max(all_ids)
    events = stage_by_run.get(target, [])
    return _build_stage_summary(events, target)


@app.get("/summary")
async def get_summary(
    run_id: int | None = Query(default=None),
    _: str = Depends(verify_key),
):
    rid = _resolve_run(run_id, metrics_by_run)
    if rid is None:
        return {}
    return _build_summary(metrics_by_run.get(rid, []))


@app.get("/alerts")
async def get_alerts(
    run_id: int | None = Query(default=None),
    _: str = Depends(verify_key),
):
    rid = _resolve_run(run_id, metrics_by_run)
    if rid is None:
        return []
    return _compute_alerts(metrics_by_run.get(rid, []))


@app.get("/token_budget")
async def get_token_budget(
    run_id: int | None = Query(default=None),
    _: str = Depends(verify_key),
):
    rid = _resolve_run(run_id, metrics_by_run)
    if rid is None:
        return {}
    return _compute_token_budget(metrics_by_run.get(rid, []))


@app.post("/clear")
async def clear(_: str = Depends(verify_key)):
    metrics_by_run.clear()
    stage_by_run.clear()
    return {"status": "cleared"}


# ────────────────────────────────────────────────────────────
# Dashboard + health
# ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    with open("static/index.html") as f:
        return f.read()


@app.get("/health")
async def health():
    all_runs = set(list(metrics_by_run.keys()) + list(stage_by_run.keys()))
    return {
        "status":        "ok",
        "service":       "librarian-dashboard",
        "timestamp":     time.time(),
        "runs_tracked":  len(all_runs),
        "metrics_total": sum(len(v) for v in metrics_by_run.values()),
        "stages_total":  sum(len(v) for v in stage_by_run.values()),
        "alert_thresholds": {
            "grad_explode":   ALERT_GRAD_EXPLODE,
            "loss_spike":     ALERT_LOSS_SPIKE,
            "stall_secs":     ALERT_STALL_SECS,
            "target_tokens":  TOTAL_TARGET_TOKENS,
        }
    }


app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
