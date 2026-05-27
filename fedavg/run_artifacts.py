"""Persist federated run outputs: timestamped model, metrics JSON, run config."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from flwr.app import Context
from flwr.serverapp.strategy.result import Result


def _dataset_slug(context: Context) -> str:
    """Filesystem-safe dataset label from run config."""
    raw = str(context.run_config.get("dataset-name", "unknown")).strip()
    return raw.replace(" ", "_").replace("/", "-") or "unknown"


def _rounds_with_labels(
    num_rounds: int, result: Result
) -> list[dict[str, int | str | dict[str, float | int | list] | None]]:
    """One entry per round with ``round`` as ``i/n`` (e.g. ``1/20``) plus metrics."""
    out: list[dict[str, int | str | dict[str, float | int | list] | None]] = []
    for i in range(1, num_rounds + 1):
        train = result.train_metrics_clientapp.get(i)
        ev_c = result.evaluate_metrics_clientapp.get(i)
        ev_s = result.evaluate_metrics_serverapp.get(i)
        out.append(
            {
                "round": f"{i}/{num_rounds}",
                "round_index": i,
                "train_metrics": dict(train) if train is not None else None,
                "evaluate_metrics_clientapp": dict(ev_c) if ev_c is not None else None,
                "evaluate_metrics_serverapp": dict(ev_s) if ev_s is not None else None,
            }
        )
    return out


def save_run_artifacts(
    result: Result,
    context: Context,
    num_rounds: int,
    *,
    runs_root: Path | str = "flwr_runs",
    latest_model_path: Path | str = "final_model.pt",
) -> Path:
    """Write ``{dataset}_{timestamp}/`` with model, metrics, and run config (all prefixed by dataset).

    Also saves the latest model (``{dataset}_final_model.pt`` when default path is used).

    Returns the directory created for this run (under ``runs_root``).
    """
    print("\nSaving final model and metrics to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    dataset = _dataset_slug(context)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{dataset}_{stamp}"
    run_dir = Path(runs_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model_name = f"{dataset}_final_model.pt"
    metrics_name = f"{dataset}_metrics.json"
    config_name = f"{dataset}_run_config.json"

    torch.save(state_dict, run_dir / model_name)
    artifact = {
        "dataset": dataset,
        "saved_at_utc": stamp,
        "run_id": run_id,
        "num_rounds": num_rounds,
        "rounds": _rounds_with_labels(num_rounds, result),
    }
    (run_dir / metrics_name).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    run_cfg = {k: context.run_config[k] for k in context.run_config}
    (run_dir / config_name).write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    latest = Path(latest_model_path)
    if latest.name == "final_model.pt":
        latest = latest.with_name(model_name)
    torch.save(state_dict, latest)
    print(f"Artifacts: {run_dir.resolve()}")
    return run_dir
