"""Persist federated run outputs: timestamped model, metrics JSON, run config."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from flwr.app import Context
from flwr.serverapp.strategy.result import Result


def _rounds_with_labels(
    num_rounds: int,
    result: Result,
    wait_times: dict[int, float],
    client_times: dict[int, dict[str, float]],
    client_eval: dict[int, dict[str, dict[str, float]]],
    server_eval_times: dict[int, float],
) -> list[dict]:
    """One entry per round with ``round`` as ``i/n`` (e.g. ``1/20``) plus metrics."""
    out: list[dict] = []
    for i in range(1, num_rounds + 1):
        train = result.train_metrics_clientapp.get(i)
        ev_c = result.evaluate_metrics_clientapp.get(i)
        ev_s = result.evaluate_metrics_serverapp.get(i)
        wait = wait_times.get(i)
        per_client = client_times.get(i) or {}
        per_client_eval = client_eval.get(i) or {}
        # The round ends with the slowest client, so that is what the wait covers.
        slowest = max(per_client.values()) if per_client else None
        # Spread between the best and worst client on their own data. Under IID
        # partitions this stays small; a widening spread indicates client drift.
        accs = [
            v["eval_acc"] for v in per_client_eval.values() if "eval_acc" in v
        ]
        out.append(
            {
                "round": f"{i}/{num_rounds}",
                "round_index": i,
                "train_metrics": dict(train) if train is not None else None,
                # Weighted mean over clients, as FedAvg aggregates it.
                "evaluate_metrics_clientapp": dict(ev_c) if ev_c is not None else None,
                # Each client on its own local test split, kept separate so that
                # divergence between clients remains visible.
                "client_eval_metrics": per_client_eval or None,
                "client_eval_acc_spread": (
                    max(accs) - min(accs) if len(accs) > 1 else None
                ),
                # The global model scored on data held out from every client.
                "evaluate_metrics_serverapp": dict(ev_s) if ev_s is not None else None,
                "server_eval_time": server_eval_times.get(i),
                "server_wait_time": wait,
                "client_train_times": per_client or None,
                "slowest_train_time": slowest,
                # Everything the server waited for that was not client training:
                # transfer over the link + ClientApp startup. This is the number
                # that should grow with extra network hops.
                "overhead_time": (
                    wait - slowest if wait is not None and slowest is not None else None
                ),
            }
        )
    return out


def save_run_artifacts(
    result: Result,
    context: Context,
    num_rounds: int,
    *,
    wait_times: dict[int, float] | None = None,
    client_times: dict[int, dict[str, float]] | None = None,
    client_eval: dict[int, dict[str, dict[str, float]]] | None = None,
    server_eval_times: dict[int, float] | None = None,
    total_time: float | None = None,
    runs_root: Path | str = "flwr_runs",
    latest_model_path: Path | str = "final_model.pt",
) -> Path:
    """Write timestamped ``final_model.pt``, ``metrics.json``, ``run_config.json``; also latest model path.

    Returns the directory created for this run (under ``runs_root``).
    """
    print("\nSaving final model and metrics to disk...")
    wait_times = wait_times or {}
    client_times = client_times or {}
    client_eval = client_eval or {}
    server_eval_times = server_eval_times or {}

    state_dict = result.arrays.to_torch_state_dict()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(runs_root) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(state_dict, run_dir / "final_model.pt")
    rounds = _rounds_with_labels(
        num_rounds, result, wait_times, client_times, client_eval, server_eval_times
    )
    train_times = [
        r["train_metrics"]["train_time"]
        for r in rounds
        if r["train_metrics"] and "train_time" in r["train_metrics"]
    ]
    overheads = [r["overhead_time"] for r in rounds if r["overhead_time"] is not None]

    per_client_time: dict[str, list[float]] = {}
    for rnd in client_times.values():
        for node, t in rnd.items():
            per_client_time.setdefault(node, []).append(t)

    # Final-round accuracy of each client on its own data, kept per node.
    final_client_eval = rounds[-1]["client_eval_metrics"] if rounds else None
    final_server_eval = rounds[-1]["evaluate_metrics_serverapp"] if rounds else None
    # The strategy also evaluates the initial model, before round 1.
    initial_server_eval = result.evaluate_metrics_serverapp.get(0)

    artifact = {
        "saved_at_utc": stamp,
        "num_rounds": num_rounds,
        "totals": {
            # Wall clock for the whole federated run (all rounds, train + evaluate).
            "total_time": total_time,
            # Time the server spent waiting on clients (compute + network).
            "total_server_wait_time": sum(wait_times.values()) or None,
            # Client-side compute only, summed over rounds (average over clients).
            "total_train_time": sum(train_times) or None,
            # Transfer + ClientApp startup, i.e. wait minus the slowest client.
            "total_overhead_time": sum(overheads) or None,
            "mean_overhead_per_round": (
                sum(overheads) / len(overheads) if overheads else None
            ),
            # Server-side evaluation runs on the server and is not part of the
            # client wait; kept separate so the round budget stays interpretable.
            "total_server_eval_time": sum(server_eval_times.values()) or None,
            # Mean training time of each individual client across the run.
            "mean_train_time_per_client": {
                node: sum(v) / len(v) for node, v in sorted(per_client_time.items())
            }
            or None,
            # Global model on held-out data, before training and after the last round.
            "initial_server_eval": (
                dict(initial_server_eval) if initial_server_eval is not None else None
            ),
            "final_server_eval": final_server_eval,
            # Each client on its own data after the last round.
            "final_client_eval": final_client_eval,
        },
        "rounds": rounds,
    }
    (run_dir / "metrics.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    run_cfg = {k: context.run_config[k] for k in context.run_config}
    (run_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    torch.save(state_dict, latest_model_path)
    print(f"Artifacts: {run_dir.resolve()}")
    return run_dir
