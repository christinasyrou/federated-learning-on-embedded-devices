"""Post-hoc evaluation of saved federated final models (accuracy + macro-F1).

Loads each run's ``{dataset}_final_model.pt``, runs ``test()`` on every client
partition, and writes ``{dataset}_posthoc_eval.json`` into the run folder.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from fedavg.task import create_model, load_data_from_disk, test

DEFAULT_RUN_DIRS = [
    "flwr_runs/cifar10_20260529_114124",
    "flwr_runs/cifar10_20260528_144603",
    "flwr_runs/cifar10_20260530_210916",
    "flwr_runs/cifar10_20260528_232524",
]


def _dataset_slug(run_dir: Path) -> str:
    prefix = run_dir.name.split("_")[0]
    return prefix or "unknown"


def evaluate_run(
    run_dir: Path,
    *,
    partitions_root: Path,
    num_partitions: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    """Evaluate one saved final model on all client test splits."""
    dataset = _dataset_slug(run_dir)
    config_path = run_dir / f"{dataset}_run_config.json"
    model_path = run_dir / f"{dataset}_final_model.pt"

    if not config_path.is_file():
        raise FileNotFoundError(f"Missing run config: {config_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing final model: {model_path}")

    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    architecture = run_config["model-architecture"]

    model = create_model(architecture)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)

    per_client: list[dict] = []
    total_examples = 0
    weighted_acc = 0.0
    weighted_f1 = 0.0
    weighted_loss = 0.0

    for partition_id in range(1, num_partitions + 1):
        partition_path = partitions_root / f"cifar10_part_{partition_id}"
        if not partition_path.is_dir():
            raise FileNotFoundError(f"Missing partition: {partition_path}")

        _, valloader = load_data_from_disk(str(partition_path), batch_size)
        eval_loss, eval_acc, eval_f1 = test(model, valloader, device)
        num_examples = len(valloader.dataset)

        per_client.append(
            {
                "partition": f"cifar10_part_{partition_id}",
                "dataset_path": str(partition_path.as_posix()),
                "num_examples": num_examples,
                "eval_loss": eval_loss,
                "eval_acc": eval_acc,
                "eval_f1": eval_f1,
            }
        )

        total_examples += num_examples
        weighted_acc += eval_acc * num_examples
        weighted_f1 += eval_f1 * num_examples
        weighted_loss += eval_loss * num_examples

    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir.as_posix()),
        "dataset": dataset,
        "model_architecture": architecture,
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "partitions_root": str(partitions_root.as_posix()),
        "batch_size": batch_size,
        "device": str(device),
        "aggregate": {
            "num_examples": total_examples,
            "eval_loss": weighted_loss / total_examples,
            "eval_acc": weighted_acc / total_examples,
            "eval_f1": weighted_f1 / total_examples,
        },
        "per_client": per_client,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved final models on all client test partitions."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        dest="run_dirs",
        help="Run directory under flwr_runs/ (repeatable). Defaults to the four documented CIFAR-10 runs.",
    )
    parser.add_argument(
        "--partitions-root",
        type=Path,
        default=Path("datasets/5-nodes-partition"),
        help="Directory containing cifar10_part_1 … cifar10_part_N (default: datasets/5-nodes-partition).",
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=5,
        help="Number of client partitions to evaluate (default: 5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Evaluation batch size (default: 32).",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("flwr_runs/posthoc_eval_summary.json"),
        help="Combined summary JSON path (default: flwr_runs/posthoc_eval_summary.json).",
    )
    args = parser.parse_args()

    run_dirs = [Path(p) for p in (args.run_dirs or DEFAULT_RUN_DIRS)]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    results: list[dict] = []
    for run_dir in run_dirs:
        print(f"Evaluating {run_dir} …")
        result = evaluate_run(
            run_dir,
            partitions_root=args.partitions_root,
            num_partitions=args.num_partitions,
            batch_size=args.batch_size,
            device=device,
        )
        results.append(result)

        out_name = f"{result['dataset']}_posthoc_eval.json"
        out_path = run_dir / out_name
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        agg = result["aggregate"]
        print(
            f"  acc={agg['eval_acc']:.4f}  f1={agg['eval_f1']:.4f}  "
            f"loss={agg['eval_loss']:.4f}  -> {out_path}"
        )

    summary = {
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "partitions_root": str(args.partitions_root.as_posix()),
        "runs": results,
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary written to {args.summary_out.resolve()}")


if __name__ == "__main__":
    main()
