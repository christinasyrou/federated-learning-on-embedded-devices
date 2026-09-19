"""fedavg: Flower server app using the FedAvg aggregation strategy."""

import time
from collections.abc import Callable, Iterable
from logging import INFO, WARNING
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord
from flwr.common import log
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fedavg.run_artifacts import save_run_artifacts
from fedavg.task import ResNet, load_server_test_from_disk
from fedavg.task import test as test_fn

# Client metrics recorded per node, in addition to the aggregated mean.
_CLIENT_EVAL_KEYS = ("eval_loss", "eval_acc", "num-examples")


class TimedFedAvg(FedAvg):
    """FedAvg that keeps per-node values the aggregated metrics would average away.

    Metrics returned by clients are combined into a weighted *mean*, which hides
    two things this project needs. A round ends only when the *slowest* client
    replies, so the mean of ``train_time`` describes no client and understates the
    wait. And a mean accuracy cannot show one client diverging from another, which
    is the signature of client drift under non-IID data.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.wait_times: dict[int, float] = {}
        # round -> {node_id: train_time}
        self.client_times: dict[int, dict[str, float]] = {}
        # round -> {node_id: {eval_loss, eval_acc, num-examples}}
        self.client_eval: dict[int, dict[str, dict[str, float]]] = {}
        self._t0 = 0.0

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        # Build the messages first so model serialization is not counted as waiting.
        messages = list(super().configure_train(server_round, arrays, config, grid))
        self._t0 = time.perf_counter()
        return messages

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        self.wait_times[server_round] = time.perf_counter() - self._t0
        replies = list(replies)  # materialize: read below, then aggregate
        self.client_times[server_round] = {
            str(m.metadata.src_node_id): float(m.content["metrics"]["train_time"])
            for m in replies
            if not m.has_error() and "train_time" in m.content["metrics"]
        }
        return super().aggregate_train(server_round, replies)

    def aggregate_evaluate(
        self, server_round: int, replies: Iterable[Message]
    ) -> MetricRecord | None:
        replies = list(replies)
        per_node: dict[str, dict[str, float]] = {}
        for msg in replies:
            if msg.has_error():
                continue
            metrics = msg.content["metrics"]
            values = {k: float(metrics[k]) for k in _CLIENT_EVAL_KEYS if k in metrics}
            if values:
                per_node[str(msg.metadata.src_node_id)] = values
        self.client_eval[server_round] = per_node
        return super().aggregate_evaluate(server_round, replies)


def _should_evaluate(server_round: int, num_rounds: int, every: int) -> bool:
    """Decide whether the global model is scored after ``server_round``.

    Round 0 (the untrained baseline) and the final round are always scored, so a
    run never ends without a score for the model it produced; in between, every
    ``every``-th round is.
    """
    if server_round in (0, num_rounds):
        return True
    return server_round % every == 0


def make_evaluate_fn(
    test_path: str,
    batch_size: int,
    durations: dict[int, float],
    num_rounds: int,
    every: int = 1,
) -> Callable[[int, ArrayRecord], MetricRecord | None] | None:
    """Build a server-side evaluation function, or ``None`` if unavailable.

    The global model is scored on the held-out split written by
    ``generate_dataset.py``, which no client holds. This is the centralised
    measure used in the federated learning literature, and complements the
    per-client scores by saying how the model generalises rather than how it
    performs on each participant's own data.

    Evaluation costs a full forward pass over the held-out set on the server, so
    ``every`` controls how often it runs: 1 for every round, ``k`` for every k-th
    round, 0 to disable it entirely.

    Disabling is the right choice when only the end result matters: ``evaluation.py``
    scores the saved model afterwards, with a per-class breakdown and a confusion
    matrix, and without adding work to the run being timed.

    Returns ``None`` when disabled or when the test split is absent, so a run
    proceeds without centralized evaluation rather than failing.
    """
    if every <= 0:
        log(
            INFO,
            "Server-side evaluation disabled (server-eval-every=0). Score the saved "
            "model afterwards with: python evaluation.py",
        )
        return None

    if not Path(test_path).exists():
        log(
            WARNING,
            "Server test set not found at '%s'; skipping server-side evaluation. "
            "Generate it with `python generate_dataset.py` and copy it to this node.",
            test_path,
        )
        return None

    testloader = load_server_test_from_disk(test_path, batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet().to(device)
    schedule = "every round" if every == 1 else f"every {every} rounds"
    log(INFO, "Server-side evaluation enabled (%s): %d images from '%s'",
        schedule, len(testloader.dataset), test_path)

    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord | None:
        if not _should_evaluate(server_round, num_rounds, every):
            return None
        t0 = time.perf_counter()
        model.load_state_dict(arrays.to_torch_state_dict())
        loss, accuracy = test_fn(model, testloader, device)
        durations[server_round] = time.perf_counter() - t0
        return MetricRecord({"eval_loss": loss, "eval_acc": accuracy})

    return evaluate


# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    server_test_path = str(
        context.run_config.get("server-test-path", "datasets/fashionmnist_server_test")
    )
    server_eval_batch = int(context.run_config.get("server-eval-batch-size", 64))
    server_eval_every = int(context.run_config.get("server-eval-every", 1))

    # Load global model
    global_model = ResNet()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = TimedFedAvg(fraction_evaluate=fraction_evaluate)

    # Centralized evaluation of the global model, if the held-out split is present
    server_eval_times: dict[int, float] = {}
    evaluate_fn = make_evaluate_fn(
        server_test_path,
        server_eval_batch,
        server_eval_times,
        num_rounds,
        server_eval_every,
    )

    # Start strategy, running FedAvg for `num_rounds`
    t0 = time.perf_counter()
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        evaluate_fn=evaluate_fn,
    )
    total_time = time.perf_counter() - t0

    save_run_artifacts(
        result,
        context,
        num_rounds,
        wait_times=strategy.wait_times,
        client_times=strategy.client_times,
        client_eval=strategy.client_eval,
        server_eval_times=server_eval_times,
        total_time=total_time,
    )
