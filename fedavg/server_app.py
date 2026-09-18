"""fedavg: Flower server app using the FedAvg aggregation strategy."""

import time
from collections.abc import Iterable

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fedavg.run_artifacts import save_run_artifacts
from fedavg.task import ResNet


class TimedFedAvg(FedAvg):
    """FedAvg that records the server's wait plus each client's own train time.

    ``train_time`` in the aggregated metrics is a weighted *average* over clients,
    but a round ends only when the *slowest* client replies. Keeping the per-node
    values makes a straggler (e.g. a node several hops away) visible as itself.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.wait_times: dict[int, float] = {}
        # round -> {node_id: train_time}
        self.client_times: dict[int, dict[str, float]] = {}
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


# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]

    # Load global model
    global_model = ResNet()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = TimedFedAvg(fraction_evaluate=fraction_evaluate)

    # Start strategy, running FedAvg for `num_rounds`
    t0 = time.perf_counter()
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )
    total_time = time.perf_counter() - t0

    save_run_artifacts(
        result,
        context,
        num_rounds,
        wait_times=strategy.wait_times,
        client_times=strategy.client_times,
        total_time=total_time,
    )
