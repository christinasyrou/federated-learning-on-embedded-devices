"""fedavg: Flower client app (local training aligned with FedAvg rounds)."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fedavg.task import ResNet, load_data_from_disk
from fedavg.task import test as test_fn
from fedavg.task import train as train_fn

app = ClientApp()

# Cache dataloaders across rounds (dataset on disk does not change during a run).
_LOADER_CACHE: dict[tuple[str, int, int], tuple] = {}


def _get_loaders(dataset_path: str, batch_size: int, max_train_samples: int):
    key = (dataset_path, batch_size, max_train_samples)
    if key not in _LOADER_CACHE:
        _LOADER_CACHE[key] = load_data_from_disk(
            dataset_path, batch_size, max_train_samples
        )
    return _LOADER_CACHE[key]


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]
    batch_size = context.run_config["batch-size"]
    max_train_samples = int(context.run_config.get("max-train-samples", 0))

    model = ResNet()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset_path = context.node_config["dataset-path"]
    trainloader, _ = _get_loaders(dataset_path, batch_size, max_train_samples)

    train_loss = train_fn(
        model,
        trainloader,
        local_epochs,
        learning_rate,
        device,
    )

    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    batch_size = context.run_config["batch-size"]
    max_train_samples = int(context.run_config.get("max-train-samples", 0))

    model = ResNet()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset_path = context.node_config["dataset-path"]
    _, valloader = _get_loaders(dataset_path, batch_size, max_train_samples)

    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
