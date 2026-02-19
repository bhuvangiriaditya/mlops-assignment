from contextlib import contextmanager
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import src.train as train_module


def _mock_loaders():
    train_x = torch.rand(4, 3, 8, 8)
    train_y = torch.tensor([0, 1, 0, 1])
    val_x = torch.rand(2, 3, 8, 8)
    val_y = torch.tensor([0, 1])
    test_x = torch.rand(2, 3, 8, 8)
    test_y = torch.tensor([1, 0])

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=2, shuffle=False)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=1, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=1, shuffle=False)
    return train_loader, val_loader, test_loader


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 1),
        )

    def forward(self, x):
        return self.network(x)


def test_train_runs_and_saves_model(monkeypatch, tmp_path):
    logged_params = {}
    logged_metrics = []
    logged_model_paths = []

    @contextmanager
    def fake_run():
        yield

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_module, "SimpleCNN", TinyModel)
    monkeypatch.setattr(train_module, "get_dataloaders", lambda **_kwargs: _mock_loaders())
    monkeypatch.setattr(train_module.mlflow, "set_experiment", lambda _name: None)
    monkeypatch.setattr(train_module.mlflow, "start_run", fake_run)
    monkeypatch.setattr(train_module.mlflow, "log_param", lambda key, value: logged_params.setdefault(key, value))
    monkeypatch.setattr(
        train_module.mlflow,
        "log_metric",
        lambda key, value, step=None: logged_metrics.append((key, value, step)),
    )
    monkeypatch.setattr(
        train_module.mlflow,
        "pytorch",
        SimpleNamespace(log_model=lambda _model, path: logged_model_paths.append(path)),
    )
    monkeypatch.setattr(train_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train_module.torch.backends.mps, "is_available", lambda: False)

    train_module.train(epochs=1, batch_size=2, lr=0.01)

    metric_names = {name for name, _, _ in logged_metrics}
    assert logged_params["epochs"] == 1
    assert logged_params["batch_size"] == 2
    assert "train_loss" in metric_names
    assert "val_accuracy" in metric_names
    assert "test_accuracy" in metric_names
    assert "model" in logged_model_paths
    assert (tmp_path / "model.pth").exists()


def test_train_returns_if_dataset_missing(monkeypatch):
    @contextmanager
    def fake_run():
        yield

    monkeypatch.setattr(train_module, "get_dataloaders", lambda **_kwargs: (None, None, None))
    monkeypatch.setattr(train_module.mlflow, "set_experiment", lambda _name: None)
    monkeypatch.setattr(train_module.mlflow, "start_run", fake_run)
    monkeypatch.setattr(train_module.mlflow, "log_param", lambda *_args, **_kwargs: None)

    train_module.train(epochs=1, batch_size=2, lr=0.01)
