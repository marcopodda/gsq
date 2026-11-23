from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.nn import MeanAggregation

from gsq.model import (
    SLD,
    APPNPEncoder,
    DeepSetsPooler,
    DirichletHead,
    GMNetPooler,
    KDEyMaximumLikelihood,
    MSEHead,
    NodeEncoder,
    PrevalenceCalibrator,
    ReservoirEncoder,
)
from gsq.model.network import (
    AggregativeQuantificationNetwork,
    QuantificationNetwork,
)
from gsq.model.reservoir_cache import load_or_compute_reservoir


@pytest.fixture
def graph():
    x = torch.randn(8, 4)
    edges = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 0, 4], [1, 2, 3, 0, 5, 6, 7, 4, 4, 0]]
    )
    return x, edges


@pytest.mark.parametrize("kind", ["mlp", "gcn"])
def test_trainable_encoders_follow_the_node_contract(graph, kind):
    x, edges = graph
    encoder = NodeEncoder(4, 6, 2, encoder_type=kind, dropout=0)
    output = encoder(x, edges)
    assert output.shape == (8, 6)
    output.sum().backward()
    assert all(parameter.grad is not None for parameter in encoder.parameters())


def test_appnp_encoder_follows_the_node_contract(graph):
    x, edges = graph
    output = APPNPEncoder(4, 6, 2, propagation_steps=3)(x, edges)
    assert output.shape == (8, 6)


@pytest.mark.parametrize("pooler_type", [DeepSetsPooler, GMNetPooler])
def test_learned_poolers_are_permutation_invariant(pooler_type):
    torch.manual_seed(0)
    kwargs = {"channels": 4}
    if pooler_type is GMNetPooler:
        kwargs.update(num_views=2, num_gaussians=3, latent_channels=2, dropout=0)
    pooler = pooler_type(**kwargs).eval()
    x = torch.randn(6, 4)
    groups = torch.tensor([0, 0, 0, 1, 1, 1])
    permutation = torch.tensor([2, 0, 1, 5, 3, 4])
    torch.testing.assert_close(
        pooler(x, groups), pooler(x[permutation], groups[permutation])
    )


@pytest.mark.parametrize("head_type", [MSEHead, DirichletHead])
def test_heads_return_simplex_prevalences(head_type):
    head = head_type(4, 3)
    target = torch.tensor([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]])
    loss, prediction = head(torch.randn(2, 4), target)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert torch.all(prediction >= 0)
    torch.testing.assert_close(prediction.sum(1), torch.ones(2))


@pytest.mark.parametrize("classwise", [False, True])
def test_calibrators_preserve_the_simplex(classwise):
    calibrator = PrevalenceCalibrator(3, classwise=classwise)
    prediction = calibrator(torch.tensor([[0.2, 0.3, 0.5], [0.8, 0.1, 0.1]]))
    assert torch.all(prediction >= 0)
    torch.testing.assert_close(prediction.sum(1), torch.ones(2))


def test_gesn_cache_is_reusable(graph, tmp_path):
    x, edge_index = graph
    data = Data(x=x, edge_index=edge_index)
    encoder = ReservoirEncoder(4, 5, recurrent_scale=1, input_scale=0.5, max_iterations=2)
    path = tmp_path / "gesn.pt"
    first, hit = load_or_compute_reservoir(encoder, data, path)
    second, hit_again = load_or_compute_reservoir(
        ReservoirEncoder(4, 5, recurrent_scale=1, input_scale=0.5, max_iterations=2),
        data,
        path,
    )
    assert not hit and hit_again
    torch.testing.assert_close(first, second)


def test_aggregators_return_simplex_prevalences():
    probabilities = torch.tensor(
        [[0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.1, 0.9]],
        dtype=torch.float64,
    )
    labels = torch.tensor([0, 0, 1, 1])
    groups = torch.tensor([0, 0, 1, 1])
    for rule in (SLD(2), KDEyMaximumLikelihood(2)):
        rule.fit(probabilities, labels, probabilities, labels)
        prediction = rule.predict_many(probabilities, groups, 2)
        assert prediction.shape == (2, 2)
        assert torch.all(prediction >= 0)
        torch.testing.assert_close(prediction.sum(1), torch.ones(2, dtype=torch.float64))


def test_mean_pooler_remains_compatible():
    x = torch.randn(4, 3)
    groups = torch.tensor([0, 0, 1, 1])
    assert MeanAggregation()(x, groups).shape == (2, 3)


def _training_graph() -> Data:
    data = Data(
        x=torch.randn(24, 4),
        y=torch.arange(24) % 2,
        edge_index=torch.stack((torch.arange(24), torch.arange(24))),
    )
    data.train_mask = torch.arange(24) < 12
    data.cal_mask = (torch.arange(24) >= 6) & (torch.arange(24) < 12)
    data.val_mask = (torch.arange(24) >= 12) & (torch.arange(24) < 18)
    data.test_mask = torch.arange(24) >= 18
    return data


def _protocol():
    return OmegaConf.create(
        {
            "name": "app",
            "sample_size": 4,
            "repeats": 2,
            "train_repeats": 2,
            "validation_repeats": 2,
        }
    )


def test_symmetric_network_trains_and_calibrates():
    model = QuantificationNetwork(
        encoder=OmegaConf.create(
            {
                "_target_": "gsq.model.NodeEncoder",
                "in_channels": 4,
                "hidden_channels": 5,
                "num_layers": 1,
                "encoder_type": "mlp",
                "dropout": 0,
            }
        ),
        pooler=OmegaConf.create({"_target_": "torch_geometric.nn.MeanAggregation"}),
        head=OmegaConf.create(
            {"_target_": "gsq.model.MSEHead", "in_channels": 5, "num_classes": 2}
        ),
        optimizer=OmegaConf.create({"_target_": "torch.optim.AdamW", "lr": 1e-3}),
        scheduler=OmegaConf.create({}),
        num_classes=2,
        calibration_mode="scalar",
        calibration_max_steps=2,
        bag_protocol=_protocol(),
    )
    data = _training_graph()
    assert torch.isfinite(model.training_step(data, 0))
    assert set(model.fit_calibrators(data)) == {"ae/cal", "ae/val"}


def test_aggregative_network_trains_and_calibrates():
    model = AggregativeQuantificationNetwork(
        encoder=OmegaConf.create(
            {
                "_target_": "gsq.model.NodeEncoder",
                "in_channels": 4,
                "hidden_channels": 5,
                "num_layers": 1,
                "encoder_type": "mlp",
                "dropout": 0,
            }
        ),
        classifier=OmegaConf.create(
            {"_target_": "torch.nn.Linear", "in_features": 5, "out_features": 2}
        ),
        quantifier=OmegaConf.create({"_target_": "gsq.model.SLD", "num_classes": 2}),
        optimizer=OmegaConf.create({"_target_": "torch.optim.AdamW", "lr": 1e-3}),
        scheduler=OmegaConf.create({}),
        num_classes=2,
        calibration_mode="scalar",
        calibration_max_steps=2,
        bag_protocol=_protocol(),
    )
    data = _training_graph()
    assert torch.isfinite(model.training_step(data, 0))
    assert set(model.fit_calibrators(data)) == {"ae/cal", "ae/val"}
