from PIL import Image
import pytest
import torch


from digit_classification.model import DigitClassifier

def test_model_output_shape():
    print("Testing Output shape")
    model = DigitClassifier()

    x = torch.randn(8, 1, 28, 28)
    logits = model(x)

    assert logits.shape == (8, 3)



@pytest.mark.parametrize(
    "hidden_dims",
    [
        [],
        [32],
        [64, 32],
        [128, 64, 32],
    ],
)
def test_model_architecture(hidden_dims):
    print("Testing Hidden Dim Param")
    model = DigitClassifier(hidden_dims=hidden_dims, pool_indexes=[])

    x = torch.randn(4, 1, 28, 28)
    logits = model(x)

    assert logits.shape == (4, 3)

def test_pool_indexes():
    print("Testing pool_indexes Param")
    model = DigitClassifier(hidden_dims=[], pool_indexes=[0, 1])

    x = torch.randn(4, 1, 28, 28)
    logits = model(x)

    assert logits.shape == (4, 3)


def test_model_loss():
    print("Testing loss function")
    model = DigitClassifier()

    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 3, (8,))

    logits = model(x)
    loss = model.loss_fn(logits, y)

    assert loss.ndim == 0
    assert torch.isfinite(loss)

def test_training_step():
    print("Testing training step")
    model = DigitClassifier()

    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 3, (8,))

    loss = model.training_step((x, y), 0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
