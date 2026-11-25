import numpy as np
import pytest

from engine.perturbation.adversarial import optimize_protection


def test_zero_budget_returns_exact_copy_without_loading_torch():
    image = np.random.default_rng(4).integers(0, 256, size=(12, 10, 3), dtype=np.uint8)
    result = optimize_protection(image, models=None, budget=0)
    assert np.array_equal(result.image, image)
    assert result.max_channel_delta == 0


@pytest.mark.parametrize("kwargs", [
    {"budget": -1}, {"budget": 33}, {"iterations": 0},
    {"iterations": 201}, {"eot_samples": 0}, {"eot_samples": 17},
])
def test_invalid_protection_parameters_are_rejected(kwargs):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        optimize_protection(image, models=None, **kwargs)


def test_invalid_image_is_rejected():
    with pytest.raises(ValueError):
        optimize_protection(np.zeros((8, 8), dtype=np.uint8), models=None)
