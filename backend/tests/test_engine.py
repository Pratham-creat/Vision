import numpy as np
from PIL import Image

from engine.color_space.conversions import lab_to_rgb, rgb_to_lab
from engine.metrics.difference_map import make_difference_map
from engine.metrics.perceptual_metrics import calculate_metrics
from engine.perturbation.processors import PerturbationConfig, apply_perturbation


def sample() -> np.ndarray:
    return np.full((12, 16, 3), [80, 120, 160], dtype=np.uint8)


def test_rgb_bounds_and_dimensions():
    result = apply_perturbation(sample(), PerturbationConfig(method="rgb", budget=10, seed=3))
    assert result.shape == sample().shape
    assert result.min() >= 0 and result.max() <= 255


def test_seed_is_reproducible():
    config = PerturbationConfig(method="chrominance", budget=4, seed=9)
    assert np.array_equal(apply_perturbation(sample(), config), apply_perturbation(sample(), config))


def test_color_round_trip():
    image = sample()
    round_trip = lab_to_rgb(rgb_to_lab(image))
    assert np.mean(np.abs(image.astype(int) - round_trip.astype(int))) < 3


def test_metrics_and_difference_map():
    original = sample()
    modified = apply_perturbation(original, PerturbationConfig(method="rgb", budget=2))
    metrics = calculate_metrics(original, modified)
    assert 0 <= metrics["ssim"] <= 1
    assert metrics["maximum_pixel_difference"] >= 0
    assert make_difference_map(original, modified).size == (16, 12)


def test_export_input_unchanged():
    original = sample()
    before = original.copy()
    apply_perturbation(original, PerturbationConfig(method="perceptual", budget=2))
    assert np.array_equal(original, before)
