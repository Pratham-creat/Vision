import math

import numpy as np

from engine.color_space.conversions import lab_to_rgb, rgb_to_lab, rgb_to_ycrcb, ycrcb_to_rgb
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
    lab_round_trip = lab_to_rgb(rgb_to_lab(image))
    assert np.mean(np.abs(image.astype(int) - lab_round_trip.astype(int))) < 3
    ycrcb_round_trip = ycrcb_to_rgb(rgb_to_ycrcb(image))
    assert np.mean(np.abs(image.astype(int) - ycrcb_round_trip.astype(int))) < 1


def test_ycrcb_round_trip_on_varied_pixels():
    image = np.random.default_rng(17).integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
    round_trip = ycrcb_to_rgb(rgb_to_ycrcb(image))
    assert np.mean(np.abs(image.astype(np.int16) - round_trip.astype(np.int16))) < 1


def test_metrics_and_difference_map():
    original = sample()
    modified = apply_perturbation(original, PerturbationConfig(method="rgb", budget=2))
    metrics = calculate_metrics(original, modified)
    assert 0 <= metrics["ssim"] <= 1
    assert metrics["maximum_pixel_difference"] >= 0
    assert make_difference_map(original, modified).size == (16, 12)


def test_zero_budget_and_small_image_metrics():
    for size in (1, 2, 5, 7):
        original = np.full((size, size, 3), [80, 120, 160], dtype=np.uint8)
        for method in ("rgb", "chrominance", "perceptual"):
            modified = apply_perturbation(original, PerturbationConfig(method=method, budget=0))
            assert np.array_equal(original, modified)
            metrics = calculate_metrics(original, modified)
            assert metrics["psnr"] == 100.0
            assert math.isfinite(metrics["ssim"])
            assert metrics["ssim"] == 1.0


def test_export_input_unchanged():
    original = sample()
    before = original.copy()
    apply_perturbation(original, PerturbationConfig(method="perceptual", budget=2))
    assert np.array_equal(original, before)
