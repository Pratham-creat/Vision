# Model-aware image protection research

Vision's `adversarial` mode is an optional, local PyTorch optimization path. It optimizes a small, bounded RGB perturbation against two differentiable surrogate models:

- **Identity representation:** lower cosine similarity between the source and protected image embeddings.
- **Facial geometry:** increase normalized landmark / face-structure divergence.

The optimizer uses an L-infinity pixel budget, a small total-variation regularizer, and expectation-over-transformations (small resize and photometric changes). Original uploads are retained; the protected result is written to the normal modified-image slot and receives the normal image-quality metrics.

## Configure local models

Install the optional PyTorch dependency using a wheel appropriate for the machine's CPU or CUDA runtime:

```powershell
python -m pip install -r backend/requirements-protection.txt
```

Create a local Python adapter module. Its `load_models()` factory must return an object exposing two PyTorch modules:

- `identity_model`: accepts BCHW RGB float tensors in [0, 1], returns a batch-first identity embedding tensor.
- `geometry_model`: accepts BCHW RGB float tensors in [0, 1], returns a batch-first landmarks or face-structure tensor.

Set the environment variable before starting the API:

```powershell
$env:VISION_PROTECTION_ADAPTER = "my_vision_models:load_models"
uvicorn backend.app.main:app --reload --port 8000
```

The adapter and model weights are intentionally not bundled or downloaded. Select models whose licenses permit this use. Keep model preprocessing inside the adapter modules; do not silently normalize or resize in a way that invalidates the [0, 1] RGB adapter contract.

## API

`POST /api/v1/protection/optimize`

```json
{
  "image_id": "uploaded-image-uuid",
  "budget": 4,
  "iterations": 20,
  "seed": 42,
  "eot_samples": 3
}
```

The response includes image URLs, perceptual metrics, the maximum per-channel displacement, and before/after surrogate objective values. A missing adapter returns HTTP 503. It does not silently fall back to random noise.

## Evaluation protocol

Use consented images and compare original, RGB, chrominance, perceptual, combined, and adversarial outputs. Keep the same detector / recognition / morphing model, model version, prompt, preprocessing, and transformation settings. Report:

- Identity cosine similarity and verification TAR/FAR at fixed thresholds.
- Landmark normalized mean error and face detection / alignment success.
- Morph / swap success rate on a consented evaluation set.
- PSNR, SSIM, mean CIEDE2000, and human visual inspection.
- Robustness after JPEG recompression, resize, crop, screenshot, and color-space conversion.
- Held-out model performance; do not evaluate only on the models used to optimize the perturbation.

A successful reduction on the two surrogate models is not proof of transfer to arbitrary AI systems. This implementation is not a universal shield and cannot make an uploaded static image dynamically disintegrate. It can only test whether bounded changes reduce performance of specified models.
