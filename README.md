# VISION

VISION is a local, research-oriented image perturbation workstation. It applies bounded RGB, chrominance, or CIELAB-constrained perturbations and reports measurable similarity results. It does **not** claim universal protection against machine-vision systems; AI evaluation is an optional adapter surface and is clearly reported as unavailable until configured.

## Architecture

- `engine/` — independent NumPy/Pillow/OpenCV/scikit-image processing modules.
- `backend/` — FastAPI routes, Pydantic validation, SQLite experiment history, and safe image storage.
- `frontend/` — React + Vite + TypeScript + Tailwind UI with a compact imaging-workstation layout.

## Requirements and installation

Python 3.11+ and Node.js 20+ are recommended.

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cd frontend
npm install
```

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
Set-Location frontend
npm install
```

## Run

From the repository root, start the API:

```bash
uvicorn backend.app.main:app --reload --port 8000
```

In another terminal:

```bash
cd frontend
npm run dev
```

PowerShell equivalents:

```powershell
uvicorn backend.app.main:app --reload --port 8000
Set-Location frontend
npm run dev
```

Open `http://localhost:5173`. Configure `VITE_API_URL` if the backend is hosted elsewhere. The backend stores files and the SQLite database under `storage/`; the original upload is never overwritten.

## API

- `GET /health`
- `GET /api/v1/capabilities`
- `POST /api/v1/images/upload` multipart field `file`
- `POST /api/v1/images/process` JSON: `image_id`, `method`, `budget`, `iterations`, `seed`
- `POST /api/v1/images/compare`
- `GET /api/v1/images/history`
- `GET /api/v1/images/{image_id}`
- `GET /api/v1/images/{image_id}/download?variant=original|modified|difference`
- `POST /api/v1/evaluation/metrics`
- `POST /api/v1/evaluation/ai`

Interactive OpenAPI documentation is available at `/docs`.

## Perturbation methods

RGB mode applies reproducible, bounded channel noise. Chrominance mode changes YCrCb chroma while leaving luminance separate. Perceptual mode proposes CIELAB chroma changes, rescales them against CIEDE2000, then clips safely back to RGB. These methods are image-processing controls, not guaranteed model defenses.

## Metrics

The backend computes PSNR, SSIM, mean CIEDE2000, mean absolute RGB difference, maximum channel difference, changed-pixel percentage, dimensions, file sizes, and processing duration. LPIPS is intentionally not required so the core application remains lightweight and deterministic; it can be added as a separate optional adapter.

## Testing

```bash
pytest backend/tests
```

```powershell
python -m pytest backend\tests
```

Tests cover perturbation bounds, dimensions, reproducibility, color conversion, metrics, difference maps, upload validation, API processing, export behavior, and preservation of the input array.

## Research methodology and limitations

Record the method, budget, iterations, seed, metrics, and processing time for each experiment. Compare original and modified images under controlled settings, then evaluate compatible models through an explicit adapter that returns labels and confidence values. No API key is bundled, no model result is fabricated, and an unavailable adapter is reported as “AI evaluation not configured.” Results are image/model/dataset specific and should not be a generalized without an appropriate benchmark.
