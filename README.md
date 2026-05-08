# Project: Active Contour Descriptor Evaluation

You are helping build a course project for a graduate deformable models class. The project evaluates how well shape descriptors derived from Kass-Witkin active contours hold up under controlled image deformations, compared to embeddings from a pretrained CNN. Evaluation is done via shape retrieval on a subset of MPEG-7.

This is an academic project. Code clarity, reproducibility, and correctness matter more than performance. Total budget is ~16–20 hours of human work across a 4-person team, so do not over-engineer.

---

## Build Order

Build the project in this order. Do NOT jump ahead. After each step, stop and let me verify it works before moving on.

1. Repo scaffolding (pyproject.toml, directory structure, config module)
2. Data loading and preprocessing
3. Active contour implementation + sanity test on a synthetic disk
4. Fourier descriptor extraction
5. Curvature scale-space descriptor extraction
6. CNN embedding extraction
7. Image deformation functions
8. Retrieval index and metrics
9. Experiment orchestrator
10. Top-level scripts that run end-to-end
11. Figure generation

**Stop after step 3 for a manual sanity check.** The snake must converge on a disk before anything else is worth building.

---

## Tech Stack

- Python 3.10+, managed with `uv` (preferred) or `poetry`
- `numpy`, `scipy`, `scikit-image`, `opencv-python`, `matplotlib`, `pandas`, `tqdm`
- `pyefd` for elliptic Fourier descriptors
- `torch` + `timm` for CNN embeddings (CPU-only, no training)
- `sklearn` for nearest neighbor (no FAISS — overkill at this scale)
- `pytest` for tests

No GPU required. No model training. No web UI. No interactive demos.

---

## Repository Layout

```
.
├── README.md
├── pyproject.toml
├── data/
│   ├── raw/                # MPEG-7 dump
│   └── subset/             # generated 300-shape subset
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── snake.py
│   ├── descriptors.py
│   ├── embeddings.py
│   ├── deformations.py
│   ├── retrieval.py
│   └── experiments.py
├── scripts/
│   ├── 00_download_dataset.py
│   ├── 01_build_subset.py
│   ├── 02_extract_contours.py
│   ├── 03_extract_descriptors.py
│   ├── 04_extract_embeddings.py
│   ├── 05_run_deformation_experiments.py
│   └── 06_generate_figures.py
├── results/
│   ├── tables/
│   └── figures/
└── tests/
    ├── test_snake.py
    ├── test_descriptors.py
    └── test_deformations.py
```

---

## Module Specs

### `src/config.py`

A single `@dataclass` called `Config` with all hyperparameters and paths. No magic numbers anywhere else in the codebase. Include:

- **Paths:** `data_raw_dir`, `data_subset_dir`, `results_tables_dir`, `results_figures_dir` (all `pathlib.Path`)
- **Subset:** `n_classes = 30`, `shapes_per_class = 10`, `subset_seed = 42`
- **Snake:** `alpha_grid = [0.001, 0.01, 0.1]`, `beta_grid = [0.1, 1.0, 10.0]`, `n_iterations = 250`, `n_contour_points = 200`, `gamma = 0.1`, `tau = 1.0`
- **Fourier descriptor:** `n_fourier_harmonics = 20`
- **Curvature scale-space:** `css_n_levels = 30`, `css_sigma_step = 1.0`, `css_grid_shape = (32, 30)`
- **Deformations:** `rotation_angles = [0, 30, 60, 90]`, `scale_factors = [0.5, 0.75, 1.0, 1.5]`, `shear_factors = [0.0, 0.1, 0.2, 0.3]`, `noise_sigmas = [0.0, 0.05, 0.1, 0.2]`
- **Embedding:** `cnn_models = ["resnet50", "dinov2_vits14"]`, `embedding_image_size = 224`
- **Retrieval:** `k_for_map = 10`, `k_for_precision = 5`, `bullseye_multiplier = 2`

### `src/data.py`

- `download_mpeg7(cfg)` — fetch MPEG-7 CE-Shape-1. Try academic torrent first; if unavailable, print instructions for manual download and exit cleanly.
- `build_subset(cfg) -> None` — deterministically pick `n_classes` classes (alphabetical), then `shapes_per_class` shapes per class (sorted by filename, take first N). Save to `data/subset/` as `{class_name}_{idx}.png`.
- `load_subset(cfg) -> List[Tuple[np.ndarray, str]]` — return list of `(binary_mask, class_label)` pairs.
- `preprocess_mask(mask: np.ndarray) -> np.ndarray` — ensure binary uint8, find largest connected component, center the shape on a 256×256 canvas, scale so the bounding box fits within ~80% of the canvas.

### `src/snake.py`

Implement Kass-Witkin snake **from scratch** using the standard implicit Euler formulation. Do NOT just wrap `skimage.segmentation.active_contour` — that defeats the course point. You may use it as a reference for sanity-checking output.

Required:

- `class Snake:`
  - `__init__(alpha, beta, gamma=0.1, tau=1.0, n_iterations=250)`
  - `fit(image: np.ndarray, init_contour: np.ndarray) -> np.ndarray` — returns final `(N, 2)` contour
  - Builds banded sparse pentadiagonal matrix `A = α·D2 + β·D4` where `D2`, `D4` are second/fourth circular difference matrices of size `N × N`
  - Implicit Euler step: `x_new = (γI + τA)^-1 · (γ·x_old - τ·F_external(x_old))`
  - External force: gradient of `-|∇(G_σ * I)|²` (negative gradient of Gaussian-smoothed edge magnitude)
  - Resample contour to maintain uniform arc-length spacing every few iterations
- `default_init_contour(mask, n_points) -> np.ndarray` — circle around mask centroid with radius slightly larger than the mask's bounding box
- Document the energy function in the class docstring with the actual math (LaTeX-style ASCII is fine)

**Sanity test (in `tests/test_snake.py`):** running `Snake(alpha=0.01, beta=1.0)` on a 256×256 disk image should converge to a near-circular contour. Mean radial deviation from the disk's true radius should be < 5 pixels.

### `src/descriptors.py`

- `elliptic_fourier(contour: np.ndarray, n_harmonics: int = 20, normalize: bool = True) -> np.ndarray`
  - Wrap `pyefd.elliptic_fourier_descriptors` and `pyefd.normalize_efd`
  - Return flat 1D vector, L2-normalized
- `curvature_scale_space(contour: np.ndarray, n_levels: int = 30, sigma_step: float = 1.0, grid_shape: Tuple[int, int] = (32, 30)) -> np.ndarray`
  - Implement from scratch:
    1. Parameterize contour by arc length, resample to uniform spacing
    2. For σ in `[1, 2, ..., n_levels]`, smooth contour coordinates with 1D Gaussian (separately on x and y)
    3. Compute curvature κ at each point: `κ = (x'·y'' - y'·x'') / (x'² + y'²)^(3/2)`
    4. Find zero-crossings of κ at each scale
    5. Bin (normalized_arc_position, scale_index) pairs into a `grid_shape` 2D histogram
    6. Flatten and L2-normalize

Both functions return 1D `numpy` arrays.

### `src/embeddings.py`

- `extract_embeddings(masks: List[np.ndarray], model_name: str = "resnet50", cfg: Config) -> np.ndarray`
- Support `resnet50` (via `timm.create_model("resnet50", pretrained=True, num_classes=0)`) and `dinov2_vits14` (via `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")`)
- Convert binary masks to 3-channel RGB by tiling, resize to `cfg.embedding_image_size`, normalize with ImageNet stats
- Take the model's pooled output (no classification head)
- L2-normalize each embedding
- Cache results to `results/cache/embeddings_{model_name}.npy` keyed by a hash of the input

### `src/deformations.py`

All functions take a binary mask `(H, W)` and return a binary mask of the same size. All take an explicit `seed` argument where randomness is involved.

- `rotate(mask, angle_degrees)` — `cv2.warpAffine` with nearest-neighbor interpolation, preserve binary nature
- `scale(mask, factor)` — scale image, then center-pad or center-crop to original size
- `affine_warp(mask, shear_x, shear_y)` — small shears via `cv2.warpAffine`
- `tps_perturb(mask, n_control_points=8, max_displacement=5, seed=None)` — thin-plate spline with random small displacements at control points; use `scipy.interpolate.RBFInterpolator` or implement directly
- `add_noise_to_grayscale(image, sigma, seed=None)` — additive Gaussian noise on a grayscale rendering of the mask (use the mask itself as the grayscale image)

All deformations must be deterministic given a seed.

### `src/retrieval.py`

- `class RetrievalIndex:`
  - `__init__(vectors: np.ndarray, labels: List[str], metric: str = "cosine")`
  - `query(vector: np.ndarray, k: int) -> List[Tuple[int, float, str]]` — returns list of `(index, similarity, label)`
  - Use `sklearn.neighbors.NearestNeighbors` with `metric="cosine"`
- `mean_average_precision(index, queries, query_labels, k=10) -> float`
- `precision_at_k(index, queries, query_labels, k=5) -> float`
- `bullseye_score(index, queries, query_labels, class_size=10, multiplier=2) -> float`
  - For each query, count how many of the top `multiplier × class_size` retrievals share its label
  - Divide by `class_size`, clip to `[0, 1]`, average across queries

### `src/experiments.py`

The orchestrator. All functions write structured CSV to `results/tables/` with columns: `representation, alpha, beta, deformation_type, deformation_level, metric_name, metric_value`.

- `run_alpha_beta_sweep(cfg)` — for each `(α, β)` in the grid, fit snakes on full subset, extract both descriptors, store
- `run_deformation_sweep(cfg, representation_type)` — apply each deformation level, re-run pipeline (snake-based descriptors must be re-extracted from deformed images), measure retrieval drop
- `run_baseline_comparison(cfg)` — same deformation sweep but with CNN embeddings instead of descriptors

---

## Scripts (Run in Order)

Each script must be **idempotent** — re-running it should not redo cached work.

1. `00_download_dataset.py` — pull MPEG-7, unpack to `data/raw/`
2. `01_build_subset.py` — pick 30 × 10 = 300 shapes, save to `data/subset/`
3. `02_extract_contours.py` — run snake with default α/β on all shapes, save contours to `results/contours_default.npz`. Log convergence stats.
4. `03_extract_descriptors.py` — for each `(α, β)` in the sweep grid, fit snakes and compute both descriptor types. Save to `results/descriptors_{alpha}_{beta}_{type}.npz`.
5. `04_extract_embeddings.py` — run ResNet-50 and DINOv2 on all shapes (and on deformed versions). Save to `results/embeddings_{model}.npz`.
6. `05_run_deformation_experiments.py` — main experiment. For each deformation type and level, re-extract descriptors and embeddings on deformed images, compute retrieval metrics, dump to CSV.
7. `06_generate_figures.py` — produce final figures: bullseye-vs-deformation curves, parameter heatmaps, per-class disagreement examples.

---

## Tests

`tests/test_snake.py`:
- Snake on a perfect disk converges to a circle (mean radial deviation < 5px)
- Snake is invariant to translation of the input mask
- Higher β produces smoother contours (measure via mean absolute curvature)

`tests/test_descriptors.py`:
- Fourier descriptors of a shape and a rotated copy are nearly identical after normalization (cosine similarity > 0.99)
- CSS signatures of a shape and a translated copy are identical
- Both descriptors return fixed-length vectors regardless of contour length

`tests/test_deformations.py`:
- Rotating by 360° returns to original (within interpolation noise; IoU > 0.95)
- Deformations are deterministic given a seed

Use `pytest`. All tests should run in under 30 seconds total.

---

## Constraints and Style

- Type hints on every function signature
- One-line docstring minimum on every function; complex math gets the equation in the docstring
- All hyperparameters live in `config.py` — no magic numbers in modules
- All file paths go through `pathlib.Path`
- Random operations take an explicit `seed` argument
- Use `tqdm` for any loop over the dataset
- No GPU code paths — everything runs on CPU including embedding extraction (ResNet-50 on 300 small images is fine on CPU)
- Cache any step that takes more than 5 minutes to disk

---

## Deliverables

By end of project:

1. Working repo with all code, runnable end-to-end via a top-level `make run` or shell script
2. CSV tables in `results/tables/`:
   - `alpha_beta_sensitivity.csv` — retrieval metrics across the (α, β) grid
   - `deformation_robustness.csv` — retrieval drop curves for each (representation, deformation) pair
   - `headline_comparison.csv` — single table with bullseye, mAP@10, P@5 for snake-Fourier, snake-CSS, ResNet-50, DINOv2
3. Figures in `results/figures/`:
   - α/β heatmap (one per descriptor type)
   - Robustness curves (bullseye vs deformation level, lines for each method)
   - Per-class disagreement examples (qualitative grid of queries where snake-based and CNN-based methods retrieve different shapes)

The final writeup is a separate deliverable, not in this spec.

---

## Out of Scope

- Training any model
- Fine-tuning SAM, U-Net, or any segmenter
- Real-time or interactive demos
- Web UI
- Comparing more than 2 CNN backbones
- Datasets other than MPEG-7
- 3D shapes
- Level sets, geodesic active contours, or any deformable model other than Kass-Witkin

---

## Working Style Instructions

- After each numbered step in the build order, stop and summarize what you built. Wait for confirmation before proceeding.
- If a design question is ambiguous, ask before implementing.
- If you find a contradiction in this spec, flag it explicitly rather than picking one interpretation silently.
- Prefer well-tested standard library and `numpy`/`scipy` solutions over clever custom code.
- When implementing math, include the equation in the docstring so a reviewer doesn't have to reverse-engineer the code.
