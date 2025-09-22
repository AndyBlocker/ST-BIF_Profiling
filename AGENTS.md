# Repository Guidelines

## Project Structure & Module Organization
- `snn/` contains conversion logic, spike-aware layers, and neuron primitives for the ST-BIF stack.
- `wrapper/` implements ANN→SNN orchestration (encoding, reset, attention conversion); scripts in `examples/` import it directly.
- Reference networks, experiment presets, and pretrained checkpoints sit in `models/`, `configs/`, and `checkpoints/`; store new assets in matching subfolders.
- CUDA kernels live in `neuron_cupy/`; legacy fallbacks remain in `legacy/`. Use `tests/` for quick sanity checks that avoid full training cycles.

## Build, Test, & Development Commands
- Bootstrap a clean environment, then install deep-learning dependencies: `pip install torch torchvision timm cupy-cuda11x` (swap the CuPy wheel for your CUDA version).
- Validate the full conversion path: `python examples/ann_to_snn_conversion.py --batch-size 64 --time-step 8`.
- Profile kernels before performance claims: `python neuron_cupy/profile_kernels.py --device cuda`.
- Run GPU/CPU parity checks: `pytest -q neuron_cupy/test_snn_operator.py` (automatically falls back to CPU when CUDA is unavailable).

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indents; keep functions/modules in `snake_case`, classes in `PascalCase`, constants uppercase.
- Add type hints (`Tensor`, `Tuple[int, ...]`) when they clarify APIs, and mirror the short docstrings already in `snn/neurons`.
- Guard CUDA branches with `torch.cuda.is_available()` and retain numerically stable CPU paths.
- Place helpers next to the feature they support; promote shared utilities to `utils/` only when reused across multiple modules.

## Testing Guidelines
- Use `pytest`; co-locate unit tests with the module (`snn/.../tests`) or drop cross-cutting checks under `tests/`.
- Parameterize over shapes, dtypes, and seeds as in `neuron_cupy/test_snn_operator.py`, and assert tensor closeness with explicit tolerances.
- After changing quantization or neuron logic, capture an ANN/QANN/SNN regression run and report accuracy deltas in your PR.

## Commit & Pull Request Guidelines
- Commit messages start with an action verb and may be English or Chinese (e.g., `Add ST-BIF SNN optimization framework`); keep them focused and under 72 characters when possible.
- PRs should summarize functional changes, list required configs or datasets, and attach profiling or accuracy metrics for model-impacting updates.
- Mention related issues or experiment IDs and call out any non-default dependencies (CUDA version, dataset path) in the description.

## Profiling & Large Artifacts
- Keep generated artifacts out of git; extend `.gitignore` for new profiler dumps under `profile/` or dataset mirrors like `cifar10/`.
- Store reusable weights in `checkpoints/<model>/` and document training recipes with a brief `README.md` inside the folder.
