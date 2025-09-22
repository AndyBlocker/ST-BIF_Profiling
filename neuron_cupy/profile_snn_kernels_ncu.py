#!/usr/bin/env python3
"""Run Nsight Compute profiling on legacy and new CUDA SNN kernels.

This script orchestrates Nsight Compute (``ncu``) runs across representative
workloads inspired by ViT Base and LLaMA configurations. Each run exercises the
forward and backward passes in FP16 and FP32 precision for both
``cuda_snn_kernels.cu`` and ``cuda_snn_kernels_new.cu``.

Two execution modes are supported:
  * Orchestrator mode (default): launches ``ncu`` for every requested
    combination.
  * Runner mode: internal helper invoked under ``ncu`` that instantiates input
    tensors, executes forward/backward once, and synchronizes the device.
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EXPORT_DIR = ROOT_DIR / "profile" / "ncu_snn"
DEFAULT_NCU_BINARY = "ncu"


@dataclass(frozen=True)
class KernelSpec:
    name: str
    impl: str  # "cupy" or "extension"
    path: Path


DEFAULT_KERNELS: List[KernelSpec] = [
    KernelSpec(
        name="baseline",
        impl="cupy",
        path=ROOT_DIR / "neuron_cupy" / "cuda_snn_kernels.cu",
    ),
    KernelSpec(
        name="optimized",
        impl="extension",
        path=ROOT_DIR / "neuron_cupy" / "cuda_snn_kernels_new.cu",
    ),
]


DEFAULT_MODEL_SPECS: Dict[str, Dict[str, object]] = {
    # ViT Base: 14x14 patches + CLS token, hidden size 768, batch 32
    "vit_base": {
        "batch_size": 32,
        "feature_shape": (197, 768),
    },
    # LLaMA-7B: context length 2048, hidden size 4096, batch 4
    "llama7b": {
        "batch_size": 4,
        "feature_shape": (2048, 4096),
    },
}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(mode="orchestrate")
    subparsers = parser.add_subparsers(dest="mode", required=False)

    orchestrator = subparsers.add_parser("orchestrate", help="Launch Nsight Compute runs (default)")
    orchestrator.add_argument(
        "--ncu-binary", default=DEFAULT_NCU_BINARY,
        help="Nsight Compute CLI binary (default: %(default)s)",
    )
    orchestrator.add_argument(
        "--export-dir", type=Path, default=DEFAULT_EXPORT_DIR,
        help="Directory to store Nsight Compute reports (default: %(default)s)",
    )
    orchestrator.add_argument(
        "--kernels", nargs="*", choices=[k.name for k in DEFAULT_KERNELS],
        help="Subset of kernels to profile (default: both)",
    )
    orchestrator.add_argument(
        "--models", nargs="*", choices=list(DEFAULT_MODEL_SPECS.keys()),
        help="Subset of model-inspired workloads to run (default: all)",
    )
    orchestrator.add_argument(
        "--time-steps", nargs="*", type=int, choices=[4, 8, 16],
        help="Override time steps (default: 4 8 16)",
    )
    orchestrator.add_argument(
        "--precisions", nargs="*", choices=["fp16", "fp32"],
        help="Override precisions (default: fp16 fp32)",
    )
    orchestrator.add_argument(
        "--device", default="cuda",
        help="Torch device string passed to runner (default: %(default)s)",
    )
    orchestrator.add_argument(
        "--cuda-visible-devices",
        help="Optional CUDA_VISIBLE_DEVICES value for spawned runs",
    )
    orchestrator.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them",
    )
    orchestrator.set_defaults(mode="orchestrate")

    runner = subparsers.add_parser("runner", help="Internal entry point used under Nsight Compute")
    runner.add_argument("--kernel-impl", choices=["cupy", "extension"], required=True)
    runner.add_argument("--kernel-path", type=Path, required=True)
    runner.add_argument("--dtype", choices=["fp16", "fp32"], required=True)
    runner.add_argument("--time-steps", type=int, required=True)
    runner.add_argument("--batch-size", type=int, required=True)
    runner.add_argument("--feature-shape", nargs="+", type=int, required=True)
    runner.add_argument("--device", default="cuda")
    runner.add_argument("--v-th", type=float, default=0.1)
    runner.add_argument("--t-max", type=float, default=4.0)
    runner.add_argument("--t-min", type=float, default=-4.0)
    runner.add_argument("--prefire", type=float, default=0.0)
    runner.set_defaults(mode="runner")

    argv_list = list(argv)
    if not argv_list:
        argv_list = ["orchestrate"]
    else:
        first = argv_list[0]
        if first not in {"orchestrate", "runner"}:
            argv_list = ["orchestrate"] + argv_list

    args = parser.parse_args(argv_list)
    return args


def discover_kernel_specs(selected: Iterable[str] | None) -> List[KernelSpec]:
    if not selected:
        return DEFAULT_KERNELS
    selected_set = set(selected)
    return [spec for spec in DEFAULT_KERNELS if spec.name in selected_set]


def discover_model_specs(selected: Iterable[str] | None) -> Dict[str, Dict[str, object]]:
    if not selected:
        return DEFAULT_MODEL_SPECS
    return {name: DEFAULT_MODEL_SPECS[name] for name in selected}


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def validate_cuda_env(device: str) -> None:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "PyTorch must be installed to profile the CUDA kernels"
        ) from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        cuda_ok = torch.cuda.is_available()

    if not cuda_ok:
        raise RuntimeError(
            "CUDA device is required for profiling. Check driver installation "
            "and CUDA_VISIBLE_DEVICES settings."
        )

    try:
        torch.empty(1, device=device)
    except RuntimeError as exc:  # pragma: no cover - device dependent
        raise RuntimeError(
            f"Unable to allocate tensor on device '{device}': {exc}"
        ) from exc


def orchestrate(args: argparse.Namespace) -> None:
    kernels = discover_kernel_specs(args.kernels)
    models = discover_model_specs(args.models)
    time_steps = args.time_steps or [4, 8, 16]
    precisions = args.precisions or ["fp16", "fp32"]

    export_root = args.export_dir.resolve()
    ensure_directory(export_root)

    if not args.dry_run:
        validate_cuda_env(args.device)

    ncu_path = Path(args.ncu_binary)
    if not ncu_path.is_file():
        resolved = shutil.which(args.ncu_binary)
        if resolved is None:
            raise FileNotFoundError(f"Unable to locate Nsight Compute binary '{args.ncu_binary}'")
        ncu_path = Path(resolved)

    script_path = Path(__file__).resolve()

    for kernel_spec in kernels:
        for model_name, cfg in models.items():
            batch_size = int(cfg["batch_size"])
            feature_shape = [int(v) for v in cfg["feature_shape"]]

            for ts in time_steps:
                for precision in precisions:
                    run_id = f"{kernel_spec.name}_{model_name}_T{ts}_{precision}"
                    export_base = export_root / run_id
                    log_file = export_root / f"{run_id}.log"

                    runner_args = [
                        str(script_path),
                        "runner",
                        "--kernel-impl",
                        kernel_spec.impl,
                        "--kernel-path",
                        str(kernel_spec.path.resolve()),
                        "--dtype",
                        precision,
                        "--time-steps",
                        str(ts),
                        "--batch-size",
                        str(batch_size),
                        "--feature-shape",
                    ] + [str(dim) for dim in feature_shape] + [
                        "--device",
                        args.device,
                    ]

                    cmd = [
                        str(ncu_path),
                        "--set",
                        "full",
                        "--target-processes",
                        "all",
                        "--cache-control",
                        "all",
                        "--force-overwrite",
                        "--export",
                        str(export_base),
                        "--log-file",
                        str(log_file),
                        sys.executable,
                    ] + runner_args

                    env = os.environ.copy()
                    if args.cuda_visible_devices:
                        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

                    print(f"[profile] {run_id}")
                    if args.dry_run:
                        print(" ".join(cmd))
                        continue

                    subprocess.run(cmd, check=True, env=env, cwd=str(ROOT_DIR))


def _load_cupy_operator(kernel_path: Path):
    os.environ["CUDA_SNN_KERNELS_PATH"] = str(kernel_path.resolve())
    module = importlib.import_module("neuron_cupy.cuda_operator")
    module = importlib.reload(module)
    return module.ST_BIFNodeATGF_MS_CUDA.apply


def _load_extension_operator(kernel_path: Path):
    os.environ["CUDA_SNN_KERNELS_PATH"] = str(kernel_path.resolve())
    module = importlib.import_module("neuron_cupy.cuda_operator_new")
    module = importlib.reload(module)
    module.ST_BIFNodeATGF_MS_CUDA._built = False
    module.ST_BIFNodeATGF_MS_CUDA._ext_mod = None
    return module.ST_BIFNodeATGF_MS_CUDA.apply


def runner(args: argparse.Namespace) -> None:
    import torch

    validate_cuda_env(args.device)

    dtype_map = {
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    feature_shape = tuple(int(v) for v in args.feature_shape)
    shape = (args.time_steps, args.batch_size, *feature_shape)

    torch.manual_seed(123)
    x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    v_th = torch.tensor([args.v_th], device=device, dtype=dtype)
    t_max = torch.tensor([args.t_max], device=device, dtype=dtype)
    t_min = torch.tensor([args.t_min], device=device, dtype=dtype)
    prefire = torch.tensor([args.prefire], device=device, dtype=dtype)

    if args.kernel_impl == "cupy":
        op_apply = _load_cupy_operator(args.kernel_path)
    else:
        op_apply = _load_extension_operator(args.kernel_path)

    spike_seq, v_out, t_seq = op_apply(x, v_th, t_max, t_min, prefire)

    loss = (
        spike_seq.sum(dtype=torch.float32)
        + v_out.sum(dtype=torch.float32)
        + t_seq.sum(dtype=torch.float32)
    )
    loss.backward()
    torch.cuda.synchronize(device)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    if args.mode == "runner":
        runner(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
