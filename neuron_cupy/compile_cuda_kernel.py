#!/usr/bin/env python3
"""Helper script to sanity-check nvcc compilation of cuda_snn_kernels_new.cu.

This does *not* attempt to link (the source provides kernels only), instead it
invokes nvcc in compile-only mode and reports compiler diagnostics. Use the
optional flags if you need a non-default nvcc binary or architecture.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nvcc",
        default="nvcc",
        help="nvcc executable to invoke (default: %(default)s)",
    )
    parser.add_argument(
        "--arch",
        default="sm_70",
        help="GPU architecture passed to nvcc's -arch flag (default: %(default)s)",
    )
    parser.add_argument(
        "--src",
        default=str(Path(__file__).resolve().parent / "cuda_snn_kernels_new.cu"),
        help="CUDA source file to compile (default: %(default)s)",
    )
    parser.add_argument(
        "--keep-object",
        action="store_true",
        help="Keep the generated object file instead of deleting the temp dir",
    )
    parser.add_argument(
        "--extra-nvcc-flags",
        nargs=argparse.REMAINDER,
        help="Additional flags appended to the nvcc command",
    )
    return parser.parse_args()


def ensure_nvcc_exists(nvcc: str) -> None:
    if shutil.which(nvcc) is None:
        sys.stderr.write(f"[compile_cuda_kernel] nvcc not found: {nvcc}\n")
        sys.exit(127)


def main() -> int:
    args = parse_args()
    ensure_nvcc_exists(args.nvcc)

    src_path = Path(args.src).resolve()
    if not src_path.exists():
        sys.stderr.write(f"[compile_cuda_kernel] Source file not found: {src_path}\n")
        return 1

    cmd = [
        args.nvcc,
        "-c",
        str(src_path),
        "-std=c++14",
        f"-arch={args.arch}",
        "-Xcompiler",
        "-fPIC",
    ]

    if args.extra_nvcc_flags:
        cmd.extend(args.extra_nvcc_flags)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "cuda_snn_kernels_new.o"
        cmd.extend(["-o", str(out_path)])

        print("[compile_cuda_kernel] Running:", " ".join(cmd))
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if proc.stdout:
            print(proc.stdout.rstrip())
        if proc.stderr:
            sys.stderr.write(proc.stderr)

        if proc.returncode != 0:
            print(f"[compile_cuda_kernel] nvcc exited with code {proc.returncode}")
            return proc.returncode

        if args.keep_object:
            keep_path = src_path.with_suffix(".o")
            keep_path.write_bytes(out_path.read_bytes())
            print(f"[compile_cuda_kernel] Saved object file to {keep_path}")

    print("[compile_cuda_kernel] nvcc compilation succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
