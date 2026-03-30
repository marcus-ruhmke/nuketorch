#!/usr/bin/env python3
"""Export a tiny ONNX model (x -> x * 2) and build a TensorRT engine via trtexec.

Requires: torch, onnx, trtexec on PATH (TensorRT bin), NVIDIA GPU + driver for trtexec.

Usage:
  gen_tiny_trt_engine.py <out.engine> [--trtexec /path/to/trtexec]

Then run C++ tests with:
  NUKETORCH_TRT_TEST_ENGINE=/abs/path/to/out.engine
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_engine", type=Path, help="Output .engine path")
    ap.add_argument(
        "--trtexec",
        type=Path,
        default=Path("trtexec"),
        help="Path to trtexec (default: trtexec on PATH)",
    )
    args = ap.parse_args()
    args.out_engine.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        import torch.onnx
    except ImportError as e:
        print("torch is required to export ONNX", file=sys.stderr)
        raise SystemExit(2) from e

    class DoubleModule(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 2.0

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        onnx_path = td_path / "tiny_double.onnx"
        m = DoubleModule()
        example = torch.randn(1, 1, 4, 4)
        torch.onnx.export(
            m,
            example,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
        )
        cmd = [
            str(args.trtexec),
            f"--onnx={onnx_path}",
            f"--saveEngine={args.out_engine.resolve()}",
        ]
        print("Running:", " ".join(cmd), file=sys.stderr)
        r = subprocess.run(cmd, check=False)
        if r.returncode != 0:
            print(f"trtexec failed with code {r.returncode}", file=sys.stderr)
            raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
