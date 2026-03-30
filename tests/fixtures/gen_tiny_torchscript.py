#!/usr/bin/env python3
"""Writes a minimal TorchScript file for C++ tests (trace of x -> x * 2)."""
import sys

import torch


class DoubleModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: gen_tiny_torchscript.py <out.pt>", file=sys.stderr)
        sys.exit(2)
    out = sys.argv[1]
    m = DoubleModule()
    example = torch.randn(1, 1, 4, 4)
    traced = torch.jit.trace(m, example)
    traced.save(out)


if __name__ == "__main__":
    main()
