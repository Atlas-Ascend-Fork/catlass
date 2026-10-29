# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Run predicate ``tla.pack`` with a checked output buffer.

Each source register contributes one half of a destination register. LOW
writes the low destination half and HIGH writes the high destination half;
the other half is zeroed by the ``ppack.z`` instruction.
"""

from __future__ import annotations

import argparse
from typing import Any

from catlass import tla

_MASK_CONFIGS = {
    "64": (tla.Int32, tla.Int16, "int16", 64),
    "128": (tla.Int16, tla.Int8, "int8", 128),
}
_MASK_DTYPE = tla.Int32
_MASK_DST_DTYPE = tla.Int16
_MASK_DST_TORCH_NAME = "int16"
_MASK_SRC_LANES = 64
_MASK_DST_LANES = 128
_MASK_PACK_PART = tla.RegPart.LOW


def _make_tensor(length: int, dtype: type[Any]) -> tla.Tensor:
    ptr = tla.allocate(length, dtype, tla.AddressSpace.ub, 256)
    layout = tla.make_layout(
        shape=tla.make_shape(length),
        stride=tla.make_stride(1),
    )
    return tla.make_tensor(ptr, layout)


@tla.kernel
def mask_pack_op(mem_out: tla.Tensor) -> None:
    done = tla.flag("mask_done", tla.arch.VECTOR, tla.arch.MTE3)

    out_gm = tla.tile_view(mem_out, tla.make_shape(_MASK_DST_LANES), tla.make_coord(0))
    out_ub = _make_tensor(_MASK_DST_LANES, _MASK_DST_DTYPE)

    with tla.vector():
        with tla.vec.func(mode="simd"):
            source = tla.create_mask(pattern=tla.mask.H, dtype=_MASK_DTYPE)
            packed = tla.pack(source, part=_MASK_PACK_PART)
            ones = tla.full(1, _MASK_DST_DTYPE)
            zeros = tla.full(0, _MASK_DST_DTYPE)
            out_ub.store(tla.where(packed, ones, zeros))

        tla.set_flag(done)
        tla.wait_flag(done)
        tla.copy(out_gm, out_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _configure_mask(mask_width: str, part_name: str) -> str:
    global _MASK_DTYPE, _MASK_DST_DTYPE, _MASK_DST_TORCH_NAME
    global _MASK_SRC_LANES, _MASK_DST_LANES, _MASK_PACK_PART

    mask_dtype, dst_dtype, dst_torch_name, src_lanes = _MASK_CONFIGS[mask_width]
    _MASK_DTYPE = mask_dtype
    _MASK_DST_DTYPE = dst_dtype
    _MASK_DST_TORCH_NAME = dst_torch_name
    _MASK_SRC_LANES = src_lanes
    _MASK_DST_LANES = src_lanes * 2
    _MASK_PACK_PART = tla.RegPart.LOW if part_name == "low" else tla.RegPart.HIGH
    return dst_torch_name


def _expected_mask(*, mask_width: str, part_name: str, device: Any) -> Any:
    import torch

    dst_torch_name = _configure_mask(mask_width, part_name)
    dtype = getattr(torch, dst_torch_name)
    output = torch.zeros((_MASK_DST_LANES,), dtype=dtype, device=device)
    start = 0 if part_name == "low" else _MASK_SRC_LANES
    output[start : start + _MASK_SRC_LANES // 2] = 1
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-width", choices=tuple(_MASK_CONFIGS), default="64")
    parser.add_argument("--part", choices=("low", "high"), default="low")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    import torch
    import torch_npu

    del torch_npu
    torch.npu.set_device(args.device)
    dst_torch_name = _configure_mask(args.mask_width, args.part)
    dst_torch_dtype = getattr(torch, dst_torch_name)
    output = torch.zeros((_MASK_DST_LANES,), dtype=dst_torch_dtype, device="npu")
    tla_output = tla.from_dlpack(output, layout_tag=tla.arch.RowMajor)
    artifact = tla.compile(mask_pack_op, tla_output, options="--npu-arch 3510")
    artifact(tla_output, block_num=1)
    torch.npu.synchronize()
    expected = _expected_mask(
        mask_width=args.mask_width,
        part_name=args.part,
        device=output.device,
    )
    matches = output.eq(expected)
    ok = bool(matches.all())
    mismatch = matches.logical_not().nonzero(as_tuple=False)
    first_mismatch = None
    if mismatch.numel():
        pos = int(mismatch[0].item())
        first_mismatch = {
            "index": pos,
            "actual": output[pos].item(),
            "expected": expected[pos].item(),
        }

    print(
        "compile_ok=True host=torch_npu op=mask_pack "
        f"mask_width={args.mask_width} part={args.part}"
    )
    print(f"kernel.o path={artifact.kernel_binary_path}")
    print("launch_ok=True")
    print(f"output equals expected mask pack? {ok}")
    print(f"first mismatch={first_mismatch}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
