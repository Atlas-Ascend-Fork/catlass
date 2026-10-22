# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PostMode UPDATE e2e: aligned ``tile.load`` / ``tile.store`` (``vldsx1.post`` / ``vstsx1.post``).

``add`` — full trips unmasked; when ``VECTOR_ELE % VL != 0``, one epilogue
``update_mask(tail)``. ``--dynamic-extent`` — tile extent is a kernel ``index``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import catlass.tla as tla
from catlass.params import NormalLoadParams, NormalStoreParams, PostMode

from vector_op_harness import (
    DirectVectorOpConfig,
    DirectVectorOpHarness,
    make_tla_tensor,
    require_torch_npu,
    runtime_kwargs,
    vector_kernel_config,
)

VECTOR_ELE = 400
VL_ELE = 64
LOOPS = (VECTOR_ELE + VL_ELE - 1) // VL_ELE
ALL_DTYPES = ("f32",)

_KERNEL_DTYPE = tla.Float32
_KERNEL_ELEMENT_BYTES = 4
_KERNEL_SHAPE = (VECTOR_ELE,)

_LOAD_U = NormalLoadParams(
    post_mode=PostMode.POST_MODE_UPDATE, post_update_stride=VL_ELE
)
_STORE_U = NormalStoreParams(
    post_mode=PostMode.POST_MODE_UPDATE, post_update_stride=VL_ELE
)


@tla.kernel
def load_store_update_op(
    mem_x: tla.Tensor, mem_y: tla.Tensor, mem_z: tla.Tensor
) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    x_gm = tla.tile_view(mem_x, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    y_gm = tla.tile_view(mem_y, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    z_gm = tla.tile_view(mem_z, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    x_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    y_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    z_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)

    x_ub = tla.make_tensor_like(x_ub_ptr, x_gm, tla.arch.RowMajor)
    y_ub = tla.make_tensor_like(y_ub_ptr, y_gm, tla.arch.RowMajor)
    z_ub = tla.make_tensor_like(z_ub_ptr, z_gm, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(x_ub, x_gm)
        tla.copy(y_ub, y_gm)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            full_loops = VECTOR_ELE // VL_ELE
            tail_elems = VECTOR_ELE % VL_ELE
            x_tile = tla.tile_view(x_ub, tla.make_shape(VL_ELE), tla.make_coord(0))
            y_tile = tla.tile_view(y_ub, tla.make_shape(VL_ELE), tla.make_coord(0))
            z_tile = tla.tile_view(z_ub, tla.make_shape(VL_ELE), tla.make_coord(0))
            for _ in tla.range(full_loops):
                x_reg, x_tile = x_tile.load(_LOAD_U)
                y_reg, y_tile = y_tile.load(_LOAD_U)
                z_tile = z_tile.store(tla.add(x_reg, y_reg), _STORE_U)
            if tail_elems != 0:
                x_tail, x_tile = x_tile.load(_LOAD_U)
                y_tail, y_tile = y_tile.load(_LOAD_U)
                tail, _ = tla.update_mask(tail_elems, dtype=_KERNEL_DTYPE)
                z_tile = z_tile.store(tla.add(x_tail, y_tail), _STORE_U, mask=tail)
        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(z_gm, z_ub)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def load_store_update_dynamic_extent_op(
    mem_x: tla.Tensor,
    mem_y: tla.Tensor,
    mem_z: tla.Tensor,
    vl: "index",  # noqa: F821
) -> None:
    """UPDATE scan with dynamic ``make_shape(vl)``."""
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    x_gm = tla.tile_view(mem_x, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    y_gm = tla.tile_view(mem_y, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    z_gm = tla.tile_view(mem_z, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    x_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    y_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    z_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)

    x_ub = tla.make_tensor_like(x_ub_ptr, x_gm, tla.arch.RowMajor)
    y_ub = tla.make_tensor_like(y_ub_ptr, y_gm, tla.arch.RowMajor)
    z_ub = tla.make_tensor_like(z_ub_ptr, z_gm, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(x_ub, x_gm)
        tla.copy(y_ub, y_gm)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            x_tile = tla.tile_view(x_ub, tla.make_shape(vl), tla.make_coord(0))
            y_tile = tla.tile_view(y_ub, tla.make_shape(vl), tla.make_coord(0))
            z_tile = tla.tile_view(z_ub, tla.make_shape(vl), tla.make_coord(0))
            for _ in tla.range(LOOPS):
                x_reg, x_tile = x_tile.load(_LOAD_U)
                y_reg, y_tile = y_tile.load(_LOAD_U)
                z_tile = z_tile.store(tla.add(x_reg, y_reg), _STORE_U)
        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(z_gm, z_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {"add": {"op": "add", "dtypes": ALL_DTYPES, "default_atol": 1e-4}}


def _set_kernel_config(
    op_name: str, dtype_name: str, shape: tuple[int, ...] | None = None
) -> tuple[type[Any], Any, float | int]:
    global VL_ELE, LOOPS, VECTOR_ELE, _KERNEL_DTYPE, _KERNEL_ELEMENT_BYTES
    global _KERNEL_SHAPE, _LOAD_U, _STORE_U
    del op_name
    config = vector_kernel_config(dtype_name, shape, ALL_DTYPES)
    VECTOR_ELE = config.vector_elements
    _KERNEL_SHAPE = shape if shape is not None else (VECTOR_ELE,)
    VL_ELE = config.lanes
    LOOPS = config.loops
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_ELEMENT_BYTES = config.element_bytes
    _LOAD_U = NormalLoadParams(
        post_mode=PostMode.POST_MODE_UPDATE, post_update_stride=VL_ELE
    )
    _STORE_U = NormalStoreParams(
        post_mode=PostMode.POST_MODE_UPDATE, post_update_stride=VL_ELE
    )
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, Any]:
    _set_kernel_config("add", dtype_name, args.shape)
    dtype = vector_kernel_config(dtype_name, args.shape, ALL_DTYPES).torch_dtype
    x = torch.arange(VECTOR_ELE, dtype=dtype, device="npu") + 1
    y = (torch.arange(VECTOR_ELE, dtype=dtype, device="npu") * 0.5) + 2
    return x, y


def _expected(_op_name: str, inputs: tuple[Any, ...]) -> Any:
    x, y = inputs
    return x + y


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description="PostMode UPDATE load/store sliding-window add E2E",
        kernel=load_store_update_op,
        all_dtypes=ALL_DTYPES,
        operator_specs=_operator_specs,
        set_kernel_config=_set_kernel_config,
        get_vector_elements=lambda: VECTOR_ELE,
        get_kernel_shape=lambda: _KERNEL_SHAPE,
        make_inputs=_make_inputs,
        expected=_expected,
        unsupported_case=lambda _o, _d: False,
        print_skip=lambda o, d, s: print(f"skip op={o} dtype={d} shape={s}"),
        script_path=Path(__file__).resolve(),
        float_dtypes=frozenset({"f32"}),
        input_count=2,
    )
)


def _run_dynamic_extent(args: argparse.Namespace) -> int:
    torch = require_torch_npu(args.device, Path(__file__).stem)
    torch.npu.set_device(args.device)
    shape = args.shape if args.shape is not None else (256,)
    tla_dtype, torch_dt, sentinel = _set_kernel_config("add", "f32", shape)
    if VECTOR_ELE % VL_ELE != 0:
        raise SystemExit(
            f"--dynamic-extent needs vector length multiple of VL={VL_ELE}, "
            f"got {VECTOR_ELE}"
        )
    x = torch.arange(VECTOR_ELE, dtype=torch_dt, device="npu") + 1
    y = (torch.arange(VECTOR_ELE, dtype=torch_dt, device="npu") * 0.5) + 2
    z = torch.full((VECTOR_ELE,), sentinel, dtype=torch_dt, device="npu")
    expected = x + y
    tla_x = make_tla_tensor(x, tla_dtype, _KERNEL_SHAPE)
    tla_y = make_tla_tensor(y, tla_dtype, _KERNEL_SHAPE)
    tla_z = make_tla_tensor(z, tla_dtype, _KERNEL_SHAPE)
    vl = tla.Int32(VL_ELE)
    artifact = tla.compile(
        load_store_update_dynamic_extent_op,
        tla_x,
        tla_y,
        tla_z,
        vl,
        **runtime_kwargs(args),
    )
    artifact(tla_x, tla_y, tla_z, vl, block_num=args.block_num)
    torch.npu.synchronize()
    ok = bool(torch.isclose(z, expected, rtol=0.0, atol=args.atol).all())
    mismatch = (~torch.isclose(z, expected, rtol=0.0, atol=args.atol)).nonzero(
        as_tuple=False
    )
    first = None
    if mismatch.numel():
        i = int(mismatch[0].item())
        first = {"index": i, "actual": z[i].item(), "expected": expected[i].item()}
    print(
        f"compile_ok=True host=torch_npu op=add dtype=f32 shape={shape} "
        f"dynamic_extent=True vl={VL_ELE}"
    )
    print(f"kernel.o path={artifact.kernel_binary_path}")
    print("launch_ok=True")
    print(f"outputs equal expected add? {ok}")
    print(f"first mismatch={first}")
    return 0 if ok else 1


def main() -> int:
    if "--dynamic-extent" in sys.argv:
        parser = argparse.ArgumentParser(
            description="UPDATE e2e with dynamic tile extent"
        )
        parser.add_argument("op", nargs="?", default="add")
        parser.add_argument("--device", type=int, default=2)
        parser.add_argument("--block-num", type=int, default=1)
        parser.add_argument("--shape", type=int, nargs="+", default=None)
        parser.add_argument("--atol", type=float, default=1e-4)
        parser.add_argument("--dynamic-extent", action="store_true")
        parser.add_argument("--sentinel", type=float, default=None)
        args = parser.parse_args()
        if args.shape is not None:
            args.shape = tuple(args.shape)
        return _run_dynamic_extent(args)
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
