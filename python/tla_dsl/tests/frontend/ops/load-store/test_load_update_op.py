# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend TLAIR coverage for aligned ``post_mode=update`` load/store."""

from __future__ import annotations

from typing import Any

import pytest

import catlass.tla as tla
from catlass.execution_lowering import TlaLoweringError, UnsupportedExecutionLowering
from catlass.params import NormalLoadParams, NormalStoreParams, PostMode
from catlass.tla.runtime import make_fake_tensor

_LOAD_UPD = NormalLoadParams(
    post_mode=PostMode.POST_MODE_UPDATE, post_update_stride=64
)
_STORE_UPD = NormalStoreParams(
    post_mode=PostMode.POST_MODE_UPDATE, post_update_stride=64
)


def _ub_tensor(
    dtype: type[tla.Numeric] = tla.Float32,
    extent: int = 64,
) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (extent,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(extent,),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def load_post_mode_update_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            x_reg = src_tile.load(
                NormalLoadParams(post_mode=PostMode.POST_MODE_UPDATE)
            )
            dst_tile.store(x_reg)


@tla.kernel
def load_post_update_stride_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(
                src_tile.load(NormalLoadParams(post_update_stride=4))
            )


@tla.kernel
def load_post_mode_update_outside_loop_kernel(
    src: tla.Tensor, dst: tla.Tensor
) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            x_reg, _src_tile = src_tile.load(_LOAD_UPD)
            dst_tile.store(x_reg)


@tla.kernel
def load_post_mode_update_ok_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
            dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
            for _ in tla.range(2):
                x_reg, src_tile = src_tile.load(_LOAD_UPD)
                dst_tile = dst_tile.store(x_reg, _STORE_UPD)


@tla.kernel
def load_post_mode_update_local_params_kernel(
    src: tla.Tensor, dst: tla.Tensor
) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
            dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
            load_params = NormalLoadParams(
                post_mode=PostMode.POST_MODE_UPDATE, post_update_stride=64
            )
            store_params = NormalStoreParams(
                post_mode=PostMode.POST_MODE_UPDATE, post_update_stride=64
            )
            for _ in tla.range(2):
                x_reg, src_tile = src_tile.load(load_params)
                dst_tile = dst_tile.store(x_reg, store_params)


@tla.kernel
def load_post_mode_update_nonzero_coord_in_loop_kernel(
    src: tla.Tensor, dst: tla.Tensor
) -> None:
    """Illegal: UPDATE rejects nonzero coord (must prefold to coord=0)."""
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(1))
            dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(1))
            for _ in tla.range(2):
                x_reg, src_tile = src_tile.load(_LOAD_UPD)
                dst_tile = dst_tile.store(x_reg, _STORE_UPD)


@tla.kernel
def load_post_mode_update_nonzero_coord_outside_loop_kernel(
    src: tla.Tensor, dst: tla.Tensor
) -> None:
    """Illegal: UPDATE rejects nonzero coord even outside a loop."""
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(1))
            dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
            x_reg, _src_tile = src_tile.load(_LOAD_UPD)
            dst_tile.store(x_reg)


@tla.kernel
def load_update_then_ordinary_load_kernel(
    src: tla.Tensor, dst: tla.Tensor
) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
            out = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
            _x, tile = tile.load(_LOAD_UPD)
            y = tile.load()
            out.store(y)


@tla.kernel
def ordinary_loop_carries_nonzero_coord_tile_kernel(
    src: tla.Tensor, dst: tla.Tensor
) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(1))
            out = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
            for _ in tla.range(2):
                x = tile.load()
                out.store(x)
                tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(1))


@tla.kernel
def load_update_propagates_origin_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            win = tla.make_layout(
                tla.make_shape(64),
                tla.make_stride(1),
                origin_shape=tla.make_shape(40),
                layoutTag=tla.arch.RowMajor,
            )
            src_tile = tla.make_tensor(src.ptr, win, tla.make_coord(0))
            dst_tile = tla.make_tensor(dst.ptr, win, tla.make_coord(0))
            for _ in tla.range(2):
                x_reg, src_tile = src_tile.load(_LOAD_UPD)
                dst_tile = dst_tile.store(x_reg, _STORE_UPD)


@tla.kernel
def load_update_nonunit_stride_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    """Illegal: UPDATE rebuilds with make_stride(1); non-unit layout rejected."""
    with tla.vector():
        with tla.vec.func(mode="simd"):
            # ``src`` is typed with stride=(2,); do not retile away the layout.
            x_reg, _src_tile = src.load(_LOAD_UPD)
            dst.store(x_reg)


@tla.kernel
def load_update_tail_mask_epilogue_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
            dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
            x_full, src_tile = src_tile.load(_LOAD_UPD)
            dst_tile = dst_tile.store(x_full, _STORE_UPD)
            x_tail, src_tile = src_tile.load(_LOAD_UPD)
            tail, _ = tla.update_mask(16, dtype=tla.Float32)
            dst_tile = dst_tile.store(x_tail, _STORE_UPD, mask=tail)


@tla.kernel
def load_post_mode_update_dynamic_stride_kernel(
    src: tla.Tensor, dst: tla.Tensor, stride: "index"  # noqa: F821
) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            step = tla.as_numeric(stride)
            src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
            dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
            for _ in tla.range(1):
                x_reg, src_tile = src_tile.load(
                    NormalLoadParams(
                        post_mode=PostMode.POST_MODE_UPDATE,
                        post_update_stride=step,
                    )
                )
                dst_tile = dst_tile.store(
                    x_reg,
                    NormalStoreParams(
                        post_mode=PostMode.POST_MODE_UPDATE,
                        post_update_stride=step,
                    ),
                )


@pytest.mark.parametrize(
    ("kernel", "match"),
    [
        (load_post_mode_update_kernel, "post_update_stride"),
        (load_post_update_stride_kernel, "post_update_stride"),
    ],
)
def test_load_update_rejects_illegal_params(
    compiler_tlair: Any, kernel: Any, match: str
) -> None:
    with pytest.raises(
        (TlaLoweringError, UnsupportedExecutionLowering, NotImplementedError),
        match=match,
    ):
        compiler_tlair(kernel, type_args=(_ub_tensor(), _ub_tensor()))


def test_load_post_mode_update_outside_loop_emits_tlair(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        load_post_mode_update_outside_loop_kernel,
        type_args=(_ub_tensor(), _ub_tensor()),
    )
    assert "tla.load_update" in mlir
    assert "scf.for" not in mlir


@pytest.mark.parametrize(
    "kernel",
    [
        load_post_mode_update_ok_kernel,
        load_post_mode_update_local_params_kernel,
    ],
)
def test_load_store_post_mode_update_emits_tlair(
    compiler_tlair: Any, kernel: Any
) -> None:
    mlir = compiler_tlair(
        kernel, type_args=(_ub_tensor(extent=128), _ub_tensor(extent=128))
    )
    assert "tla.load_update" in mlir
    assert "tla.store_update" in mlir
    assert "scf.for" in mlir
    assert "!tla.ptr" in mlir
    if kernel is load_post_mode_update_ok_kernel:
        assert "64 : i32" in mlir
        assert "!tla.coord<0>" in mlir
        assert "tla.tile_view" in mlir


@pytest.mark.parametrize(
    "kernel",
    [
        load_post_mode_update_nonzero_coord_in_loop_kernel,
        load_post_mode_update_nonzero_coord_outside_loop_kernel,
    ],
)
def test_load_store_post_mode_update_nonzero_coord_rejected(
    compiler_tlair: Any, kernel: Any
) -> None:
    with pytest.raises(TlaLoweringError, match="origin window"):
        compiler_tlair(
            kernel,
            type_args=(_ub_tensor(extent=256), _ub_tensor(extent=256)),
        )


def test_load_store_post_mode_update_nonunit_stride_rejected(
    compiler_tlair: Any,
) -> None:
    strided = make_fake_tensor(
        tla.Float32,
        (64,),
        (2,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(64,),
        layout_tag=tla.arch.RowMajor,
    )
    with pytest.raises(TlaLoweringError, match="all strides == 1"):
        compiler_tlair(
            load_update_nonunit_stride_kernel,
            type_args=(strided, _ub_tensor(extent=64)),
        )


def test_load_store_post_mode_update_rejects_unsupported_dtype(
    compiler_tlair: Any,
) -> None:
    with pytest.raises(TlaLoweringError, match="f16/bf16/f32 only"):
        compiler_tlair(
            load_post_mode_update_ok_kernel,
            type_args=(
                _ub_tensor(dtype=tla.Int32, extent=256),
                _ub_tensor(dtype=tla.Int32, extent=256),
            ),
        )


def test_load_update_then_ordinary_load_uses_next_window_tile(
    compiler_tlair: Any,
) -> None:
    mlir = compiler_tlair(
        load_update_then_ordinary_load_kernel,
        type_args=(_ub_tensor(extent=256), _ub_tensor(extent=256)),
    )
    assert "tla.load_update" in mlir
    assert '"tla.load"' in mlir
    assert "tla.make_tensor" in mlir
    load_update_i = mlir.index("tla.load_update")
    make_tensor_i = mlir.index("tla.make_tensor", load_update_i)
    ordinary_i = mlir.index('"tla.load"', make_tensor_i)
    assert load_update_i < make_tensor_i < ordinary_i


def test_ordinary_loop_carries_nonzero_coord_tile_without_update(
    compiler_tlair: Any,
) -> None:
    mlir = compiler_tlair(
        ordinary_loop_carries_nonzero_coord_tile_kernel,
        type_args=(_ub_tensor(extent=256), _ub_tensor(extent=256)),
    )
    assert "scf.for" in mlir
    assert "tla.load_update" not in mlir
    assert "tla.ptr_add" not in mlir
    assert "!tla.coord<64>" in mlir


def test_load_update_propagates_origin_shape(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        load_update_propagates_origin_kernel,
        type_args=(_ub_tensor(extent=256), _ub_tensor(extent=256)),
    )
    assert "tla.load_update" in mlir
    assert "tla.store_update" in mlir
    assert "tla.make_tensor" in mlir
    assert mlir.count("!tla.shape<40>") >= 2
    assert (
        "!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<40>, RowMajor>" in mlir
    )


def test_load_update_tail_mask_epilogue_emits_update_mask(
    compiler_tlair: Any,
) -> None:
    mlir = compiler_tlair(
        load_update_tail_mask_epilogue_kernel,
        type_args=(_ub_tensor(extent=80), _ub_tensor(extent=80)),
    )
    assert "tla.load_update" in mlir
    assert "tla.store_update" in mlir
    assert "tla.update_mask" in mlir
    assert mlir.count("tla.update_mask") == 1


def test_load_store_post_mode_update_dynamic_stride_emits_tlair(
    compiler_tlair: Any,
) -> None:
    mlir = compiler_tlair(
        load_post_mode_update_dynamic_stride_kernel,
        type_args=(_ub_tensor(), _ub_tensor(), 64),
    )
    assert "tla.load_update" in mlir
    assert "tla.store_update" in mlir
    assert "%arg2" in mlir
