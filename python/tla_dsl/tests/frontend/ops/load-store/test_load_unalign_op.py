from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor


from typing import Any

import pytest

import catlass.tla as tla
from catlass.execution_lowering import TlaLoweringError, UnsupportedExecutionLowering
from catlass.params import (
    LoadDist,
    NormalLoadParams,
    PostMode,
    UnalignLoadParams,
    UnalignStoreParams,
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


def _gm_tensor(
    dtype: type[tla.Numeric] = tla.Float32,
    extent: int = 64,
) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (extent,),
        (1,),
        origin_shape=(extent,),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def load_unalign_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(src_tile.load(UnalignLoadParams()))


@tla.kernel
def load_normal_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(src_tile.load())


@tla.kernel
def load_unalign_loop_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            for i in tla.range(2):
                src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(i))
                dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(i))
                dst_tile.store(src_tile.load(UnalignLoadParams()))


@tla.kernel
def load_unalign_outside_vec_func(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        _ = src_tile.load(UnalignLoadParams())


@tla.kernel
def load_from_gm_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(1), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            _ = src_tile.load(NormalLoadParams(load_dist=LoadDist.DIST_BRC_B32))


def test_load_unalign_emits_tlair(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        load_unalign_kernel,
        type_args=(_ub_tensor(), _ub_tensor()),
    )

    assert "tla.load" in mlir
    assert "unaligned_ub_access" in mlir
    assert "tla.load_unalign" not in mlir
    assert "load_unalign_pre" not in mlir
    assert "unalign_reg" not in mlir


def test_load_normal_has_no_unalign_attr(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        load_normal_kernel,
        type_args=(_ub_tensor(), _ub_tensor()),
    )

    assert "tla.load" in mlir
    assert "unaligned_ub_access" not in mlir


def test_load_unalign_loop_emits_tlair(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        load_unalign_loop_kernel,
        type_args=(_ub_tensor(extent=128), _ub_tensor(extent=128)),
    )

    assert "scf.for" in mlir
    assert "tla.load" in mlir
    assert "unaligned_ub_access" in mlir


@tla.kernel
def load_brc_b32_add_kernel(
    x: tla.Tensor,
    y: tla.Tensor,
    z: tla.Tensor,
) -> None:
    x_tile = tla.tile_view(x, tla.make_shape(1), tla.make_coord(0))
    y_tile = tla.tile_view(y, tla.make_shape(64), tla.make_coord(0))
    z_tile = tla.tile_view(z, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            x_v = x_tile.load(NormalLoadParams(load_dist=LoadDist.DIST_BRC_B32))
            y_v = y_tile.load()
            z_tile.store(x_v + y_v)


def test_load_brc_b32_add_emits_tlair(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        load_brc_b32_add_kernel,
        type_args=(_ub_tensor(extent=64), _ub_tensor(), _ub_tensor()),
    )

    assert "#tla.load_dist<brc_b32>" in mlir
    assert "tla.add" in mlir
    assert "!tla.shape<64>" in mlir


def test_load_brc_b32_emits_tlair(compiler_tlair: Any) -> None:
    @tla.kernel
    def load_brc_b32_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
        src_tile = tla.tile_view(src, tla.make_shape(1), tla.make_coord(0))
        dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                dst_tile.store(
                    src_tile.load(NormalLoadParams(load_dist=LoadDist.DIST_BRC_B32))
                )

    mlir = compiler_tlair(
        load_brc_b32_kernel,
        type_args=(_ub_tensor(extent=64), _ub_tensor()),
    )

    assert "#tla.load_dist<brc_b32>" in mlir
    assert "tla.load" in mlir


@tla.kernel
def load_dintlv_b32_kernel(
    src: tla.Tensor,
    dst0: tla.Tensor,
    dst1: tla.Tensor,
) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(128), tla.make_coord(0))
    dst0_tile = tla.tile_view(dst0, tla.make_shape(64), tla.make_coord(0))
    dst1_tile = tla.tile_view(dst1, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            v0, v1 = src_tile.load(NormalLoadParams(load_dist=LoadDist.DIST_DINTLV_B32))
            dst0_tile.store(v0)
            dst1_tile.store(v1)


def test_load_dintlv_b32_emits_tlair(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        load_dintlv_b32_kernel,
        type_args=(
            _ub_tensor(extent=128),
            _ub_tensor(),
            _ub_tensor(),
        ),
    )

    assert "#tla.load_dist<dintlv_b32>" in mlir
    load_lines = [line for line in mlir.splitlines() if "tla.load" in line]
    assert len(load_lines) == 1
    assert "->" in load_lines[0]
    # Dual-destination: two result types after ->
    assert load_lines[0].count("!") >= 3


@tla.kernel
def load_dintlv_b32_wrong_dtype_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(256), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(128), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            v0, _v1 = src_tile.load(
                NormalLoadParams(load_dist=LoadDist.DIST_DINTLV_B32)
            )
            dst_tile.store(v0)


def test_load_dintlv_b32_rejects_non_f32(compiler_tlair: Any) -> None:
    with pytest.raises(Exception, match="f32|DIST_DINTLV_B32"):
        compiler_tlair(
            load_dintlv_b32_wrong_dtype_kernel,
            type_args=(
                _ub_tensor(dtype=tla.Float16, extent=256),
                _ub_tensor(dtype=tla.Float16, extent=128),
            ),
        )


@tla.kernel
def load_dintlv_b32_i32_kernel(
    src: tla.Tensor,
    dst0: tla.Tensor,
    dst1: tla.Tensor,
) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(128), tla.make_coord(0))
    dst0_tile = tla.tile_view(dst0, tla.make_shape(64), tla.make_coord(0))
    dst1_tile = tla.tile_view(dst1, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            v0, v1 = src_tile.load(NormalLoadParams(load_dist=LoadDist.DIST_DINTLV_B32))
            dst0_tile.store(v0)
            dst1_tile.store(v1)


def test_load_dintlv_b32_rejects_i32(compiler_tlair: Any) -> None:
    with pytest.raises(Exception, match="f32|DIST_DINTLV_B32"):
        compiler_tlair(
            load_dintlv_b32_i32_kernel,
            type_args=(
                _ub_tensor(dtype=tla.Int32, extent=128),
                _ub_tensor(dtype=tla.Int32, extent=64),
                _ub_tensor(dtype=tla.Int32, extent=64),
            ),
        )


@tla.kernel
def load_unalign_update_dual_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    """UnAlign UPDATE emits TLA update ops; no public Pre/Post/ureg."""
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            x_reg, src_tile = src_tile.load(
                UnalignLoadParams(post_mode=PostMode.POST_MODE_UPDATE)
            )
            dst_tile = dst_tile.store(
                x_reg,
                UnalignStoreParams(post_mode=PostMode.POST_MODE_UPDATE),
            )


@tla.kernel
def load_unalign_post_update_stride_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(src_tile.load(UnalignLoadParams(post_update_stride=8)))


def test_load_unalign_requires_vec_func() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="vec.func"):
        load_unalign_outside_vec_func.dump_mlir(type_args=(_ub_tensor(),))


def test_load_unalign_update_emits_update_ops(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        load_unalign_update_dual_kernel,
        type_args=(_ub_tensor(), _ub_tensor()),
    )
    assert "tla.load_unalign_update" in mlir
    assert "tla.store_unalign_update" in mlir
    assert "!tla.ureg" not in mlir
    assert "tla.unalign_ureg_load_pre" not in mlir
    assert "tla.unalign_ureg_store_empty" not in mlir


@tla.kernel
def load_unalign_update_loop_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    """Sliding-window UnAlign UPDATE: natural for carries tiles only."""
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            for _ in tla.range(3):
                x_reg, src_tile = src_tile.load(
                    UnalignLoadParams(post_mode=PostMode.POST_MODE_UPDATE)
                )
                dst_tile = dst_tile.store(
                    x_reg,
                    UnalignStoreParams(post_mode=PostMode.POST_MODE_UPDATE),
                )


def test_load_unalign_update_loop_emits_ops(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        load_unalign_update_loop_kernel,
        type_args=(_ub_tensor(extent=256), _ub_tensor(extent=256)),
    )
    assert "tla.load_unalign_update" in mlir
    assert "tla.store_unalign_update" in mlir
    assert "scf.for" in mlir
    assert "!tla.ureg" not in mlir


@tla.kernel
def load_unalign_update_loop_with_sidecar_tile_kernel(
    src: tla.Tensor, dst: tla.Tensor, side: tla.Tensor
) -> None:
    """Loop carries UPDATE windows plus an ordinary (non-UPDATE) sidecar tile.

    Regression for VectorRegion ``scf.for`` init lowering: sidecar / TensorDesc
    bases may still be abstract ``!tla.ptr`` in ``valueMap`` when the for is
    rebuilt; they must rematerialize to i64 rather than hard-fail.
    """
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    side_tile = tla.tile_view(side, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            for _ in tla.range(3):
                x_reg, src_tile = src_tile.load(
                    UnalignLoadParams(post_mode=PostMode.POST_MODE_UPDATE)
                )
                dst_tile = dst_tile.store(
                    x_reg,
                    UnalignStoreParams(post_mode=PostMode.POST_MODE_UPDATE),
                )
                side_tile.store(x_reg)
                side_tile = tla.tile_view(side, tla.make_shape(64), tla.make_coord(0))


def test_load_unalign_update_loop_with_sidecar_tile(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        load_unalign_update_loop_with_sidecar_tile_kernel,
        type_args=(
            _ub_tensor(extent=256),
            _ub_tensor(extent=256),
            _ub_tensor(extent=256),
        ),
    )
    assert "tla.load_unalign_update" in mlir
    assert "tla.store_unalign_update" in mlir
    assert "scf.for" in mlir
    assert "!tla.ureg" not in mlir


@tla.kernel
def load_unalign_update_inplace_loop_kernel(buf: tla.Tensor) -> None:
    """Same-tile load+store UPDATE (in-place) inside a natural for."""
    tile = tla.tile_view(buf, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            for _ in tla.range(3):
                cur = tile
                x_reg, _ = cur.load(
                    UnalignLoadParams(post_mode=PostMode.POST_MODE_UPDATE)
                )
                tile = cur.store(
                    x_reg,
                    UnalignStoreParams(post_mode=PostMode.POST_MODE_UPDATE),
                )


def test_load_unalign_update_inplace_loop_emits_ops(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        load_unalign_update_inplace_loop_kernel,
        type_args=(_ub_tensor(extent=256),),
    )
    assert "tla.load_unalign_update" in mlir
    assert "tla.store_unalign_update" in mlir
    assert "scf.for" in mlir
    assert "!tla.ureg" not in mlir


def test_load_unalign_update_rejects_nonzero_coord(compiler_tlair: Any) -> None:
    @tla.kernel
    def bad_origin(src: tla.Tensor, dst: tla.Tensor) -> None:
        src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(1))
        dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                x_reg, src_tile = src_tile.load(
                    UnalignLoadParams(post_mode=PostMode.POST_MODE_UPDATE)
                )
                dst_tile.store(x_reg)

    with pytest.raises(Exception, match="origin window|coord"):
        compiler_tlair(
            bad_origin,
            type_args=(_ub_tensor(extent=256), _ub_tensor(extent=256)),
        )


def test_load_unalign_update_rejects_explicit_post_stride(
    compiler_tlair: Any,
) -> None:
    """UnAlign UPDATE forbids explicit post_update_stride; emit uses origin_shape."""

    @tla.kernel
    def bad_stride(src: tla.Tensor, dst: tla.Tensor) -> None:
        src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                src_tile.load(
                    UnalignLoadParams(
                        post_mode=PostMode.POST_MODE_UPDATE,
                        post_update_stride=64,
                    )
                )

    with pytest.raises(TlaLoweringError, match="do not pass post_update_stride"):
        compiler_tlair(bad_stride, type_args=(_ub_tensor(), _ub_tensor()))


def test_load_rejects_gm_tensor(compiler_tlair: Any) -> None:
    with pytest.raises(
        tla.TlaCoreAPIError,
        match="invalid argument 'source'.*expected addrspace ub, got gm",
    ):
        compiler_tlair(load_from_gm_kernel, type_args=(_gm_tensor(),))


@pytest.mark.parametrize(
    ("kernel", "match"),
    [
        (load_unalign_post_update_stride_kernel, "post_update_stride"),
    ],
)
def test_load_rejects_unsupported_params(
    compiler_tlair: Any, kernel: Any, match: str
) -> None:
    with pytest.raises(
        (TlaLoweringError, UnsupportedExecutionLowering, NotImplementedError),
        match=match,
    ):
        compiler_tlair(
            kernel,
            type_args=(_ub_tensor(), _ub_tensor()),
        )
