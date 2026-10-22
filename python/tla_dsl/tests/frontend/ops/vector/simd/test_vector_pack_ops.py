from __future__ import annotations

import pytest

from catlass import tla
from catlass.core_api import MaskSSA
from catlass.tla.runtime import make_fake_tensor


@tla.kernel
def _mask_pack(
    src_dtype: tla.Constexpr[type],
    dst_dtype: tla.Constexpr[type],
    part: tla.Constexpr[object],
) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            source = tla.create_mask(pattern=tla.mask.H, dtype=src_dtype)
            packed = tla.pack(source, part=part)
            assert isinstance(packed, MaskSSA)
            one = tla.full(1, dst_dtype)
            zero = tla.full(0, dst_dtype)
            tla.where(packed, one, zero)


@pytest.mark.parametrize(
    ("src_dtype", "dst_dtype", "src_lanes", "dst_lanes"),
    [
        (tla.Int32, tla.Int16, 64, 128),
        (tla.Int16, tla.Int8, 128, 256),
    ],
    ids=["mask64-mask128", "mask128-mask256"],
)
@pytest.mark.parametrize("part", [tla.RegPart.LOW, tla.RegPart.HIGH])
def test_mask_pack_doubles_logical_width(
    src_dtype, dst_dtype, src_lanes, dst_lanes, part
) -> None:
    mlir = _mask_pack.dump_mlir(type_args=(src_dtype, dst_dtype, part))

    (line,) = [line for line in mlir.splitlines() if "tla.pack" in line]
    assert f"<{part.value}>" in line
    assert f"!tla.mask<{src_lanes}>" in line
    assert f"!tla.mask<{dst_lanes}>" in line
    assert "tla.where" in mlir


@tla.kernel
def _mask_pack_compile_kernel(
    src: tla.Tensor,
    dst: tla.Tensor,
    value_dtype: tla.Constexpr[type],
    source_mask_dtype: tla.Constexpr[type],
    lanes: tla.Constexpr[int],
) -> None:
    loaded = tla.flag("mask_pack_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    computed = tla.flag("mask_pack_computed", tla.arch.VECTOR, tla.arch.MTE3)
    src_ptr = tla.allocate(lanes, value_dtype, tla.AddressSpace.ub, 256)
    dst_ptr = tla.allocate(lanes, value_dtype, tla.AddressSpace.ub, 256)
    ub_src = tla.make_tensor_like(src_ptr, src, tla.arch.RowMajor)
    ub_dst = tla.make_tensor_like(dst_ptr, dst, tla.arch.RowMajor)

    with tla.vector():  # noqa: SIM117 - DSL regions must remain syntactically nested.
        tla.copy(ub_src, src)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            value = ub_src.load()
            zero = tla.sub(value, value)
            source = tla.create_mask(pattern=tla.mask.H, dtype=source_mask_dtype)
            packed = tla.pack(source, part=tla.RegPart.LOW)
            ub_dst.store(tla.where(packed, value, zero))
        tla.set_flag(computed)
        tla.wait_flag(computed)
        tla.copy(dst, ub_dst)
        tla.pipe_barrier(tla.pipes.ALL)


@pytest.mark.parametrize(
    ("value_dtype", "source_mask_dtype", "lanes"),
    [
        (tla.Int16, tla.Int32, 128),
        (tla.Int8, tla.Int16, 256),
    ],
    ids=["mask64-mask128", "mask128-mask256"],
)
def test_mask_pack_compiles_to_kernel_object(
    isolated_compile_cache, value_dtype, source_mask_dtype, lanes
) -> None:
    del isolated_compile_cache
    tensors = tuple(
        make_fake_tensor(
            value_dtype,
            (lanes,),
            (1,),
            origin_shape=(lanes,),
            layout_tag=tla.arch.RowMajor,
        )
        for _ in range(2)
    )

    compiled = tla.compile(
        _mask_pack_compile_kernel,
        *tensors,
        value_dtype,
        source_mask_dtype,
        lanes,
        options="--npu-arch 3510",
    )

    assert compiled.kernel_binary_path.is_file()
    assert compiled.kernel_binary_path.stat().st_size > 0


@tla.kernel
def _pack_with_invalid_part(part: tla.Constexpr[object]) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            source = tla.create_mask(pattern=tla.mask.H, dtype=tla.Int32)
            tla.pack(source, part=part)


def test_pack_rejects_plain_integer_part() -> None:
    with pytest.raises(
        tla.TlaCoreAPIError,
        match=r"expected tla\.RegPart\.LOW or tla\.RegPart\.HIGH",
    ):
        _pack_with_invalid_part.dump_mlir(type_args=(0,))


@tla.kernel
def _pack_unsupported_mask(dtype: tla.Constexpr[type]) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            mask = tla.create_mask(pattern=tla.mask.ALL, dtype=dtype)
            tla.pack(mask, part=tla.RegPart.LOW)


@pytest.mark.parametrize(
    ("dtype", "width"),
    [(tla.Int64, 32), (tla.Int8, 256)],
)
def test_pack_rejects_unsupported_mask_widths(dtype, width) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match=f"unsupported mask width {width}"):
        _pack_unsupported_mask.dump_mlir(type_args=(dtype,))


@tla.kernel
def _pack_in_simt() -> None:
    with tla.vector():
        with tla.vec.func(mode="simt"):
            source = tla.create_mask(pattern=tla.mask.H, dtype=tla.Int32)
            tla.pack(source, part=tla.RegPart.LOW)


def test_pack_rejects_simt_mode() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="only supported.*simd"):
        _pack_in_simt.dump_mlir()


@pytest.mark.parametrize("src_lanes", [64, 128])
@pytest.mark.parametrize("part", ["low", "high"])
def test_pack_verifier_accepts_lowered_func(src_lanes, part) -> None:
    import catlass.runtime as runtime
    from catlass._mlir import ir

    with runtime._eager_capture():
        module = ir.Module.parse(
            f"""
            module {{
              func.func @lowered(%src: !tla.mask<{src_lanes}>)
                  -> !tla.mask<{2 * src_lanes}> {{
                %result = tla.pack %src, <{part}>
                    : !tla.mask<{src_lanes}> -> !tla.mask<{2 * src_lanes}>
                return %result : !tla.mask<{2 * src_lanes}>
              }}
            }}
            """
        )
        assert module.operation.verify()
