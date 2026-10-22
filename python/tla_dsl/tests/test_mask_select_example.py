from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from catlass import tla
from catlass.tla.runtime import make_fake_tensor


@pytest.fixture
def mask_select_example(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "examples/end_to_end/vector_ops"))
    return importlib.import_module("mask_select_op")


@pytest.mark.parametrize("dtype_name", ["i8", "f16", "f32"])
def test_example_cpu_reference(mask_select_example, dtype_name, monkeypatch) -> None:
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    torch = pytest.importorskip("torch")
    example = mask_select_example
    _, dtype, _ = example._set_kernel_config("mask_select", dtype_name, (512,))
    data_a = torch.full((512,), 3, dtype=dtype)
    data_b = torch.full((512,), -2, dtype=dtype)

    result = example.HARNESS.config.expected("mask_select", (data_a, data_b))

    lane = torch.arange(example.VL_ELE)
    selected = ((lane < example.VL_ELE // 2) & (lane < example.VL_ELE // 4)) | (
        (lane >= example.VL_ELE // 2) & ((lane % 4) == 0)
    )
    expected = torch.where(selected.repeat(example.LOOPS), data_a, data_b)
    assert torch.equal(result, expected)
    chunks = result.reshape(-1, example.VL_ELE)
    assert bool((chunks == 3).any(dim=1).all())
    assert bool((chunks == -2).any(dim=1).all())


@pytest.mark.parametrize("dtype_name", ["i8", "f16", "f32"])
def test_example_dumps_mask_select_mlir(mask_select_example, dtype_name) -> None:
    example = mask_select_example
    dtype, _, _ = example._set_kernel_config("mask_select", dtype_name, (512,))
    tensor = make_fake_tensor(
        dtype,
        (512,),
        (1,),
        addrspace=tla.AddressSpace.gm,
        origin_shape=(512,),
        layout_tag=tla.arch.RowMajor,
    )

    mlir = example.mask_select_op.dump_mlir(type_args=(tensor, tensor, tensor))
    assert "tla.where" in mlir
    assert "tla.create_mask" in mlir
