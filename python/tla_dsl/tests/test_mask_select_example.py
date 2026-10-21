from __future__ import annotations

import importlib
import os
import subprocess
import sys
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
def test_example_compiles_to_kernel_object(
    mask_select_example, dtype_name, request
) -> None:
    if not os.environ.get("ASCEND_HOME_PATH"):
        pytest.skip("CANN environment required: source set_env.sh")
    request.getfixturevalue("isolated_compile_cache")
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

    artifact = tla.compile(
        example.mask_select_op,
        tensor,
        tensor,
        tensor,
        options="--npu-arch 3510",
    )

    binary = Path(artifact.kernel_binary_path)
    assert binary.is_file()
    assert binary.stat().st_size > 0


@pytest.mark.parametrize("dtype_name", ["i8", "f16", "f32"])
def test_example_on_npu(mask_select_example, dtype_name, request, monkeypatch) -> None:
    if not os.environ.get("ASCEND_HOME_PATH"):
        pytest.skip("CANN environment required: source set_env.sh")
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu", exc_type=ImportError)
    if not torch.npu.is_available():
        pytest.skip("No accessible Ascend NPU")
    request.getfixturevalue("isolated_compile_cache")

    result = subprocess.run(
        [
            sys.executable,
            "-u",
            mask_select_example.__file__,
            "mask_select",
            "--dtype",
            dtype_name,
            "--shape",
            "512",
            "--device",
            os.environ.get("CATLASS_TEST_NPU_DEVICE", "0"),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    print(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "launch_ok=True" in result.stdout
    assert "outputs equal expected mask_select? True" in result.stdout
