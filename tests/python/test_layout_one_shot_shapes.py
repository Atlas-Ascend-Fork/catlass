# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Import the complete real layout module and exercise one-shot shape inputs."""
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "python/catlass_cppgen/catlass_cppgen/catlass/layout/layout.py"


def layout_class(name):
    spec = importlib.util.spec_from_file_location("cppgen_layout_under_test", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


@pytest.mark.parametrize("name", ["PrivateLayout", "nZ", "zN", "zZ", "nN"])
@pytest.mark.parametrize("iterator_kind", ["generator", "list_iterator"])
def test_one_shot_shapes_work_with_default_strides(name, iterator_kind):
    dims = [2, 3, 4]
    shape = (value for value in dims) if iterator_kind == "generator" else iter(dims)
    layout = layout_class(name)(shape)
    assert layout.shape == (2, 3, 4)
    assert layout.stride == (1, 1, 1)
    assert layout.capacity == 24
    assert list(shape) == []


@pytest.mark.parametrize("shape", [(2, 3), [2, 3], (), []])
def test_sequence_and_empty_controls_keep_existing_defaults(shape):
    layout = layout_class("PrivateLayout")(shape)
    assert layout.shape == tuple(shape)
    assert layout.stride == (1,) * len(shape)
    assert layout.capacity == (6 if shape else 1)


def test_explicit_stride_accepts_one_shot_iterables_without_changing_values():
    layout = layout_class("nZ")(iter([2, 3]), iter([3, 1]))
    assert layout.shape == (2, 3)
    assert layout.stride == (3, 1)
    assert layout.capacity == 6


@pytest.mark.parametrize("name,expected_stride", [("RowMajor", (3, 1)), ("ColumnMajor", (1, 2))])
def test_matrix_layout_sequence_controls(name, expected_stride):
    layout = layout_class(name)((2, 3))
    assert layout.shape == (2, 3)
    assert layout.stride == expected_stride
    assert layout.capacity == 6
