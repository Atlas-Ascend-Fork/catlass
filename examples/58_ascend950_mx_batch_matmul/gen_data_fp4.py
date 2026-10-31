#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# FP4 MX Batch Matmul 数据生成脚本：
# A/B 为 float4_e2m1（每 32 元素一组 e8m0 缩放因子），输出 float golden。
# 与 fp8_mx_batch_matmul/gen_data.py 结构对应，仅量化器替换为 FP4 E2M1。

import os
import argparse
from typing import Dict, Tuple

import torch

WORKSPACE = os.path.dirname(os.path.abspath(__file__))

_BLOCK_SIZE = 32
_EPSILON = 1e-12
_MIN_SCALE_EXP = -128
_MAX_SCALE_EXP = 127

_FP4_FORMATS: Dict[str, Dict[str, float]] = {
    "E2M1": {
        "exp_bits": 2,
        "mantissa_bits": 1,
        "bias": 1,
        "emax": 2,
        "max_value": 6.0,
        "min_value": -6.0,
    },
    "E1M2": {
        "exp_bits": 1,
        "mantissa_bits": 2,
        "bias": 1,
        "emax": 0,
        "max_value": 1.75,
        "min_value": -1.75,
    },
}


def _build_fp4_lut(format_name: str) -> torch.Tensor:
    config = _FP4_FORMATS[format_name]
    exp_bits = int(config["exp_bits"])
    mantissa_bits = int(config["mantissa_bits"])
    bias = float(config["bias"])

    values = []
    for i in range(16):
        sign = (i >> 3) & 0x01
        exp = (i >> mantissa_bits) & ((1 << exp_bits) - 1)
        mantissa = i & ((1 << mantissa_bits) - 1)

        if exp == 0:
            if mantissa == 0:
                value = 0.0
            else:
                value = (mantissa / float(1 << mantissa_bits)) * (2.0 ** (1.0 - bias))
        else:
            value = (1.0 + mantissa / float(1 << mantissa_bits)) * (
                2.0 ** (float(exp) - bias)
            )

        if sign == 1:
            value = -value
        values.append(value)

    return torch.tensor(values, dtype=torch.float32)


_FP4_LUT = {
    "E2M1": _build_fp4_lut("E2M1"),
    "E1M2": _build_fp4_lut("E1M2"),
}


def _quantize_to_fp4_lut(
    values: torch.Tensor, format_name: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    lut = _FP4_LUT[format_name].to(values.device)
    min_value = _FP4_FORMATS[format_name]["min_value"]
    max_value = _FP4_FORMATS[format_name]["max_value"]

    clamped = values.clamp(min_value, max_value)

    distances = (clamped.unsqueeze(-1) - lut).abs()
    indices = torch.argmin(distances, dim=-1)
    quantized = lut[indices]

    return quantized, indices.to(torch.uint8)


def _pack_fp4_nibbles(index_matrix: torch.Tensor) -> torch.Tensor:
    rows, cols = index_matrix.shape
    if cols % 2 != 0:
        index_matrix = torch.cat(
            [
                index_matrix,
                torch.zeros((rows, 1), dtype=torch.uint8, device=index_matrix.device),
            ],
            dim=1,
        )

    low = index_matrix[:, 0::2]
    high = index_matrix[:, 1::2] << 4
    packed = low | high
    return packed.to(torch.uint8)


def _quantize_axis_last(
    matrix: torch.Tensor, format_name: str, block_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = matrix.shape
    padded_n = ((n + block_size - 1) // block_size) * block_size
    num_blocks = padded_n // block_size

    if padded_n != n:
        padded = torch.zeros((m, padded_n), dtype=matrix.dtype, device=matrix.device)
        padded[:, :n] = matrix
    else:
        padded = matrix

    blocks = padded.view(m, num_blocks, block_size)
    max_abs = blocks.abs().amax(dim=-1)

    exp = (
        torch.floor(torch.log2(torch.clamp(max_abs, min=_EPSILON)))
        - _FP4_FORMATS[format_name]["emax"]
    )
    exp = torch.where(max_abs < _EPSILON, torch.zeros_like(exp), exp)
    exp = exp.clamp(_MIN_SCALE_EXP, _MAX_SCALE_EXP)
    scale = torch.pow(torch.tensor(2.0, dtype=torch.float32, device=matrix.device), exp)

    scaled = blocks / scale.unsqueeze(-1)
    quantized_blocks, _ = _quantize_to_fp4_lut(scaled, format_name)
    dequant_blocks = quantized_blocks * scale.unsqueeze(-1)

    quantized = quantized_blocks.reshape(m, padded_n)
    dequantized = dequant_blocks.reshape(m, padded_n)
    if padded_n != n:
        quantized = quantized[:, :n].contiguous()
        dequantized = dequantized[:, :n].contiguous()

    padded_blocks = ((num_blocks + 1) // 2) * 2
    if padded_blocks != num_blocks:
        scale_padded = torch.ones(
            (m, padded_blocks), dtype=torch.float32, device=matrix.device
        )
        scale_padded[:, :num_blocks] = scale
        scale = scale_padded

    return quantized, scale, dequantized


def _quantize_axis_first(
    matrix: torch.Tensor, format_name: str, block_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quantized_t, scale_t, dequantized_t = _quantize_axis_last(
        matrix.t().contiguous(), format_name, block_size
    )
    return (
        quantized_t.t().contiguous(),
        scale_t.t().contiguous(),
        dequantized_t.t().contiguous(),
    )


def _quantize(
    matrix: torch.Tensor, format_name: str, axis: int, block_size: int = _BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if axis == 0:
        return _quantize_axis_first(matrix, format_name, block_size)
    if axis == 1:
        return _quantize_axis_last(matrix, format_name, block_size)
    raise ValueError("axis must be 0 or 1")


def gen_data_fp4_e2m1(row, col, axis):
    matrix = torch.randn((row, col), dtype=torch.float32)
    quantized_matrix, scale_matrix, dequantized_matrix = _quantize(matrix, "E2M1", axis)
    return quantized_matrix, scale_matrix, dequantized_matrix


def gen_data(batch, m, n, k, trans_a, trans_b):
    # FP4 数据使用独立目录，避免与 FP8 的 data/ 相互覆盖
    data_dir = os.path.join(WORKSPACE, "data_fp4")
    input_dir = os.path.join(data_dir, "input")
    golden_dir = os.path.join(data_dir, "golden")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(golden_dir, exist_ok=True)

    a_uint8_list = []
    b_uint8_list = []
    a_scale_list = []
    b_scale_list = []
    c_fp32_list = []

    for _ in range(batch):
        # A: axis=1 (列方向，K 维分块)
        a_fp4, a_scale, a_fp32 = gen_data_fp4_e2m1(m, k, 1)
        # B: axis=0 (行方向，K 维分块)
        b_fp4, b_scale, b_fp32 = gen_data_fp4_e2m1(k, n, 0)

        # 对齐 scale 的 K/32 维度到 2
        a_scale = a_scale.reshape(a_scale.shape[0], a_scale.shape[1] // 2, 2)
        b_scale = b_scale.reshape(b_scale.shape[0] // 2, 2, b_scale.shape[1])

        if trans_a == 1:
            a_fp4 = a_fp4.t().contiguous()
            a_scale = a_scale.permute(1, 0, 2)

        if trans_b == 1:
            b_fp4 = b_fp4.t().contiguous()
            b_scale = b_scale.permute(2, 0, 1)
        else:
            b_scale = b_scale.permute(0, 2, 1)

        # FP4 索引打包为 uint8（两值一字节）
        _, a_indices = _quantize_to_fp4_lut(a_fp4, "E2M1")
        a_uint8 = _pack_fp4_nibbles(a_indices)
        _, b_indices = _quantize_to_fp4_lut(b_fp4, "E2M1")
        b_uint8 = _pack_fp4_nibbles(b_indices)

        a_uint8_list.append(a_uint8)
        b_uint8_list.append(b_uint8)
        a_scale_list.append(a_scale)
        b_scale_list.append(b_scale)

        c_fp32 = a_fp32 @ b_fp32
        c_fp32_list.append(c_fp32)

    a_uint8_batch = torch.stack(a_uint8_list, dim=0)
    b_uint8_batch = torch.stack(b_uint8_list, dim=0)
    a_scale_batch = torch.stack(a_scale_list, dim=0)
    b_scale_batch = torch.stack(b_scale_list, dim=0)
    c_fp32_batch = torch.stack(c_fp32_list, dim=0)

    a_np = torch.tensor(
        a_uint8_batch.flatten().untyped_storage(), dtype=torch.int8
    ).numpy()
    b_np = torch.tensor(
        b_uint8_batch.flatten().untyped_storage(), dtype=torch.int8
    ).numpy()
    a_np.tofile(os.path.join(input_dir, "a_4.bin"))
    b_np.tofile(os.path.join(input_dir, "b_4.bin"))

    a_scale_np = torch.tensor(
        a_scale_batch.to(torch.float8_e8m0fnu).flatten().untyped_storage(),
        dtype=torch.int8,
    ).numpy()
    b_scale_np = torch.tensor(
        b_scale_batch.to(torch.float8_e8m0fnu).flatten().untyped_storage(),
        dtype=torch.int8,
    ).numpy()
    a_scale_np.tofile(os.path.join(input_dir, "a_scale.bin"))
    b_scale_np.tofile(os.path.join(input_dir, "b_scale.bin"))

    c_np = c_fp32_batch.numpy()
    c_np.tofile(os.path.join(golden_dir, "expected_data.bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate MX-FP4 batch matmul inputs and FP32 golden."
    )
    parser.add_argument("batch", type=int)
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("trans_a", type=int)
    parser.add_argument("trans_b", type=int)
    args = parser.parse_args()
    gen_data(args.batch, args.m, args.n, args.k, args.trans_a, args.trans_b)
