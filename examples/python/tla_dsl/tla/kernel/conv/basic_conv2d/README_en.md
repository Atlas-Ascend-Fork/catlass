# BasicConv2d Example Readme

## Description

- Function: performs a two-dimensional convolution as an implicit GEMM, on one core with a single K partition and a single N partition.
- Formula:

  $$
    O_{co,ho,wo} = \sum_{ci}\sum_{kh}\sum_{kw}
      I_{ci,\; ho \cdot s_h + kh \cdot d_h - p_t,\; wo \cdot s_w + kw \cdot d_w - p_l}
      \cdot W_{co,ci,kh,kw}
  $$

  where $s$ is the stride, $d$ the dilation, and $p$ the padding; out-of-range positions are padded with zero by the hardware.

- The point of this example is not the convolution algorithm but that **a convolution is an ordinary GEMM plus one instruction**. `tla.experimental.img2col` performs the img2col expansion while it loads the NC1HWC0 feature map from L1 into L0A, and `tla.mmad` then does what it always does:

  | Implicit GEMM dimension | Value | Description |
  | --- | --- | --- |
  | `M` | `ho * wo` | Number of output spatial positions, `64` in this example |
  | `N` | `cout` | Number of output channels, `32` in this example |
  | `K` | `cin1 * kh * kw * C0` | The reduction axis, `288` in this example |

- Three conventions deserve particular attention:

  | Convention | Description |
  | --- | --- |
  | NC1HWC0 needs no new layout | It is the existing `zNUnAlign` tree `((h*w, 1), (C0, c1))`, where the `c1` leaf carries the feature map's `h*w*C0` pitch |
  | `C0` is counted in **bytes** | `C0` is 32 bytes, which is 16 elements for `f16`/`bf16` but 32 for `int8` |
  | The `K` axis is ordered `(cin1, kh, kw, c0)` | The filter matrix layout must agree with the `K` order img2col emits, and this is the one convention here that cannot be settled by reading the headers |

- The filter side needs no new capability at all: a filter matrix of `K` rows and `N` columns is already a valid B operand, and is loaded into L0B as `nZ`.
- The output operand of `tla.experimental.img2col` has the logical shape `(M, K)`, which is neither the feature map's shape nor the filter's, so it needs a separate reference tile for `make_tensor_like` to derive the `zN` tree from. That reference is metadata only; nothing is ever read or written through its pointer.
- Supported products: Ascend 950 series products

## Parameters

| Parameter | Attribute | Shape | dtype | Description |
| --- | --- | --- | --- | --- |
| `fmap` | Input | `(128, 16)` | `f16` | Feature map, a `RowMajor` buffer holding the flattened NC1HWC0 order; it corresponds to a `(1, 32, 8, 8)` input |
| `filt` | Input | `(288, 32)` | `f16` | Filter matrix; the layout is `RowMajor` and the `K` axis is ordered `(cin1, kh, kw, c0)` |
| `out` | Output | `(64, 32)` | `f32` | Convolution result; the layout is `RowMajor` and the shape is `(ho * wo, cout)` |

The convolution parameters are constants inside the example; changing them requires a re-run so that the kernel is recompiled:

| Constant | Value | Description |
| --- | --- | --- |
| `HI` / `WI` | `8` / `8` | Height and width of the input feature map |
| `CIN1` / `C0` / `CIN` | `2` / `16` / `32` | Fractal split of the input channels, and the total channel count |
| `COUT` | `32` | Number of output channels |
| `KH` / `KW` | `3` / `3` | Height and width of the convolution kernel |
| `PAD_T` / `PAD_B` / `PAD_L` / `PAD_R` | `1` | Padding on each of the four sides |
| `STRIDE_H` / `STRIDE_W` | `1` / `1` | Stride along the height and the width |
| `DIL_H` / `DIL_W` | `1` / `1` | Dilation along the height and the width |

## Usage Scope

This example is the minimal demonstration of `tla.experimental.img2col`. It contains no optimizations such as tiling, double buffering or multi-core partitioning, and it launches with `block_num=1`.

The size of a single tile is bounded by the L0A capacity. L0A is 64 KB, so `m_round * k * itemsize <= 65536` must hold; this example needs `64 * 288 * 2 = 36864` bytes, which leaves headroom. Exceeding that ceiling requires partitioning along `K`, and because the `K` axis is ordered `(cin1, kh, kw, c0)`, a partition boundary must fall on a whole `C1` slice rather than at an arbitrary channel.

The hardware can only pad with a constant, so padding modes such as reflect or causal cannot be expressed through the `pad` parameter and have to be materialized before the data is loaded.

The precision check uses the `k` form of `common.compare`, which scales the relative tolerance by the reduction length `K`, consistently with the other GEMM-like examples.

### Command Line Parameters

```bash
basic_conv2d.py [--device deviceId]
```

The command line parameters are described as follows:

| Parameter | Default Value | Description |
| --- | --- | --- |
| `deviceId` | `0` | ID of the device on which the program runs |

### Example

1. Go to the project root directory and set up the runtime environment for the sample code. For details, see [Quick Start](../../../../../../../python/tla_dsl/docs/zh/quick_start.md).

2. Run the operator sample program from the project root directory.

    ```bash
    python examples/python/tla_dsl/tla/kernel/conv/basic_conv2d/basic_conv2d.py --device 0
    ```

3. If the following result is displayed, the sample is successfully executed and the precision verification passes:

    ```text
    passed=True kernel.o=...
    ```
