# BasicConv2d Example Readme

## 功能说明

- 算子功能：以隐式 GEMM（implicit GEMM）的方式完成二维卷积，单核、单 K 分块、单 N 分块。
- 计算公式：

  $$
    O_{co,ho,wo} = \sum_{ci}\sum_{kh}\sum_{kw}
      I_{ci,\; ho \cdot s_h + kh \cdot d_h - p_t,\; wo \cdot s_w + kw \cdot d_w - p_l}
      \cdot W_{co,ci,kh,kw}
  $$

  其中 $s$ 为 stride、$d$ 为 dilation、$p$ 为 padding；越界位置由硬件补 0。

- 本样例的重点不在卷积算法本身，而在于**卷积就是一次普通的 GEMM 加一条指令**。`tla.experimental.img2col` 在把 L1 上的 NC1HWC0 特征图搬入 L0A 的同时完成 img2col 展开，`tla.mmad` 随后执行的就是它一向执行的矩阵乘：

  | 隐式 GEMM 维度 | 取值 | 说明 |
  | --- | --- | --- |
  | `M` | `ho * wo` | 输出的空间位置数，本样例为 `64` |
  | `N` | `cout` | 输出通道数，本样例为 `32` |
  | `K` | `cin1 * kh * kw * C0` | 归约轴，本样例为 `288` |

- 三个需要特别注意的约定：

  | 约定 | 说明 |
  | --- | --- |
  | NC1HWC0 无需新 layout | 它就是已有的 `zNUnAlign` 树 `((h*w, 1), (C0, c1))`，其中 `c1` 叶子的 stride 为特征图的 `h*w*C0` |
  | `C0` 的单位是**字节** | `C0` 为 32 字节，对 `f16`/`bf16` 是 16 个元素，对 `int8` 则是 32 个 |
  | `K` 轴的顺序是 `(cin1, kh, kw, c0)` | filter 矩阵的排布必须与 img2col 发出的 `K` 顺序一致，这是本样例中唯一无法靠阅读头文件确定的约定 |

- filter 侧不需要任何新能力：一个 `K` 行、`N` 列的滤波器矩阵本身就是合法的 B 操作数，按 `nZ` 搬入 L0B 即可。
- `tla.experimental.img2col` 的输出操作数逻辑形状为 `(M, K)`，既不是特征图的形状也不是 filter 的形状，因此需要一个独立的参考 tile 供 `make_tensor_like` 推导 `zN` 树；该参考仅作为元信息使用，不会通过其指针读写数据。
- 支持产品型号：Ascend 950 系列产品

## 样例参数说明

| 参数 | 属性 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `fmap` | Input | `(128, 16)` | `f16` | 特征图，按 NC1HWC0 展平后的 `RowMajor` 缓冲区；对应 `(1, 32, 8, 8)` 的输入 |
| `filt` | Input | `(288, 32)` | `f16` | 滤波器矩阵，`layout` 为 `RowMajor`，`K` 轴按 `(cin1, kh, kw, c0)` 排列 |
| `out` | Output | `(64, 32)` | `f32` | 卷积结果，`layout` 为 `RowMajor`，形状为 `(ho * wo, cout)` |

卷积参数均为样例内的常量，修改后需重新运行以触发重新编译：

| 常量 | 取值 | 说明 |
| --- | --- | --- |
| `HI` / `WI` | `8` / `8` | 输入特征图的高和宽 |
| `CIN1` / `C0` / `CIN` | `2` / `16` / `32` | 输入通道的分形拆分与总通道数 |
| `COUT` | `32` | 输出通道数 |
| `KH` / `KW` | `3` / `3` | 卷积核的高和宽 |
| `PAD_T` / `PAD_B` / `PAD_L` / `PAD_R` | `1` | 四个方向的填充量 |
| `STRIDE_H` / `STRIDE_W` | `1` / `1` | 高、宽方向的步长 |
| `DIL_H` / `DIL_W` | `1` / `1` | 高、宽方向的膨胀系数 |

## 使用范围说明

本样例为 `tla.experimental.img2col` 的最小示例，不含 tiling、double buffer、多核切分等优化，启动时 `block_num=1`。

单块的规模受 L0A 容量限制。L0A 为 64 KB，因此必须满足 `m_round * k * itemsize <= 65536`；本样例为 `64 * 288 * 2 = 36864` 字节，留有余量。超出该上限时需要按 `K` 分块，且由于 `K` 轴按 `(cin1, kh, kw, c0)` 排列，分块边界必须落在完整的 `C1` 切片上，而不能落在任意通道处。

硬件只能以常量填充，因此 reflect、causal 等填充方式无法由 `pad` 参数表达，需要在搬入前显式构造好数据。

精度校验使用 `common.compare` 的 `k` 形式，即按归约长度 `K` 放缩相对误差阈值，与其他 GEMM 类样例一致。

### 命令行参数

```bash
basic_conv2d.py [--device deviceId]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `deviceId` | `0` | 指定运行设备 ID |

### 执行示例

1. 进入项目根目录，参考[快速上手](../../../docs/zh/quick_start.md)准备样例代码的运行环境。

2. 在项目根目录下执行算子样例程序。

    ```bash
    python python/tla_dsl/examples/end_to_end/basic_conv2d/basic_conv2d.py --device 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    passed=True kernel.o=...
    ```
