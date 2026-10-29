# FP4MxMatmul Example Readme

## 功能说明

- 算子功能：完成 MX（微缩放）量化 FP4 矩阵乘计算，不含 Bias。
- 计算公式：

  $$
    \begin{aligned}
    C &= \hat{A} \times \hat{B} \\
    \hat{A}_{i,k} &= A_{i,k} \cdot MxScaleA_{i,\lfloor k/32 \rfloor} \\
    \hat{B}_{k,j} &= B_{k,j} \cdot MxScaleB_{\lfloor k/32 \rfloor,j}
    \end{aligned}
  $$

  其中 $A$ 和 $B$ 分别是形如 `(m,k)`、`(k,n)` 的输入矩阵，$MxScaleA$、$MxScaleB$ 为对应的 MX 缩放因子，沿 K 轴每 32 个元素共用一个缩放因子；$C$ 是形如 `(m,n)` 的输出矩阵。

- 实现变体：本样例提供两个可执行文件，对应同一 MX FP4 矩阵乘算子的两种 block 调度实现：
  - `fp4_mx_matmul`：采用恒等块 swizzle 调度。
  - `fp4_mx_matmul_aswt`：采用 ASWT 窗口调度，支持尾部 tile 多核拆分。

- 支持产品型号：Ascend 950PR&950DT系列产品

## 样例参数说明

| 参数       | 属性   | shape             | dtype             | 说明                                                             |
| ---------- | ------ | ----------------- | ----------------- | ---------------------------------------------------------------- |
| `A`        | Input  | `(m,k)`           | `float4_e2m1x2_t` / `float4_e1m2x2_t` | 左矩阵，layout 支持 `RowMajor`（默认）和`ColumnMajor`，K 轴按 64 对齐显式补零        |
| `B`        | Input  | `(k,n)`           | `float4_e2m1x2_t` / `float4_e1m2x2_t` | 右矩阵，layout 支持 `ColumnMajor`（默认）和`RowMajor`，K 轴按 64 对齐显式补零     |
| `MxScaleA` | Input  | `(m, ceil(k/32))` | `float8_e8m0_t`   | A 的 MX 缩放因子，沿 K 轴每 32 个元素分组，存储时按 2 对齐补齐（补 scale=1.0）   |
| `MxScaleB` | Input  | `(ceil(k/32), n)` | `float8_e8m0_t`   | B 的 MX 缩放因子，沿 K 轴每 32 个元素分组，存储时按 2 对齐补齐（补 scale=1.0）   |
| `C`        | Output | `(m,n)`           | `fp32`            | 量化结果矩阵，layout 支持 `RowMajor`                         |

## 使用范围说明

推荐范围：

- `m`、`n` 对齐 256，`k` 对齐 512，与 L1 分块 256 * 256 * 512 一致。

## 代码组织

```text
54_ascend950_fp4_mx_matmul
├── README.md
├── gen_data.py  # 生成测试数据（input/ 与 golden/）
├── fp4_mx_matmul.cpp  # 恒等块 swizzle 调度实现
├── fp4_mx_matmul_aswt.cpp  # ASWT 窗口调度实现
└── CMakeLists.txt
```

## 使用示例

### 命令行参数

执行样例命令如下：

```bash
# 恒等块 swizzle 调度
54_ascend950_fp4_mx_matmul [m] [n] [k] [deviceId]
```

```bash
# ASWT 窗口调度
54_ascend950_fp4_mx_matmul_aswt [m] [n] [k] [deviceId]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `m` | 无 | A 矩阵的 m 轴大小 |
| `n` | 无 | B 矩阵的 n 轴大小 |
| `k` | 无 | A/B 矩阵的 k 轴大小 |
| `deviceId` | `0` | 指定运行设备ID |

生成测试数据命令行如下：

```bash
python3 gen_data.py [m] [n] [k] [trans_a] [trans_b] [--type-a TYPE] [--type-b TYPE] [--data-root DIR]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `m` | 无 | 矩阵 m 轴大小 |
| `n` | 无 | 矩阵 n 轴大小 |
| `k` | 无 | 矩阵 k 轴大小 |
| `trans_a` | 无 | A 矩阵是否转置（0 不转置 / 1 转置） |
| `trans_b` | 无 | B 矩阵是否转置（0 不转置 / 1 转置） |
| `--type-a` | `e2m1` | A 矩阵 FP4 类型（`e2m1` 或 `e1m2`） |
| `--type-b` | `e2m1` | B 矩阵 FP4 类型（`e2m1` 或 `e1m2`） |
| `--data-root` | 脚本所在目录 | 生成数据的根目录，`input/` 与 `golden/` 子目录创建在其下 |

其中 `MxScaleA`、`MxScaleB` 的 shape 分两层理解：

- 参数表中的 `(m, ceil(k/32))`、`(ceil(k/32), n)` 为逻辑 shape，表示沿 K 轴每 32 个元素共用一个 scale。
- 实际存储时按 2 字节对齐：把 scale 分组数补齐到偶数后，将相邻 2 个 scale 打包成一组，物理存储维度展开为三维（`ceil(k/32)` 补偶后等于 `2 × ceil(k/64)`）。各 layout 下的存储 shape 如下：

  - A 为 `RowMajor`：`MxScaleA` 存储 shape 为 `(m, ceil(k/64), 2)`
  - A 为 `ColumnMajor`：`MxScaleA` 存储 shape 为 `(ceil(k/64), m, 2)`
  - B 为 `RowMajor`：`MxScaleB` 存储 shape 为 `(ceil(k/64), n, 2)`
  - B 为 `ColumnMajor`：`MxScaleB` 存储 shape 为 `(n, ceil(k/64), 2)`

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件（Ascend 950 算子，编译时需加 `-DCATLASS_ARCH=3510`）。

    ```bash
    # 恒等块 swizzle 调度
    bash scripts/build.sh 54_ascend950_fp4_mx_matmul -DCATLASS_ARCH=3510
    # ASWT 窗口调度
    bash scripts/build.sh 54_ascend950_fp4_mx_matmul_aswt -DCATLASS_ARCH=3510
    ```

2. 生成测试数据（在 `examples/54_ascend950_fp4_mx_matmul/data` 下生成 `input/` 与 `golden/`）。

    ```bash
    python3 examples/54_ascend950_fp4_mx_matmul/gen_data.py 256 512 1024 0 1 --type-a e2m1 --type-b e2m1
    ```

3. 切换到可执行文件的编译目录 `output/bin`，执行算子样例程序。

    ```bash
    cd output/bin
    # 恒等块 swizzle 调度
    ./54_ascend950_fp4_mx_matmul 256 512 1024 0
    # ASWT 窗口调度
    ./54_ascend950_fp4_mx_matmul_aswt 256 512 1024 0
    ```

4. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```

## 补充说明

`MxMatmulTla` 与 `BlockMmadTla` 搭配使用的 DispatchPolicy 为 **`Gemm::MmadMx`**（定义见 `include/catlass/gemm/dispatch_policy.hpp`），模板参数顺序与默认值如下：

| 模板参数             | 默认值  | 参数说明                                                                                                            |
| -------------------- | ------- | ------------------------------------------------------------------------------------------------------------------- |
| `ArchTag`            | 无      | 架构标签，例如 `Arch::Ascend950`                                                                                    |
| `ENABLE_UNIT_FLAG`   | `false` | 是否开启 UnitFlag；当 `L0C_STAGES > 1`（L0C 多缓冲）时必须为 `false`                                                |
| `L1_SCALE_FACTOR_K`  | `16`    | GM→L1 的 MX scale 一次驻留所覆盖的 **L1 K 方向条带个数**；为 `1` 时表示每个 L1 K 条带各搬一次 scale（见类型内注释） |
| `ENABLE_L1_RESIDENT` | `false` | 是否启用 L1 常驻                                                                                                    |
| `L1A_STAGES`         | `2`     | L1 上加载矩阵 A 的 buffer 数量                                                                                      |
| `L1B_STAGES`         | `2`     | L1 上加载矩阵 B 的 buffer 数量                                                                                      |
| `L0A_STAGES`         | `2`     | L0 上加载矩阵 A 的 buffer 数量                                                                                      |
| `L0B_STAGES`         | `2`     | L0 上加载矩阵 B 的 buffer 数量                                                                                      |

设矩阵Shape为`M N K`, L1上的分块大小为`m1 n1 k1`，M方向的分块数量`mTiles = CeilDiv(M, m1)`，N方向的分块数量`nTiles = CeilDiv(N, n1)`，总任务数为`taskBlocks = mTiles * nTiles`，在以下两种情况下可以选择开启enableL1Resident：

1.`mTiles = 1`，且`nTiles > CoreNum`，且`K < 2 * k1`。此时还可以设置`l0CStages=2`(需要关闭enableUnitFlag)，如果空间不足无法设置`l0CStages=2`，则将`n1`设置为原来的一半。

2.`nTiles = 1`，且`mTiles > CoreNum`, 且`K < 2 * k1`。此时还可以设置`l0CStages=2`(需要关闭enableUnitFlag)，如果空间不足无法设置`l0CStages=2`，则将`m1`设置为原来的一半。
