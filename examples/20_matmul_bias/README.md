# MatmulBias Example Readme

## 功能说明

- 算子功能：完成矩阵乘并在结果上叠加一维偏置
- 计算公式：

  $$
    \begin{aligned}
    D &= A \times B + bias \\
    D_{i,j} &= \Sigma_{k} A_{i,k}B_{k,j} + bias_{j}
    \end{aligned}
  $$

  其中$A$和$B$分别是形如`(m,k)`，`(k,n)`的输入矩阵，$bias$是形如`(n,)`的偏置向量，$D$是形如`(m,n)`的输出矩阵。

- 支持产品型号：Atlas A2/A3系列产品

## 样例参数说明

| 参数       | 属性 |shape|dtype| 说明                                                             |
| ---------- | ------ | --------|---------|----------------------------------------------- |
| `A`        | Input  | `(m,k)` | `float16/bfloat16` | 左矩阵，layout支持`RowMajor`（默认）和`ColumnMajor`  |
| `B`        | Input  | `(k,n)` | `float16/bfloat16` | 右矩阵，layout支持`RowMajor`（默认）和`ColumnMajor`，数据类型与左矩阵一致 |
| `bias`     | Input  | `(n,)`  | `float16` | 一维偏置，长度等于`n`|
| `D`        | Output | `(m,n)` | `float16/bfloat16` | 矩阵乘加偏置的结果，layout仅支持`RowMajor`，数据类型与左矩阵一致 |

## 使用范围说明

本样例在基础矩阵乘的基础上融合一维偏置。

## 使用示例

### 命令行参数

```bash
20_matmul_bias [m] [n] [k] [deviceId]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| --- | --- | --- |
| `m` | 无 | A 矩阵的 m 轴大小 |
| `n` | 无 | B 矩阵的 n 轴大小，同时是 bias 的长度 |
| `k` | 无 | A/B 矩阵的 k 轴大小 |
| `deviceId` | `0` | 指定运行设备ID |

### 执行示例

1. 进入项目根目录，编译样例代码生成相应的算子可执行文件。

    ```bash
    bash scripts/build.sh 20_matmul_bias
    ```

2. 切换到可执行文件的编译目录 `output/bin`，执行算子样例程序。

    ```bash
    cd output/bin
    ./20_matmul_bias 256 512 1024 0
    ```

3. 执行结果如下，说明样例执行成功，精度通过：

    ```text
    Compare success.
    ```
