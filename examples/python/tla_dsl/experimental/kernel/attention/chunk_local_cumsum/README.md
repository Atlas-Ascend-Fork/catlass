# ChunkLocalCumsum Example Readme

## 功能说明

- 算子功能：对门控张量 `g` 执行 chunk 内局部前缀累加（cumsum），基于 CATLASS DSL 实现的 AIV 算子。
- 计算公式：

  $$
    out[i, :] = \sum_{j=0}^{i} g[j, :] \quad (i = 0, 1, \ldots, BT-1)
  $$
  若指定 `scale`，则输出乘以 `scale`：$out = out \times scale$。

  其中$g$是输入张量，$out$是输出张量，$BT$为 chunk 大小。

- 支持产品型号：Ascend 950PR&950DT系列产品

## 参数说明

ChunkLocalCumsum 的输入输出参数如下：

| 参数 | 属性 | shape | dtype | 说明 |
| ----- | ----- | ----- | ----- | ----- |
| `g` | Input | `(B, T, H, D)`（BSND）或`(T, H, D)`（TND） | `float16`/`bfloat16` | 门控张量，D=64或128 |
| `out` | Output | 与`g`一致 | `float16`/`float32` | 累加结果，dtype由`output_dtype`指定 |

## 约束说明

BSND、TND 模式及维度应满足下述约束条件。

| 模式 | 约束 |
| ----- | -------- |
| BSND | `T` 能被 `chunk_size` 整除 |
| TND | 每个序列长度能被 `chunk_size` 整除，`cu_seqlens` 首元素为 0、末元素等于 T、严格递增 |
| D=128 | 内部切分为两次 `D=64` 调用 |

不支持 `reverse=True`、`head_first=True` 和显式 `chunk_indices`。


## 使用示例

### 命令行参数

```bash
python3 chunk_local_cumsum.py B T H D [deviceId] [选项]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 参数说明 |
| ----- | -------- | ------ |
| `B` | 必填 | 输入的 batch 大小。BSND 模式的输入 shape 为 `(B, T, H, D)`；TND 模式下该位置参数仍需填写，建议填 1，不参与输入 shape 构造。 |
| `T` | 必填 | 输入的序列长度。BSND 模式下为每个 batch 的序列长度，必须能被 `--chunk-size` 整除；TND 模式下为所有序列的 token 总数。 |
| `H` | 必填 | 输入的 head 数。 |
| `D` | 必填 | 输入的特征维度，支持 64 或 128。 |
| `deviceId` | `0` | 指定运行设备 ID；省略时默认使用设备 0。 |
| `--mode` | `bsnd` | 输入布局，支持 `bsnd` 和 `tnd`。TND 模式使用 `(T, H, D)` 输入，并要求指定 `--cu-seqlens`。 |
| `--chunk-size` | `64` | chunk 大小，支持 64 或 128。 |
| `--dtype` | `float16` | 输入 dtype，支持 `float16` 或 `bfloat16`。 |
| `--output-dtype` | `float32` | 输出 dtype，支持 `float16` 或 `float32`。 |
| `--scale` | `None` | 可选缩放因子；默认不缩放，指定后输出为局部累加结果乘以该因子。 |
| `--cu-seqlens` | `None` | TND 模式必填的累计序列边界，以逗号分隔，例如 `0,64,128`。首元素为 0，末元素等于 T，严格递增，且各序列长度均能被 chunk 大小整除；BSND 模式不使用该参数。 |
| `--block` | `160` | kernel 启动的 block 数量。 |
| `--seed` | `42` | 生成随机输入所用的种子。 |
| `--force-recompile` | `False` | 是否强制重新编译 kernel；指定该选项时为 `True`。 |
| `--no-cache` | `False` | 是否禁用 kernel 编译缓存；默认启用缓存，指定该选项时禁用。 |
| `--cache-dir` | 脚本目录下的 `artifacts/runtime-cache` | kernel 编译缓存根目录。显式传入相对路径时，相对于当前工作目录解析。 |

### 执行示例

准备好 PyTorch、torch_npu、CATLASS DSL 和配套 CANN 环境。当前样例已在 CANN 9.2.0 正式版、`ascend-catlass-dsl 2.1.1.dev20261027` 环境中验证；该 DSL 版本需要 CANN 提供 `acl.rt.launch_kernel_with_host_args` 接口。

进入项目根目录，切换到样例所在目录，执行算子样例程序。脚本根据位置参数生成输入，完成 kernel 编译、NPU 执行和精度比较，无需预先编译可执行文件，也无需 `workload.json`。

```bash
cd examples/python/tla_dsl/experimental/kernel/attention/chunk_local_cumsum
python3 chunk_local_cumsum.py 1 64 1 64 --chunk-size 64 --force-recompile --no-cache
```

上述命令省略 `deviceId`，默认使用设备 0，对 shape 为 `(1, 64, 1, 64)` 的 FP16 输入进行 chunk 内累加，chunk 大小为 64，输出类型为 FP32。直接参数模式不需要额外指定 `--run`。

执行结果如下，说明样例执行成功，精度通过：

```text
accuracy uuid=direct-bsnd-float16-bt64 pass=True max_abs=4.76837e-07 atol=0.02 rtol=0.02
validation uuid=direct-bsnd-float16-bt64 pass=True
```

其中 `max_abs` 为最大绝对误差，具体数值可能随输入和运行环境变化，以 `pass=True` 表示精度比较通过。

运行 BF16、带缩放的 BSND 用例：

```bash
python3 chunk_local_cumsum.py 1 128 2 64 0 --chunk-size 64 --dtype bfloat16 --output-dtype float32 --scale 0.125
```

运行 TND 变长用例，两个序列的长度均为 64：

```bash
python3 chunk_local_cumsum.py 1 128 1 64 0 --mode tnd --cu-seqlens 0,64,128 --chunk-size 64
```
