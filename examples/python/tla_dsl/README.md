# CATLASS Python DSL 样例

本目录收录 **CATLASS Python** 的端到端算子样例与 DSL 教学示例，与 `examples/` 下的 C++ 样例并列；框架代码与文档位于 [`python/tla_dsl`](../../../python/tla_dsl/README.md)。

## 目录结构

```text
examples/python/tla_dsl
├── tla/kernel/       # TLA 算子样例：matmul / attention / conv / gmm / vec
├── dsl_tutorials/    # DSL 语言特性与向量 / SIMT / 标量教学示例
├── experimental/     # 实验性算子样例，目录结构与 tla/ 一致
└── helpers/          # 样例共用的参数、精度比对与工具
```

| 目录 | 概述 |
|------|------|
| [`tla/kernel/`](tla/kernel/README.md) | 以 TLA 抽象实现的完整算子样例，按算子类别分组，每个样例可独立运行。 |
| [`dsl_tutorials/`](dsl_tutorials/README.md) | 单个 DSL 特性或指令类别的教学示例，聚焦一种用法。 |
| `experimental/` | 尚未稳定的实验性样例，组织方式与 `tla/kernel/` 一致。 |
| `helpers/` | 样例共用的参数解析、精度比对与工具函数，样例中通过 `from helpers import ...` 引用。 |

## 运行方式

以下命令均在 **CATLASS 仓库根目录** 执行：

```bash
python examples/python/tla_dsl/tla/kernel/matmul/basic_mmad/basic_matmul.py --device 0
python examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/binary_op.py add --device 0
```

每个样例目录下的 README 给出了该样例可用的完整参数、文件组织与约束说明。环境准备与工具链安装见[快速开始](../../../python/tla_dsl/docs/zh/quick_start.md)。

## 批量回归

仓库根目录的 `tests/run_dsl_test.sh` 会构建 DSL 并运行 `tests/dsl_battery` 中的上板用例，样例根目录由 `EXAMPLES_DIR` 指向本目录：

```bash
bash tests/run_dsl_test.sh --device 0
```
