#!/usr/bin/env python3
"""Generate TLA DSL Kernel API reference docs from ``core_api.py``.

Signatures, parameters and the `Directory:` path come from the source AST. The
docstrings come from the imported module instead, because an op can append to
its own docstring at import time (`mmad` / `mmad_mx` add the table of dtypes they
accept), which the source text does not show. Importing needs the MLIR python
bindings on the path.
Section order and blurbs live in ``DIRECTORY_SECTIONS`` below.
Writes English Markdown to ``docs/en/api/kernel_api_reference.md``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType

from common import (
    APIEntry,
    PACKAGE_ROOT,
    const_str,
    directory_path,
    function_entry,
    has_dsl_user_op,
    parse_all_names,
    render_reference,
    run_cli,
)


CORE_API_PATH = PACKAGE_ROOT / "catlass" / "core_api.py"
CORE_API_QUALNAME_PREFIX = "catlass.core_api."
TENSOR_API_PATH = PACKAGE_ROOT / "catlass" / "tla" / "tensor.py"
TYPING_API_PATH = PACKAGE_ROOT / "catlass" / "base_dsl" / "typing.py"
OUTPUT_PATH = PACKAGE_ROOT / "docs" / "en" / "api" / "kernel_api_reference.md"
GENERATED_BY = "python/tla_dsl/tools/generate_kernel_api_reference.py"

DIRECTORY_SECTIONS: list[tuple[str, str]] = [
    (
        "Basic Data Types and Operations",
        "Construction and views for front-end structured values such as "
        "Shape / Coord / Stride / Layout / Tensor, plus pointer helpers.",
    ),
    (
        "Numeric Scalars",
        "`Numeric` is the base of TLA DSL **scalar** types and values used "
        "inside `@tla.kernel`. Concrete types include integers "
        "(`tla.Int8` / `Int16` / `Int32` / `Int64` and unsigned variants), "
        "floats (`tla.Float16` / `BFloat16` / `Float32`, plus narrower "
        "formats), and `tla.Bool`. Construct values with the type, for "
        "example `tla.Int32(0)` or `tla.Float32(1.5)`.\n"
        "\n"
        "Use `Numeric` values for loop indices, coordinates, and other "
        "single-element state updated under runtime `if` / `tla.range` / "
        "`while`. Promote Python literals with `as_numeric`; change type "
        "with `Numeric.to`; reinterpret bits with `Numeric.bitcast`.\n"
        "\n"
        "The operators `+` `-` `*` `/` `//` `**` and the comparisons are\n"
        "evaluated on the core's own scalar unit, except inside a\n"
        '`tla.vec.func(mode="simt")` region, where each becomes the\n'
        "corresponding per-thread op (`tla.simt_add`, `tla.simt_cmp`, …). See\n"
        "**SIMT Vector Compute**.\n"
        "\n"
        "Constraints:\n"
        "\n"
        "- Prefer concrete types (`Int32`, `Float32`, …). The abstract bases "
        "`Numeric` / `Integer` / `Float` are not constructible.\n"
        "- Must be used inside a `@tla.kernel` body when the value "
        "participates in device-side control flow or computation.\n"
        "- A name reassigned under runtime `if` / `tla.range` / `while` must "
        "hold a `Numeric` (from `as_numeric` or a concrete type such as "
        "`tla.Int32(...)`). Plain Python `int` / `float` / `bool` cannot be "
        "carried as runtime state across that control flow.",
    ),
    (
        "Data Movement",
        "Tensor copies between on-chip and global memory, and UB register load/store.",
    ),
    ("Matrix Compute", "Cube-side matrix multiply-accumulate (`tla.mmad`)."),
    (
        "SIMD Vector Compute",
        "The default vector execution mode, entered with "
        '`tla.vec.func(mode="simd")`. The body operates on whole vector '
        "registers: data is staged into UB, viewed as tiles, loaded into "
        "`VectorSSA` values, and every op covers a full register under an "
        "optional lane mask. This is the path for regular, dense work, and the "
        "one every op below belongs to.\n"
        "\n"
        "Several entries here also have a per-thread form, used when the "
        'enclosing region is `tla.vec.func(mode="simt")`; each says so under '
        "`Constraints:`, and **SIMT Vector Compute** documents that mode.",
    ),
    ("SIMD Vector Compute / Mask Compute", "Mask creation and tail-mask updates."),
    (
        "SIMD Vector Compute / Basic Arithmetic",
        "Element-wise arithmetic and unary math ops. `VectorSSA` overloads "
        "`+` / `-` / `*` / `/` for `add` / `sub` / `mul` / `div` when no "
        "`mask=` is needed.",
    ),
    (
        "SIMD Vector Compute / Logical Compute",
        "Bitwise and logical ops on Mask / Vector.",
    ),
    (
        "SIMD Vector Compute / Compare and Select",
        "Vector compares that produce masks, and masked select.",
    ),
    (
        "SIMD Vector Compute / Data Fill",
        "Constant fill and lane-index sequence construction.",
    ),
    (
        "SIMD Vector Compute / Discrete and Aggregate",
        "Gather elements from a UB tensor by index, and masked reductions "
        "(`VectorSSA.reduce`).",
    ),
    (
        "SIMD Vector Compute / Data Rearrange",
        "Interleave / deinterleave and related lane reshuffles.",
    ),
    (
        "SIMD Vector Compute / Data Compress",
        "Compress valid lanes under a mask.",
    ),
    (
        "SIMD Vector Compute / Type Conversion",
        "Element-type conversion on the register-vector path.",
    ),
    (
        "SIMD Vector Compute / Histogram",
        "Histogram accumulation on register-vector data.",
    ),
    (
        "SIMT Vector Compute",
        "The per-thread vector execution mode, entered with "
        '`tla.vec.func(mode="simt", thread_block_dim=...)`. The body is the '
        "program of *one thread*: each thread addresses memory by index "
        "through ordinary element access, with no tiles, no vector registers "
        "and no UB staging.\n"
        "\n"
        "```python\n"
        "with tla.vector():\n"
        '    with tla.vec.func(mode="simt", thread_block_dim=400):\n'
        "        tid, _, _ = tla.arch.thread_idx()\n"
        "        nthreads, _, _ = tla.arch.thread_block_dim()\n"
        "        for i in tla.range(tid, 400, nthreads):\n"
        "            gm_c[i] = gm_a[i] + gm_b[i]\n"
        "    tla.pipe_barrier(tla.pipes.ALL)\n"
        "```\n"
        "\n"
        "The subsections below follow the same order as **SIMD Vector "
        "Compute**: what the region is and what may appear in it, then how a "
        "thread identifies itself, moves data, and computes.",
    ),
    (
        "SIMT Vector Compute / Region Scope",
        "The region is declared by `tla.vec.func` (see **Scopes and Control "
        "Flow**), which owns the geometry rules, the 2048 threads-per-block "
        "limit and the scalar-only capture rule. On-chip scratch is allocated "
        "outside the region with `tla.allocate` (see **Resource "
        "Management**), and a pipe barrier goes after it, inside "
        "`tla.vector()`, as the example above does.\n"
        "\n"
        "**What a SIMT region may contain**\n"
        "\n"
        "The body is per-thread scalar code. These are available:\n"
        "\n"
        "- Element access on a `gm` or `ub` tensor (`tensor[i]`, "
        "`tensor[i] = value`) -- see **Element Access** below.\n"
        "- Scalar arithmetic, comparison, selection and conversion on "
        "`Numeric` values, each emitting its per-thread op -- see **Basic "
        "Arithmetic**, **Compare and Select** and **Type Conversion** below.\n"
        "- `tla.range` loops and runtime `if`.\n"
        "- `tla.arch.thread_idx` / `thread_block_dim` / `sync_threads`, and "
        "the launch-level `tla.arch.block_idx` / `block_num` -- see **Thread "
        "Indexing** and **Thread Synchronization** below.\n"
        "\n"
        "These are not, and are rejected when the kernel is compiled:\n"
        "\n"
        "- Register-vector and mask ops -- `VectorSSA` / `MaskSSA` values, the "
        "arithmetic and mask APIs under **SIMD Vector Compute**, "
        "`Tensor.load` / `Tensor.store`, `tla.full`, `tla.create_mask`. A SIMT "
        "thread has no vector register of its own.\n"
        "- `tla.copy`: a DMA needs the tensor itself, and a SIMT region may "
        "capture only scalar runtime values. Stage data before the region, or "
        "let each thread address global memory directly.\n"
        "- `tla.local_mem_bar`, `tla.pipe_barrier` and `tla.print`.\n"
        "- `while` loops, in any `tla.vec.func`; carry the iteration through "
        "`tla.range` instead.\n"
        "- A nested `tla.vec.func`.",
    ),
    (
        "SIMT Vector Compute / Thread Indexing",
        "A thread learns which elements are its own from its index within the "
        "block and the block's extents, and typically walks a strided slice "
        "of the data with `tla.range`, as the chapter example shows. "
        "`thread_block_dim` is the number of threads in this block; the number "
        "of blocks in the launch is `tla.arch.block_num` (see **System "
        "Variable Access**).",
    ),
    (
        "SIMT Vector Compute / Thread Synchronization",
        "Barrier across the threads of one block. Cross-core and pipe "
        "synchronization stay outside the region -- see **Sync Control**.",
    ),
    (
        "SIMT Vector Compute / Element Access",
        "A thread moves data by element access on a `gm` or `ub` tensor: "
        "`value = tensor[i]` and `tensor[i] = value`, which emit the "
        "per-thread `tla.simt_load` / `tla.simt_store`. The layout must be "
        "`RowMajor` or `ColumnMajor`, the logical rank 1 or 2, and sub-byte "
        "element types other than `Bool` are not supported. Both are "
        "documented as `Tensor.__getitem__` / `Tensor.__setitem__` under "
        "**Data Movement**, which is also where the SIMD staging copies "
        "live.\n"
        "\n"
        "```python\n"
        'with tla.vec.func(mode="simt", thread_block_dim=256):\n'
        "    tid, _, _ = tla.arch.thread_idx()\n"
        "    gm_out[tid] = gm_in[tid]\n"
        "```",
    ),
    (
        "SIMT Vector Compute / Basic Arithmetic",
        "Arithmetic and unary math on `Numeric` values keep their SIMD "
        "spelling inside a SIMT region but emit the per-thread op: `+` / `-` / "
        "`*` / `/` become `tla.simt_add` / `simt_sub` / `simt_mul` / "
        "`simt_div`, and `tla.exp` / `log` / `abs` / `sqrt` / `max` / `min` "
        "become their `simt_` forms. The entries are shared with **SIMD "
        "Vector Compute / Basic Arithmetic**, where each one states its SIMT "
        "form under `Constraints:`.\n"
        "\n"
        "`tla.exp` and `tla.log` are available *only* in this mode: the "
        "core's own scalar unit has neither, so the scalar form is rejected "
        "outside a SIMT region.\n"
        "\n"
        "```python\n"
        'with tla.vec.func(mode="simt", thread_block_dim=256):\n'
        "    tid, _, _ = tla.arch.thread_idx()\n"
        "    x = gm_in[tid]\n"
        "    gm_out[tid] = tla.sqrt(tla.exp(x) + tla.abs(x))\n"
        "```",
    ),
    (
        "SIMT Vector Compute / Compare and Select",
        "The comparisons emit `tla.simt_cmp` and produce a `Bool`, not a "
        "`MaskSSA`; `tla.where` on a `Bool` condition emits the per-thread "
        "`tla.simt_where`. The entries are shared with **SIMD Vector Compute "
        "/ Compare and Select**.\n"
        "\n"
        "```python\n"
        'with tla.vec.func(mode="simt", thread_block_dim=256):\n'
        "    tid, _, _ = tla.arch.thread_idx()\n"
        "    a = gm_a[tid]\n"
        "    gm_out[tid] = tla.where(a > 0.0, a, -a)\n"
        "```",
    ),
    (
        "SIMT Vector Compute / Type Conversion",
        "`Numeric.to` emits the per-thread `tla.simt_cast` inside a SIMT "
        "region and the core's own scalar conversion outside one; the entry "
        "is under **Numeric Scalars**. There is no per-thread equivalent of "
        "the register-vector `tla.cast`.\n"
        "\n"
        "```python\n"
        'with tla.vec.func(mode="simt", thread_block_dim=256):\n'
        "    tid, _, _ = tla.arch.thread_idx()\n"
        "    gm_f32[tid] = gm_f16[tid].to(tla.Float32)\n"
        "```",
    ),
    (
        "Sync Control",
        "In-core / cross-core flags, pipe barriers, mutexes, and local-memory "
        "barriers.",
    ),
    (
        "System Variable Access",
        "Architecture attributes on `tla.arch` (layout tags, pipe identifiers, "
        "block helpers, etc.).",
    ),
    ("Resource Management", "On-chip scratch allocation via `allocate`."),
    (
        "Debug APIs",
        "In-kernel scalar / tensor debug printing.",
    ),
    (
        "Scopes and Control Flow",
        "Cube / Vector / `vec.func` regions and kernel-side loop ranges.",
    ),
]
DIRECTORY_ORDER = [path for path, _ in DIRECTORY_SECTIONS]
DIRECTORY_INTROS = dict(DIRECTORY_SECTIONS)


_MAKE_UNARY_HELPERS = frozenset({"_make_unary_op", "_make_scalar_aware_unary_op"})


def _unary_op_template(tree: ast.Module) -> APIEntry | None:
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "_make_unary_op":
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.FunctionDef)
                and child.name == "_unary"
                and has_dsl_user_op(child.decorator_list)
            ):
                return function_entry("_unary", child)
    return None


def _doc_from_make_unary_call(call: ast.Call) -> str:
    for kw in call.keywords:
        if kw.arg == "doc":
            return const_str(kw.value) or ""
    if len(call.args) >= 2:
        return const_str(call.args[1]) or ""
    return ""


def _function_defs_by_name(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def _collect_typing_helpers() -> dict[str, APIEntry]:
    """Collect typing.py free helpers and types that carry ``Directory:``.

    Opt-in via docstring (same rule as documented class methods); no name allowlist.
    Private ``_``-prefixed names are skipped. Class entries are marked ``is_class``.
    """
    if not TYPING_API_PATH.is_file():
        raise FileNotFoundError(f"typing module not found: {TYPING_API_PATH}")
    tree = ast.parse(
        TYPING_API_PATH.read_text(encoding="utf-8"), filename=str(TYPING_API_PATH)
    )
    entries: dict[str, APIEntry] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            doc = ast.get_docstring(node) or ""
            if directory_path(doc) is None:
                continue
            entries[node.name] = APIEntry(
                name=node.name,
                qualified_name=f"catlass.base_dsl.typing.{node.name}",
                source_line=node.lineno,
                docstring=doc,
                is_class=True,
                source_path=TYPING_API_PATH,
            )
            continue
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        doc = ast.get_docstring(node) or ""
        if directory_path(doc) is None:
            continue
        entries[node.name] = function_entry(
            node.name,
            node,
            qualified_name=f"catlass.base_dsl.typing.{node.name}",
            source_path=TYPING_API_PATH,
        )
    return entries


def _collect_core_functions(
    tree: ast.Module, exported: set[str]
) -> dict[str, APIEntry]:
    entries: dict[str, APIEntry] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name == "_vec_func" and has_dsl_user_op(node.decorator_list):
            entry = function_entry("vec.func", node)
            entry.qualified_name = "catlass.core_api._vec_func"
            entries["vec.func"] = entry
            continue
        if node.name not in exported:
            continue
        if node.name == "print" or has_dsl_user_op(node.decorator_list):
            entries[node.name] = function_entry(node.name, node)
    return entries


def _collect_unary_aliases(
    tree: ast.Module, exported: set[str], template: APIEntry | None
) -> dict[str, APIEntry]:
    if template is None:
        return {}
    entries: dict[str, APIEntry] = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target, value = node.targets[0], node.value
        if not isinstance(target, ast.Name) or target.id not in exported:
            continue
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in _MAKE_UNARY_HELPERS
        ):
            continue
        name = target.id
        entries[name] = APIEntry(
            name=name,
            qualified_name=f"catlass.core_api.{name}",
            source_line=node.lineno,
            params=list(template.params),
            returns=template.returns,
            docstring=_doc_from_make_unary_call(value),
        )
    return entries


def _collect_scalar_aware_binary_aliases(
    tree: ast.Module, exported: set[str]
) -> dict[str, APIEntry]:
    """Public names built by ``_make_scalar_aware_binary_op`` (e.g. ``max`` / ``min``).

    Docs and signatures live on the vector-form helper (``_vector_max`` / ``_vector_min``);
    the Assign line is the public symbol users import.
    """
    defs = _function_defs_by_name(tree)
    entries: dict[str, APIEntry] = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target, value = node.targets[0], node.value
        if not isinstance(target, ast.Name) or target.id not in exported:
            continue
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "_make_scalar_aware_binary_op"
        ):
            continue
        if len(value.args) < 2 or not isinstance(value.args[1], ast.Name):
            continue
        vector_def = defs.get(value.args[1].id)
        if vector_def is None:
            continue
        name = target.id
        entry = function_entry(
            name,
            vector_def,
            qualified_name=f"catlass.core_api.{name}",
        )
        entry.source_line = node.lineno
        entries[name] = entry
    return entries


def _collect_arch_namespace(
    tree: ast.Module, exported: set[str]
) -> dict[str, APIEntry]:
    if "arch" not in exported:
        return {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target, value = node.targets[0], node.value
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "arch"
            and target.attr == "__doc__"
        ):
            return {
                "arch": APIEntry(
                    name="arch",
                    qualified_name="catlass.core_api.arch",
                    source_line=node.lineno,
                    docstring=const_str(value) or "",
                    is_namespace=True,
                )
            }
    return {
        "arch": APIEntry(
            name="arch",
            qualified_name="catlass.core_api.arch",
            source_line=None,
            docstring="",
            is_namespace=True,
        )
    }


# The ``arch`` helpers hang off the ``arch`` namespace object rather than being
# exported names of their own, so ``_collect_core_functions`` skips them. Their
# public spelling is not hand-listed here: it is read from the
# ``arch._set("<public>", <function>)`` calls at the end of ``core_api.py``, the
# same source of truth the runtime uses.
_ARCH_SIMT_DIRECTORY_ROOT = "SIMT Vector Compute"


def _arch_set_bindings(tree: ast.Module) -> dict[str, str]:
    """Map ``arch._set("public", func)`` to ``{func_name: "arch.public"}``."""
    bindings: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Expr):
            continue
        call = node.value
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "_set"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "arch"
            and len(call.args) == 2
        ):
            continue
        public, target = call.args
        public_name = const_str(public)
        if public_name is None or not isinstance(target, ast.Name):
            continue
        bindings[target.id] = f"arch.{public_name}"
    return bindings


def _collect_arch_simt_functions(tree: ast.Module) -> dict[str, APIEntry]:
    """Document the ``arch`` helpers whose ``Directory:`` is a SIMT chapter.

    The other ``arch`` callables are described as bullets in the namespace's own
    docstring rather than as entries; restricting the collector to the SIMT
    chapters keeps that as it is.
    """
    bindings = _arch_set_bindings(tree)
    entries: dict[str, APIEntry] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        public = bindings.get(node.name)
        if public is None or not has_dsl_user_op(node.decorator_list):
            continue
        directory = directory_path(ast.get_docstring(node) or "") or ""
        if not directory.startswith(_ARCH_SIMT_DIRECTORY_ROOT):
            continue
        entries[public] = function_entry(
            public,
            node,
            qualified_name=f"{CORE_API_QUALNAME_PREFIX}{node.name}",
        )
    return entries


def _collect_class_methods(
    path: Path,
    class_names: set[str],
    *,
    module: str,
    label: str | None = None,
) -> dict[str, APIEntry]:
    """Document methods of ``class_names`` that are ``@dsl_user_op`` and have ``Directory:``.

    Every matching class is scanned. Entry keys use ``label.method`` when
    ``label`` is set (e.g. document ``_Tensor`` as ``Tensor``); otherwise
    ``ClassName.method``.
    """
    if not path.is_file():
        return {}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in class_names
    ]
    if not classes:
        return {}

    entries: dict[str, APIEntry] = {}
    for cls in classes:
        entry_label = label if label is not None else cls.name
        for node in cls.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if not has_dsl_user_op(node.decorator_list):
                continue
            doc = ast.get_docstring(node) or ""
            if directory_path(doc) is None:
                continue
            key = f"{entry_label}.{node.name}"
            entries[key] = function_entry(
                key,
                node,
                qualified_name=f"{module}.{cls.name}.{node.name}",
                source_path=path.resolve(),
                drop_self=True,
            )
    return entries


def _import_core_api() -> ModuleType:
    """Import the op module, which is where the ``__modify_doc__`` hooks live."""
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    try:
        import catlass.core_api as core_api
    except ImportError as exc:
        raise RuntimeError(
            "generating the reference needs `catlass.core_api` importable (the "
            f"MLIR python bindings on PYTHONPATH): {exc}"
        ) from exc
    return core_api


def _apply_doc_hooks(entries: dict[str, APIEntry]) -> None:
    """Let ops that declare one extend their own docstring for the reference.

    ``copy`` / ``mmad`` / ``mmad_mx`` carry a ``__modify_doc__`` hook that
    appends the routes or dtypes they accept. Importing does not run it -- the
    runtime docstring stays as written -- so the reference asks for it here. A
    hook edits ``fn.__doc__`` in place, so the original is put back afterwards:
    a second pass must not stack the same table twice.
    """
    module = _import_core_api()
    for entry in entries.values():
        if not entry.qualified_name.startswith(CORE_API_QUALNAME_PREFIX):
            continue
        attr = entry.qualified_name[len(CORE_API_QUALNAME_PREFIX) :]
        op = getattr(module, attr, None)
        hook = getattr(op, "__modify_doc__", None)
        if hook is None:
            continue
        original = op.__doc__
        hook(op)
        entry.docstring = op.__doc__ or ""
        op.__doc__ = original


def parse_core_api(path: Path) -> dict[str, APIEntry]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    exported = parse_all_names(tree)
    entries: dict[str, APIEntry] = {}
    entries.update(_collect_core_functions(tree, exported))
    entries.update(_collect_unary_aliases(tree, exported, _unary_op_template(tree)))
    entries.update(_collect_scalar_aware_binary_aliases(tree, exported))
    entries.update(_collect_arch_namespace(tree, exported))
    entries.update(_collect_typing_helpers())
    # Per-class labels (Numeric.to / Float.sqrt / ...); no shared alias.
    entries.update(
        _collect_class_methods(
            TYPING_API_PATH,
            {"Numeric", "Integer", "Float"},
            module="catlass.base_dsl.typing",
        )
    )
    entries.update(_collect_arch_simt_functions(tree))
    entries.update(
        _collect_class_methods(
            TENSOR_API_PATH,
            {"Tensor", "_Tensor"},
            label="Tensor",
            module="catlass.tla.tensor",
        )
    )
    entries.update(
        _collect_class_methods(
            path, {"VectorSSA"}, label="VectorSSA", module="catlass.core_api"
        )
    )
    _apply_doc_hooks(entries)
    return entries


def generate(*, docs_dir: Path | None = None) -> str:
    if not CORE_API_PATH.is_file():
        raise FileNotFoundError(f"core_api not found: {CORE_API_PATH}")

    docs_dir = docs_dir or OUTPUT_PATH.parent
    return render_reference(
        parse_core_api(CORE_API_PATH),
        docs_dir=docs_dir,
        title="TLA DSL Kernel API Reference",
        intro=[
            "This document describes the **TLA DSL kernel-side Core APIs** "
            "(typically imported as `import catlass.tla as tla`). "
            "It covers data structures, compute / sync helpers, on-chip resources, "
            "and debug printing. Host-side compile / launch / tensor binding are in "
            "`docs/en/api/host_api_reference.md`; environment variables are in "
            "`docs/zh/kernel_development/core_concepts/env_vars.md`.",
            "Interface descriptions and examples come from each op's source docstring "
            "(`Directory:` plus `Description:` / `Parameters:` / `Constraints:` / `Example:`).",
            "All APIs must be called inside a `@tla.kernel`-decorated kernel function body.",
            "The vector unit has two execution modes, and they are separate "
            "chapters below. **SIMD Vector Compute** is the default: whole "
            "vector registers over data staged in UB, for regular dense work. "
            "**SIMT Vector Compute** is one thread per element addressing "
            "memory by index, for work too irregular or scattered for tiling "
            "to pay off. A number of scalar APIs behave differently depending "
            "on which mode encloses them -- element access and the scalar "
            "arithmetic / compare / select / conversion ops all emit a "
            "per-thread op inside a SIMT region, and each entry says so under "
            "`Constraints:`.",
        ],
        header_sources=(
            "Do not edit manually. Update docstrings/Examples in catlass/core_api.py",
            "(or the defining module for imported types) instead.",
        ),
        leftovers_title="Other Core APIs",
        leftovers_blurb=(
            "APIs still exported by the current source but not yet filed under "
            "the directory tree above."
        ),
        directory_order=DIRECTORY_ORDER,
        directory_intros=DIRECTORY_INTROS,
        default_source_path=CORE_API_PATH,
        generated_by=GENERATED_BY,
    )


def main() -> int:
    return run_cli(
        description=__doc__,
        default_output=OUTPUT_PATH,
        generate_fn=generate,
    )


if __name__ == "__main__":
    raise SystemExit(main())
