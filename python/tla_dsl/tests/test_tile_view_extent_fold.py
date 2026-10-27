"""Frontend tests for the tile view row-extent fold.

The lit test tile-view-row-extent-fold.mlir checks the fold on hand-written
MLIR. These tests check that a Python kernel built with ``tla.range`` and
``tla.tile_view`` produces IR that the fold recognises.

Each test prints the IR after tla-lower-tensor-desc and reads the
``origin_shape`` of every strip descriptor. A folded extent is a constant; a
cropped extent is the result of an ``arith.minsi``.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor

_LANES = 64  # elements in one vector register
_STRIPS = 2  # strips per row; a single-iteration loop would be folded away


def _require_tla_compile() -> pathlib.Path:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    tla_compile = repo_root / "csrc" / "mlir" / "build" / "tools" / "tla-compile" / "TlaCompile"
    if not tla_compile.exists():
        pytest.skip("TlaCompile binary not found; build csrc/mlir first")
    return tla_compile


@tla.kernel
def strip_loop_kernel(gm: tla.Tensor) -> None:
    """Copy a UB tile onto itself row by row, one vector register at a time."""
    c0 = 0
    c1 = 1
    rows = gm.origin_shape[0]
    cols = gm.origin_shape[1]
    ub_ptr = tla.allocate(128 * 128, tla.Float32, tla.AddressSpace.ub, 256)
    ub = tla.make_tensor(
        ub_ptr,
        tla.make_layout(
            tla.make_shape(rows, cols),
            tla.make_stride(cols, c1),
            layoutTag=tla.arch.RowMajor,
        ),
    )
    with tla.vector():
        tla.copy(ub, gm)
        # The row crop computes `rows - row_i`. The loop runs up to `rows`, so
        # that is at least 1 and the fold applies.
        with tla.vec.func(mode="simd"):
            for row_i in tla.range(c0, rows, c1):
                for col_j in tla.range(c0, _STRIPS, c1):
                    strip = tla.tile_view(
                        ub, tla.make_shape(c1, _LANES), tla.make_coord(row_i, col_j)
                    )
                    strip.store(strip.load())
        tla.copy(gm, ub)


def _strip_descriptor_extents(pass_output: str) -> list[tuple[str, str]]:
    """Return the (rows, cols) extents of every one-row strip descriptor."""
    out = []
    for line in pass_output.splitlines():
        if " = tla.tensor_desc " not in line:
            continue
        # Keep only the strip views, not the descriptor of the whole plane.
        if "!tla.shape<1,64>" not in line:
            continue
        match = re.search(r"origin_shape\[([^,\]]+),\s*([^\]]+)\]", line)
        assert match, line
        out.append((match.group(1).strip(), match.group(2).strip()))
    return out


def _lower(tmp_path: pathlib.Path) -> str:
    """Compile the kernel and return the IR printed after tla-lower-tensor-desc."""
    gm = make_fake_tensor(
        tla.Float32, (128, 128), (128, 1), layout_tag=tla.arch.RowMajor
    ).mark_layout_dynamic()
    path = tmp_path / "strip_loop.mlir"
    path.write_text(strip_loop_kernel.dump_mlir(type_args=(gm,)))
    result = subprocess.run(
        [
            str(_require_tla_compile()),
            str(path),
            "-o",
            "-",
            "--mlir-print-ir-after=tla-lower-tensor-desc",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    # --mlir-print-ir-after prints the IR on stderr.
    return result.stderr


def test_row_extent_folds_when_loop_counts_up_to_that_extent(tmp_path) -> None:
    """The row crop is folded away, but the column crop stays."""
    pass_output = _lower(tmp_path)
    strips = _strip_descriptor_extents(pass_output)
    assert strips, pass_output
    for rows, cols in strips:
        # Folded: the extent is the requested constant 1, not a minsi result.
        assert rows.startswith("%c1"), (rows, pass_output)
        # The column crop must stay. For example, a row that is 100 elements
        # wide takes two strips, and the second one holds only 36 elements:
        # min(64, 100 - 64) = 36.
        assert not cols.startswith("%c"), (cols, pass_output)
