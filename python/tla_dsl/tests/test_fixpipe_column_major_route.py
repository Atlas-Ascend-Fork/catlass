# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""What the L0C -> GM fixpipe routes do and do not accept.

The ColumnMajor GM destination is the transposing (nz2dn) store. These pin its
parity with the RowMajor (nz2nd) store.
"""

from __future__ import annotations

import pytest

from catlass.core_api import _SUPPORTED_CUBE_FIXPIPE_ROUTES

_GM_KEY_NO_RELU = ("gm", "{layout}", "-", "NO_QUANT", False)
_GM_KEY_RELU = ("gm", "{layout}", "-", "NO_QUANT", True)


def _gm_routes(layout: str):
    return {
        key: value
        for key, value in _SUPPORTED_CUBE_FIXPIPE_ROUTES.items()
        if key[0] == "gm" and key[1] == layout
    }


def test_column_major_gm_route_exists():
    """The transposing store is reachable at all."""
    assert _gm_routes("ColumnMajor"), "no L0C -> GM ColumnMajor fixpipe route"


@pytest.mark.parametrize("relu", [False, True])
def test_column_major_matches_row_major_dtypes(relu):
    """Same accumulator dtypes in, same destination dtypes out, relu included."""
    row = _SUPPORTED_CUBE_FIXPIPE_ROUTES[("gm", "RowMajor", "-", "NO_QUANT", relu)]
    col = _SUPPORTED_CUBE_FIXPIPE_ROUTES[("gm", "ColumnMajor", "-", "NO_QUANT", relu)]
    assert col == row, (
        "the ColumnMajor GM fixpipe route must accept exactly the dtypes the "
        f"RowMajor one does; row={row} col={col}"
    )
