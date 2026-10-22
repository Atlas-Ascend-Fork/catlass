"""Host-side MX quantization for the mx_mmad examples.

Importable and torch-only: nothing here knows about TLA, so the same code can
feed ``mx_matmul.py`` / ``mx_matmul_bias.py`` and be exercised on its own. The
device side only ever sees the bytes this module produces -- e8m0 scale blocks
for the MX copy routes, quantized mantissas for the operands.

Two layers:

* the primitives -- ``quantize_mx_fp8`` / ``quantize_mx_fp4`` -- which split one
  matrix into quantized values, an e8m0 block scale, and the dequantized fp32 the
  golden is built from. fp8 and fp4 diverge here and nowhere else: fp8 quantises
  with torch's float8 dtypes, while fp4 needs a 16-entry LUT and nibble packing.
* ``gen_mx_matmul_inputs``, which runs both sides, applies the host-side
  orientation each ``--layout-*`` asks for, and hands back buffers ready to bind
  plus the fp32 reference. Both examples call this, so a change to the
  quantization cannot leave one of them behind.

Scale blocks go over as the plain (M, ceil(K/32)) and (ceil(K/32), N) matrices,
padded to an even group count. The GM->L1 DMA does the zZMxScale / nNMxScale
fractal reorder, steered by the GM-side layout tag, so nothing here
pre-swizzles them into fractal order -- only into the group pairing the
non-transposing copy needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# One e8m0 shared exponent per this many elements along K (OCP microscaling).
MX_SCALE_GROUP_NUM = 32

# Two e8m0 groups per 16-bit unit: the GM MxScale layouts move the scale block
# as 16-bit elements, which is why every block count below is rounded to even.
MX_SCALE_ELE_NUM_PER_C0 = 2

# fp8 format constants, mirroring examples/53_ascend950_fp8_mx_matmul/gen_data.py.
# Keyed by the torch dtype, which is what selects the quantization grid.
FP8_FORMATS = {
    torch.float8_e4m3fn: {"emax": 8, "max_value": 448.0},
    torch.float8_e5m2: {"emax": 15, "max_value": 57344.0},
}

# fp8 case name -> torch dtype. The TLA element type is the kernel side's
# business and stays in the examples.
FP8_DTYPES = {
    "f8e4m3fn": torch.float8_e4m3fn,
    "f8e5m2": torch.float8_e5m2,
}

# fp4 format constants, mirroring examples/54_ascend950_fp4_mx_matmul/gen_data.py.
FP4_FORMATS = {
    "f4e2m1": {
        "exp_bits": 2,
        "mantissa_bits": 1,
        "bias": 1,
        "emax": 2,
        "max_value": 6.0,
    },
    "f4e1m2": {
        "exp_bits": 1,
        "mantissa_bits": 2,
        "bias": 1,
        "emax": 0,
        "max_value": 1.75,
    },
}


# ---------------------------------------------------------------------------
# Shared pieces: the e8m0 block scale is a byte per 32 elements whatever the
# operand format is.
# ---------------------------------------------------------------------------


def e8m0_exp(max_abs: torch.Tensor, emax: int) -> torch.Tensor:
    """Per-block e8m0 exponent, bit-exact with gen_data.py's ``e8m0_exp``.

    ``floor(log2(x))`` for positive finite fp32 ``x`` is exactly
    ``biased_exp(x) - 127``, so it is read off the exponent field rather than
    computed in floating point.
    """
    zero_mask = max_abs < 1e-30
    safe = torch.where(zero_mask, torch.ones_like(max_abs), max_abs).float()
    bits = safe.contiguous().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) - 127 - emax
    exp = exp.clamp(-128, 127)
    return torch.where(zero_mask, torch.zeros_like(exp), exp)


def scale_bytes(scale: torch.Tensor, rows: int, nb: int) -> torch.Tensor:
    """e8m0 storage bytes: torch encodes exp2(e) as the biased exponent 127+e."""
    return torch.tensor(
        scale.to(torch.float8_e8m0fnu).flatten().untyped_storage(), dtype=torch.uint8
    ).reshape(rows, nb)


def pad_scale_to_even_groups(scale: torch.Tensor, nb: int) -> torch.Tensor:
    """Pad the block count to an even number of groups.

    The GM MxScale layout moves e8m0 as 16-bit units -- two groups per unit --
    so an odd group total cannot be moved at all. Padded groups carry the e8m0
    byte for 1.0 and land on operand pad that is quantized zeros (the bc-side K
    alignment zeroes the L1 pad too), so 1.0-scaled zeros add nothing.
    """
    sk_pad = (nb + 1) // 2 * 2
    if sk_pad == nb:
        return scale
    rows = scale.shape[0]
    return torch.cat(
        [scale, scale.new_full((rows, sk_pad - nb), 127)],
        dim=1,
    )


# ---------------------------------------------------------------------------
# fp8
# ---------------------------------------------------------------------------


def quantize_mx_fp8(matrix: torch.Tensor, axis: int, fp8_dtype):
    """Split ``matrix`` into fp8 mantissas plus e8m0 block scales along ``axis``.

    K is padded up to a whole scale group before quantising, then the pad is
    trimmed back off the quantized and dequantized matrices -- only the scale
    keeps the padded group count, rounded up to an even number of groups for
    the 16-bit e8m0 DMA units (gen_data.py's pad-then-trim plus its
    ``padded_blocks`` rounding). The operands go over at their real K and the
    DMA adapts to the unaligned tail; the pad elements, quantized zeros, exist
    only so the boundary block can be built.

    Returns ``(fp8_values, e8m0_scale_bytes, dequantized_fp32)``. The
    dequantized tensor is what the device actually multiplies, so the reference
    is built from it -- only accumulation order then differs.
    """
    fmt = FP8_FORMATS[fp8_dtype]
    work = matrix if axis == 1 else matrix.transpose(0, 1)
    work = work.float().contiguous()
    rows, cols = work.shape
    nb = (cols + MX_SCALE_GROUP_NUM - 1) // MX_SCALE_GROUP_NUM
    padded_cols = nb * MX_SCALE_GROUP_NUM
    if padded_cols != cols:
        work = torch.cat([work, work.new_zeros(rows, padded_cols - cols)], dim=1)

    blocks = work.view(rows, nb, MX_SCALE_GROUP_NUM)
    exp = e8m0_exp(blocks.abs().amax(dim=-1), fmt["emax"])
    scale = torch.exp2(exp.float())
    scaled = (blocks / scale.unsqueeze(-1)).clamp(-fmt["max_value"], fmt["max_value"])
    q = scaled.to(fp8_dtype)
    dequant = (q.float() * scale.unsqueeze(-1)).reshape(rows, padded_cols)
    q = q.reshape(rows, padded_cols)
    # Trim the pad back off; the scale keeps the padded group count.
    if padded_cols != cols:
        q = q[:, :cols].reshape(rows, cols)
        dequant = dequant[:, :cols].reshape(rows, cols)
    scale_bytes_out = pad_scale_to_even_groups(scale_bytes(scale, rows, nb), nb)

    if axis == 1:
        return q, scale_bytes_out, dequant
    return (
        q.transpose(0, 1).contiguous(),
        scale_bytes_out.transpose(0, 1).contiguous(),
        dequant.transpose(0, 1).contiguous(),
    )


# ---------------------------------------------------------------------------
# fp4
# ---------------------------------------------------------------------------


def fp4_lut(fmt: str) -> torch.Tensor:
    """The 16 representable fp4 values, in encoding order."""
    cfg = FP4_FORMATS[fmt]
    exp_bits, mbits, bias = cfg["exp_bits"], cfg["mantissa_bits"], float(cfg["bias"])
    values = []
    for i in range(16):
        sign = (i >> 3) & 0x01
        exp = (i >> mbits) & ((1 << exp_bits) - 1)
        mant = i & ((1 << mbits) - 1)
        if exp == 0:
            v = 0.0 if mant == 0 else (mant / float(1 << mbits)) * (2.0 ** (1.0 - bias))
        else:
            v = (1.0 + mant / float(1 << mbits)) * (2.0 ** (float(exp) - bias))
        values.append(-v if sign else v)
    return torch.tensor(values, dtype=torch.float32)


def quantize_mx_fp4(matrix: torch.Tensor, axis: int, fmt: str):
    """Split ``matrix`` into fp4 nibble indices plus e8m0 block scales.

    Returns ``(packed_uint8, e8m0_scale_bytes, dequantized_fp32)``. Packing is
    two elements per byte, element 2i in the low nibble (as gen_data.py does);
    an odd K puts the last element in a final byte whose high nibble carries
    the zero pad element.

    As in the fp8 path, K is padded up to a whole scale group before quantising
    and the pad is trimmed back off afterwards -- the packed operand goes over
    at its real K (``ceil(K/2)`` bytes per row) and only the scale keeps the
    padded group count, rounded up to an even number of groups for the 16-bit
    e8m0 DMA units.
    """
    cfg = FP4_FORMATS[fmt]
    lut = fp4_lut(fmt)
    work = (matrix if axis == 1 else matrix.transpose(0, 1)).float().contiguous()
    rows, cols = work.shape
    nb = (cols + MX_SCALE_GROUP_NUM - 1) // MX_SCALE_GROUP_NUM
    padded_cols = nb * MX_SCALE_GROUP_NUM
    if padded_cols != cols:
        work = torch.cat([work, work.new_zeros(rows, padded_cols - cols)], dim=1)

    blocks = work.view(rows, nb, MX_SCALE_GROUP_NUM)
    max_abs = blocks.abs().amax(dim=-1)
    exp = torch.floor(torch.log2(max_abs.clamp(min=1e-30))) - cfg["emax"]
    exp = torch.where(max_abs > 0, exp, torch.zeros_like(exp)).clamp(-127, 127)
    scale = torch.exp2(exp)

    scaled = (blocks / scale.unsqueeze(-1)).clamp(-cfg["max_value"], cfg["max_value"])
    # Nearest representable value; ties go to the lower index, matching argmin.
    # The index IS the fp4 bit pattern (the LUT is addressed by encoding:
    # sign/exp/mantissa), so the nibbles packed below are what the device
    # stores -- gen_data.py recovers the same indices by re-running the LUT
    # argmin over the quantized values, an identity except for -0 -> +0, which
    # this argmin never picks anyway (the tie goes to +0's lower index).
    idx = (scaled.unsqueeze(-1) - lut).abs().argmin(dim=-1)
    dequant = (lut[idx] * scale.unsqueeze(-1)).reshape(rows, padded_cols)
    idx = idx.reshape(rows, padded_cols)
    # Trim the pad back off, then pack whole bytes: an odd tail element takes a
    # zero pad nibble in the high half of its byte.
    if padded_cols != cols:
        dequant = dequant[:, :cols].reshape(rows, cols)
        idx = idx[:, :cols]
    if cols % 2 != 0:
        idx = torch.cat([idx, torch.zeros(rows, 1, dtype=idx.dtype)], dim=1)
    idx = idx.to(torch.uint8)

    packed = (idx[:, 0::2] | (idx[:, 1::2] << 4)).to(torch.uint8)
    scale_bytes_out = pad_scale_to_even_groups(scale_bytes(scale, rows, nb), nb)

    if axis == 1:
        return packed, scale_bytes_out, dequant
    # For the B side the caller wants (K, N) orientation back.
    return (
        packed,  # already (N, K/2): B is column-major
        scale_bytes_out.transpose(0, 1).contiguous(),  # (K/32, N)
        dequant.transpose(0, 1).contiguous(),  # (K, N)
    )


# ---------------------------------------------------------------------------
# Scale block group pairing, for the non-transposing GM scale routes.
# ---------------------------------------------------------------------------


def pair_interleave_a(scale: torch.Tensor) -> torch.Tensor:
    """(M, G) -> pairs of groups, M fastest between.

    ColMajorMxScaleA reaches L1 through a plain ND2NZ, so every 16-bit unit has
    to hold two groups of the *same* row. A plain transpose would pair two
    different rows and could never be right.
    """
    rows, groups = scale.shape
    inter = (
        scale.view(rows, groups // MX_SCALE_ELE_NUM_PER_C0, MX_SCALE_ELE_NUM_PER_C0)
        .permute(1, 0, 2)
        .contiguous()
        .reshape(-1)
    )
    return inter.view(rows, groups)


def pair_interleave_b(scale: torch.Tensor) -> torch.Tensor:
    """(G, N) -> pairs of groups, N fastest between.

    The B-side counterpart of ``pair_interleave_a``, for RowMajorMxScaleB.
    """
    groups, cols = scale.shape
    inter = (
        scale.view(groups // MX_SCALE_ELE_NUM_PER_C0, MX_SCALE_ELE_NUM_PER_C0, cols)
        .permute(0, 2, 1)
        .contiguous()
        .reshape(-1)
    )
    return inter.view(groups, cols)


# ---------------------------------------------------------------------------
# The whole host side of one mx matmul.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MxInputs:
    """Quantized operands, oriented for their layouts, plus the fp32 golden.

    Every tensor is a contiguous CPU buffer already byte-viewed as ``int8`` --
    the operand element type lives in the TLA descriptor, not the buffer -- so a
    caller only has to move them to the device and bind the matching GM tags.
    """

    #: A, oriented for ``--layout-a``: (M, K) row-major, or (K, M) when col.
    a: torch.Tensor
    #: B, oriented for ``--layout-b``. fp4 always comes back as (N, ceil(K/2))
    #: because the nibble pairing runs along K, which is B's column-major form.
    b: torch.Tensor
    #: e8m0 block scale for A, (M, groups), paired for a non-transposing route.
    scale_a: torch.Tensor
    #: e8m0 block scale for B, (groups, N), or (N, groups) when B is row-major.
    scale_b: torch.Tensor
    #: Padded group count both scale blocks carry, for the descriptor's extent.
    scale_groups: int
    #: ``dequant(A) @ dequant(B)`` in fp32 -- what the device should produce.
    ref: torch.Tensor


def random_operands(m: int, n: int, k: int, spread: float):
    return (
        torch.randn(m, k, device="cpu") * spread,
        torch.randn(k, n, device="cpu") * spread,
    )


def gen_mx_matmul_inputs(
    m: int,
    n: int,
    k: int,
    *,
    fp4_format: str | None = None,
    dtype_a: str | None = None,
    dtype_b: str | None = None,
    layout_a: str = "row",
    layout_b: str = "col",
    seed: int = 0,
) -> MxInputs:
    """Quantize one mx matmul's operands and build its fp32 golden.

    ``fp4_format`` selects the packed fp4 path (and then ``dtype_a`` /
    ``dtype_b`` must be omitted); otherwise both are fp8 case names from
    ``FP8_DTYPES``. The layouts decide the physical orientation the operands and
    their scale blocks are handed over in.
    """
    is_fp4 = fp4_format is not None
    if is_fp4:
        if fp4_format not in FP4_FORMATS:
            raise ValueError(f"unknown fp4 encoding: {fp4_format!r}")
        if dtype_a is not None or dtype_b is not None:
            raise ValueError(
                "an fp4 case takes no fp8 dtypes: "
                f"fp4_format={fp4_format!r} dtype_a={dtype_a!r} dtype_b={dtype_b!r}"
            )
    else:
        for side, name in (("a", dtype_a), ("b", dtype_b)):
            if name not in FP8_DTYPES:
                raise ValueError(
                    f"dtype_{side} must be one of {', '.join(sorted(FP8_DTYPES))}, "
                    f"got {name!r}"
                )

    torch.manual_seed(seed)

    # fp4's tighter dynamic range wants a smaller spread than fp8's.
    a_f, b_f = random_operands(m, n, k, 4.0 if is_fp4 else 10.0)
    if not is_fp4:
        a_f, b_f = a_f - 5.0, b_f - 5.0

    # Quantise the whole operand at once. An MX scale group is 32 elements along
    # K and never straddles a K tile, so this is identical to quantising per tile
    # and concatenating -- which is what this used to do, back when the host also
    # pre-swizzled each tile's scale block into zZMxScale / nNMxScale. The DMA does
    # that reorder now, so the blocks go over as the plain (M, K/32) and (K/32, N)
    # matrices and there is nothing left for the host to chunk.
    if is_fp4:
        qa, sa_packed, adq = quantize_mx_fp4(a_f, 1, fp4_format)
        qb, sb_packed, bdq = quantize_mx_fp4(b_f, 0, fp4_format)
    else:
        qa, sa_packed, adq = quantize_mx_fp8(a_f, 1, FP8_DTYPES[dtype_a])
        qb, sb_packed, bdq = quantize_mx_fp8(b_f, 0, FP8_DTYPES[dtype_b])

    ref = adq @ bdq

    # Each operand is physically transposed to match its --layout-*, the same
    # convention create_tla_tensor expects and basic_matmul uses: a column-major
    # operand is handed over as the transposed contiguous buffer.
    if layout_a == "col":
        qa = qa.permute(1, 0).contiguous()
    # fp4's B side comes back already packed as (N, ceil(K/2)) -- the
    # column-major form, because the nibble pairing runs along K. fp8's is
    # (K, N).
    if (layout_b == "row") if is_fp4 else (layout_b == "col"):
        qb = qb.permute(1, 0).contiguous()

    # Each GM tag says how its logical scale matrix is laid out. ColMajorMxScaleA
    # and RowMajorMxScaleB reach L1 through ND2NZ and need the groups interleaved
    # in C0-sized pairs; their DN2NZ counterparts take the block as it comes.
    sa = sa_packed.view(torch.int8).contiguous()
    sb = sb_packed.view(torch.int8).contiguous()
    scale_groups = sa.shape[1]
    if layout_a == "col":
        sa = pair_interleave_a(sa)
    if layout_b == "row":
        sb = pair_interleave_b(sb)
    else:
        sb = sb.permute(1, 0).contiguous()

    return MxInputs(
        a=qa.contiguous().view(torch.int8),
        b=qb.contiguous().view(torch.int8),
        scale_a=sa,
        scale_b=sb,
        scale_groups=int(scale_groups),
        ref=ref,
    )
