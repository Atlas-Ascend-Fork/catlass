#include "catlass/catlass.hpp"
#include "../common.h"

namespace Catlass::Gemm {

template <class ElementA, class ElementB, class ElementAccumulator>
__aicore__ __attribute__((always_inline)) void Mmad(
    __ca__ ElementA* a, __cb__ ElementB* b, __cc__ ElementAccumulator* c, uint32_t m, uint32_t n, uint32_t k,
    bool initC = true, uint8_t unitFlag = 0)
{
    AscendC::LocalTensor<ElementA> l0a{
        AscendC::TPosition::A2, (uint32_t) reinterpret_cast<int64_t>(a), (uint32_t)(m * k)};
    AscendC::LocalTensor<ElementB> l0b{
        AscendC::TPosition::B2, (uint32_t) reinterpret_cast<int64_t>(b), (uint32_t)(n * k)};
    AscendC::LocalTensor<ElementAccumulator> l0c{
        AscendC::TPosition::CO1, (uint32_t) reinterpret_cast<int64_t>(c), (uint32_t)(m * n)};

    AscendC::MmadParams mmadParams;
    mmadParams.m = m;
    mmadParams.n = n;
    mmadParams.k = k;
    mmadParams.unitFlag = unitFlag;
    mmadParams.cmatrixInitVal = initC;
    mmadParams.disableGemv = true;

    AscendC::Mmad(l0c, l0a, l0b, mmadParams);

    const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
    if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {
        AscendC::PipeBarrier<PIPE_M>();
    }
}

template <class ElementA, class ElementB, class ElementAccumulator>
__aicore__ __attribute__((always_inline)) void MmadBias(
    __ca__ ElementA* a, __cb__ ElementB* b, __cc__ ElementAccumulator* c, __biasbuf__ ElementAccumulator* bias,
    uint32_t m, uint32_t n, uint32_t k, bool initC = true, uint8_t unitFlag = 0)
{
    AscendC::LocalTensor<ElementA> l0a{
        AscendC::TPosition::A2, (uint32_t) reinterpret_cast<int64_t>(a), (uint32_t)(m * k)};
    AscendC::LocalTensor<ElementB> l0b{
        AscendC::TPosition::B2, (uint32_t) reinterpret_cast<int64_t>(b), (uint32_t)(n * k)};
    AscendC::LocalTensor<ElementAccumulator> l0c{
        AscendC::TPosition::CO1, (uint32_t) reinterpret_cast<int64_t>(c), (uint32_t)(m * n)};
    AscendC::LocalTensor<ElementAccumulator> l0bias{
        AscendC::TPosition::C2, (uint32_t) reinterpret_cast<int64_t>(bias), (uint32_t)n};

    AscendC::MmadParams mmadParams;
    mmadParams.m = m;
    mmadParams.n = n;
    mmadParams.k = k;
    mmadParams.unitFlag = unitFlag;
    mmadParams.cmatrixInitVal = false;
    mmadParams.disableGemv = true;

    AscendC::Mmad(l0c, l0a, l0b, l0bias, mmadParams);

    const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
    if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {
        AscendC::PipeBarrier<PIPE_M>();
    }
}

} // namespace Catlass::Gemm

extern "C" {
// MLIR lowering passes memref<?xT, strided<[?], offset:?>, #ca/cb/cc> (rank 1); only GM uses rank 2.
// Cube MMAD on dav-c310 uses fp32 L0C (__cc__); f16/bf16 outputs use copy_l0c_to_gm_RowMajor_* stubs.
//
// Every route below comes from one of the three registries, and each registry
// emits both wrappers together: the plain mmad and the mmad_bias that reads a
// 1 x N tile from the bias table buffer (C2). Registering them as a pair is
// what keeps a route from gaining one without the other.
//
// A registry pastes the type token into the symbol name as well as the
// signature, so every element type has to spell the symbol the lowering asks
// for.

// bf16 and the integer route are registered under their short names, so they
// need an alias that is both a valid C++ type and the symbol spelling
// (mmad_bf16_bf16_float rather than mmad_bfloat16_t_bfloat16_t_float). float,
// half and the fp8 types already spell themselves.
using bf16 = bfloat16_t;
using int8 = int8_t;
using int32 = int32_t;

// Plain routes: f32, f16, bf16, i8 and the two fp8 formats. The bias element
// type follows the accumulator, so TypeAcc fills both roles -- f32 on the
// fp32-L0C routes, int32 on i8 x i8 -> i32.
#define REGISTER_MMAD(TypeA, TypeB, TypeAcc)                                                                         \
    __aicore__ __attribute__((always_inline)) void _mlir_ciface_mmad_##TypeA##_##TypeB##_##TypeAcc(                  \
        memref_t<__ca__ TypeA, 1>* a, memref_t<__cb__ TypeB, 1>* b, memref_t<__cc__ TypeAcc, 1>* c, int64_t m,       \
        int64_t n, int64_t k, bool initC = true, uint8_t unitFlag = 0)                                               \
    {                                                                                                                \
        Catlass::Gemm::Mmad<TypeA, TypeB, TypeAcc>(                                                                  \
            a->aligned + a->offset, b->aligned + b->offset, c->aligned + c->offset, m, n, k, initC, unitFlag);       \
    }                                                                                                                \
    __aicore__ __attribute__((always_inline)) void _mlir_ciface_mmad_bias_##TypeA##_##TypeB##_##TypeAcc(             \
        memref_t<__ca__ TypeA, 1>* a, memref_t<__cb__ TypeB, 1>* b, memref_t<__cc__ TypeAcc, 1>* c, int64_t m,       \
        int64_t n, int64_t k, bool initC, uint8_t unitFlag, memref_t<__biasbuf__ TypeAcc, 1>* bias)                  \
    {                                                                                                                \
        Catlass::Gemm::MmadBias<TypeA, TypeB, TypeAcc>(                                                              \
            a->aligned + a->offset, b->aligned + b->offset, c->aligned + c->offset, bias->aligned + bias->offset, m, \
            n, k, initC, unitFlag);                                                                                  \
    }

REGISTER_MMAD(float, float, float)
REGISTER_MMAD(half, half, float)
REGISTER_MMAD(bf16, bf16, float)

// Integer route: int8 operands accumulate into an int32 L0C (not fp32). This is
// the non-MX `mad` path, same intrinsic as the float routes -- only the L0C
// element type, and with it the bias element type, differs.
REGISTER_MMAD(int8, int8, int32)

#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))
// The PLAIN fp8 routes, not the mx_fp8_* ones -- MmadCal only takes the mad_mx
// path for mx_fp8_*/fp4 operands, so these land on the same `mad` intrinsic as
// the f16/bf16/f32 routes and need no MX scale setup. A and B formats are
// independent, hence four symbols.
REGISTER_MMAD(fp8_e4m3fn_t, fp8_e4m3fn_t, float)
REGISTER_MMAD(fp8_e5m2_t, fp8_e5m2_t, float)
REGISTER_MMAD(fp8_e4m3fn_t, fp8_e5m2_t, float)
REGISTER_MMAD(fp8_e5m2_t, fp8_e4m3fn_t, float)

// MX fp8. The L0 tiles were loaded as mx_fp8_*, which is what makes MmadCal
// pick mad_mx; the scale was consumed by that load, so neither wrapper takes a
// scale operand. The bias operand does not affect the choice either: it is made
// from the L0 element types alone, so a biased mx_fp8_* pair still reaches
// mad_mx.
//
// The MX operand types need re-declaring at global scope under the same
// spelling: AscendC::mx_fp8_e4m3_t contains a `::`, which cannot be pasted into
// a symbol name.
using mx_fp8_e4m3_t = AscendC::mx_fp8_e4m3_t;
using mx_fp8_e5m2_t = AscendC::mx_fp8_e5m2_t;

#define REGISTER_MMADMX_FP8(TypeA, TypeB)                                                                            \
    __aicore__ __attribute__((always_inline)) void _mlir_ciface_mmadmx_##TypeA##_##TypeB##_float(                    \
        memref_t<__ca__ TypeA, 1>* a, memref_t<__cb__ TypeB, 1>* b, memref_t<__cc__ float, 1>* c, int64_t m,         \
        int64_t n, int64_t k, bool initC = true, uint8_t unitFlag = 0)                                               \
    {                                                                                                                \
        Catlass::Gemm::Mmad<TypeA, TypeB, float>(                                                                    \
            a->aligned + a->offset, b->aligned + b->offset, c->aligned + c->offset, m, n, k, initC, unitFlag);       \
    }                                                                                                                \
    __aicore__ __attribute__((always_inline)) void _mlir_ciface_mmadmx_bias_##TypeA##_##TypeB##_float(               \
        memref_t<__ca__ TypeA, 1>* a, memref_t<__cb__ TypeB, 1>* b, memref_t<__cc__ float, 1>* c, int64_t m,         \
        int64_t n, int64_t k, bool initC, uint8_t unitFlag, memref_t<__biasbuf__ float, 1>* bias)                    \
    {                                                                                                                \
        Catlass::Gemm::MmadBias<TypeA, TypeB, float>(                                                                \
            a->aligned + a->offset, b->aligned + b->offset, c->aligned + c->offset, bias->aligned + bias->offset, m, \
            n, k, initC, unitFlag);                                                                                  \
    }

REGISTER_MMADMX_FP8(mx_fp8_e4m3_t, mx_fp8_e4m3_t)
REGISTER_MMADMX_FP8(mx_fp8_e5m2_t, mx_fp8_e5m2_t)
REGISTER_MMADMX_FP8(mx_fp8_e4m3_t, mx_fp8_e5m2_t)
REGISTER_MMADMX_FP8(mx_fp8_e5m2_t, mx_fp8_e4m3_t)

// MX fp4. Unlike fp8, the L0 element type is the fp4 type itself -- MmadCal's
// isMx list matches fp4x2_e2m1_t/fp4x2_e1m2_t directly, with no mx_* variant --
// so the same pair of wrappers works, on int8_t storage that both reinterpret
// (two fp4 per byte). The encoding is part of the registered type, so a mixed
// pair names both encodings in its symbol.
#define REGISTER_MMADMX_FP4(TypeA, TypeB)                                                                      \
    __aicore__ __attribute__((always_inline)) void _mlir_ciface_mmadmx_##TypeA##_##TypeB##_float(              \
        memref_t<__ca__ int8_t, 1>* a, memref_t<__cb__ int8_t, 1>* b, memref_t<__cc__ float, 1>* c, int64_t m, \
        int64_t n, int64_t k, bool initC = true, uint8_t unitFlag = 0)                                         \
    {                                                                                                          \
        Catlass::Gemm::Mmad<TypeA, TypeB, float>(                                                              \
            reinterpret_cast<__ca__ TypeA*>(a->aligned + a->offset),                                           \
            reinterpret_cast<__cb__ TypeB*>(b->aligned + b->offset), c->aligned + c->offset, m, n, k, initC,   \
            unitFlag);                                                                                         \
    }                                                                                                          \
    __aicore__ __attribute__((always_inline)) void _mlir_ciface_mmadmx_bias_##TypeA##_##TypeB##_float(         \
        memref_t<__ca__ int8_t, 1>* a, memref_t<__cb__ int8_t, 1>* b, memref_t<__cc__ float, 1>* c, int64_t m, \
        int64_t n, int64_t k, bool initC, uint8_t unitFlag, memref_t<__biasbuf__ float, 1>* bias)              \
    {                                                                                                          \
        Catlass::Gemm::MmadBias<TypeA, TypeB, float>(                                                          \
            reinterpret_cast<__ca__ TypeA*>(a->aligned + a->offset),                                           \
            reinterpret_cast<__cb__ TypeB*>(b->aligned + b->offset), c->aligned + c->offset,                   \
            bias->aligned + bias->offset, m, n, k, initC, unitFlag);                                           \
    }

REGISTER_MMADMX_FP4(float4_e2m1x2_t, float4_e2m1x2_t)
REGISTER_MMADMX_FP4(float4_e1m2x2_t, float4_e1m2x2_t)
REGISTER_MMADMX_FP4(float4_e2m1x2_t, float4_e1m2x2_t)
REGISTER_MMADMX_FP4(float4_e1m2x2_t, float4_e2m1x2_t)
#endif
}
