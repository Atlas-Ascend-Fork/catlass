#include "catlass/catlass.hpp"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510

#include "vector_reg_utils.h"

extern "C" {

__aiv__ __attribute__((always_inline)) ave_preg tla_mask_select(ave_preg src0, ave_preg src1, ave_preg selectMask)
{
    vector_bool dst;
    psel(dst, convertAVEPregToVecBool(src0), convertAVEPregToVecBool(src1), convertAVEPregToVecBool(selectMask));

    ave_preg result;
    *reinterpret_cast<vector_bool*>(&result) = dst;
    return result;
}

} // extern "C"

#endif
