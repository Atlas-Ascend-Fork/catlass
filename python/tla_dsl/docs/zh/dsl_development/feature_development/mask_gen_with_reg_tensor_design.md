## mask_gen_with_reg_tensor

### Function

将RegTensor按offset取一段存入MaskReg

### Checking

- `template <typename T = DefaultType, int16_t offset, typename U> __simd_callee__ inline void MaskGenWithRegTensor(MaskReg& dst, U& srcReg)` in `asc-devkit`
- No similar interface in `CuTeDSL`

#### Definition

```py
@dsl_user_op
def mask_gen(src: VectorSSA, *, offset: tla.ConstExpr[int]) -> tla.MaskSSA:
    # ...

# 可选
class VectorSSA:
    @dsl_user_op
    def gen_mask(offset: tla.ConstExpr[int]) -> tla.MaskSSA:
        # ...
```

### Implementation

```cpp
template <typename T = DefaultType, int16_t offset, typename U>
__simd_callee__ inline void MaskGenWithRegTensorImpl(MaskReg& dst, U& srcReg)
{
    using ActualT = typename U::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportBytes<ActualT, 2, 4>(), "MaskGenWithRegTensor only support type b16/b32 on current device");
    if constexpr (sizeof(ActualT) == 2) {
        static_assert((offset >= 0) && (offset <= 15), "MaskGenWithRegTensor offset must be in 0~15 when T is b16");
        movvp(dst, (RegTensor<uint16_t>&)srcReg, offset);
    } else if constexpr (sizeof(ActualT) == 4) {
        static_assert((offset >= 0) && (offset <= 31), "MaskGenWithRegTensor offset must be in 0~31 when T is b32");
        movvp(dst, (RegTensor<uint32_t>&)srcReg, offset);
    }
}
```

使用`hivm_regbaseintrins::MovvpInstrOp`实现