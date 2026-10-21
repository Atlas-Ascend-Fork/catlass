// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --implicit-check-not=tla.where --implicit-check-not=tla_mask_select

!fvec = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>

module {
  func.func @vector_where(
      %src_memref: memref<64xf32, #hivm.address_space<ub>>,
      %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !fvec
    "tla.vec.func"() ({
      %x = tla.load %src : !fvec -> !tla.vector<64xf32>
      %y = tla.load %dst : !fvec -> !tla.vector<64xf32>
      %mask = "tla.create_mask"() {pattern = "H", dtype = f32} : () -> !tla.mask<64>
      %selected = "tla.where"(%mask, %x, %y) : (!tla.mask<64>, !tla.vector<64xf32>, !tla.vector<64xf32>) -> !tla.vector<64xf32>
      tla.store %dst, %selected : !fvec, !tla.vector<64xf32>
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vector_region_
// CHECK: ave.hir.vsel
