// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --implicit-check-not=tla.pack --implicit-check-not=ave.hir.ppack

!t16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<i16, ub, 2>>
!t8 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 1>>

module {
  func.func @pack_mask64_low(%lhs_memref: memref<128xi16, #hivm.address_space<ub>>, %rhs_memref: memref<128xi16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %lhs = tla.tensor_desc %lhs_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !t16
    %rhs = tla.tensor_desc %rhs_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xi16, #hivm.address_space<ub>> -> !t16
    "tla.vec.func"() ({
      %lhs_value = tla.load %lhs : !t16 -> !tla.vector<128xi16>
      %rhs_value = tla.load %rhs : !t16 -> !tla.vector<128xi16>
      %source = "tla.create_mask"() {pattern = "H", dtype = i32} : () -> !tla.mask<64>
      %packed = tla.pack %source, <low> : !tla.mask<64> -> !tla.mask<128>
      %selected = "tla.where"(%packed, %lhs_value, %rhs_value) : (!tla.mask<128>, !tla.vector<128xi16>, !tla.vector<128xi16>) -> !tla.vector<128xi16>
      tla.store %lhs, %selected : !t16, !tla.vector<128xi16>
    }) {mode = "simd"} : () -> ()
    return
  }

  func.func @pack_mask128_high(%lhs_memref: memref<256xi8, #hivm.address_space<ub>>, %rhs_memref: memref<256xi8, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %lhs = tla.tensor_desc %lhs_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !t8
    %rhs = tla.tensor_desc %rhs_memref shape [%c1, %c256, %c1, %c1] stride [%c256, %c1, %c1, %c1] origin_shape [%c1, %c256] coord [%c0, %c0] : memref<256xi8, #hivm.address_space<ub>> -> !t8
    "tla.vec.func"() ({
      %lhs_value = tla.load %lhs : !t8 -> !tla.vector<256xi8>
      %rhs_value = tla.load %rhs : !t8 -> !tla.vector<256xi8>
      %source = "tla.create_mask"() {pattern = "Q", dtype = i16} : () -> !tla.mask<128>
      %packed = tla.pack %source, <high> : !tla.mask<128> -> !tla.mask<256>
      %selected = "tla.where"(%packed, %lhs_value, %rhs_value) : (!tla.mask<256>, !tla.vector<256xi8>, !tla.vector<256xi8>) -> !tla.vector<256xi8>
      tla.store %lhs, %selected : !t8, !tla.vector<256xi8>
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK-LABEL: func.func @pack_mask64_low
// CHECK-LABEL: func.func private @vector_region_
// CHECK: %[[LOW:.*]] = arith.constant 0 : i32
// CHECK: %[[PACKED:.*]] = "hivm_regbaseintrins.intr.hivm.ppack.z"({{.*}}, %[[LOW]]) : (vector<256xi1>, i32) -> vector<256xi1>
// CHECK: ave.hir.vsel %[[PACKED]]
// CHECK: return

// CHECK-LABEL: func.func @pack_mask128_high
// CHECK-LABEL: func.func private @vector_region_
// CHECK: %[[HIGH:.*]] = arith.constant 1 : i32
// CHECK: %[[PACKED:.*]] = "hivm_regbaseintrins.intr.hivm.ppack.z"({{.*}}, %[[HIGH]]) : (vector<256xi1>, i32) -> vector<256xi1>
// CHECK: ave.hir.vsel %[[PACKED]]
// CHECK: return
