// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --implicit-check-not=tla.mask_gen

// tla.mask_gen lowers straight to hivm_regbaseintrins.movvp. movvp is a
// polymorphic intrinsic but the backend only selects the integer overloads
// (the CCE front end likewise defines movvp only for vector_u16/vector_u32), so
// a float or signed source is first vector.bitcast to the same-width integer
// vector: b32 -> vector<Nxi32>, b16 -> vector<Nxi16>. The element width fixes
// the lane count (b16 -> mask<128>, b32 -> mask<64>) and the offset becomes an
// i32 immediate operand.

!t_f32 = !tla.tensor<!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<64>, RowMajor>, !tla.coord<0>, !tla.ptr<f32, ub, 4>>
!t_f16 = !tla.tensor<!tla.layout<!tla.shape<128>, !tla.stride<1>, !tla.shape<128>, RowMajor>, !tla.coord<0>, !tla.ptr<f16, ub, 2>>

module {
  // CHECK-LABEL: func.func @mask_gen_b32
  // CHECK: vector.bitcast %{{.*}} : vector<64xf32> to vector<64xi32>
  // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}) : (vector<64xi32>, i32) -> vector<256xi1>
  func.func @mask_gen_b32(%src_memref: memref<64xf32, #hivm.address_space<ub>>, %dst_memref: memref<64xf32, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !t_f32
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c64, %c1, %c1] stride [%c64, %c1, %c1, %c1] origin_shape [%c1, %c64] coord [%c0, %c0] : memref<64xf32, #hivm.address_space<ub>> -> !t_f32
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<64>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!t_f32, !tla.shape<64>, !tla.coord<0>) -> !t_f32
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!t_f32, !tla.shape<64>, !tla.coord<0>) -> !t_f32
      %v = "tla.load"(%src_tile) : (!t_f32) -> !tla.vector<64xf32>
      %m = "tla.mask_gen"(%v) {offset = 7 : i32} : (!tla.vector<64xf32>) -> !tla.mask<64>
      %s = "tla.where"(%m, %v, %v) : (!tla.mask<64>, !tla.vector<64xf32>, !tla.vector<64xf32>) -> !tla.vector<64xf32>
      "tla.store"(%dst_tile, %s) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (!t_f32, !tla.vector<64xf32>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }

  // CHECK-LABEL: func.func @mask_gen_b16
  // CHECK: vector.bitcast %{{.*}} : vector<128xf16> to vector<128xi16>
  // CHECK: "hivm_regbaseintrins.intr.hivm.movvp"({{.*}}) : (vector<128xi16>, i32) -> vector<256xi1>
  func.func @mask_gen_b16(%src_memref: memref<128xf16, #hivm.address_space<ub>>, %dst_memref: memref<128xf16, #hivm.address_space<ub>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %src = tla.tensor_desc %src_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !t_f16
    %dst = tla.tensor_desc %dst_memref shape [%c1, %c128, %c1, %c1] stride [%c128, %c1, %c1, %c1] origin_shape [%c1, %c128] coord [%c0, %c0] : memref<128xf16, #hivm.address_space<ub>> -> !t_f16
    "tla.vec.func"() ({
      %shape = "tla.make_shape"() : () -> !tla.shape<128>
      %coord = "tla.make_coord"() : () -> !tla.coord<0>
      %src_tile = "tla.tile_view"(%src, %shape, %coord) : (!t_f16, !tla.shape<128>, !tla.coord<0>) -> !t_f16
      %dst_tile = "tla.tile_view"(%dst, %shape, %coord) : (!t_f16, !tla.shape<128>, !tla.coord<0>) -> !t_f16
      %v = "tla.load"(%src_tile) : (!t_f16) -> !tla.vector<128xf16>
      %m = "tla.mask_gen"(%v) {offset = 15 : i32} : (!tla.vector<128xf16>) -> !tla.mask<128>
      %s = "tla.where"(%m, %v, %v) : (!tla.mask<128>, !tla.vector<128xf16>, !tla.vector<128xf16>) -> !tla.vector<128xf16>
      "tla.store"(%dst_tile, %s) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (!t_f16, !tla.vector<128xf16>) -> ()
    }) {mode = "simd"} : () -> ()
    return
  }
}
