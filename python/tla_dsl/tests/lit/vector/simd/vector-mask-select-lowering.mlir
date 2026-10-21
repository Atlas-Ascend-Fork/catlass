// RUN: %tla_compile %s --mlir-print-ir-after=tla-vector-region-simd -o %t 2>&1 | %filecheck %s --implicit-check-not=tla.where --implicit-check-not=ave.hir.vsel --implicit-check-not=hivm.vsel

!bits = !tla.tensor<!tla.layout<!tla.shape<4>, !tla.stride<1>, !tla.shape<4>, RowMajor>, !tla.coord<0>, !tla.ptr<i8, ub, 1>>
!mask = !tla.mask<32>

module {
  func.func @mask_select_carriers(
      %mask_memref: memref<4xi8, #hivm.address_space<ub>>,
      %count: index,
      %condition: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %bits = tla.tensor_desc %mask_memref shape [%c1, %c4, %c1, %c1] stride [%c4, %c1, %c1, %c1] origin_shape [%c1, %c4] coord [%c0, %c0] : memref<4xi8, #hivm.address_space<ub>> -> !bits
    "tla.vec.func"() ({
      %selector = "tla.create_mask"() {pattern = "H", dtype = i64} : () -> !mask
      %x = "tla.create_mask"() {pattern = "Q", dtype = i64} : () -> !mask
      %y = "tla.create_mask"() {pattern = "M4", dtype = i64} : () -> !mask
      %first = "tla.where"(%selector, %x, %y) : (!mask, !mask, !mask) -> !mask
      %chained = "tla.where"(%first, %y, %x) : (!mask, !mask, !mask) -> !mask
      %combined = "tla.bitwise_xor"(%chained, %selector) : (!mask, !mask) -> !mask
      %loop = scf.for %i = %c0 to %count step %c1 iter_args(%carried = %combined) -> (!mask) {
        %selected = scf.if %condition -> (!mask) {
          %nested = "tla.where"(%carried, %x, %y) : (!mask, !mask, !mask) -> !mask
          scf.yield %nested : !mask
        } else {
          scf.yield %carried : !mask
        }
        scf.yield %selected : !mask
      }
      %last = "tla.where"(%loop, %x, %y) : (!mask, !mask, !mask) -> !mask
      tla.store %bits, %last : !bits, !mask
    }) {mode = "simd"} : () -> ()
    return
  }
}

// CHECK-COUNT-1: func.func private @tla_mask_select(vector<256xi1>, vector<256xi1>, vector<256xi1>) -> vector<256xi1>
// CHECK-LABEL: func.func private @vector_region_
// CHECK: %[[FIRST:.*]] = call @tla_mask_select(%[[X:.*]], %[[Y:.*]], %[[SELECTOR:.*]]) : (vector<256xi1>, vector<256xi1>, vector<256xi1>) -> vector<256xi1>
// CHECK: %[[CHAINED:.*]] = call @tla_mask_select(%[[Y]], %[[X]], %[[FIRST]]) : (vector<256xi1>, vector<256xi1>, vector<256xi1>) -> vector<256xi1>
// CHECK: scf.for {{.*}} iter_args(%[[CARRIED:.*]] = %{{.*}}) -> (vector<256xi1>)
// CHECK: scf.if {{.*}} -> (vector<256xi1>)
// CHECK: call @tla_mask_select(%[[X]], %[[Y]], %[[CARRIED]]) : (vector<256xi1>, vector<256xi1>, vector<256xi1>) -> vector<256xi1>
// CHECK: call @tla_mask_select(%[[X]], %[[Y]], %{{.*}}) : (vector<256xi1>, vector<256xi1>, vector<256xi1>) -> vector<256xi1>
// CHECK: builtin.unrealized_conversion_cast {{.*}} : vector<256xi1> to vector<32xi1>
// CHECK: ave.hir.masked_store <NORM_B8>
