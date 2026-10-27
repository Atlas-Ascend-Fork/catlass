// RUN: sed 's/index, 64 : i32/index, 32 : i32/' %s > %t.idx32.mlir
// RUN: %tla_compile %t.idx32.mlir --mlir-print-ir-after=tla-lower-tensor-desc -o %t 2>&1 | %filecheck %s --check-prefixes=CHECK,IDX32
// RUN: %tla_compile %s --mlir-print-ir-after=tla-lower-tensor-desc -o %t 2>&1 | %filecheck %s --check-prefixes=CHECK,IDX64
//
// The row crop folds only when the cast from the loop type to index cannot
// truncate. The two functions are identical except for the loop type: an i32
// bound folds with either index width, an i64 bound only with a 64-bit index.
// The first RUN line rewrites the data layout below to a 32-bit index.

!strip = !tla.tensor<!tla.layout<!tla.shape<1,64>, !tla.stride<?,1>, !tla.shape<?,?>, RowMajor>, !tla.coord<?,?>, !tla.ptr<f32, ub, 256>>
!plane = !tla.tensor<!tla.layout<!tla.shape<?,?>, !tla.stride<?,1>, !tla.shape<?,?>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>
!layout = !tla.layout<!tla.shape<?,?>, !tla.stride<?,1>, !tla.shape<?,?>, RowMajor>


module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64 : i32>>} {
  tla.func @row_iv_i32(%rows: i32, %cols: i32, %row_stride: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %rows_idx = arith.index_cast %rows : i32 to index
    %cols_idx = arith.index_cast %cols : i32 to index
    %stride_idx = arith.index_cast %row_stride : i32 to index
    %raw = "tla.alloc_ptr"() {size_bytes = 65536 : i64} : () -> !tla.ptr<i8, ub, 256>
    %ptr = "tla.recast_ptr"(%raw) : (!tla.ptr<i8, ub, 256>) -> !tla.ptr<f32, ub, 256>
    %shape = "tla.make_shape"(%rows_idx, %cols_idx) : (index, index) -> !tla.shape<?,?>
    %stride = "tla.make_stride"(%stride_idx) : (index) -> !tla.stride<?,1>
    %layout = "tla.make_layout"(%shape, %stride) : (!tla.shape<?,?>, !tla.stride<?,1>) -> !layout
    %origin = "tla.make_coord"() : () -> !tla.coord<0,0>
    %plane = "tla.make_tensor"(%ptr, %layout, %origin) : (!tla.ptr<f32, ub, 256>, !layout, !tla.coord<0,0>) -> !plane
    "tla.vector"() ({
      "tla.vec.func"() ({
        scf.for %row = %c0_i32 to %rows step %c1_i32 : i32 {
          // Two strips per row: a loop with a single iteration would be folded away.
          scf.for %col = %c0_i32 to %c2_i32 step %c1_i32 : i32 {
            %row_idx = arith.index_cast %row : i32 to index
            %col_idx = arith.index_cast %col : i32 to index
            %strip_shape = "tla.make_shape"() : () -> !tla.shape<1,64>
            %coord = "tla.make_coord"(%row_idx, %col_idx) : (index, index) -> !tla.coord<?,?>
            %tile = "tla.tile_view"(%plane, %strip_shape, %coord) : (!plane, !tla.shape<1,64>, !tla.coord<?,?>) -> !strip
            %v = tla.load %tile : !strip -> !tla.vector<?xf32>
            tla.store %tile, %v : !strip, !tla.vector<?xf32>
          }
        }
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    "tla.return"() : () -> ()
  }


  tla.func @row_iv_i64(%rows: i64, %cols: i64, %row_stride: i64) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %rows_idx = arith.index_cast %rows : i64 to index
    %cols_idx = arith.index_cast %cols : i64 to index
    %stride_idx = arith.index_cast %row_stride : i64 to index
    %raw = "tla.alloc_ptr"() {size_bytes = 65536 : i64} : () -> !tla.ptr<i8, ub, 256>
    %ptr = "tla.recast_ptr"(%raw) : (!tla.ptr<i8, ub, 256>) -> !tla.ptr<f32, ub, 256>
    %shape = "tla.make_shape"(%rows_idx, %cols_idx) : (index, index) -> !tla.shape<?,?>
    %stride = "tla.make_stride"(%stride_idx) : (index) -> !tla.stride<?,1>
    %layout = "tla.make_layout"(%shape, %stride) : (!tla.shape<?,?>, !tla.stride<?,1>) -> !layout
    %origin = "tla.make_coord"() : () -> !tla.coord<0,0>
    %plane = "tla.make_tensor"(%ptr, %layout, %origin) : (!tla.ptr<f32, ub, 256>, !layout, !tla.coord<0,0>) -> !plane
    "tla.vector"() ({
      "tla.vec.func"() ({
        scf.for %row = %c0_i64 to %rows step %c1_i64 : i64 {
          // Two strips per row: a loop with a single iteration would be folded away.
          scf.for %col = %c0_i64 to %c2_i64 step %c1_i64 : i64 {
            %row_idx = arith.index_cast %row : i64 to index
            %col_idx = arith.index_cast %col : i64 to index
            %strip_shape = "tla.make_shape"() : () -> !tla.shape<1,64>
            %coord = "tla.make_coord"(%row_idx, %col_idx) : (index, index) -> !tla.coord<?,?>
            %tile = "tla.tile_view"(%plane, %strip_shape, %coord) : (!plane, !tla.shape<1,64>, !tla.coord<?,?>) -> !strip
            %v = tla.load %tile : !strip -> !tla.vector<?xf32>
            tla.store %tile, %v : !strip, !tla.vector<?xf32>
          }
        }
      }) {mode = "simd"} : () -> ()
    }) : () -> ()
    "tla.return"() : () -> ()
  }

}

// CHECK-LABEL: func.func @row_iv_i32
// CHECK: scf.for
// CHECK: scf.for
// CHECK: %[[COL32:.*]] = arith.minsi
// CHECK-NOT: arith.minsi
// CHECK: tla.tensor_desc %{{.*}} origin_shape[%c1{{[^,]*}}, %[[COL32]]]

// CHECK-LABEL: func.func @row_iv_i64
// CHECK: scf.for
// CHECK: scf.for
// IDX64: %[[COL64:.*]] = arith.minsi
// IDX64-NOT: arith.minsi
// IDX64: tla.tensor_desc %{{.*}} origin_shape[%c1{{[^,]*}}, %[[COL64]]]
// IDX32: %[[ROW64:.*]] = arith.minsi
// IDX32: %[[COL64B:.*]] = arith.minsi
// IDX32: tla.tensor_desc %{{.*}} origin_shape[%[[ROW64]], %[[COL64B]]]
