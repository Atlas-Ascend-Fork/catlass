// RUN: %tla_compile %s --mlir-print-ir-after=tla-lower-tensor-desc -o %t 2>&1 | %filecheck %s
//
// A one-row tile view whose row index is a loop counter running up to the
// parent's own row extent always fits, so its row crop is folded away.
// The functions below differ only in the upper bound or the step of the row loop.

// One row, one vector register wide: the view that a SIMD epilogue builds.
!strip = !tla.tensor<!tla.layout<!tla.shape<1,64>, !tla.stride<?,1>, !tla.shape<?,?>, RowMajor>, !tla.coord<?,?>, !tla.ptr<f32, ub, 256>>
!plane = !tla.tensor<!tla.layout<!tla.shape<?,?>, !tla.stride<?,1>, !tla.shape<?,?>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>
!layout = !tla.layout<!tla.shape<?,?>, !tla.stride<?,1>, !tla.shape<?,?>, RowMajor>

module {
  // The row loop counts up to the plane's own row extent, so the row crop folds.
  tla.func @row_iv_over_own_extent(%rows: i32, %cols: i32, %row_stride: i32) {
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

  // Same body, but the loop bound is unrelated to the row extent, so the row crop remains.
  tla.func @row_iv_over_foreign_bound(%rows: i32, %trip: i32, %cols: i32, %row_stride: i32) {
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
        scf.for %row = %c0_i32 to %trip step %c1_i32 : i32 {
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

  // The loop counts up to the row extent but its step is a run-time value, so nothing
  // proves it counts upwards and the row crop remains.
  tla.func @row_iv_over_dynamic_step(%rows: i32, %step: i32, %cols: i32, %row_stride: i32) {
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
        scf.for %row = %c0_i32 to %rows step %step : i32 {
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
}

// Exactly one minsi remains. It is the column crop, which the last strip of a
// row needs. The CHECK-NOT fails if the row crop comes back.
// CHECK-LABEL: func.func @row_iv_over_own_extent
// CHECK: scf.for
// CHECK: scf.for
// CHECK: %[[COLCROP:.*]] = arith.minsi
// CHECK-NOT: arith.minsi
// CHECK: tla.tensor_desc %{{.*}} origin_shape[%c1{{[^,]*}}, %[[COLCROP]]]

// Both crops stay here. Without this case, a fold that always fired would still
// pass.
// CHECK-LABEL: func.func @row_iv_over_foreign_bound
// CHECK: scf.for
// CHECK: scf.for
// CHECK: %[[ROWCROP:.*]] = arith.minsi
// CHECK: %[[COLCROP2:.*]] = arith.minsi
// CHECK: tla.tensor_desc %{{.*}} origin_shape[%[[ROWCROP]], %[[COLCROP2]]]

// A step that is not a positive constant keeps both crops.
// CHECK-LABEL: func.func @row_iv_over_dynamic_step
// CHECK: scf.for
// CHECK: scf.for
// CHECK: %[[ROWCROP3:.*]] = arith.minsi
// CHECK: %[[COLCROP3:.*]] = arith.minsi
// CHECK: tla.tensor_desc %{{.*}} origin_shape[%[[ROWCROP3]], %[[COLCROP3]]]
