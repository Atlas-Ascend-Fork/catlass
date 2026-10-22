// The VectorLayout tag is the explicit one-dimensional, unit-stride tile
// shape: it is what `tla::detail::isVector` matches on the device side, so the
// VectorLayout TileCopyTla specializations serve it directly.
//
// Two things are checked here:
//   1. a rank-1 VectorLayout layout parses and verifies (it stays
//      one-dimensional, it is not rewritten as a (1, N) RowMajor tile);
//   2. both flat copy routes resolve to their VectorLayout symbols.

// RUN: split-file %s %t
// RUN: %tla_compile %t/vector-copy.mlir -o - 2>&1 | %filecheck %s
// RUN: not %tla_compile %t/vector-rank2.mlir -o - 2>&1 | %filecheck %s --check-prefix=RANK2

//--- vector-copy.mlir

!gm = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>
!l1 = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, l1, 512>>
!bias = !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, biasbuf, 512>>

module {
  "tla.func"() ({
  ^bb0(%arg0: !gm):
    %0 = "tla.alloc_ptr"() {size_bytes = 1024 : i64} : () -> !tla.ptr<f32, l1, 512>
    %1 = "tla.alloc_ptr"() {size_bytes = 1024 : i64} : () -> !tla.ptr<f32, biasbuf, 512>
    %2 = "tla.make_tensor_like"(%0, %arg0) {layoutTag = #tla.layout_tag<VectorLayout>} : (!tla.ptr<f32, l1, 512>, !gm) -> !l1
    %3 = "tla.make_tensor_like"(%1, %2) {layoutTag = #tla.layout_tag<VectorLayout>} : (!tla.ptr<f32, biasbuf, 512>, !l1) -> !bias
    "tla.cube"() ({
      "tla.copy"(%2, %arg0) : (!l1, !gm) -> ()
      "tla.copy"(%3, %2) : (!bias, !l1) -> ()
    }) : () -> ()
    "tla.return"() : () -> ()
  }) {function_type = (!gm) -> (), sym_name = "vector_flat_copy"} : () -> ()
}

// CHECK-DAG: func.func private @copy_gm_VectorLayout_to_l1_VectorLayout_float
// CHECK-DAG: func.func private @copy_l1_VectorLayout_to_biasbuf_VectorLayout_float
// CHECK: func.func @vector_flat_copy
// CHECK-COUNT-1: call @copy_gm_VectorLayout_to_l1_VectorLayout_float
// CHECK-COUNT-1: call @copy_l1_VectorLayout_to_biasbuf_VectorLayout_float
// CHECK-NOT: "tla.copy"

//--- vector-rank2.mlir

!bad = !tla.tensor<!tla.layout<!tla.shape<1,256>, !tla.stride<256,1>, !tla.shape<1,256>, VectorLayout>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>

module {
  "tla.func"() ({
  ^bb0(%arg0: !bad):
    "tla.return"() : () -> ()
  }) {function_type = (!bad) -> (), sym_name = "vector_rank2"} : () -> ()
}

// Vector is one-dimensional by definition; anything else would silently address
// only the leading index.
// RANK2: tla.layout Vector shape must be one-dimensional
