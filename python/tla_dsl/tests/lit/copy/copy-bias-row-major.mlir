// The bias staging pipeline as it is spelled today: a one-dimensional,
// contiguous run of elements tagged VectorLayout. That is the tag that matches
// `tla::detail::isVector` on the device side, so the VectorLayout TileCopyTla
// specializations do the work: copy_gm_to_l1.hpp for the GM->L1 stage and
// copy_l1_to_bt.hpp for the bias table.
//
// Neither copy may turn its tile coordinate into a memref.subview: the copy
// routes ship the coordinate in the descriptor payload and the bc layer adds
// it to the base address, so folding it into the memref as well would offset
// the tile twice.

// RUN: %tla_compile %s -o - | %filecheck %s

module {
  "tla.func"() ({
  ^bb0(%arg0: !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>):
    %0 = "tla.alloc_ptr"() {size_bytes = 1024 : i64} : () -> !tla.ptr<f32, l1, 512>
    %1 = "tla.alloc_ptr"() {size_bytes = 1024 : i64} : () -> !tla.ptr<f32, biasbuf, 512>
    %2 = "tla.make_tensor_like"(%0, %arg0) {layoutTag = #tla.layout_tag<VectorLayout>} : (!tla.ptr<f32, l1, 512>, !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>) -> !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, l1, 512>>
    %3 = "tla.make_tensor_like"(%1, %2) {layoutTag = #tla.layout_tag<VectorLayout>} : (!tla.ptr<f32, biasbuf, 512>, !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, l1, 512>>) -> !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, biasbuf, 512>>
    "tla.cube"() ({
      "tla.copy"(%2, %arg0) : (!tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, l1, 512>>, !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>) -> ()
      "tla.copy"(%3, %2) : (!tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, biasbuf, 512>>, !tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, l1, 512>>) -> ()
    }) : () -> ()
    "tla.return"() : () -> ()
  }) {function_type = (!tla.tensor<!tla.layout<!tla.shape<256>, !tla.stride<1>, !tla.shape<256>, VectorLayout>, !tla.coord<0>, !tla.ptr<f32, gm, 4>>) -> (), sym_name = "copy_bias_vector"} : () -> ()
}

// Both runtime symbols the VectorLayout tag protects must resolve, and the retired
// RowMajor spelling must not appear.
// CHECK-DAG: func.func private @copy_gm_VectorLayout_to_l1_VectorLayout_float
// CHECK-DAG: func.func private @copy_l1_VectorLayout_to_biasbuf_VectorLayout_float
// CHECK: func.func @copy_bias_vector
// CHECK-DAG: memref<256xf32, #hivm.address_space<cbuf>>
// CHECK-DAG: memref<256xf32, #hivm.address_space<biasbuf>>
// CHECK-COUNT-1: call @copy_gm_VectorLayout_to_l1_VectorLayout_float
// CHECK-COUNT-1: call @copy_l1_VectorLayout_to_biasbuf_VectorLayout_float
// CHECK-NOT: memref.subview
// CHECK-NOT: "tla.copy"
// CHECK-NOT: copy_gm_RowMajor_to_l1_RowMajor
// CHECK-NOT: copy_l1_RowMajor_to_biasbuf_RowMajor
