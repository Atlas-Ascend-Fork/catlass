// tla.img2col lowers to the img2col runtime call on the cube core: an L1 NC1HWC0
// feature map in, an expanded (M, K) L0A operand out. The convolution geometry
// arrives as i32 operands rather than attributes, which is what lets one
// compiled kernel serve any pad/stride/dilation.
//
// The IR below is the frontend's own output for a 3x3 pad-1 convolution over a
// 8x8 feature map with 32 input channels (2 C1 slices), captured from
// dump_mlir, so this doubles as a golden for the op's assembly format.

// RUN: %tla_compile %s -o - | %filecheck %s

// The feature-map copy and the img2col load are separate runtime symbols: the
// GM->L1 move is an ordinary contiguous copy and knows nothing about convolution.
// CHECK: func.func private @img2col_l1_zNUnAlign_to_l0a_zN_half
// CHECK-SAME: #hivm.address_space<cbuf>
// CHECK-SAME: #hivm.address_space<ca>
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK: func.func private @copy_gm_zNUnAlign_to_l1_zNUnAlign_half
// CHECK: func.func @img2col_min(%arg0: memref<64x288xf16, #hivm.address_space<gm>>)
// CHECK: call @copy_gm_zNUnAlign_to_l1_zNUnAlign_half
// CHECK: call @img2col_l1_zNUnAlign_to_l0a_zN_half
// The op must be fully lowered -- nothing from the tla dialect may survive.
// CHECK-NOT: tla.img2col

module {
  tla.func @img2col_min(%arg0: !tla.tensor<!tla.layout<!tla.shape<64,288>, !tla.stride<288,1>, !tla.shape<64,288>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>>) {
    %0 = tla.alloc_ptr{size_bytes = 4096} -> !tla.ptr<f16, l1, 512>
    %1 = tla.alloc_ptr{size_bytes = 36864} -> !tla.ptr<f16, l0a, 512>
    tla.cube {
      %2 = tla.tensor_ptr %arg0 : !tla.tensor<!tla.layout<!tla.shape<64,288>, !tla.stride<288,1>, !tla.shape<64,288>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>> -> !tla.ptr<f16, gm, 2>
      %3 = tla.make_shape -> !tla.shape<(64,1),(16,2)>
      %4 = tla.make_stride -> !tla.stride<(16,1024),(1,1024)>
      %5 = tla.make_shape -> !tla.shape<64,32>
      %6 = tla.make_layout %3, %4 origin %5 {layoutTag = #tla.layout_tag<zNUnAlign>} : !tla.shape<(64,1),(16,2)>, !tla.stride<(16,1024),(1,1024)> origin !tla.shape<64,32> -> !tla.layout<!tla.shape<(64,1),(16,2)>, !tla.stride<(16,1024),(1,1024)>, !tla.shape<64,32>, zNUnAlign>
      %7 = tla.make_coord -> !tla.coord<0,0>
      %8 = tla.make_tensor %2, %6, %7 : !tla.ptr<f16, gm, 2>, !tla.layout<!tla.shape<(64,1),(16,2)>, !tla.stride<(16,1024),(1,1024)>, !tla.shape<64,32>, zNUnAlign>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<(64,1),(16,2)>, !tla.stride<(16,1024),(1,1024)>, !tla.shape<64,32>, zNUnAlign>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>>
      %9 = tla.make_tensor_like %0 like %8 layoutTag(<zNUnAlign>) : !tla.ptr<f16, l1, 512>, !tla.tensor<!tla.layout<!tla.shape<(64,1),(16,2)>, !tla.stride<(16,1024),(1,1024)>, !tla.shape<64,32>, zNUnAlign>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>> -> !tla.tensor<!tla.layout<!tla.shape<(64,1),(16,2)>, !tla.stride<(16,1024),(1,1024)>, !tla.shape<64,32>, zNUnAlign>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>
      tla.copy %9, %8 : !tla.tensor<!tla.layout<!tla.shape<(64,1),(16,2)>, !tla.stride<(16,1024),(1,1024)>, !tla.shape<64,32>, zNUnAlign>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>, !tla.tensor<!tla.layout<!tla.shape<(64,1),(16,2)>, !tla.stride<(16,1024),(1,1024)>, !tla.shape<64,32>, zNUnAlign>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>>
      %10 = tla.tensor_ptr %arg0 : !tla.tensor<!tla.layout<!tla.shape<64,288>, !tla.stride<288,1>, !tla.shape<64,288>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>> -> !tla.ptr<f16, gm, 2>
      %11 = tla.make_shape -> !tla.shape<64,288>
      %12 = tla.make_stride -> !tla.stride<288,1>
      %13 = tla.make_layout %11, %12 : !tla.shape<64,288>, !tla.stride<288,1> -> !tla.layout<!tla.shape<64,288>, !tla.stride<288,1>, !tla.shape<64,288>, RowMajor>
      %14 = tla.make_coord -> !tla.coord<0,0>
      %15 = tla.make_tensor %10, %13, %14 : !tla.ptr<f16, gm, 2>, !tla.layout<!tla.shape<64,288>, !tla.stride<288,1>, !tla.shape<64,288>, RowMajor>, !tla.coord<0,0> -> !tla.tensor<!tla.layout<!tla.shape<64,288>, !tla.stride<288,1>, !tla.shape<64,288>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>>
      %16 = tla.make_tensor_like %1 like %15 layoutTag(<zN>) : !tla.ptr<f16, l0a, 512>, !tla.tensor<!tla.layout<!tla.shape<64,288>, !tla.stride<288,1>, !tla.shape<64,288>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f16, gm, 2>> -> !tla.tensor<!tla.layout<!tla.shape<(16,4),(16,18)>, !tla.stride<(16,256),(1,1024)>, !tla.shape<64,288>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l0a, 512>>
      %c8 = arith.constant 8 : index
      %c8_0 = arith.constant 8 : index
      %c3 = arith.constant 3 : index
      %c3_1 = arith.constant 3 : index
      %c1 = arith.constant 1 : index
      %c1_2 = arith.constant 1 : index
      %c1_3 = arith.constant 1 : index
      %c1_4 = arith.constant 1 : index
      %c1_5 = arith.constant 1 : index
      %c1_6 = arith.constant 1 : index
      %c1_7 = arith.constant 1 : index
      %c1_8 = arith.constant 1 : index
      tla.img2col %16, %9, fmap(%c8, %c8_0) filter(%c3, %c3_1) stride(%c1, %c1_2) dilation(%c1_3, %c1_4) pad(%c1_5, %c1_6, %c1_7, %c1_8) {pad_value = 0 : i32} : !tla.tensor<!tla.layout<!tla.shape<(16,4),(16,18)>, !tla.stride<(16,256),(1,1024)>, !tla.shape<64,288>, zN>, !tla.coord<0,0>, !tla.ptr<f16, l0a, 512>>, !tla.tensor<!tla.layout<!tla.shape<(64,1),(16,2)>, !tla.stride<(16,1024),(1,1024)>, !tla.shape<64,32>, zNUnAlign>, !tla.coord<0,0>, !tla.ptr<f16, l1, 512>>
    }
    tla.return
  }
}
