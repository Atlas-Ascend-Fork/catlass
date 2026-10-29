// tla.copy_mx lowers to a bc runtime call that takes the descriptor's *base*
// memref plus the tile coordinate in its 12-field payload; the bc layer applies
// that coordinate itself (makeTlaTileCoord). Handing over a tile subview with
// the coordinate already folded in would apply the offset twice, so the
// lowering must pass the base memref unchanged -- exactly like tla.copy.
//
// The source tile below sits at coordinate (32,0), so a folding bug shows up as
// a memref.subview on the way into the call.
//
// RUN: %tla_compile %s -o - | %filecheck %s

module {
  tla.func @copy_mx_tile_coord() {
    %src_ptr = tla.alloc_ptr{size_bytes = 8192} -> !tla.ptr<f8E4M3FN, l1, 512>
    %shape = tla.make_shape -> !tla.shape<(16,8),(32,2)>
    %stride = tla.make_stride -> !tla.stride<(32,512),(1,4096)>
    %origin = tla.make_shape -> !tla.shape<128,64>
    %layout = tla.make_layout %shape, %stride origin %origin {layoutTag = #tla.layout_tag<zN>} :
      !tla.shape<(16,8),(32,2)>, !tla.stride<(32,512),(1,4096)> origin !tla.shape<128,64> ->
      !tla.layout<!tla.shape<(16,8),(32,2)>, !tla.stride<(32,512),(1,4096)>, !tla.shape<128,64>, zN>
    %coord = tla.make_coord -> !tla.coord<32,0>
    %src = tla.make_tensor %src_ptr, %layout, %coord :
      !tla.ptr<f8E4M3FN, l1, 512>,
      !tla.layout<!tla.shape<(16,8),(32,2)>, !tla.stride<(32,512),(1,4096)>, !tla.shape<128,64>, zN>,
      !tla.coord<32,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(16,8),(32,2)>, !tla.stride<(32,512),(1,4096)>, !tla.shape<128,64>, zN>, !tla.coord<32,0>, !tla.ptr<f8E4M3FN, l1, 512>>
    %dst_ptr = tla.alloc_ptr{size_bytes = 8192} -> !tla.ptr<f8E4M3FN, l0a, 512>
    %zero = tla.make_coord -> !tla.coord<0,0>
    %dst = tla.make_tensor %dst_ptr, %layout, %zero :
      !tla.ptr<f8E4M3FN, l0a, 512>,
      !tla.layout<!tla.shape<(16,8),(32,2)>, !tla.stride<(32,512),(1,4096)>, !tla.shape<128,64>, zN>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(16,8),(32,2)>, !tla.stride<(32,512),(1,4096)>, !tla.shape<128,64>, zN>, !tla.coord<0,0>, !tla.ptr<f8E4M3FN, l0a, 512>>
    // A-side e8m0 scale block, one exponent per 32 elements along K.
    %scale_ptr = tla.alloc_ptr{size_bytes = 256} -> !tla.ptr<!tla.f8e8m0, l1, 512>
    %scale_shape = tla.make_shape -> !tla.shape<(16,8),(2,1)>
    %scale_stride = tla.make_stride -> !tla.stride<(2,32),(1,32)>
    %scale_origin = tla.make_shape -> !tla.shape<128,2>
    %scale_layout = tla.make_layout %scale_shape, %scale_stride origin %scale_origin {layoutTag = #tla.layout_tag<zZMxScale>} :
      !tla.shape<(16,8),(2,1)>, !tla.stride<(2,32),(1,32)> origin !tla.shape<128,2> ->
      !tla.layout<!tla.shape<(16,8),(2,1)>, !tla.stride<(2,32),(1,32)>, !tla.shape<128,2>, zZMxScale>
    %scale = tla.make_tensor %scale_ptr, %scale_layout, %zero :
      !tla.ptr<!tla.f8e8m0, l1, 512>,
      !tla.layout<!tla.shape<(16,8),(2,1)>, !tla.stride<(2,32),(1,32)>, !tla.shape<128,2>, zZMxScale>,
      !tla.coord<0,0> ->
      !tla.tensor<!tla.layout<!tla.shape<(16,8),(2,1)>, !tla.stride<(2,32),(1,32)>, !tla.shape<128,2>, zZMxScale>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, l1, 512>>
    tla.cube {
      tla.copy_mx %dst, %src, %scale :
        !tla.tensor<!tla.layout<!tla.shape<(16,8),(32,2)>, !tla.stride<(32,512),(1,4096)>, !tla.shape<128,64>, zN>, !tla.coord<0,0>, !tla.ptr<f8E4M3FN, l0a, 512>>,
        !tla.tensor<!tla.layout<!tla.shape<(16,8),(32,2)>, !tla.stride<(32,512),(1,4096)>, !tla.shape<128,64>, zN>, !tla.coord<32,0>, !tla.ptr<f8E4M3FN, l1, 512>>,
        !tla.tensor<!tla.layout<!tla.shape<(16,8),(2,1)>, !tla.stride<(2,32),(1,32)>, !tla.shape<128,2>, zZMxScale>, !tla.coord<0,0>, !tla.ptr<!tla.f8e8m0, l1, 512>>
    }
    tla.return
  }
}

// CHECK: func.func private @copy_mx_l1_zN_to_l0a_zN_fp8_e4m3fn_t(
// CHECK-SAME: memref<?xf8E4M3FN, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
// CHECK-SAME: memref<?xf8E4M3FN, strided<[?], offset: ?>, #hivm.address_space<ca>>
// CHECK-SAME: memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
// CHECK: func.func @copy_mx_tile_coord
// CHECK-NOT: memref.subview
// CHECK: call @copy_mx_l1_zN_to_l0a_zN_fp8_e4m3fn_t(
// CHECK-NOT: memref.subview
// CHECK-NOT: "tla.copy_mx"
