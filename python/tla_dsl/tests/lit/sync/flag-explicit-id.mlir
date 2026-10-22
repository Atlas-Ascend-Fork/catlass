// RUN: %tla_compile %s -o - | %filecheck %s

// Reserve explicit ids before processing any uses. Automatic flags must skip
// them even if an automatic flag is declared/used first. Explicit reuse is
// permitted, and different pipe pairs have independent id spaces.
module {
  tla.func @explicit_and_automatic_ids() {
    "tla.vector"() ({
      %auto0 = tla.flag "auto0" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte1>} -> !tla.flag
      %zero = tla.flag "zero" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte1>, id = 0 : i64} -> !tla.flag
      %three = tla.flag "three" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte1>, id = 3 : i64} -> !tla.flag
      %seven = tla.flag "seven" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte1>, id = 7 : i64} -> !tla.flag
      %reuse = tla.flag "reuse" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte1>, id = 3 : i64} -> !tla.flag
      %auto1 = tla.flag "auto1" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte1>, id = -1 : i64} -> !tla.flag
      %auto2 = tla.flag "auto2" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte1>} -> !tla.flag
      %other = tla.flag "other" {src_pipe = #tla.pipe<mte1>, dst_pipe = #tla.pipe<mte2>, id = 3 : i64} -> !tla.flag
      %other_auto = tla.flag "other_auto" {src_pipe = #tla.pipe<mte1>, dst_pipe = #tla.pipe<mte2>} -> !tla.flag
      tla.set_flag %auto0 : !tla.flag
      tla.wait_flag %auto0 : !tla.flag
      tla.set_flag %zero : !tla.flag
      tla.wait_flag %zero : !tla.flag
      tla.set_flag %three : !tla.flag
      tla.wait_flag %three : !tla.flag
      tla.set_flag %seven : !tla.flag
      tla.wait_flag %seven : !tla.flag
      tla.set_flag %reuse : !tla.flag
      tla.wait_flag %reuse : !tla.flag
      tla.set_flag %auto1 : !tla.flag
      tla.wait_flag %auto1 : !tla.flag
      tla.set_flag %auto2 : !tla.flag
      tla.wait_flag %auto2 : !tla.flag
      tla.set_flag %other : !tla.flag
      tla.wait_flag %other : !tla.flag
      tla.set_flag %other_auto : !tla.flag
      tla.wait_flag %other_auto : !tla.flag
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func @explicit_and_automatic_ids
// CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
// CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
// CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
// CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
// CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID3>]
// CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID3>]
// CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID7>]
// CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID7>]
// CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID3>]
// CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID3>]
// CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID2>]
// CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID2>]
// CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID4>]
// CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID4>]
// CHECK: hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
// CHECK: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
// CHECK: hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
// CHECK: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
