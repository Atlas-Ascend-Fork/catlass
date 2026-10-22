// RUN: %tla_compile %s --emit=tla-mlir -o - | %filecheck %s --implicit-check-not='"dead"' --implicit-check-not='"local_only"'

// Unknown dialect consumers keep their flag handles and receive resolved ids.
// A flag shared by set/wait and an external consumer keeps the same id in both
// representations. Flags without remaining consumers must disappear.
module {
  tla.func @flag_external_consumers() {
    %explicit = tla.flag "explicit" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte3>, id = 3 : i64} -> !tla.flag
    %auto = tla.flag "external_auto" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte3>} -> !tla.flag
    %shared = tla.flag "shared" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte3>} -> !tla.flag
    %local = tla.flag "local_only" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte3>} -> !tla.flag
    %dead = tla.flag "dead" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte3>} -> !tla.flag
    "tla.vector"() ({
      tla.set_flag %shared : !tla.flag
      tla.wait_flag %shared : !tla.flag
      tla.set_flag %local : !tla.flag
      tla.wait_flag %local : !tla.flag
    }) : () -> ()
    "test.consume_flag"(%explicit) : (!tla.flag) -> ()
    "test.consume_flag"(%auto) : (!tla.flag) -> ()
    "test.consume_flag"(%shared) : (!tla.flag) -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func @flag_external_consumers
// CHECK-DAG: %[[EXPLICIT:.*]] = tla.flag "explicit" {{.*}}id = 3 : i64
// CHECK-DAG: %[[AUTO:.*]] = tla.flag "external_auto" {{.*}}id = 2 : i64
// CHECK-DAG: %[[SHARED:.*]] = tla.flag "shared" {{.*}}id = 0 : i64
// CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
// CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
// CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
// CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
// CHECK: "test.consume_flag"(%[[EXPLICIT]])
// CHECK: "test.consume_flag"(%[[AUTO]])
// CHECK: "test.consume_flag"(%[[SHARED]])
