// Learning aid: a cycle-by-cycle trace of a 2x2 bpu_sa_array. Not a pass/fail test
// (tb_bpu_sa_array.sv is the self-checking testbench); it prints what every part of
// the array holds in each clock cycle so you can follow one operation end to end.
//
// What it does (same values as the walkthrough in chat):
//   cycles 1-2   load tile A1 = [1 2; 3 4], one row per cycle, LAST row first
//   cycles 3-4   stream B columns [5 6] and [7 8]; the first one switches to A1
//   cycles 4-7   load tile A2 = [-1 0; 0 2] in the background (waits out the hold-off)
//   cycle  8     stream B column [5 6]; it switches to A2
//   outputs      cycle 8: C = [17 39]   cycle 9: C = [23 53]   cycle 13: C = [-5 12]
//                (each column comes out Lat = Rows + Cols + 1 = 5 cycles after it went in)
//
// Each block of output is the state DURING that cycle, just before the rising clock
// edge that ends it. Inputs are driven on the falling edge, so the values shown are
// exactly what the design samples at the next rising edge.
//
// Registers that reset does not clear (the A values, partial sums, output register)
// print as x in ModelSim/Questa until data reaches them. That is the "garbage after
// reset" the design relies on c_valid_o to hide. Verilator, which is 2-state, shows 0.
//
// ModelSim / Questa, text trace:
//   vlog -sv bpu_sa_array.sv bpu_sa_array_trace_tb.sv
//   vsim -c bpu_sa_array_trace_tb -do "run -all; quit -f"
// ModelSim / Questa, waveforms (+acc keeps internal signals visible):
//   vsim -voptargs=+acc bpu_sa_array_trace_tb
//   add wave -r /*
//   run -all

module bpu_sa_array_trace_tb;
  timeunit 1ns;
  timeprecision 1ps;

  localparam int R = 2, C = 2, W = 32;

  logic clk   = 1'b0;
  logic rst_n = 1'b0;
  always #5 clk = ~clk;

  logic           mode    = 1'b0;
  logic           a_valid = 1'b0;
  logic           b_valid = 1'b0;
  logic [C*8-1:0] a_row   = '0;
  logic [C*8-1:0] b_col   = '0;
  logic           err_mode, a_ready, a_full, b_ready, c_valid;
  logic [R*W-1:0] c_col;

  bpu_sa_array #(.Rows(R), .Cols(C), .AccW(W)) dut (
    .clk_i      (clk),
    .rst_ni     (rst_n),
    .mode_i     (mode),
    .err_mode_o (err_mode),
    .a_valid_i  (a_valid),
    .a_ready_o  (a_ready),
    .a_row_i    (a_row),
    .a_full_o   (a_full),
    .b_valid_i  (b_valid),
    .b_ready_o  (b_ready),
    .b_col_i    (b_col),
    .c_valid_o  (c_valid),
    .c_col_o    (c_col)
  );

  // Two int8 elements into one bus: element k goes in bits [8k+7:8k].
  function automatic logic [15:0] pack2(input int e0, input int e1);
    return {8'(e1), 8'(e0)};
  endfunction

`define PE(i, k)   dut.g_row[i].g_col[k].u_pe
`define PSUM(i, k) $signed(`PE(i, k).sum_o + `PE(i, k).carry_o)
`define SHOW_PE(i, k) \
    $write("  PE(%0d,%0d)  A active %3d  shadow %3d  |  b in %3d  switch %0d  |  sum out %4d\n", \
           i, k, $signed(`PE(i, k).a_active_q), $signed(`PE(i, k).a_shadow_q), \
           $signed(`PE(i, k).b_i), `PE(i, k).swap_i, `PSUM(i, k))

  int cyc = 0;

  // Print the state held during cycle `cyc`, 1 ns before the rising edge that ends it.
  always @(negedge clk) begin
    #4;
    $display("---------------------------------------------------------------- cycle %0d", cyc);
    $display("  reset %0s | A port: a_valid %0d  a_row [%0d %0d]  a_ready %0d%0s",
             rst_n ? "off" : "ON", a_valid, $signed(a_row[7:0]), $signed(a_row[15:8]),
             a_ready, (a_valid && a_ready) ? "  -> accepted at this edge" : "");
    $display("  control: rows loaded %0d  a_full %0d  hold-off %0d | B port: b_valid %0d  b_col [%0d %0d]  swap %0d",
             dut.a_cnt_q, a_full, dut.hold_q, b_valid, $signed(b_col[7:0]), $signed(b_col[15:8]),
             dut.swap);
    $display("  into the top row (after skew): B [%0d %0d]  switch [%0d %0d]",
             $signed(dut.top_b[0]), $signed(dut.top_b[1]), dut.top_swap[0], dut.top_swap[1]);
    `SHOW_PE(0, 0);
    `SHOW_PE(0, 1);
    `SHOW_PE(1, 0);
    `SHOW_PE(1, 1);
    $display("  right edge: row adders [%0d %0d]  after deskew [%0d %0d] | c_valid %0d  c_col [%0d %0d]",
             $signed(dut.row_c[0]), $signed(dut.row_c[1]), $signed(dut.row_al[0]),
             $signed(dut.row_al[1]), c_valid, $signed(c_col[31:0]), $signed(c_col[63:32]));
    cyc++;
  end

  // Inputs change on the falling edge; the design samples them on the next rising edge.
  task automatic step();
    @(negedge clk);
  endtask

  initial begin
    step();
    step();                                   // cycle 0: reset held
    rst_n   = 1'b1;
    a_valid = 1'b1;
    a_row   = pack2(3, 4);                    // cycle 1: A1's last row first
    step();
    a_row   = pack2(1, 2);                    // cycle 2: A1's first row
    step();
    a_valid = 1'b0;                           // cycle 3: A1 is full
    b_valid = 1'b1;
    b_col   = pack2(5, 6);                    //          B column 1 switches to A1
    step();
    b_col   = pack2(7, 8);                    // cycle 4: B column 2
    a_valid = 1'b1;
    a_row   = pack2(0, 2);                    //          offer A2's last row
    step();
    b_valid = 1'b0;
    b_col   = '0;
    while (!a_ready) step();                  // hold A2's row until the hold-off ends
    step();
    a_row   = pack2(-1, 0);                   // A2's first row
    while (!a_ready) step();
    step();
    a_valid = 1'b0;                           // A2 is full
    a_row   = '0;
    b_valid = 1'b1;
    b_col   = pack2(5, 6);                    // B column 3 switches to A2
    step();
    b_valid = 1'b0;
    b_col   = '0;
    repeat (6) step();
    $display("----------------------------------------------------------------");
    $display("Expected C columns: [17 39] (cycle 8), [23 53] (cycle 9), [-5 12] (cycle 13)");
    $finish;
  end

endmodule
