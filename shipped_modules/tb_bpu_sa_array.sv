// Self-checking testbench for bpu_sa_array.sv.
//
//   pe_tester  every signed int8 x int8 pair through one PE, with random incoming
//              sum/carry, through both the shadow (swap) and active A paths.
//   sa_tester  random A tiles and B columns through a whole array, with gaps,
//              overlapped tile loads and tile switches, checked column by column
//              against a reference model. Run at several shapes and widths.
//
// ModelSim / Questa:
//   vlog -sv bpu_sa_array.sv tb_bpu_sa_array.sv
//   vsim -c tb_bpu_sa_array -do "run -all; quit -f"

module pe_tester (
  input  logic clk,
  input  logic rst_n,
  output bit   done,
  output int   errors
);

  logic        load, swap, swap_o;
  logic [7:0]  load_a, load_a_o, b, b_o;
  logic [31:0] sum_i, carry_i, sum_o, carry_o;

  bpu_sa_pe #(.AccW(32)) dut (
    .clk_i(clk), .rst_ni(rst_n), .mode_i(1'b0),
    .load_i(load), .load_a_i(load_a), .load_a_o(load_a_o),
    .b_i(b), .swap_i(swap), .b_o(b_o), .swap_o(swap_o),
    .sum_i(sum_i), .carry_i(carry_i), .sum_o(sum_o), .carry_o(carry_o)
  );

  initial begin
    logic [31:0] exp_v;
    logic [7:0]  exp_b;
    logic        exp_sw, pend;
    done = 0; errors = 0; pend = 0;
    load = 0; swap = 0; load_a = '0; b = '0; sum_i = '0; carry_i = '0;
    void'($urandom(7));
    @(posedge rst_n);
    for (int a = -128; a < 128; a++) begin
      @(negedge clk);
      pend = 0;
      load = 1; load_a = 8'(a); swap = 0; b = '0;
      for (int bv = -128; bv < 129; bv++) begin
        @(negedge clk);
        if (pend) begin
          if (sum_o + carry_o !== exp_v || b_o !== exp_b || swap_o !== exp_sw) begin
            errors++;
            if (errors < 10)
              $display("PE MISMATCH a=%0d: got %h want %h", a, sum_o + carry_o, exp_v);
          end
          if (load_a_o !== 8'(a)) errors++;
        end
        if (bv == 128) break;
        load    = 0;
        swap    = ((bv & 1) == 0);           // first B swaps; then alternate
        b       = 8'(bv);
        sum_i   = $urandom;
        carry_i = $urandom;
        exp_v   = sum_i + carry_i + 32'(a * bv);
        exp_b   = 8'(bv);
        exp_sw  = swap;
        pend    = 1;
      end
    end
    done = 1;
  end

endmodule


module sa_tester #(
  parameter int Rows   = 8,
  parameter int Cols   = 8,
  parameter int AccW   = 32,
  parameter int Cycles = 4000,
  parameter int Seed   = 1
) (
  input  logic clk,
  input  logic rst_n,
  output bit   done,
  output int   errors,
  output int   checked,
  output int   switches
);

  localparam int Lat = Rows + Cols + 1;
  localparam int QN  = 128;

  logic                 mode, err_mode, a_valid, a_ready, a_full;
  logic                 b_valid, b_ready, c_valid;
  logic [Cols*8-1:0]    a_row, b_col;
  logic [Rows*AccW-1:0] c_col;

  bpu_sa_array #(.Rows(Rows), .Cols(Cols), .AccW(AccW)) dut (
    .clk_i(clk), .rst_ni(rst_n), .mode_i(mode), .err_mode_o(err_mode),
    .a_valid_i(a_valid), .a_ready_o(a_ready), .a_row_i(a_row), .a_full_o(a_full),
    .b_valid_i(b_valid), .b_ready_o(b_ready), .b_col_i(b_col),
    .c_valid_o(c_valid), .c_col_o(c_col)
  );

  int     nt  [Rows][Cols];      // tile being loaded
  int     act [Rows][Cols];      // tile in use
  int     bv  [Cols];
  longint expq [QN][Rows];
  int     head, tail, cnt;
  bit     started;

  function automatic int r8();
    int u;
    u = $urandom_range(0, 19);
    if (u == 0) return -128;
    if (u == 1) return 127;
    return int'($urandom_range(0, 255)) - 128;
  endfunction

  task automatic new_tile();
    for (int i = 0; i < Rows; i++)
      for (int k = 0; k < Cols; k++) nt[i][k] = r8();
  endtask

  task automatic check_out();
    logic [AccW-1:0] want;
    if (c_valid !== 1'b1) return;
    if (head == tail) begin
      errors++;
      $display("[%0dx%0d] unexpected output column", Rows, Cols);
      return;
    end
    for (int i = 0; i < Rows; i++) begin
      want = AccW'(expq[head % QN][i]);
      if (c_col[i*AccW +: AccW] !== want) begin
        errors++;
        if (errors < 10)
          $display("[%0dx%0d] MISMATCH col %0d row %0d: got %0d want %0d", Rows, Cols,
                   head, i, $signed(c_col[i*AccW +: AccW]), $signed(want));
      end
    end
    head++;
    checked++;
  endtask

  initial begin
    done = 0; errors = 0; checked = 0; switches = 0;
    head = 0; tail = 0; cnt = 0; started = 0;
    mode = 0; a_valid = 0; b_valid = 0; a_row = '0; b_col = '0;
    void'($urandom(Seed));
    new_tile();
    @(posedge rst_n);

    for (int cyc = 0; cyc < Cycles + Lat + 4; cyc++) begin
      @(negedge clk);
      check_out();
      if (a_full !== (cnt == Rows)) begin
        errors++;
        $display("[%0dx%0d] a_full mismatch", Rows, Cols);
      end
      if (err_mode !== 1'b0 || b_ready !== 1'b1) errors++;

      // Stimulus for the coming rising edge (stops after Cycles, then drains).
      a_valid = (cyc < Cycles) && (cnt < Rows) && ($urandom_range(0, 99) < 60);
      if (a_valid)
        for (int k = 0; k < Cols; k++) a_row[8*k +: 8] = 8'(nt[Rows-1-cnt][k]);
      b_valid = (cyc < Cycles) && (started || cnt == Rows) && ($urandom_range(0, 99) < 55);
      for (int k = 0; k < Cols; k++) begin
        bv[k] = r8();
        b_col[8*k +: 8] = 8'(bv[k]);
      end

      // Reference model of what that edge does.
      if (b_valid && cnt == Rows) begin           // tile switch
        act = nt;
        cnt = 0;
        started = 1;
        switches++;
        new_tile();
      end else if (a_valid && a_ready) begin
        cnt++;
      end
      if (b_valid) begin
        for (int i = 0; i < Rows; i++) begin
          longint s;
          s = 0;
          for (int k = 0; k < Cols; k++) s += longint'(act[i][k]) * longint'(bv[k]);
          expq[tail % QN][i] = s;
        end
        tail++;
      end
    end

    if (head != tail) begin
      errors++;
      $display("[%0dx%0d] %0d columns never came out", Rows, Cols, tail - head);
    end
    @(negedge clk);
    mode = 1;
    #1 if (err_mode !== 1'b1) errors++;
    done = 1;
  end

endmodule


module tb_bpu_sa_array;

  logic clk = 1'b0, rst_n = 1'b0;
  always #5 clk = ~clk;
  initial #23 rst_n = 1'b1;

  bit pe_done, d0, d1, d2, d3;
  int pe_err, e0, e1, e2, e3, n0, n1, n2, n3, s0, s1, s2, s3;

  pe_tester u_pe (.clk, .rst_n, .done(pe_done), .errors(pe_err));
  sa_tester #(.Rows(8), .Cols(8), .AccW(32), .Seed(11)) u_8x8 (
    .clk, .rst_n, .done(d0), .errors(e0), .checked(n0), .switches(s0));
  sa_tester #(.Rows(3), .Cols(5), .AccW(32), .Seed(22)) u_3x5 (
    .clk, .rst_n, .done(d1), .errors(e1), .checked(n1), .switches(s1));
  sa_tester #(.Rows(5), .Cols(2), .AccW(17), .Seed(33)) u_5x2 (
    .clk, .rst_n, .done(d2), .errors(e2), .checked(n2), .switches(s2));
  sa_tester #(.Rows(1), .Cols(1), .AccW(16), .Seed(44)) u_1x1 (
    .clk, .rst_n, .done(d3), .errors(e3), .checked(n3), .switches(s3));

  initial begin
    wait (pe_done && d0 && d1 && d2 && d3);
    $display("PE  exhaustive int8 x int8 (65,536 pairs x 2 paths): %0d errors", pe_err);
    $display("8x8 AccW 32: %0d columns, %0d tile switches, %0d errors", n0, s0, e0);
    $display("3x5 AccW 32: %0d columns, %0d tile switches, %0d errors", n1, s1, e1);
    $display("5x2 AccW 17: %0d columns, %0d tile switches, %0d errors", n2, s2, e2);
    $display("1x1 AccW 16: %0d columns, %0d tile switches, %0d errors", n3, s3, e3);
    if (pe_err + e0 + e1 + e2 + e3 == 0) $display("PASS");
    else                                 $display("FAIL");
    $finish;
  end

endmodule
