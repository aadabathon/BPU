// Equivalence of bpu_qmv_dot with the plain definition of a dot product, for
// every possible input (combinational, so a depth-1 check is a complete proof).
//   W4: sum_i int4(w[i]) * int8(x[i])
//   W8: sum_j int8(w byte j) * int8(x[half*Lanes/2 + j])
module bpu_qmv_dot_fv #(
  parameter int unsigned Lanes = 8
) (
  input logic               w8_i,
  input logic               x_hi_i,
  input logic [Lanes*4-1:0] w_i,
  input logic [Lanes*8-1:0] x_i
);

  localparam int unsigned OutW = 17 + $clog2(Lanes);

  logic signed [OutW-1:0] dut_sum, ref_sum;
  logic                   clk_unused = 1'b0;

  bpu_qmv_dot #(.Lanes(Lanes), .ProdReg(1'b0), .TreeRegEvery(0)) dut (
    .clk_i(clk_unused), .w8_i, .x_hi_i, .w_i, .x_i, .sum_o(dut_sum)
  );

  always_comb begin
    ref_sum = '0;
    if (!w8_i) begin
      for (int i = 0; i < Lanes; i++)
        ref_sum += OutW'($signed(w_i[4*i +: 4])) * OutW'($signed(x_i[8*i +: 8]));
    end else begin
      for (int j = 0; j < Lanes / 2; j++)
        ref_sum += OutW'($signed(w_i[8*j +: 8]))
                 * OutW'($signed(x_i[8*((x_hi_i ? Lanes/2 : 0) + j) +: 8]));
    end
    assert (dut_sum == ref_sum);
  end

endmodule
