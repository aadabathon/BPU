// Signed integer to IEEE-754 binary32. Exact (no rounding) because InW <= 25
// keeps every magnitude at or below 2^24. Zero converts to +0.
module bpu_int2fp32 #(
  parameter int unsigned InW = 22,
  parameter bit          Reg = 1'b1   // register the output
) (
  input  logic           clk_i,
  input  logic           rst_ni,
  input  logic           valid_i,
  input  logic [InW-1:0] x_i,         // two's complement
  output logic           valid_o,
  output logic [31:0]    y_o
);

  logic signed [31:0] x_ext;
  logic [31:0]        mag, norm;
  logic [5:0]         lz;
  logic [31:0]        y_d;
  logic               unused_norm;

  assign x_ext = 32'($signed(x_i));
  assign mag   = x_ext[31] ? 32'(-x_ext) : 32'(x_ext);

  bpu_lzc #(.Width(32)) u_lzc (.in_i(mag), .cnt_o(lz));

  assign norm = mag << lz;
  // Exponent = 127 + (31 - lz). Every nonzero bit lands in norm[31:8] because
  // |x| <= 2^24, so dropping norm[7:0] is exact; norm[31] is the hidden bit.
  assign y_d  = (mag == '0) ? 32'd0 : {x_ext[31], 8'(8'd158 - {2'b00, lz}), norm[30:8]};
  assign unused_norm = ^{norm[31], norm[7:0]};

  bpu_delay #(.Width(1), .Depth(int'(Reg)), .Reset(1'b1)) u_v (
    .clk_i, .rst_ni, .d_i(valid_i), .q_o(valid_o)
  );
  bpu_delay #(.Width(32), .Depth(int'(Reg))) u_d (
    .clk_i, .rst_ni, .d_i(y_d), .q_o(y_o)
  );

`ifndef SYNTHESIS
  initial begin
    if (InW < 2 || InW > 25) $fatal(1, "bpu_int2fp32: InW=%0d must be in [2, 25] for exact conversion", InW);
  end
`endif

endmodule
