// |a| rounded half-to-even to an integer and clamped to 127 (the magnitude part
// of VQCLAMP). NaN is handled by the caller; +-inf clamps to 127.
module bpu_fvu_qround (
  input  logic [30:0] a_i,         // magnitude bits (sign handled by the caller)
  output logic [7:0]  mag_o
);

  logic [7:0]  e;
  logic [23:0] sig;
  logic [4:0]  sh;          // 23 - E for E in [-1, 6]
  logic [23:0] ip, rem, half;
  logic        up;
  logic [8:0]  r;

  assign e   = a_i[30:23];
  assign sig = {1'b1, a_i[22:0]};
  assign sh  = 5'(8'd150 - e);                  // 23 - (e - 127), valid for e in [126, 133]

  assign ip   = sig >> sh;
  assign rem  = sig & ((24'd1 << sh) - 24'd1);
  assign half = 24'd1 << (sh - 5'd1);
  assign up   = (rem > half) || ((rem == half) && ip[0]);
  assign r    = 9'(ip[7:0]) + 9'(up);          // ip < 128 whenever r is used
  logic unused_ip;
  assign unused_ip = ^ip[23:8];

  always_comb begin
    if (e >= 8'd134)      mag_o = 8'd127;       // |a| >= 128 (incl. inf)
    else if (e < 8'd126)  mag_o = 8'd0;         // |a| < 0.5
    else if (r > 9'd127)  mag_o = 8'd127;
    else                  mag_o = r[7:0];
  end

endmodule
