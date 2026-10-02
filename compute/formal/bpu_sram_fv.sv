// Formal harness for bpu_sram_shared: three read and two write clients making any
// requests (held stable until accepted). Checks, beyond the module's own asserts:
//   - every request is accepted within the round-robin bound,
//   - a read's response comes exactly RdLat cycles after its acceptance,
//   - data: for one symbolic word address, every read returns the last write
//     accepted in an earlier cycle (zero for addresses past the capacity).
module bpu_sram_fv #(
  parameter bit OutReg = 1'b0,
  parameter bit Hash   = 1'b1
) (
  input logic           clk_i,
  input logic           rst_ni,
  input logic [2:0]     rd_valid_i,
  input logic [3*3-1:0] rd_addr_i,
  input logic [1:0]     wr_valid_i,
  input logic [2*3-1:0] wr_addr_i,
  input logic [1:0]     wr_mask_i,
  input logic [2*32-1:0] wr_data_i,
  input logic [2:0]     trk_i                     // the tracked word address
);

  localparam int unsigned RdLat = 1 + int'(OutReg);
  localparam int unsigned Cap   = 4;              // 2 banks x 2 words; addresses 4..7 are out of range

  logic [2:0]      rd_ready, rd_rvalid, rd_oob;
  logic [3*32-1:0] rd_rdata;
  logic [1:0]      wr_ready, wr_oob;

  bpu_sram_shared #(
    .NBanks(2), .BankWords(2), .Lanes(1), .NRd(3), .NWr(2), .AW(3), .OutReg(OutReg), .Hash(Hash)
  ) dut (
    .clk_i, .rst_ni,
    .rd_valid_i, .rd_ready_o(rd_ready), .rd_addr_i, .rd_rvalid_o(rd_rvalid), .rd_rdata_o(rd_rdata),
    .rd_oob_o(rd_oob),
    .wr_valid_i, .wr_ready_o(wr_ready), .wr_addr_i, .wr_mask_i, .wr_data_i, .wr_oob_o(wr_oob),
    .perf_rd_wait_o(), .perf_wr_wait_o());

  // ---------------------------------------------------------------------------
  // Environment: reset, stable requests, a fixed tracked address
  // ---------------------------------------------------------------------------
  logic init_q = 1'b1, rst_n_q;
  logic [2:0]       rv_q, rr_q;
  logic [3*3-1:0]   ra_q;
  logic [1:0]       wv_q, wr_q, wm_q;
  logic [2*3-1:0]   wa_q;
  logic [2*32-1:0]  wd_q;
  logic [2:0]       trk_q;
  always_ff @(posedge clk_i) begin
    init_q <= 1'b0;
    rst_n_q <= rst_ni;
    rv_q <= rd_valid_i; rr_q <= rd_ready; ra_q <= rd_addr_i;
    wv_q <= wr_valid_i; wr_q <= wr_ready; wa_q <= wr_addr_i; wm_q <= wr_mask_i; wd_q <= wr_data_i;
    trk_q <= trk_i;
  end

  always_comb begin
    if (init_q) assume (!rst_ni);
    if (!init_q && rst_n_q) begin
      assume (rst_ni);
      assume (trk_i == trk_q);
      for (int p = 0; p < 3; p++)
        if (rv_q[p] && !rr_q[p]) assume (rd_valid_i[p] && rd_addr_i[p*3 +: 3] == ra_q[p*3 +: 3]);
      for (int p = 0; p < 2; p++)
        if (wv_q[p] && !wr_q[p])
          assume (wr_valid_i[p] && wr_addr_i[p*3 +: 3] == wa_q[p*3 +: 3]
                  && wr_mask_i[p] == wm_q[p] && wr_data_i[p*32 +: 32] == wd_q[p*32 +: 32]);
    end
  end

  // ---------------------------------------------------------------------------
  // Shadow of the tracked word
  // ---------------------------------------------------------------------------
  logic        known_q;
  logic [31:0] shadow_q;
  logic        wr_hit;
  logic [31:0] wr_val;
  always_comb begin
    wr_hit = 1'b0;
    wr_val = shadow_q;
    for (int p = 0; p < 2; p++)
      if (wr_valid_i[p] && wr_ready[p] && wr_addr_i[p*3 +: 3] == trk_i && wr_mask_i[p]) begin
        wr_hit = 1'b1;
        wr_val = wr_data_i[p*32 +: 32];
      end
  end

  // Per read port: expected response pipeline (valid, check, value)
  logic [2:0][RdLat-1:0] ev_q, ek_q;
  logic [2:0][RdLat-1:0][31:0] ed_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      known_q <= 1'b0;
      ev_q    <= '0;
      ek_q    <= '0;
    end else begin
      if (wr_hit) begin
        known_q  <= 1'b1;
        shadow_q <= wr_val;
      end
      for (int p = 0; p < 3; p++) begin
        logic acc, oob, hit;
        acc = rd_valid_i[p] && rd_ready[p];
        oob = 32'(rd_addr_i[p*3 +: 3]) >= Cap;
        hit = rd_addr_i[p*3 +: 3] == trk_i;
        for (int s = RdLat - 1; s > 0; s--) begin
          ev_q[p][s] <= ev_q[p][s-1];
          ek_q[p][s] <= ek_q[p][s-1];
          ed_q[p][s] <= ed_q[p][s-1];
        end
        ev_q[p][0] <= acc;
        ek_q[p][0] <= acc && (oob || (hit && known_q));      // a read sees writes of earlier cycles
        ed_q[p][0] <= oob ? 32'd0 : shadow_q;
      end
    end
  end

  // Bounded wait
  logic [2:0][2:0] rw_q;
  logic [1:0][2:0] ww_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      rw_q <= '0;
      ww_q <= '0;
    end else begin
      for (int p = 0; p < 3; p++) rw_q[p] <= (rd_valid_i[p] && !rd_ready[p]) ? rw_q[p] + 1'b1 : '0;
      for (int p = 0; p < 2; p++) ww_q[p] <= (wr_valid_i[p] && !wr_ready[p]) ? ww_q[p] + 1'b1 : '0;
    end
  end

  always_comb begin
    if (rst_ni) begin
      for (int p = 0; p < 3; p++) begin
        assert (rw_q[p] <= 3'd2);                                 // NRd - 1
        assert (rd_rvalid[p] == ev_q[p][RdLat-1]);                // fixed latency
        if (ev_q[p][RdLat-1] && ek_q[p][RdLat-1]) assert (rd_rdata[p*32 +: 32] == ed_q[p][RdLat-1]);
        assert (rd_oob[p] == (rd_valid_i[p] && 32'(rd_addr_i[p*3 +: 3]) >= Cap));
      end
      for (int p = 0; p < 2; p++) assert (ww_q[p] <= 3'd1);      // NWr - 1
      cover (ev_q[0][RdLat-1] && ek_q[0][RdLat-1] && rd_rdata[31:0] != '0);   // data flows
    end
  end

endmodule
