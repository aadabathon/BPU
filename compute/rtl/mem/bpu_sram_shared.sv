// Shared SRAM: NBanks banks of 1R1W memory behind a request/response crossbar,
// shared by every engine (the "SRAM arbitrator" + "Shared SRAM" of the BPU block
// diagram). Words are Lanes x 32 bits; addresses are word addresses.
//
// Client protocol (each read port and each write port independently):
//   request   valid/ready; addr (and mask/data for writes) stay stable until ready.
//   response  (reads) rvalid + rdata exactly RdLat = 1 + OutReg cycles after the
//             request was accepted, in request order, no backpressure: a client
//             only requests what it can take.
// Ordering and visibility: a write accepted in cycle t is seen by every read
// accepted in a later cycle; a read accepted in the same cycle as a write to the
// same word returns the old data.
//
// Each bank accepts one read and one write per cycle. Competing requests for a
// bank are granted round-robin per bank, so a waiting request is served within
// NRd (or NWr) grants of that bank. Bank = word address modulo NBanks, XOR-folded
// with the row bits when Hash is set, so streams whose bases differ by a multiple
// of NBanks still spread over different banks.
// Requests at or beyond the capacity are accepted at once without touching any
// bank: reads return zero, writes are dropped, and rd_oob_o / wr_oob_o pulse.
module bpu_sram_shared #(
  parameter int unsigned NBanks    = 4,       // power of two
  parameter int unsigned BankWords = 256,     // power of two, >= 2
  parameter int unsigned Lanes     = 4,       // 32-bit lanes per word
  parameter int unsigned NRd       = 2,
  parameter int unsigned NWr       = 2,
  parameter int unsigned AW        = 32,      // client word-address width
  parameter bit          OutReg    = 1'b0,    // register read data (RdLat = 2)
  parameter bit          Hash      = 1'b1
) (
  input  logic                    clk_i,
  input  logic                    rst_ni,

  input  logic [NRd-1:0]          rd_valid_i,
  output logic [NRd-1:0]          rd_ready_o,
  input  logic [NRd*AW-1:0]       rd_addr_i,
  output logic [NRd-1:0]          rd_rvalid_o,
  output logic [NRd*Lanes*32-1:0] rd_rdata_o,
  output logic [NRd-1:0]          rd_oob_o,

  input  logic [NWr-1:0]          wr_valid_i,
  output logic [NWr-1:0]          wr_ready_o,
  input  logic [NWr*AW-1:0]       wr_addr_i,
  input  logic [NWr*Lanes-1:0]    wr_mask_i,
  input  logic [NWr*Lanes*32-1:0] wr_data_i,
  output logic [NWr-1:0]          wr_oob_o,

  output logic [31:0]             perf_rd_wait_o,   // cycles some read waited for a bank
  output logic [31:0]             perf_wr_wait_o
);

  localparam int unsigned W   = Lanes * 32;
  localparam int unsigned BB  = $clog2(NBanks);              // bank index bits (0: one bank)
  localparam int unsigned BI  = (BB > 0) ? BB : 1;
  localparam int unsigned RB  = $clog2(BankWords);           // row bits
  localparam int unsigned CW  = BB + RB;                     // capacity = 2^CW words
  localparam int unsigned RPW = (NRd > 1) ? $clog2(NRd) : 1;
  localparam int unsigned WPW = (NWr > 1) ? $clog2(NWr) : 1;

  // ---------------------------------------------------------------------------
  // Address decode
  // ---------------------------------------------------------------------------
  function automatic logic in_range(input logic [AW-1:0] a);
    if (AW > CW) return (a >> CW) == '0;
    return 1'b1;
  endfunction
  function automatic logic [RB-1:0] row_of(input logic [AW-1:0] a);
    return RB'(a >> BB);
  endfunction
  function automatic logic [BI-1:0] bank_of(input logic [AW-1:0] a);
    logic [BI-1:0] b;
    logic [RB-1:0] r;
    if (BB == 0) return '0;
    b = BI'(a);
    r = row_of(a);
    if (Hash) for (int i = 0; i < RB; i++) b[i % BI] = b[i % BI] ^ r[i];
    return b;
  endfunction

  logic [NRd-1:0]         rd_in;
  logic [NRd-1:0][BI-1:0] rd_bank;
  logic [NRd-1:0][RB-1:0] rd_row;
  logic [NWr-1:0]         wr_in;
  logic [NWr-1:0][BI-1:0] wr_bank;
  logic [NWr-1:0][RB-1:0] wr_row;

  for (genvar p = 0; p < NRd; p++) begin : g_rdec
    assign rd_in[p]   = in_range(rd_addr_i[p*AW +: AW]);
    assign rd_bank[p] = bank_of(rd_addr_i[p*AW +: AW]);
    assign rd_row[p]  = row_of(rd_addr_i[p*AW +: AW]);
  end
  for (genvar p = 0; p < NWr; p++) begin : g_wdec
    assign wr_in[p]   = in_range(wr_addr_i[p*AW +: AW]);
    assign wr_bank[p] = bank_of(wr_addr_i[p*AW +: AW]);
    assign wr_row[p]  = row_of(wr_addr_i[p*AW +: AW]);
  end

  // ---------------------------------------------------------------------------
  // Per-bank round-robin arbitration
  // ---------------------------------------------------------------------------
  logic [NBanks-1:0][NRd-1:0] rd_gnt;
  logic [NBanks-1:0][NWr-1:0] wr_gnt;
  logic [NBanks-1:0][RPW-1:0] rd_ptr_q, rd_gidx;
  logic [NBanks-1:0][WPW-1:0] wr_ptr_q, wr_gidx;

  always_comb begin
    for (int b = 0; b < NBanks; b++) begin
      logic found;
      int   p;
      rd_gnt[b]  = '0;
      rd_gidx[b] = '0;
      found      = 1'b0;
      for (int i = 0; i < NRd; i++) begin
        p = int'(rd_ptr_q[b]) + i;
        if (p >= NRd) p = p - NRd;
        if (!found && rd_valid_i[p] && rd_in[p] && rd_bank[p] == BI'(b)) begin
          rd_gnt[b][p] = 1'b1;
          rd_gidx[b]   = RPW'(p);
          found        = 1'b1;
        end
      end
      wr_gnt[b]  = '0;
      wr_gidx[b] = '0;
      found      = 1'b0;
      for (int i = 0; i < NWr; i++) begin
        p = int'(wr_ptr_q[b]) + i;
        if (p >= NWr) p = p - NWr;
        if (!found && wr_valid_i[p] && wr_in[p] && wr_bank[p] == BI'(b)) begin
          wr_gnt[b][p] = 1'b1;
          wr_gidx[b]   = WPW'(p);
          found        = 1'b1;
        end
      end
    end
  end

  always_comb begin
    for (int p = 0; p < NRd; p++) begin
      rd_ready_o[p] = !rd_in[p];                   // out of range: accepted at once
      for (int b = 0; b < NBanks; b++) rd_ready_o[p] = rd_ready_o[p] | rd_gnt[b][p];
      rd_oob_o[p] = rd_valid_i[p] && !rd_in[p];
    end
    for (int p = 0; p < NWr; p++) begin
      wr_ready_o[p] = !wr_in[p];
      for (int b = 0; b < NBanks; b++) wr_ready_o[p] = wr_ready_o[p] | wr_gnt[b][p];
      wr_oob_o[p] = wr_valid_i[p] && !wr_in[p];
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      rd_ptr_q <= '0;
      wr_ptr_q <= '0;
    end else begin
      for (int b = 0; b < NBanks; b++) begin
        if (rd_gnt[b] != '0) rd_ptr_q[b] <= (int'(rd_gidx[b]) == NRd - 1) ? '0 : rd_gidx[b] + 1'b1;
        if (wr_gnt[b] != '0) wr_ptr_q[b] <= (int'(wr_gidx[b]) == NWr - 1) ? '0 : wr_gidx[b] + 1'b1;
      end
    end
  end

  // ---------------------------------------------------------------------------
  // Banks
  // ---------------------------------------------------------------------------
  logic [NBanks-1:0][W-1:0] bank_rdata;

  for (genvar b = 0; b < NBanks; b++) begin : g_bank
    logic [RB-1:0]    raddr, waddr;
    logic [Lanes-1:0] wmask;
    logic [W-1:0]     wdata;
    assign raddr = rd_row[rd_gidx[b]];
    assign waddr = wr_row[wr_gidx[b]];
    assign wmask = wr_mask_i[wr_gidx[b]*Lanes +: Lanes];
    assign wdata = wr_data_i[wr_gidx[b]*W +: W];
    bpu_sram_1r1w_be #(.Depth(BankWords), .Lanes(Lanes), .LaneW(32)) u_mem (
      .clk_i,
      .we_i(wr_gnt[b] != '0), .waddr_i(waddr), .wmask_i(wmask), .wdata_i(wdata),
      .re_i(rd_gnt[b] != '0), .raddr_i(raddr), .rdata_o(bank_rdata[b]));
  end

  // ---------------------------------------------------------------------------
  // Read responses: data of the bank each accepted read went to (zero if out of range)
  // ---------------------------------------------------------------------------
  logic [NRd-1:0]         r1_v_q, r1_in_q;
  logic [NRd-1:0][BI-1:0] r1_bank_q;
  logic [NRd-1:0][W-1:0]  r1_data;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) r1_v_q <= '0;
    else         r1_v_q <= rd_valid_i & rd_ready_o;
  end
  always_ff @(posedge clk_i) begin
    r1_in_q   <= rd_in;
    r1_bank_q <= rd_bank;
  end
  for (genvar p = 0; p < NRd; p++) begin : g_rsp
    assign r1_data[p] = r1_in_q[p] ? bank_rdata[r1_bank_q[p]] : '0;
  end

  if (OutReg) begin : g_outreg
    logic [NRd-1:0]        r2_v_q;
    logic [NRd-1:0][W-1:0] r2_data_q;
    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) r2_v_q <= '0;
      else         r2_v_q <= r1_v_q;
    end
    always_ff @(posedge clk_i) r2_data_q <= r1_data;
    assign rd_rvalid_o = r2_v_q;
    assign rd_rdata_o  = r2_data_q;
  end else begin : g_direct
    assign rd_rvalid_o = r1_v_q;
    assign rd_rdata_o  = r1_data;
  end

  // ---------------------------------------------------------------------------
  // Contention counters
  // ---------------------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      perf_rd_wait_o <= '0;
      perf_wr_wait_o <= '0;
    end else begin
      perf_rd_wait_o <= perf_rd_wait_o + 32'((rd_valid_i & ~rd_ready_o) != '0);
      perf_wr_wait_o <= perf_wr_wait_o + 32'((wr_valid_i & ~wr_ready_o) != '0);
    end
  end

`ifdef FORMAL
  always_comb begin
    if (rst_ni) begin
      for (int b = 0; b < NBanks; b++) begin
        assert ($onehot0(rd_gnt[b]));                    // one read and one write per bank
        assert ($onehot0(wr_gnt[b]));
        for (int p = 0; p < NRd; p++)
          if (rd_gnt[b][p]) assert (rd_valid_i[p] && rd_in[p] && rd_bank[p] == BI'(b));
        for (int p = 0; p < NWr; p++)
          if (wr_gnt[b][p]) assert (wr_valid_i[p] && wr_in[p] && wr_bank[p] == BI'(b));
      end
    end
  end
`endif

`ifndef SYNTHESIS
  initial begin
    if (NBanks < 1 || (NBanks & (NBanks - 1)) != 0)
      $fatal(1, "bpu_sram_shared: NBanks=%0d must be a power of two", NBanks);
    if (BankWords < 2 || (BankWords & (BankWords - 1)) != 0)
      $fatal(1, "bpu_sram_shared: BankWords=%0d must be a power of two >= 2", BankWords);
    if (AW < CW)
      $fatal(1, "bpu_sram_shared: AW=%0d cannot address %0d words", AW, NBanks * BankWords);
  end
`endif

endmodule
