/*
 * M2 design spike (Phase A1): *_mean/EWMA register folding.
 *
 * Isolates ONLY the open question the plan flags for M2: can the original
 * v1model EWMA-mean logic (scratchpad/old_p4_template.p4:527-544, 628-633 --
 * the pre-TNA-rewrite template, recovered from git history at commit
 * ebab8d1) be folded into TNA RegisterActions under:
 *   (a) the 4-touch-per-register cap, and
 *   (b) "one atomic read-modify-write per touch" (no separate .read()/.write()
 *       call pairs -- already established as mandatory by Part D/M1),
 *   (c) the "at most one external PHV input per RegisterAction, else a
 *       Mux/multi-field-branch restriction may bite" pattern observed in
 *       every ground-truth spike so far (G.2's num_trees>1 finding).
 *
 * Original v1model logic being ported (Flow_IAT_Mean, chosen because it's
 * this session's M2 feature-set decision -- see conversation, not the plan
 * doc, for why: M1's 3 features have no mean feature, so M2 hand-picks
 * Flow_IAT_Mean in addition to them specifically to force this design to get
 * built and validated now rather than deferred again):
 *
 *   flow_packet_count.read(packet_count, hash); (incremented earlier)
 *   flow_duration.read(iat_total, hash);         // accumulated IAT total
 *   flow_iat_mean.read(iat_mean, hash);
 *   bool isPowerOf2 = (packet_count & (packet_count-1)) == 0;
 *   if (isPowerOf2) {
 *       flow_next_bit_shift.read(bit_shift, 0);   // single GLOBAL 1-entry reg
 *       estimated_mean = iat_total >> bit_shift;
 *   } else {
 *       estimated_mean = (current_iat >> 1) + (iat_mean >> 1);
 *   }
 *   flow_iat_mean.write(hash, estimated_mean);
 *   ...later...
 *   if (isPowerOf2) {
 *       flow_next_bit_shift.read(bit_shift, 0);
 *       bit_shift = bit_shift + 1;
 *       flow_next_bit_shift.write(0, bit_shift);
 *   }
 *
 * Known pre-existing quirk in the original (not this spike's bug, not being
 * fixed here): flow_next_bit_shift is a single register shared across ALL
 * flows (size 1), and is force-reset to 1 whenever ANY flow's own count hits
 * 1 -- meaning one flow starting resets every other flow's EWMA shift. Also,
 * resetting to 1 (not 0) means the very first mean estimate is
 * total>>1 = half the first packet's own IAT, not the IAT itself -- an
 * off-by-one. This spike does NOT reproduce either quirk faithfully: see
 * the "Deliberate simplification" note below the bit-shift register.
 *
 * Disposable spike code, not wired into build_p4_script.py.
 */

#include <core.p4>
#if __TARGET_TOFINO__ == 2
#include <t2na.p4>
#else
#include <tna.p4>
#endif

#include "common/headers.p4"
#include "common/util.p4"

const bit<32> MAX_NUM_FLOWS = 4096;

struct metadata_t {
    bit<32> flow_hash;
    bit<16> now_pseudo_us;
    bit<16> current_iat;
    bit<16> flow_iat_mean_val;
}

parser SwitchIngressParser(
        packet_in pkt,
        out header_t hdr,
        out metadata_t ig_md,
        out ingress_intrinsic_metadata_t ig_intr_md) {

    TofinoIngressParser() tofino_parser;

    state start {
        tofino_parser.apply(pkt, ig_intr_md);
        transition parse_ethernet;
    }

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            ETHERTYPE_IPV4: parse_ipv4;
            default: accept;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            IP_PROTOCOLS_TCP: parse_tcp;
            default: accept;
        }
    }

    state parse_tcp {
        pkt.extract(hdr.tcp);
        transition accept;
    }
}

control SwitchIngressDeparser(
        packet_out pkt,
        inout header_t hdr,
        in metadata_t ig_md,
        in ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md) {
    apply {
        pkt.emit(hdr);
    }
}

control SwitchIngress(
        inout header_t hdr,
        inout metadata_t meta,
        in ingress_intrinsic_metadata_t ig_intr_md,
        in ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {

    Hash<bit<32>>(HashAlgorithm_t.CRC32) flow_hash_calc;

    action calc_flow_hash() {
        meta.flow_hash = flow_hash_calc.get({
            hdr.ipv4.src_addr,
            hdr.ipv4.dst_addr,
            hdr.ipv4.protocol,
            hdr.tcp.src_port,
            hdr.tcp.dst_port
        }) & 0xFFF;
    }

    action calc_timestamp() {
        meta.now_pseudo_us = (bit<16>)(ig_prsr_md.global_tstamp >> 10);
    }

    // ---- current_iat, standing in for M1's flow_last_arrival_time_reg
    // (already validated) -- needed here only as an input to the mean calc.

    Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) flow_last_arrival_time_reg;
    RegisterAction<bit<16>, bit<32>, bit<16>>(flow_last_arrival_time_reg) flow_iat_action = {
        void apply(inout bit<16> value, out bit<16> rv) {
            rv = meta.now_pseudo_us - value;
            value = meta.now_pseudo_us;
        }
    };

    // ---- Redesigned mean estimator, take 3.
    //
    // Take 1 (halve the register's OWN stored value in place via
    // `value = value >> 1;`) failed: "expression too complex for register
    // action", even with NO external input and only ONE op. Confirmed
    // against p4c's own source (backends/tofino/bf-p4c/mau/stateful_alu.h):
    // the visitor that builds register-action instructions has explicit
    // cases for Add/AddSat/Sub/SubSat/BAnd/BOr/BXor/Cmpl/Concat/compare/
    // Div-Mod, but NONE for Shl/Shr -- shift falls through to the generic
    // "too complex" catch-all. Categorical, not a complexity-budget issue.
    //
    // Take 2 (do the shift externally in a plain action, use the register
    // purely for read-then-write) failed differently: table placement
    // couldn't co-locate the read-touch and write-touch tables with the
    // register. Root cause: a physical Tofino Register can be accessed at
    // most ONCE per packet total (not once per stage) -- unlike the M1
    // `flows` register's two RegisterActions, which are mutually exclusive
    // (if/else -- only one ever fires per packet), ours needed the read AND
    // the write to BOTH fire, sequentially, for the same packet. That's not
    // schedulable on one physical register unit.
    //
    // Take 3 (this one): Tofino has a dedicated MathUnit<T> hardware
    // primitive for exactly this class of problem (scaled/reciprocal
    // arithmetic no register ALU can do) -- confirmed from the installed
    // SDE headers (tofino1_base.p4) and the compiler's own handling
    // (CreateSaluInstruction::preorder(IR::MAU::Primitive*), "math_unit.
    // execute"): its result is just one more operand fed into whatever
    // instruction is being built, so unlike a second RegisterAction, it
    // does NOT count as a second register touch -- it's part of the SAME
    // touch. MathUnit(MathOp_t.MUL, 1, 2) approximates `x/2` (a ratio A/B
    // times an identity function, via an internal exponent+8-bit-mantissa
    // LUT -- an approximation, not exact integer division, same precision
    // class as how Tofino meters compute rates). Folding the whole blend
    // into ONE expression -- (old + sample) then halved, algebraically
    // identical to old/2 + sample/2 -- keeps this to exactly ONE touch.

    MathUnit<bit<16>>(MathOp_t.MUL, 1, 2) halve_unit;

    Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) flow_iat_mean_reg;

    RegisterAction<bit<16>, bit<32>, bit<16>>(flow_iat_mean_reg) flow_iat_mean_ewma_action = {
        void apply(inout bit<16> value, out bit<16> rv) {
            value = halve_unit.execute(value + meta.current_iat);
            rv = value;
        }
    };

    apply {
        if (hdr.tcp.isValid()) {
            calc_flow_hash();
            calc_timestamp();

            meta.current_iat = flow_iat_action.execute(meta.flow_hash);

            meta.flow_iat_mean_val = flow_iat_mean_ewma_action.execute(meta.flow_hash);
        }

        ig_tm_md.ucast_egress_port = ig_intr_md.ingress_port;
        ig_tm_md.bypass_egress = 1w1;
    }
}

Pipeline(SwitchIngressParser(),
         SwitchIngress(),
         SwitchIngressDeparser(),
         EmptyEgressParser(),
         EmptyEgress(),
         EmptyEgressDeparser()) pipe;

Switch(pipe) main;
