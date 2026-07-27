/*
 * T11 Stage-1 spike, "worst-case touch count, 4 touches" variant (see
 * reviews/t11_tofino_port_and_env.md Part D.2 and Part F).
 *
 * Sibling of tna_rf_ddos_spike_worstcase.p4 (5 touches), which failed to
 * compile: p4c enforces a hard architecture limit of at most 4
 * RegisterActions per Register ("too many RegisterActions attached to the
 * Register... limits ... to 4"). This variant drops exactly the touch
 * Part D.2 already identified as *provably redundant* -- the :639 re-read,
 * which duplicates the value already fetched at :557 -- landing at 4
 * touches (2 reads + 1 write + 1 more read), right at the compiler's limit,
 * to see whether 4 touches compiles and what it costs in stages relative to
 * the 1-touch consolidated baseline.
 *
 * Simplification, stated honestly: this does not reimplement the real
 * power-of-2/EWMA logic those extra touches serve in production (that
 * would pull in Tier-2 accuracy-costing changes, out of scope here). It
 * only reproduces the TOUCH COUNT and keeps every read's result live (folded
 * into the final feature value via XOR) so the compiler cannot dead-code
 * eliminate any touch -- matching the resource cost of 4 real register
 * accesses without claiming to reproduce their semantics.
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

const bit<32> MAX_NUM_FLOWS = 4096; // matches p4/p4_code_RF_models.p4:9

struct metadata_t {
    bit<32> flow_hash;
    bit<16> flow_packet_count;
    bit<16> flow_packet_length_total;
    bit<16> flow_packet_length_max;
    bit<6>  codeword;   // 3 features x 2-bit interval code (toy sizing)
    bit<1>  ddos_class;
    bit<16> touch1_liveness;  // holds touch1's result so p4c can't dead-code it
    bit<16> touch2_liveness;  // holds touch2's result so p4c can't dead-code it
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
        }) & 0xFFF; // MAX_NUM_FLOWS = 4096 = 2^12
    }

    // ---- flow_packet_count: FIVE touches per packet, mirroring
    // p4_code_RF_models.p4's :294 (read), :486 (read), :488 (write), :557
    // (read), :639 (read) on the same register. ----

    Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) flow_packet_count_reg;

    RegisterAction<bit<16>, bit<32>, bit<16>>(flow_packet_count_reg) count_read_touch1 = {
        void apply(inout bit<16> value, out bit<16> rv) { rv = value; }        // mirrors :294
    };
    RegisterAction<bit<16>, bit<32>, bit<16>>(flow_packet_count_reg) count_read_touch2 = {
        void apply(inout bit<16> value, out bit<16> rv) { rv = value; }        // mirrors :486
    };
    RegisterAction<bit<16>, bit<32>, bit<16>>(flow_packet_count_reg) count_write_touch = {
        void apply(inout bit<16> value, out bit<16> rv) {                      // mirrors :488
            value = value + 1;
            rv = value;
        }
    };
    RegisterAction<bit<16>, bit<32>, bit<16>>(flow_packet_count_reg) count_read_touch3 = {
        void apply(inout bit<16> value, out bit<16> rv) { rv = value; }        // mirrors :557
    };
    // count_read_touch4 (mirroring :639) deliberately dropped -- Part D.2
    // already identified it as a provably redundant re-read of the same
    // value fetched at :557, and 5 touches does not compile (see file
    // header). This is exactly the Tier-1-style trim the real fix would do.

    // ---- The other two registers: unchanged, 1 touch each (already
    // Tier-1-consolidated), to isolate the effect of the one worst-case
    // register. ----

    Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) flow_packet_length_total_reg;
    RegisterAction<bit<16>, bit<32>, bit<16>>(flow_packet_length_total_reg) total_action = {
        void apply(inout bit<16> value, out bit<16> rv) {
            value = value + hdr.ipv4.total_len;
            rv = value;
        }
    };

    Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) flow_packet_length_max_reg;
    RegisterAction<bit<16>, bit<32>, bit<16>>(flow_packet_length_max_reg) max_action = {
        void apply(inout bit<16> value, out bit<16> rv) {
            if (hdr.ipv4.total_len > value) {
                value = hdr.ipv4.total_len;
            }
            rv = value;
        }
    };

    action set_code_packet_count(bit<2> code) {
        meta.codeword[1:0] = code;
    }

    table table_0_flow_packet_count {
        key = { meta.flow_packet_count : range; }
        actions = { set_code_packet_count; NoAction; }
        const default_action = NoAction();
        size = 16;
    }

    action set_code_packet_length_total(bit<2> code) {
        meta.codeword[3:2] = code;
    }

    table table_1_flow_packet_length_total {
        key = { meta.flow_packet_length_total : range; }
        actions = { set_code_packet_length_total; NoAction; }
        const default_action = NoAction();
        size = 16;
    }

    action set_code_packet_length_max(bit<2> code) {
        meta.codeword[5:4] = code;
    }

    table table_2_flow_packet_length_max {
        key = { meta.flow_packet_length_max : range; }
        actions = { set_code_packet_length_max; NoAction; }
        const default_action = NoAction();
        size = 16;
    }

    action classify_ddos(bit<1> class) {
        meta.ddos_class = class;
    }

    table get_classification_tree_ddos_0 {
        key = {
            meta.codeword         : ternary;
            meta.touch1_liveness  : ternary; // forces touch1 live w/o cross-stage ALU folding
            meta.touch2_liveness  : ternary; // forces touch2 live w/o cross-stage ALU folding
        }
        actions = { classify_ddos; NoAction; }
        const default_action = NoAction();
        size = 32;
    }

    apply {
        if (hdr.tcp.isValid()) {
            calc_flow_hash();

            // Four touches on flow_packet_count_reg. Each read's result is
            // kept live by assigning it directly to its own metadata field
            // (used as a wildcard-able extra match key below) rather than
            // combining multiple execute() results with an ALU op in one
            // action -- an earlier attempt to XOR-fold touch1..3 into a
            // single assignment hit p4c's "action spanning multiple stages"
            // restriction (results from different RegisterActions cannot be
            // combined by non-trivial logic within one action). Keeping each
            // touch's result in its own field sidesteps that without
            // eliminating any of the four touches.
            meta.touch1_liveness = count_read_touch1.execute(meta.flow_hash);
            meta.touch2_liveness = count_read_touch2.execute(meta.flow_hash);
            count_write_touch.execute(meta.flow_hash);
            meta.flow_packet_count = count_read_touch3.execute(meta.flow_hash);

            meta.flow_packet_length_total = total_action.execute(meta.flow_hash);
            meta.flow_packet_length_max = max_action.execute(meta.flow_hash);

            table_0_flow_packet_count.apply();
            table_1_flow_packet_length_total.apply();
            table_2_flow_packet_length_max.apply();
            get_classification_tree_ddos_0.apply();
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
