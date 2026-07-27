/*
 * T11 Stage-1 spike (see reviews/t11_tofino_port_and_env.md).
 *
 * Disposable, hand-ported TNA slice of p4/p4_code_RF_models.p4: one DDoS
 * tree, three single-site flow-level features (packet count, packet-length
 * total, packet-length max). These three are deliberately chosen because
 * they are read/written exactly once per packet in the original v1model
 * program; the mean/EWMA/IAT features are excluded because they require the
 * multi-site read-modify-write pattern (up to 5 register touches per packet)
 * that v1model tolerates but a single-stateful-ALU-op-per-packet TNA target
 * does not, without algorithmic changes to merge sites (see change #5 in the
 * v1model -> TNA table in the review doc).
 *
 * Goal: get bf-p4c/p4c-tofino to emit a real resource-allocation report
 * (pipe/logs/resources.json, mau.json, mau.resources.log, table_summary.log,
 * <prog>.bfa) for the register + range + ternary shape evaluation.py
 * estimates analytically, and compare stage/TCAM/SRAM counts against that
 * estimate.
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
}

// ---------------------------------------------------------------------------
// Ingress parser
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Ingress Deparser
// ---------------------------------------------------------------------------
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

    // ---- Flow hash (CRC32 over the 5-tuple, folded to MAX_NUM_FLOWS) ----
    // Mirrors p4_code_RF_models.p4:268-288 (get_flow_hash_fwd), change #6 in
    // the v1model -> TNA table: explicit "& 0xFFF" replaces the v1model
    // hash(..., base, max) modulo argument.
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

    // ---- Feature registers: ONE RegisterAction.execute() per packet per
    // array, folding the original's separate read-then-write (and, for
    // count/total/max, the extra bulk-read in update_current_flow_features)
    // into a single atomic read-modify-write, as TNA requires. ----

    Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) flow_packet_count_reg;
    RegisterAction<bit<16>, bit<32>, bit<16>>(flow_packet_count_reg) count_action = {
        void apply(inout bit<16> value, out bit<16> rv) {
            value = value + 1;
            rv = value;
        }
    };

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

    // ---- Feature-interval encoding: range-match tables, one per feature,
    // each writing a 2-bit slice of the codeword. Mirrors
    // table_0_flow_iat_max / table_1_fwd_iat_max / table_2_fwd_packet_length_max
    // in p4_code_RF_models.p4:407-435 (range match, action writes a codeword
    // slice). ----

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

    // ---- Decision table: ternary match on the codeword, one tree
    // (num_trees_ddos = 1). Mirrors get_classification_tree_ddos_0
    // (p4_code_RF_models.p4:367-405). ----

    action classify_ddos(bit<1> class) {
        meta.ddos_class = class;
    }

    table get_classification_tree_ddos_0 {
        key = { meta.codeword : ternary; }
        actions = { classify_ddos; NoAction; }
        const default_action = NoAction();
        size = 32;
    }

    apply {
        if (hdr.tcp.isValid()) {
            calc_flow_hash();

            meta.flow_packet_count = count_action.execute(meta.flow_hash);
            meta.flow_packet_length_total = total_action.execute(meta.flow_hash);
            meta.flow_packet_length_max = max_action.execute(meta.flow_hash);

            table_0_flow_packet_count.apply();
            table_1_flow_packet_length_total.apply();
            table_2_flow_packet_length_max.apply();
            get_classification_tree_ddos_0.apply();
        }

        // No forwarding logic in this spike (resource-oracle only, per
        // T11 Part B scope reminder) -- send back where it came from.
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
