/*
 * T11 Stage-1 spike, "worst-case touch count" + "Tier-3" combined variant
 * (see reviews/t11_tofino_port_and_env.md Part F).
 *
 * Combines tna_rf_ddos_spike_worstcase.p4's 5-touch flow_packet_count_reg
 * with tna_rf_ddos_spike_tier3.p4's split codeword fields (own PHV
 * container per feature instead of shared bit-slices). Tests whether
 * removing the write-write hazard on the codeword container (Tier 3) frees
 * enough stage-packing freedom to absorb the extra stage pressure the
 * worst-case register touch count would otherwise add -- i.e. whether the
 * two tiers interact rather than being independent.
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
    bit<8>  code_packet_count;         // Tier 3: own container per feature
    bit<8>  code_packet_length_total;
    bit<8>  code_packet_length_max;
    bit<1>  ddos_class;
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

    Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) flow_packet_count_reg;

    RegisterAction<bit<16>, bit<32>, bit<16>>(flow_packet_count_reg) count_read_touch1 = {
        void apply(inout bit<16> value, out bit<16> rv) { rv = value; }
    };
    RegisterAction<bit<16>, bit<32>, bit<16>>(flow_packet_count_reg) count_read_touch2 = {
        void apply(inout bit<16> value, out bit<16> rv) { rv = value; }
    };
    RegisterAction<bit<16>, bit<32>, bit<16>>(flow_packet_count_reg) count_write_touch = {
        void apply(inout bit<16> value, out bit<16> rv) {
            value = value + 1;
            rv = value;
        }
    };
    RegisterAction<bit<16>, bit<32>, bit<16>>(flow_packet_count_reg) count_read_touch3 = {
        void apply(inout bit<16> value, out bit<16> rv) { rv = value; }
    };
    RegisterAction<bit<16>, bit<32>, bit<16>>(flow_packet_count_reg) count_read_touch4 = {
        void apply(inout bit<16> value, out bit<16> rv) { rv = value; }
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

    action set_code_packet_count(bit<8> code) {
        meta.code_packet_count = code;
    }

    table table_0_flow_packet_count {
        key = { meta.flow_packet_count : range; }
        actions = { set_code_packet_count; NoAction; }
        const default_action = NoAction();
        size = 16;
    }

    action set_code_packet_length_total(bit<8> code) {
        meta.code_packet_length_total = code;
    }

    table table_1_flow_packet_length_total {
        key = { meta.flow_packet_length_total : range; }
        actions = { set_code_packet_length_total; NoAction; }
        const default_action = NoAction();
        size = 16;
    }

    action set_code_packet_length_max(bit<8> code) {
        meta.code_packet_length_max = code;
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
            meta.code_packet_count        : ternary;
            meta.code_packet_length_total : ternary;
            meta.code_packet_length_max   : ternary;
        }
        actions = { classify_ddos; NoAction; }
        const default_action = NoAction();
        size = 32;
    }

    apply {
        if (hdr.tcp.isValid()) {
            calc_flow_hash();

            bit<16> t1 = count_read_touch1.execute(meta.flow_hash);
            bit<16> t2 = count_read_touch2.execute(meta.flow_hash);
            count_write_touch.execute(meta.flow_hash);
            bit<16> t3 = count_read_touch3.execute(meta.flow_hash);
            meta.flow_packet_count = count_read_touch4.execute(meta.flow_hash) ^ (t1 ^ t2 ^ t3);

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
