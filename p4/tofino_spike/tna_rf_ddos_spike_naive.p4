/*
 * T11 Stage-1 spike, "naive" variant (see reviews/t11_tofino_port_and_env.md
 * Part D.4, "Recommended next experiment").
 *
 * Same 1-tree/1-task/3-feature shape as tna_rf_ddos_spike.p4, but each
 * register is touched TWICE per packet instead of once: a write-only
 * RegisterAction (increment/accumulate/max, return discarded) followed by a
 * separate read-only RegisterAction (bulk-read phase). This mirrors the
 * structural pattern actually used in p4/p4_code_RF_models.p4, where feature
 * registers are updated incrementally through the apply block and then
 * re-read in bulk by update_current_flow_features (:293-325), called last
 * at :761 -- i.e. write and read are two separate touches, not one atomic
 * read-modify-write.
 *
 * Purpose: quantify Tier-1 fix #1 ("collapse every register's multiple
 * touches into one atomic RegisterAction.execute() per packet ... instead of
 * re-reading") and fix #3 ("merge the bulk-read into the write-side logic so
 * each register is visited once per packet, total") from Part D.3. Compare
 * this file's stage count against tna_rf_ddos_spike.p4's already-consolidated
 * 7 stages to get a real before/after number, per Part D.4.
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

    // ---- Feature registers: TWO touches per packet per array -- a
    // write-only RegisterAction (mirrors the original's separate .write()
    // call during incremental flow tracking) and a read-only RegisterAction
    // (mirrors the bulk .read() in update_current_flow_features, called
    // last). This is the "naive" structural pattern this file exists to
    // measure, deliberately NOT collapsed into one atomic RMW. ----

    Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) flow_packet_count_reg;
    RegisterAction<bit<16>, bit<32>, bit<16>>(flow_packet_count_reg) count_write_action = {
        void apply(inout bit<16> value, out bit<16> rv) {
            value = value + 1;
            rv = value;
        }
    };
    RegisterAction<bit<16>, bit<32>, bit<16>>(flow_packet_count_reg) count_read_action = {
        void apply(inout bit<16> value, out bit<16> rv) {
            rv = value;
        }
    };

    Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) flow_packet_length_total_reg;
    RegisterAction<bit<16>, bit<32>, bit<16>>(flow_packet_length_total_reg) total_write_action = {
        void apply(inout bit<16> value, out bit<16> rv) {
            value = value + hdr.ipv4.total_len;
            rv = value;
        }
    };
    RegisterAction<bit<16>, bit<32>, bit<16>>(flow_packet_length_total_reg) total_read_action = {
        void apply(inout bit<16> value, out bit<16> rv) {
            rv = value;
        }
    };

    Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) flow_packet_length_max_reg;
    RegisterAction<bit<16>, bit<32>, bit<16>>(flow_packet_length_max_reg) max_write_action = {
        void apply(inout bit<16> value, out bit<16> rv) {
            if (hdr.ipv4.total_len > value) {
                value = hdr.ipv4.total_len;
            }
            rv = value;
        }
    };
    RegisterAction<bit<16>, bit<32>, bit<16>>(flow_packet_length_max_reg) max_read_action = {
        void apply(inout bit<16> value, out bit<16> rv) {
            rv = value;
        }
    };

    // ---- Feature-interval encoding: range-match tables, one per feature,
    // each writing a 2-bit slice of the codeword. Unchanged from the
    // consolidated spike. ----

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
    // (num_trees_ddos = 1). Unchanged from the consolidated spike. ----

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

            // ---- Write phase: update each register, discarding the RMW's
            // own return value -- mirroring the original's separate
            // .write() call in the apply block. ----
            count_write_action.execute(meta.flow_hash);
            total_write_action.execute(meta.flow_hash);
            max_write_action.execute(meta.flow_hash);

            // ---- Bulk-read phase: re-read every register into metadata --
            // mirroring update_current_flow_features (p4_code_RF_models.p4
            // :293-325), called last at :761. ----
            meta.flow_packet_count = count_read_action.execute(meta.flow_hash);
            meta.flow_packet_length_total = total_read_action.execute(meta.flow_hash);
            meta.flow_packet_length_max = max_read_action.execute(meta.flow_hash);

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
