/*
 * M1 design spike: `flows` fwd/bwd bookkeeping (M1-0) + IAT timestamp
 * rescale (ns -> pseudo-us), using the REAL M1 DDoS-only feature set
 * (flow_iat_max, fwd_iat_max, fwd_packet_length_max) instead of the
 * Part C-F toy totals/counts.
 *
 * Registers needed, traced from p4/p4_code_RF_models.p4's apply block for
 * exactly these 3 features (flow_duration/flow_packet_count/mean/EWMA
 * machinery all pruned -- none of those are selected features or their
 * dependencies for this feature set):
 *   flows_reg               (flow existence/direction bookkeeping, M1-0)
 *   flow_last_arrival_time  (needed by flow_iat_max)
 *   flow_iat_max
 *   fwd_last_arrival_time   (needed by fwd_iat_max, gated fwd==1)
 *   fwd_iat_max                                      (gated fwd==1)
 *   fwd_packet_length_max                             (gated fwd==1)
 *
 * ---- M1-0: flows fwd/bwd design ----
 * The plan's candidate (b) as literally worded ("a fwd-hash test-and-set
 * plus a conditional bwd-hash test-only") was hand-traced before writing
 * this file and found to have a real correctness bug: unconditionally
 * test-and-setting the packet's OWN hash first, before checking the
 * opposite hash, writes a stray 1 into the *reverse* direction's own slot
 * on the first reverse-direction packet of a flow. From the *second*
 * reverse packet onward, that stray bit makes the own-hash test-and-set
 * read back 1 and misclassify the packet as "forward" indexed at the wrong
 * (non-canonical) register slot -- corrupting fwd_iat_max/
 * fwd_packet_length_max for any flow with more than one reverse packet
 * (i.e. most real flows).
 *
 * The fix is an order swap that is still "at most 2 touches, at most 1
 * conditional": always read-only test the OTHER direction's hash first
 * (cheap, no register write is possible while the true state is still
 * unknown); only if that comes back unset do we touch our OWN hash's slot,
 * and only ever write there (the canonical slot, matching the original
 * v1model code's invariant that a flow's canonical index is always the
 * hash of whichever direction sent the first packet). This never writes to
 * a non-canonical slot, so it has no equivalent bug. Hand-verified correct
 * across a 4-packet forward/reverse/reverse/reverse trace before writing
 * this file (see reviews/t11_tofino_port_and_env.md Part G).
 *
 * ---- Timestamp rescale ----
 * ig_prsr_md.global_tstamp is 48-bit *nanoseconds* (confirmed from the
 * installed SDE headers); the original v1model code used
 * standard_metadata.ingress_global_timestamp[15:0], 16 bits of
 * *microseconds*. TNA stateful/action ALUs cannot synthesize a divide by
 * 1000 (non-power-of-2), only shifts -- confirmed empirically by this
 * file's own compile (see Part G). This spike uses `>> 10` (divide by
 * 1024) as a synthesizable pseudo-microsecond approximation (~2.4% off
 * true us), truncated to the low 16 bits to preserve the same wraparound
 * period as the original code. This is a resource-oracle scale
 * substitution only -- it changes the numeric meaning of IAT thresholds
 * and would need retraining/re-validation before any accuracy claim, out
 * of scope for M1.
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
    bit<32> flow_hash_self;   // this packet's own (src,dst)-ordered hash
    bit<32> flow_hash_other;  // this packet's swapped-order hash
    bit<32> flow_hash;        // canonical index used for all feature regs
    bit<1>  fwd;               // 1 if this packet is on the flow's canonical (forward) direction
    bit<16> now_pseudo_us;
    bit<16> current_iat;
    bit<16> flow_iat_max_val;
    bit<16> fwd_iat_max_val;
    bit<8>  code_flow_iat_max;         // Tier-3: own PHV container per feature
    bit<8>  code_fwd_iat_max;
    bit<8>  code_fwd_packet_length_max;
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

    // TNA requires one Hash<> instance per distinct field list -- a single
    // instance's .get() cannot be called twice with different field orders
    // (confirmed empirically: "Dynamic hashes must have the same field list
    // ... for each get call"). v1model's hash() primitive allowed this
    // (two actions calling the same stateless primitive); TNA's Hash<> is a
    // fixed-configuration object, so this needs two separate instances.
    Hash<bit<32>>(HashAlgorithm_t.CRC32) flow_hash_calc_self;
    Hash<bit<32>>(HashAlgorithm_t.CRC32) flow_hash_calc_other;

    // Each hash.get() needs its own action/table -- confirmed empirically:
    // combining both calls in one action overflows Tofino's 32-bit
    // immediate-data pathway (each hash result needs its own 32 bits, and
    // the two can't share one table's immediate-data budget).
    action calc_flow_hash_self() {
        meta.flow_hash_self = flow_hash_calc_self.get({
            hdr.ipv4.src_addr,
            hdr.ipv4.dst_addr,
            hdr.ipv4.protocol,
            hdr.tcp.src_port,
            hdr.tcp.dst_port
        }) & 0xFFF; // MAX_NUM_FLOWS = 4096 = 2^12
    }

    action calc_flow_hash_other() {
        meta.flow_hash_other = flow_hash_calc_other.get({
            hdr.ipv4.dst_addr,
            hdr.ipv4.src_addr,
            hdr.ipv4.protocol,
            hdr.tcp.dst_port,
            hdr.tcp.src_port
        }) & 0xFFF;
    }

    action calc_timestamp() {
        meta.now_pseudo_us = (bit<16>)(ig_prsr_md.global_tstamp >> 10);
    }

    // ---- M1-0: flows bookkeeping, corrected 2-touch design ----

    Register<bit<1>, bit<32>>(MAX_NUM_FLOWS) flows_reg;

    // Touch 1 (always): read-only test of the OTHER direction's slot.
    RegisterAction<bit<1>, bit<32>, bit<1>>(flows_reg) flows_test_other = {
        void apply(inout bit<1> value, out bit<1> rv) {
            rv = value;
        }
    };

    // Touch 2 (conditional, only when touch 1 read 0): unconditionally mark
    // OUR OWN slot as canonical -- never touches the other slot, so it can
    // never corrupt the opposite direction's classification.
    RegisterAction<bit<1>, bit<32>, bit<1>>(flows_reg) flows_set_self = {
        void apply(inout bit<1> value, out bit<1> rv) {
            value = 1;
            rv = value;
        }
    };

    // ---- Feature registers: one atomic RegisterAction per register,
    // combining the read-modify-write (Tier 1). ----

    Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) flow_last_arrival_time_reg;
    RegisterAction<bit<16>, bit<32>, bit<16>>(flow_last_arrival_time_reg) flow_iat_action = {
        void apply(inout bit<16> value, out bit<16> rv) {
            rv = meta.now_pseudo_us - value;
            value = meta.now_pseudo_us;
        }
    };

    Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) flow_iat_max_reg;
    RegisterAction<bit<16>, bit<32>, bit<16>>(flow_iat_max_reg) flow_iat_max_action = {
        void apply(inout bit<16> value, out bit<16> rv) {
            if (meta.current_iat > value) {
                value = meta.current_iat;
            }
            rv = value;
        }
    };

    Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) fwd_last_arrival_time_reg;
    RegisterAction<bit<16>, bit<32>, bit<16>>(fwd_last_arrival_time_reg) fwd_iat_action = {
        void apply(inout bit<16> value, out bit<16> rv) {
            rv = meta.now_pseudo_us - value;
            value = meta.now_pseudo_us;
        }
    };

    Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) fwd_iat_max_reg;
    RegisterAction<bit<16>, bit<32>, bit<16>>(fwd_iat_max_reg) fwd_iat_max_action = {
        void apply(inout bit<16> value, out bit<16> rv) {
            if (meta.current_iat > value) {
                value = meta.current_iat;
            }
            rv = value;
        }
    };

    Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) fwd_packet_length_max_reg;
    RegisterAction<bit<16>, bit<32>, bit<16>>(fwd_packet_length_max_reg) fwd_packet_length_max_action = {
        void apply(inout bit<16> value, out bit<16> rv) {
            if (hdr.ipv4.total_len > value) {
                value = hdr.ipv4.total_len;
            }
            rv = value;
        }
    };

    // ---- Tier-3 feature-interval encoding: own PHV container per feature ----

    action set_code_flow_iat_max(bit<8> code) {
        meta.code_flow_iat_max = code;
    }

    table table_0_flow_iat_max {
        key = { meta.flow_iat_max_val : range; }
        actions = { set_code_flow_iat_max; NoAction; }
        const default_action = NoAction();
        size = 16;
    }

    action set_code_fwd_iat_max(bit<8> code) {
        meta.code_fwd_iat_max = code;
    }

    table table_1_fwd_iat_max {
        key = { meta.fwd_iat_max_val : range; }
        actions = { set_code_fwd_iat_max; NoAction; }
        const default_action = NoAction();
        size = 16;
    }

    action set_code_fwd_packet_length_max(bit<8> code) {
        meta.code_fwd_packet_length_max = code;
    }

    table table_2_fwd_packet_length_max {
        key = { hdr.ipv4.total_len : range; }
        actions = { set_code_fwd_packet_length_max; NoAction; }
        const default_action = NoAction();
        size = 16;
    }

    action classify_ddos(bit<1> class) {
        meta.ddos_class = class;
    }

    table get_classification_tree_ddos_0 {
        key = {
            meta.code_flow_iat_max             : ternary;
            meta.code_fwd_iat_max              : ternary;
            meta.code_fwd_packet_length_max    : ternary;
        }
        actions = { classify_ddos; NoAction; }
        const default_action = NoAction();
        size = 32;
    }

    apply {
        if (hdr.tcp.isValid()) {
            calc_flow_hash_self();
            calc_flow_hash_other();
            calc_timestamp();

            bit<1> other_seen = flows_test_other.execute(meta.flow_hash_other);

            if (other_seen == 1) {
                meta.fwd = 0;
                meta.flow_hash = meta.flow_hash_other;
            } else {
                flows_set_self.execute(meta.flow_hash_self);
                meta.fwd = 1;
                meta.flow_hash = meta.flow_hash_self;
            }

            meta.current_iat = flow_iat_action.execute(meta.flow_hash);
            meta.flow_iat_max_val = flow_iat_max_action.execute(meta.flow_hash);

            if (meta.fwd == 1) {
                meta.current_iat = fwd_iat_action.execute(meta.flow_hash);
                meta.fwd_iat_max_val = fwd_iat_max_action.execute(meta.flow_hash);
                fwd_packet_length_max_action.execute(meta.flow_hash);
            }

            table_0_flow_iat_max.apply();
            table_1_fwd_iat_max.apply();
            table_2_fwd_packet_length_max.apply();
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
