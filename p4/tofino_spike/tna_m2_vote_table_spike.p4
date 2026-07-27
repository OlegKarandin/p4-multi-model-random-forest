/*
 * M2 optimization spike: replace the 27-branch (3^3) majority-vote if-cascade
 * (generate_voting_code's real output, already validated to compile cheaply
 * in tna_m2_numtrees3_spike.p4 -- 27 independent top-level ifs packed into
 * ~2 gateway-heavy stages) with a single small exact-match table whose
 * entries are the same 27 (tree0,tree1,tree2) -> winner mappings, computed
 * at Python/generation time (this file hand-embeds them via `const entries`,
 * mirroring how build_p4_script.py would need to emit them).
 *
 * Motivation: M2's real compile (Part H, reviews/t11_tofino_port_and_env.md)
 * showed Gateway usage more-than-tripling (11 -> 37) versus M1, attributed to
 * this exact voting cascade. A user asked whether trading Gateway/stage cost
 * for a small Exact-Match table would be cheaper. This spike isolates ONLY
 * the voting mechanism (classify actions/tables are identical to
 * tna_m2_numtrees3_spike.p4) to get an apples-to-apples comparison.
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

struct metadata_t {
    bit<2> class_tree_app_0;
    bit<2> class_tree_app_1;
    bit<2> class_tree_app_2;
    bit<2> classification_app;
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

    // ---- Identical to tna_m2_numtrees3_spike.p4: per-tree dedicated
    // classify actions, no tree param, no conditional. ----

    action classify_flow_codeword_app_0(bit<2> class) {
        meta.class_tree_app_0 = class;
    }
    action classify_flow_codeword_app_1(bit<2> class) {
        meta.class_tree_app_1 = class;
    }
    action classify_flow_codeword_app_2(bit<2> class) {
        meta.class_tree_app_2 = class;
    }

    table get_classification_tree_app_0 {
        key = { hdr.ipv4.total_len : range; }
        actions = { classify_flow_codeword_app_0; NoAction; }
        const default_action = NoAction();
        size = 32;
    }
    table get_classification_tree_app_1 {
        key = { hdr.ipv4.total_len : range; }
        actions = { classify_flow_codeword_app_1; NoAction; }
        const default_action = NoAction();
        size = 32;
    }
    table get_classification_tree_app_2 {
        key = { hdr.ipv4.total_len : range; }
        actions = { classify_flow_codeword_app_2; NoAction; }
        const default_action = NoAction();
        size = 32;
    }

    // ---- NEW: single exact-match voting table replacing the 27-branch
    // if-cascade. Entries computed with Python's statistics.mode over
    // itertools.product(range(3), repeat=3) -- identical tie-breaking
    // semantics to generate_voting_code's existing mode() call, so this
    // is a mechanism change only, not a behavior change. ----

    action set_classification_app(bit<2> winner) {
        meta.classification_app = winner;
    }

    table vote_app {
        key = {
            meta.class_tree_app_0 : exact;
            meta.class_tree_app_1 : exact;
            meta.class_tree_app_2 : exact;
        }
        actions = { set_classification_app; NoAction; }
        const default_action = NoAction();
        size = 32;
        const entries = {
            (0, 0, 0) : set_classification_app(0);
            (0, 0, 1) : set_classification_app(0);
            (0, 0, 2) : set_classification_app(0);
            (0, 1, 0) : set_classification_app(0);
            (0, 1, 1) : set_classification_app(1);
            (0, 1, 2) : set_classification_app(0);
            (0, 2, 0) : set_classification_app(0);
            (0, 2, 1) : set_classification_app(0);
            (0, 2, 2) : set_classification_app(2);
            (1, 0, 0) : set_classification_app(0);
            (1, 0, 1) : set_classification_app(1);
            (1, 0, 2) : set_classification_app(1);
            (1, 1, 0) : set_classification_app(1);
            (1, 1, 1) : set_classification_app(1);
            (1, 1, 2) : set_classification_app(1);
            (1, 2, 0) : set_classification_app(1);
            (1, 2, 1) : set_classification_app(1);
            (1, 2, 2) : set_classification_app(2);
            (2, 0, 0) : set_classification_app(0);
            (2, 0, 1) : set_classification_app(2);
            (2, 0, 2) : set_classification_app(2);
            (2, 1, 0) : set_classification_app(2);
            (2, 1, 1) : set_classification_app(1);
            (2, 1, 2) : set_classification_app(2);
            (2, 2, 0) : set_classification_app(2);
            (2, 2, 1) : set_classification_app(2);
            (2, 2, 2) : set_classification_app(2);
        }
    }

    apply {
        if (hdr.tcp.isValid()) {
            get_classification_tree_app_0.apply();
            get_classification_tree_app_1.apply();
            get_classification_tree_app_2.apply();

            vote_app.apply();
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
