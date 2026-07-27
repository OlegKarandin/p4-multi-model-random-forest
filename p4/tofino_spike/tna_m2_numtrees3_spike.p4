/*
 * M2 design spike (Phase A0): num_trees > 1 classify-action fix + the
 * 3-tree/3-class voting block's stage cost.
 *
 * G.2 (reviews/t11_tofino_port_and_env.md) found that a single shared,
 * tree-parameterized classify action --
 *   action classify_flow_codeword_app(bit<2> tree, bit<2> class) {
 *       if (tree == 0) { meta.class_tree_app_0 = class; }
 *       if (tree == 1) { meta.class_tree_app_1 = class; }
 *       if (tree == 2) { meta.class_tree_app_2 = class; }
 *   }
 * -- fails to compile on TNA (rejected IR::Mux from branching on an action
 * data parameter to decide which of several DIFFERENT metadata fields gets
 * written). M1's fix special-cased num_trees==1 to skip the branch
 * entirely, deliberately leaving num_trees>1 unvalidated.
 *
 * This spike tests the generalizing fix: since build_p4_script.py's
 * get_table_entries()/generate_P4_tables_and_apply() already instantiate
 * ONE PHYSICAL TABLE PER TREE (get_classification_tree_app_0/_1/_2, each
 * with its own entries), there is no actual need for a shared,
 * tree-parameterized action at all -- each tree's table can bind to its OWN
 * dedicated action that unconditionally writes only its own field, exactly
 * like M1's num_trees==1 case, just replicated N times instead of
 * special-cased once. No conditional, no Mux, no branch.
 *
 * Also tests generate_voting_code's real output shape for num_trees=3,
 * num_classes=3 (27 independent top-level `if (...) {...}` blocks, all
 * assigning the SAME single field meta.classification_app) -- flagged in
 * the plan as an unmeasured stage-cost risk ("does the 3-tree/3-class
 * voting block cost stages independent of the register work").
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

    // ---- Per-tree dedicated classify actions: no `tree` parameter, no
    // conditional, each writes only its own field. Stands in for a toy
    // codeword (a single ternary field of hdr.ipv4.total_len here, since
    // this spike isolates the classify-action/voting question -- the real
    // Tier-3 per-feature codeword mechanism is already validated in M1). ----

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

    apply {
        if (hdr.tcp.isValid()) {
            get_classification_tree_app_0.apply();
            get_classification_tree_app_1.apply();
            get_classification_tree_app_2.apply();

            // ---- generate_voting_code(3, 3, "app")'s real shape: 27
            // independent top-level ifs (majority vote over 3 trees x 3
            // classes), all writing the same meta.classification_app. ----
            if ((meta.class_tree_app_0 == 0) && (meta.class_tree_app_1 == 0) && (meta.class_tree_app_2 == 0)) { meta.classification_app = 0; }
            if ((meta.class_tree_app_0 == 0) && (meta.class_tree_app_1 == 0) && (meta.class_tree_app_2 == 1)) { meta.classification_app = 0; }
            if ((meta.class_tree_app_0 == 0) && (meta.class_tree_app_1 == 0) && (meta.class_tree_app_2 == 2)) { meta.classification_app = 0; }
            if ((meta.class_tree_app_0 == 0) && (meta.class_tree_app_1 == 1) && (meta.class_tree_app_2 == 0)) { meta.classification_app = 0; }
            if ((meta.class_tree_app_0 == 0) && (meta.class_tree_app_1 == 1) && (meta.class_tree_app_2 == 1)) { meta.classification_app = 1; }
            if ((meta.class_tree_app_0 == 0) && (meta.class_tree_app_1 == 1) && (meta.class_tree_app_2 == 2)) { meta.classification_app = 1; }
            if ((meta.class_tree_app_0 == 0) && (meta.class_tree_app_1 == 2) && (meta.class_tree_app_2 == 0)) { meta.classification_app = 0; }
            if ((meta.class_tree_app_0 == 0) && (meta.class_tree_app_1 == 2) && (meta.class_tree_app_2 == 1)) { meta.classification_app = 1; }
            if ((meta.class_tree_app_0 == 0) && (meta.class_tree_app_1 == 2) && (meta.class_tree_app_2 == 2)) { meta.classification_app = 2; }
            if ((meta.class_tree_app_0 == 1) && (meta.class_tree_app_1 == 0) && (meta.class_tree_app_2 == 0)) { meta.classification_app = 0; }
            if ((meta.class_tree_app_0 == 1) && (meta.class_tree_app_1 == 0) && (meta.class_tree_app_2 == 1)) { meta.classification_app = 1; }
            if ((meta.class_tree_app_0 == 1) && (meta.class_tree_app_1 == 0) && (meta.class_tree_app_2 == 2)) { meta.classification_app = 1; }
            if ((meta.class_tree_app_0 == 1) && (meta.class_tree_app_1 == 1) && (meta.class_tree_app_2 == 0)) { meta.classification_app = 1; }
            if ((meta.class_tree_app_0 == 1) && (meta.class_tree_app_1 == 1) && (meta.class_tree_app_2 == 1)) { meta.classification_app = 1; }
            if ((meta.class_tree_app_0 == 1) && (meta.class_tree_app_1 == 1) && (meta.class_tree_app_2 == 2)) { meta.classification_app = 1; }
            if ((meta.class_tree_app_0 == 1) && (meta.class_tree_app_1 == 2) && (meta.class_tree_app_2 == 0)) { meta.classification_app = 1; }
            if ((meta.class_tree_app_0 == 1) && (meta.class_tree_app_1 == 2) && (meta.class_tree_app_2 == 1)) { meta.classification_app = 1; }
            if ((meta.class_tree_app_0 == 1) && (meta.class_tree_app_1 == 2) && (meta.class_tree_app_2 == 2)) { meta.classification_app = 2; }
            if ((meta.class_tree_app_0 == 2) && (meta.class_tree_app_1 == 0) && (meta.class_tree_app_2 == 0)) { meta.classification_app = 0; }
            if ((meta.class_tree_app_0 == 2) && (meta.class_tree_app_1 == 0) && (meta.class_tree_app_2 == 1)) { meta.classification_app = 1; }
            if ((meta.class_tree_app_0 == 2) && (meta.class_tree_app_1 == 0) && (meta.class_tree_app_2 == 2)) { meta.classification_app = 2; }
            if ((meta.class_tree_app_0 == 2) && (meta.class_tree_app_1 == 1) && (meta.class_tree_app_2 == 0)) { meta.classification_app = 1; }
            if ((meta.class_tree_app_0 == 2) && (meta.class_tree_app_1 == 1) && (meta.class_tree_app_2 == 1)) { meta.classification_app = 1; }
            if ((meta.class_tree_app_0 == 2) && (meta.class_tree_app_1 == 1) && (meta.class_tree_app_2 == 2)) { meta.classification_app = 2; }
            if ((meta.class_tree_app_0 == 2) && (meta.class_tree_app_1 == 2) && (meta.class_tree_app_2 == 0)) { meta.classification_app = 2; }
            if ((meta.class_tree_app_0 == 2) && (meta.class_tree_app_1 == 2) && (meta.class_tree_app_2 == 1)) { meta.classification_app = 2; }
            if ((meta.class_tree_app_0 == 2) && (meta.class_tree_app_1 == 2) && (meta.class_tree_app_2 == 2)) { meta.classification_app = 2; }
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
