/*
 * M2 optimization spike: replace the current 2-hash "test other, then set
 * self" flow-direction design (4 stages: 2 Hash<> instances + a 2-touch
 * flows_reg lookup, see tna_m1_flows_iat_spike.p4 and Part G.1) with a
 * single SYMMETRIC/canonical hash (Part G.6's deferred candidate (a)).
 *
 * ---- Design (revised after a real compile error -- see below) ----
 * Original plan: hash over a canonically min/max-ordered field list so both
 * directions hash identically. That needed `if (hdr.ipv4.src_addr <
 * hdr.ipv4.dst_addr)` directly in the apply block to pick the ordering --
 * which failed with a NEW restriction not seen anywhere else in this repo:
 * "condition too complex... one operand must be constant" -- TNA's gateway
 * hardware (which implements bare `if` statements in a control's apply
 * block) can only compare a field against a COMPILE-TIME CONSTANT, not two
 * runtime fields against each other. Confirmed empirically (see git history
 * of this file / the session's design notes for the failed attempt).
 *
 * Fixed by using XOR instead of min/max: hash over
 * {src_addr ^ dst_addr, protocol, src_port ^ dst_port}. XOR is
 * order-independent (XOR(a,b) == XOR(b,a)) by construction, so this needs
 * NO comparison at all -- just a bitwise op, computable in a single plain
 * action (the same class of operation as the existing `& 0xFFF` hash mask
 * already used everywhere in this repo). Only ONE Hash<> instance needed
 * (one field list -- TNA's "one Hash<> per distinct field list" restriction,
 * Part G.1 finding 2, doesn't apply since there's only one ordering now).
 *
 * Caveat: XOR-based symmetric hashing has a real, different collision
 * profile than true min/max canonicalization -- two structurally unrelated
 * flows whose (src,dst) or (sport,dport) pairs happen to XOR to the same
 * value will hash identically even though min/max ordering would have kept
 * them distinct. Not measured here; flagged as a caveat, consistent with
 * this whole project's practice of noting approximations rather than
 * treating them as free.
 *
 * Since the hash itself can no longer distinguish direction, "forward" is
 * redefined structurally: the flow's canonical index now stores the actual
 * srcAddr of whichever packet was seen FIRST for that flow (not a
 * direction bit tied to hash value). A later packet is "fwd" iff its own
 * srcAddr matches the stored one.
 *
 * ---- Correctness, hand-traced BEFORE writing this file (mirroring the
 * rigor M1's flows-design fix used) ----
 * Packet 1 (A->B, first ever packet of this flow): canonical_hash =
 *   H({min(A,B), max(A,B), proto, min(portA,portB), max(portA,portB)}).
 *   Register at canonical_hash reads 0 (assume real IPs are never 0) ->
 *   conditional branch writes value = A (this packet's own srcAddr) ->
 *   rv = (value == A) = true -> fwd = 1. Correct: first packet is fwd.
 * Packet 2 (B->A, reverse direction, same flow): SAME canonical_hash (hash
 *   is symmetric). Register already holds A (from packet 1) -> value != 0,
 *   no write -> rv = (value == B's srcAddr, i.e. B == A) = false -> fwd = 0.
 *   Correct: reverse-direction packet is bwd.
 * Packet 3 (A->B again): SAME canonical_hash, register still holds A ->
 *   rv = (A == A) = true -> fwd = 1. Correct.
 * Packet 4 (B->A again): register still holds A -> rv = (B == A) = false ->
 *   fwd = 0. Correct. No mutation ever happens after the first packet (the
 *   `if (value == 0)` guard only fires once, matching the original design's
 *   "flows_set_self only ever happens on first sight" invariant) -- so this
 *   is stable indefinitely, same as the current 2-hash design's invariant.
 *
 * Caveat this spike does NOT resolve: real IPv4 addresses are never
 * literally 0.0.0.0 in practice for routed traffic, so "value == 0 means
 * unseen" is a safe sentinel for this dataset/deliverable, matching how
 * the existing design already relies on similar real-traffic assumptions
 * (e.g. TCP-only, IPv4-only). Not a general-purpose production
 * consideration, fine for a resource-oracle deliverable.
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
    bit<1>  fwd;
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

    // ---- Step 1: ONE Hash<> instance, one field list, symmetric via XOR
    // (no comparison needed at all -- see file header for why the original
    // min/max-ordering design needed this rework). ----

    Hash<bit<32>>(HashAlgorithm_t.CRC32) flow_hash_calc;

    action calc_flow_hash() {
        meta.flow_hash = flow_hash_calc.get({
            hdr.ipv4.src_addr ^ hdr.ipv4.dst_addr,
            hdr.ipv4.protocol,
            hdr.tcp.src_port ^ hdr.tcp.dst_port
        }) & 0xFFF;
    }

    // ---- Step 3: ONE register, ONE touch, stores the first-seen srcAddr
    // per canonical flow index; returns whether THIS packet's srcAddr
    // matches it (fwd) or not (bwd). ----

    Register<bit<32>, bit<32>>(MAX_NUM_FLOWS) flow_forward_srcaddr_reg;

    RegisterAction<bit<32>, bit<32>, bit<1>>(flow_forward_srcaddr_reg) flow_orientation_action = {
        void apply(inout bit<32> value, out bit<1> rv) {
            if (value == 0) {
                value = hdr.ipv4.src_addr;
            }
            rv = (value == hdr.ipv4.src_addr) ? 1w1 : 1w0;
        }
    };

    apply {
        if (hdr.tcp.isValid()) {
            calc_flow_hash();
            meta.fwd = flow_orientation_action.execute(meta.flow_hash);
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
