/* -*- P4_16 -*- */
#include <core.p4>
// Tofino-1 TNA only: the resource model's constants (p4model/target.py) and the
// export widths (program.VOTE_EXPORT_BITS) are Tofino-1's.
#include <tna.p4>

#include "p4_headers.p4"
#include "p4_util.p4"

const bit<32> MAX_NUM_FLOWS = 4096;

@pa_container_size("ingress", "ig_md.bwd_iat_total_val", 16)
@pa_container_size("ingress", "ig_md.fwd_iat_min_val", 16)
@pa_container_size("ingress", "ig_md.min_packet_length_val", 16)
@pa_container_size("ingress", "ig_md.packet_length_total_val", 16)
@pa_container_size("ingress", "ig_md.bwd_iat_min_val", 16)
@pa_container_size("ingress", "ig_md.fwd_iat_total_val", 16)
@pa_no_overlay("ingress", "ig_md.code_bwd_iat_total")
@pa_container_size("ingress", "ig_md.code_bwd_iat_total", 32)
@pa_no_overlay("ingress", "ig_md.code_app_fwd_iat_min")
@pa_container_size("ingress", "ig_md.code_app_fwd_iat_min", 8, 32)
@pa_no_overlay("ingress", "ig_md.code_min_packet_length")
@pa_container_size("ingress", "ig_md.code_min_packet_length", 8)
@pa_no_overlay("ingress", "ig_md.code_app_packet_length_total")
@pa_container_size("ingress", "ig_md.code_app_packet_length_total", 32, 32)
@pa_no_overlay("ingress", "ig_md.code_bwd_iat_min")
@pa_container_size("ingress", "ig_md.code_bwd_iat_min", 8)
@pa_no_overlay("ingress", "ig_md.code_ddos_fwd_iat_min")
@pa_container_size("ingress", "ig_md.code_ddos_fwd_iat_min", 16)
@pa_no_overlay("ingress", "ig_md.code_fwd_iat_total")
@pa_container_size("ingress", "ig_md.code_fwd_iat_total", 8)
@pa_no_overlay("ingress", "ig_md.code_ddos_packet_length_total")
@pa_container_size("ingress", "ig_md.code_ddos_packet_length_total", 8)
@pa_container_size("ingress", "ig_md.fwd_current_iat", 16)
@pa_container_size("ingress", "ig_md.bwd_current_iat", 16)
@pa_solitary("ingress", "ig_md.now_pseudo_us")
@pa_solitary("ingress", "ig_md.now_coarse")
@pa_solitary("ingress", "ig_md.flow_hash")
struct metadata_t {
    bit<12> flow_hash;
    bit<1>  fwd;
    bit<32> now_pseudo_us;

	bit<32> now_coarse;
	bit<16> fwd_current_iat;
	bit<16> bwd_current_iat;
	bit<16> bwd_iat_total_val;
	bit<16> fwd_iat_min_val;
	bit<16> min_packet_length_val;
	bit<16> packet_length_total_val;
	bit<16> bwd_iat_min_val;
	bit<16> fwd_iat_total_val;
	bit<32> code_bwd_iat_total;
	bit<40> code_app_fwd_iat_min;
	bit<8> code_min_packet_length;
	bit<64> code_app_packet_length_total;
	bit<8> code_bwd_iat_min;
	bit<16> code_ddos_fwd_iat_min;
	bit<8> code_fwd_iat_total;
	bit<8> code_ddos_packet_length_total;

}

struct dur_pair_t {
    bit<32> first;
    bit<32> dur;
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

	@symmetric("hdr.ipv4.src_addr", "hdr.ipv4.dst_addr")
	@symmetric("hdr.tcp.src_port", "hdr.tcp.dst_port")
	Hash<bit<12>>(HashAlgorithm_t.CRC32) flow_hash_calc;
	Register<bit<32>, bit<12>>(MAX_NUM_FLOWS) flow_forward_srcaddr_reg;
	Register<dur_pair_t, bit<12>>(MAX_NUM_FLOWS, {32w0xFFFFFFFF, 32w0}) bwd_iat_total_reg;
	Register<bit<32>, bit<12>>(MAX_NUM_FLOWS, 32w0xFFFFFFFF) fwd_last_arrival_time_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS, 65535) fwd_iat_min_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS, 65535) min_packet_length_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) packet_length_total_reg;
	Register<bit<32>, bit<12>>(MAX_NUM_FLOWS, 32w0xFFFFFFFF) bwd_last_arrival_time_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS, 65535) bwd_iat_min_reg;
	Register<dur_pair_t, bit<12>>(MAX_NUM_FLOWS, {32w0xFFFFFFFF, 32w0}) fwd_iat_total_reg;


	action calc_flow_hash() {
		meta.flow_hash = flow_hash_calc.get({
			hdr.ipv4.src_addr,
			hdr.ipv4.dst_addr,
			hdr.ipv4.protocol,
			hdr.tcp.src_port,
			hdr.tcp.dst_port
		});
	}

	action calc_timestamp() {
		meta.now_pseudo_us = ig_prsr_md.global_tstamp[41:10];
		meta.now_coarse = ig_intr_md.ingress_mac_tstamp[47:16];
	}

	RegisterAction<bit<32>, bit<12>, bit<1>>(flow_forward_srcaddr_reg) flow_orientation_action = {
		void apply(inout bit<32> value, out bit<1> rv) {
			if (value == 0 || value == hdr.ipv4.src_addr) {
				value = hdr.ipv4.src_addr;
				rv = 1w1;
			} else {
				rv = 1w0;
			}
		}
	};

	RegisterAction<dur_pair_t, bit<12>, bit<32>>(bwd_iat_total_reg) bwd_iat_total_action = {
		void apply(inout dur_pair_t value, out bit<32> rv) {
			if (meta.fwd == 0) {
				if (meta.now_coarse - value.first > 65535) {
					value.dur = 65535;
				} else {
					value.dur = meta.now_coarse |-| value.first;
				}
				value.first = min(value.first, meta.now_coarse);
			}
			rv = value.dur;
		}
	};

	RegisterAction<bit<32>, bit<12>, bit<32>>(fwd_last_arrival_time_reg) fwd_last_arrival_time_action = {
		void apply(inout bit<32> value, out bit<32> rv) {
			if (meta.fwd == 1) {
				if (meta.now_pseudo_us - value > 65535) {
					rv = 65535;
				} else {
					rv = meta.now_pseudo_us |-| value;
				}
				value = meta.now_pseudo_us - 1;
			} else {
				rv = 0;
			}
		}
	};

	RegisterAction<bit<16>, bit<12>, bit<16>>(fwd_iat_min_reg) fwd_iat_min_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd_current_iat != 0 && meta.fwd_current_iat <= value) {
				value = meta.fwd_current_iat - 1;
			}
			rv = value;
		}
	};

	RegisterAction<bit<16>, bit<12>, bit<16>>(min_packet_length_reg) min_packet_length_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (hdr.ipv4.total_len < value) {
				value = hdr.ipv4.total_len;
			}
			rv = value;
		}
	};

	RegisterAction<bit<16>, bit<12>, bit<16>>(packet_length_total_reg) packet_length_total_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			value = value |+| hdr.ipv4.total_len;
			rv = value;
		}
	};

	RegisterAction<bit<32>, bit<12>, bit<32>>(bwd_last_arrival_time_reg) bwd_last_arrival_time_action = {
		void apply(inout bit<32> value, out bit<32> rv) {
			if (meta.fwd == 0) {
				if (meta.now_pseudo_us - value > 65535) {
					rv = 65535;
				} else {
					rv = meta.now_pseudo_us |-| value;
				}
				value = meta.now_pseudo_us - 1;
			} else {
				rv = 0;
			}
		}
	};

	RegisterAction<bit<16>, bit<12>, bit<16>>(bwd_iat_min_reg) bwd_iat_min_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.bwd_current_iat != 0 && meta.bwd_current_iat <= value) {
				value = meta.bwd_current_iat - 1;
			}
			rv = value;
		}
	};

	RegisterAction<dur_pair_t, bit<12>, bit<32>>(fwd_iat_total_reg) fwd_iat_total_action = {
		void apply(inout dur_pair_t value, out bit<32> rv) {
			if (meta.fwd == 1) {
				if (meta.now_coarse - value.first > 65535) {
					value.dur = 65535;
				} else {
					value.dur = meta.now_coarse |-| value.first;
				}
				value.first = min(value.first, meta.now_coarse);
			}
			rv = value.dur;
		}
	};


	action classify_flow_codeword_app_0(bit<2> class){
		ig_tm_md.qid = (bit<5>) class;
	}

	action classify_flow_codeword_ddos_0(bit<1> class){
		ig_tm_md.ingress_cos = (bit<3>) class;
	}

    
    action set_code_bwd_iat_total (bit<32> code) {
        meta.code_bwd_iat_total = code;
    }
    
    action set_code_app_fwd_iat_min (bit<40> code) {
        meta.code_app_fwd_iat_min = code;
    }
    
    action set_code_min_packet_length (bit<8> code) {
        meta.code_min_packet_length = code;
    }
    
    action set_code_app_packet_length_total (bit<64> code) {
        meta.code_app_packet_length_total = code;
    }
    
    action set_code_bwd_iat_min (bit<8> code) {
        meta.code_bwd_iat_min = code;
    }
    
    action set_code_ddos_fwd_iat_min (bit<16> code) {
        meta.code_ddos_fwd_iat_min = code;
    }
    
    action set_code_fwd_iat_total (bit<8> code) {
        meta.code_fwd_iat_total = code;
    }
    
    action set_code_ddos_packet_length_total (bit<8> code) {
        meta.code_ddos_packet_length_total = code;
    }



    @placement_priority(1)
    table get_classification_tree_app_0 {
        key = {
            meta.code_bwd_iat_total[29:0] : ternary @name("code_bwd_iat_total_29_0_");
            meta.code_app_fwd_iat_min[32:0] : ternary @name("code_app_fwd_iat_min_32_0_");
            meta.code_min_packet_length[0:0] : ternary @name("code_min_packet_length_0_0_");
            meta.code_app_packet_length_total[60:0] : ternary @name("code_app_packet_length_total_60_0_");
        }
        actions = {
            classify_flow_codeword_app_0;        
        }
        size = 136;
    }
    
    @placement_priority(2)
    table get_classification_tree_ddos_0 {
        key = {
            meta.code_bwd_iat_min[4:0] : ternary @name("code_bwd_iat_min_4_0_");
            meta.code_ddos_fwd_iat_min[9:0] : ternary @name("code_ddos_fwd_iat_min_9_0_");
            meta.code_fwd_iat_total[1:0] : ternary @name("code_fwd_iat_total_1_0_");
            meta.code_ddos_packet_length_total[5:0] : ternary @name("code_ddos_packet_length_total_5_0_");
        }
        actions = {
            classify_flow_codeword_ddos_0;        
        }
        size = 24;
    }
    
    @placement_priority(3)
    table table_0_bwd_iat_total {
        key = {
            meta.bwd_iat_total_val: range;
        }
        actions = {
            set_code_bwd_iat_total;        
        }
        size = 31;
    }   
    
    @placement_priority(3)
    table table_1_app_fwd_iat_min {
        key = {
            meta.fwd_iat_min_val: range;
        }
        actions = {
            set_code_app_fwd_iat_min;        
        }
        size = 34;
    }   
    
    @placement_priority(3)
    table table_2_min_packet_length {
        key = {
            meta.min_packet_length_val: range;
        }
        actions = {
            set_code_min_packet_length;        
        }
        size = 2;
    }   
    
    @placement_priority(3)
    table table_3_app_packet_length_total {
        key = {
            meta.packet_length_total_val: range;
        }
        actions = {
            set_code_app_packet_length_total;        
        }
        size = 62;
    }   
    
    @placement_priority(3)
    table table_4_bwd_iat_min {
        key = {
            meta.bwd_iat_min_val: range;
        }
        actions = {
            set_code_bwd_iat_min;        
        }
        size = 6;
    }   
    
    @placement_priority(3)
    table table_5_ddos_fwd_iat_min {
        key = {
            meta.fwd_iat_min_val: range;
        }
        actions = {
            set_code_ddos_fwd_iat_min;        
        }
        size = 11;
    }   
    
    @placement_priority(3)
    table table_6_fwd_iat_total {
        key = {
            meta.fwd_iat_total_val: range;
        }
        actions = {
            set_code_fwd_iat_total;        
        }
        size = 3;
    }   
    
    @placement_priority(3)
    table table_7_ddos_packet_length_total {
        key = {
            meta.packet_length_total_val: range;
        }
        actions = {
            set_code_ddos_packet_length_total;        
        }
        size = 7;
    }   
    

    apply {
        // First, not last: the vote actions (and a one-tree task's tree action) write
        // ig_tm_md (qid, ingress_cos), so p4c orders a trailing assignment to ig_tm_md
        // after them, in a stage of its own.
        ig_tm_md.ucast_egress_port = ig_intr_md.ingress_port;
        ig_tm_md.bypass_egress = 1w1;

        if (hdr.tcp.isValid()) {

			calc_flow_hash();
			calc_timestamp();

			meta.fwd = flow_orientation_action.execute(meta.flow_hash);

			meta.min_packet_length_val = min_packet_length_action.execute(meta.flow_hash);
			meta.packet_length_total_val = packet_length_total_action.execute(meta.flow_hash);

			meta.fwd_current_iat = (bit<16>) fwd_last_arrival_time_action.execute(meta.flow_hash);
			meta.fwd_iat_min_val = fwd_iat_min_action.execute(meta.flow_hash) + 1;
			meta.fwd_iat_total_val = (bit<16>) fwd_iat_total_action.execute(meta.flow_hash);

			meta.bwd_iat_total_val = (bit<16>) bwd_iat_total_action.execute(meta.flow_hash);
			meta.bwd_current_iat = (bit<16>) bwd_last_arrival_time_action.execute(meta.flow_hash);
			meta.bwd_iat_min_val = bwd_iat_min_action.execute(meta.flow_hash) + 1;


			table_0_bwd_iat_total.apply();
			table_1_app_fwd_iat_min.apply();
			table_2_min_packet_length.apply();
			table_3_app_packet_length_total.apply();
			table_4_bwd_iat_min.apply();
			table_5_ddos_fwd_iat_min.apply();
			table_6_fwd_iat_total.apply();
			table_7_ddos_packet_length_total.apply();

			get_classification_tree_app_0.apply();

			get_classification_tree_ddos_0.apply();


        }
    }
}

Pipeline(SwitchIngressParser(),
         SwitchIngress(),
         SwitchIngressDeparser(),
         EmptyEgressParser(),
         EmptyEgress(),
         EmptyEgressDeparser()) pipe;

Switch(pipe) main;
