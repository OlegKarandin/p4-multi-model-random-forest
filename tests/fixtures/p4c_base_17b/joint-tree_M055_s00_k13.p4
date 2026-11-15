/* -*- P4_16 -*- */
#include <core.p4>
#if __TARGET_TOFINO__ == 2
#include <t2na.p4>
#else
#include <tna.p4>
#endif

#include "p4_headers.p4"
#include "p4_util.p4"

const bit<32> MAX_NUM_FLOWS = 4096;

@pa_solitary("ingress", "ig_md.class_tree_app_0")
@pa_solitary("ingress", "ig_md.class_tree_app_1")
@pa_solitary("ingress", "ig_md.class_tree_app_2")
@pa_solitary("ingress", "ig_md.class_tree_ddos_0")
@pa_solitary("ingress", "ig_md.class_tree_ddos_1")
@pa_solitary("ingress", "ig_md.class_tree_ddos_2")
@pa_container_size("ingress", "ig_md.bwd_iat_max_val", 16)
@pa_container_size("ingress", "ig_md.bwd_iat_mean_val", 16)
@pa_container_size("ingress", "ig_md.bwd_iat_min_val", 16)
@pa_container_size("ingress", "ig_md.bwd_packet_length_max_val", 16)
@pa_container_size("ingress", "ig_md.bwd_packet_length_mean_val", 16)
@pa_container_size("ingress", "ig_md.bwd_packet_length_min_val", 16)
@pa_container_size("ingress", "ig_md.flow_iat_mean_val", 16)
@pa_container_size("ingress", "ig_md.flow_iat_min_val", 16)
@pa_container_size("ingress", "ig_md.fwd_packet_length_max_val", 16)
@pa_container_size("ingress", "ig_md.fwd_packet_length_mean_val", 16)
@pa_container_size("ingress", "ig_md.fwd_packet_length_min_val", 16)
@pa_container_size("ingress", "ig_md.min_packet_length_val", 16)
@pa_container_size("ingress", "ig_md.packet_length_mean_val", 16)
@pa_no_overlay("ingress", "ig_md.code_pk_bwd_iat_max")
@pa_container_size("ingress", "ig_md.code_pk_bwd_iat_max", 16, 8)
@pa_no_overlay("ingress", "ig_md.code_pk_bwd_iat_mean")
@pa_container_size("ingress", "ig_md.code_pk_bwd_iat_mean", 32, 8)
@pa_no_overlay("ingress", "ig_md.code_pk_bwd_iat_min")
@pa_container_size("ingress", "ig_md.code_pk_bwd_iat_min", 32, 8)
@pa_no_overlay("ingress", "ig_md.code_pk_bwd_packet_length_max")
@pa_container_size("ingress", "ig_md.code_pk_bwd_packet_length_max", 32, 32, 16, 8)
@pa_no_overlay("ingress", "ig_md.code_pk_bwd_packet_length_mean")
@pa_container_size("ingress", "ig_md.code_pk_bwd_packet_length_mean", 32, 16, 8)
@pa_no_overlay("ingress", "ig_md.code_pk_bwd_packet_length_min")
@pa_container_size("ingress", "ig_md.code_pk_bwd_packet_length_min", 16)
@pa_no_overlay("ingress", "ig_md.code_pk_flow_iat_mean")
@pa_container_size("ingress", "ig_md.code_pk_flow_iat_mean", 32, 16, 8)
@pa_no_overlay("ingress", "ig_md.code_pk_flow_iat_min")
@pa_container_size("ingress", "ig_md.code_pk_flow_iat_min", 32, 16)
@pa_no_overlay("ingress", "ig_md.code_pk_fwd_packet_length_max")
@pa_container_size("ingress", "ig_md.code_pk_fwd_packet_length_max", 32, 32, 16, 8)
@pa_no_overlay("ingress", "ig_md.code_pk_fwd_packet_length_mean")
@pa_container_size("ingress", "ig_md.code_pk_fwd_packet_length_mean", 32, 16, 8)
@pa_no_overlay("ingress", "ig_md.code_pk_fwd_packet_length_min")
@pa_container_size("ingress", "ig_md.code_pk_fwd_packet_length_min", 16)
@pa_no_overlay("ingress", "ig_md.code_pk_min_packet_length")
@pa_container_size("ingress", "ig_md.code_pk_min_packet_length", 8)
@pa_no_overlay("ingress", "ig_md.code_pk_packet_length_mean")
@pa_container_size("ingress", "ig_md.code_pk_packet_length_mean", 32, 32, 16)
@pa_container_size("ingress", "ig_md.bwd_current_iat", 16)
@pa_container_size("ingress", "ig_md.flow_current_iat", 16)
@pa_solitary("ingress", "ig_md.now_pseudo_us")
struct metadata_t {
    bit<12> flow_hash;
    bit<1>  fwd;
    bit<16> now_pseudo_us;

	bit<16> bwd_current_iat;
	bit<16> flow_current_iat;
	bit<2> class_tree_app_0;
	bit<2> class_tree_app_1;
	bit<2> class_tree_app_2;
	bit<1> class_tree_ddos_0;
	bit<1> class_tree_ddos_1;
	bit<1> class_tree_ddos_2;
	bit<16> bwd_iat_max_val;
	bit<16> bwd_iat_mean_val;
	bit<16> bwd_iat_min_val;
	bit<16> bwd_packet_length_max_val;
	bit<16> bwd_packet_length_mean_val;
	bit<16> bwd_packet_length_min_val;
	bit<16> flow_iat_mean_val;
	bit<16> flow_iat_min_val;
	bit<16> fwd_packet_length_max_val;
	bit<16> fwd_packet_length_mean_val;
	bit<16> fwd_packet_length_min_val;
	bit<16> min_packet_length_val;
	bit<16> packet_length_mean_val;
	bit<24> code_pk_bwd_iat_max;
	bit<40> code_pk_bwd_iat_mean;
	bit<40> code_pk_bwd_iat_min;
	bit<88> code_pk_bwd_packet_length_max;
	bit<56> code_pk_bwd_packet_length_mean;
	bit<16> code_pk_bwd_packet_length_min;
	bit<56> code_pk_flow_iat_mean;
	bit<48> code_pk_flow_iat_min;
	bit<88> code_pk_fwd_packet_length_max;
	bit<56> code_pk_fwd_packet_length_mean;
	bit<16> code_pk_fwd_packet_length_min;
	bit<8> code_pk_min_packet_length;
	bit<80> code_pk_packet_length_mean;

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
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) bwd_last_arrival_time_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) bwd_iat_max_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) bwd_iat_mean_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS, 65535) bwd_iat_min_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) bwd_packet_length_max_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) bwd_packet_length_mean_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS, 65535) bwd_packet_length_min_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) flow_last_arrival_time_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) flow_iat_mean_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS, 65535) flow_iat_min_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) fwd_packet_length_max_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) fwd_packet_length_mean_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS, 65535) fwd_packet_length_min_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS, 65535) min_packet_length_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) packet_length_mean_reg;


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
		meta.now_pseudo_us = (bit<16>)(ig_prsr_md.global_tstamp >> 10);
	}

	RegisterAction<bit<32>, bit<12>, bit<1>>(flow_forward_srcaddr_reg) flow_orientation_action = {
		void apply(inout bit<32> value, out bit<1> rv) {
			if (value == 0) {
				value = hdr.ipv4.src_addr;
			}
			rv = (value == hdr.ipv4.src_addr) ? 1w1 : 1w0;
		}
	};

	RegisterAction<bit<16>, bit<12>, bit<16>>(bwd_last_arrival_time_reg) bwd_last_arrival_time_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 0) {
				rv = meta.now_pseudo_us - value;
				value = meta.now_pseudo_us;
			} else {
				rv = 0;
			}
		}
	};

	RegisterAction<bit<16>, bit<12>, bit<16>>(bwd_iat_max_reg) bwd_iat_max_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 0 && meta.bwd_current_iat > value) {
				value = meta.bwd_current_iat;
			}
			rv = value;
		}
	};

	MathUnit<bit<16>>(MathOp_t.MUL, 1, 2) bwd_iat_mean_halve_unit;

	RegisterAction<bit<16>, bit<12>, bit<16>>(bwd_iat_mean_reg) bwd_iat_mean_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 0) {
				value = bwd_iat_mean_halve_unit.execute(value + meta.bwd_current_iat);
			}
			rv = value;
		}
	};

	RegisterAction<bit<16>, bit<12>, bit<16>>(bwd_iat_min_reg) bwd_iat_min_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 0 && meta.bwd_current_iat < value) {
				value = meta.bwd_current_iat;
			}
			rv = value;
		}
	};

	RegisterAction<bit<16>, bit<12>, bit<16>>(bwd_packet_length_max_reg) bwd_packet_length_max_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 0 && hdr.ipv4.total_len > value) {
				value = hdr.ipv4.total_len;
			}
			rv = value;
		}
	};

	MathUnit<bit<16>>(MathOp_t.MUL, 1, 2) bwd_packet_length_mean_halve_unit;

	RegisterAction<bit<16>, bit<12>, bit<16>>(bwd_packet_length_mean_reg) bwd_packet_length_mean_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 0) {
				value = bwd_packet_length_mean_halve_unit.execute(value + (bit<16>)hdr.ipv4.total_len);
			}
			rv = value;
		}
	};

	RegisterAction<bit<16>, bit<12>, bit<16>>(bwd_packet_length_min_reg) bwd_packet_length_min_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 0 && hdr.ipv4.total_len < value) {
				value = hdr.ipv4.total_len;
			}
			rv = value;
		}
	};

	RegisterAction<bit<16>, bit<12>, bit<16>>(flow_last_arrival_time_reg) flow_last_arrival_time_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			rv = meta.now_pseudo_us - value;
			value = meta.now_pseudo_us;
		}
	};

	MathUnit<bit<16>>(MathOp_t.MUL, 1, 2) flow_iat_mean_halve_unit;

	RegisterAction<bit<16>, bit<12>, bit<16>>(flow_iat_mean_reg) flow_iat_mean_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			value = flow_iat_mean_halve_unit.execute(value + meta.flow_current_iat);
			rv = value;
		}
	};

	RegisterAction<bit<16>, bit<12>, bit<16>>(flow_iat_min_reg) flow_iat_min_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.flow_current_iat < value) {
				value = meta.flow_current_iat;
			}
			rv = value;
		}
	};

	RegisterAction<bit<16>, bit<12>, bit<16>>(fwd_packet_length_max_reg) fwd_packet_length_max_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 1 && hdr.ipv4.total_len > value) {
				value = hdr.ipv4.total_len;
			}
			rv = value;
		}
	};

	MathUnit<bit<16>>(MathOp_t.MUL, 1, 2) fwd_packet_length_mean_halve_unit;

	RegisterAction<bit<16>, bit<12>, bit<16>>(fwd_packet_length_mean_reg) fwd_packet_length_mean_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 1) {
				value = fwd_packet_length_mean_halve_unit.execute(value + (bit<16>)hdr.ipv4.total_len);
			}
			rv = value;
		}
	};

	RegisterAction<bit<16>, bit<12>, bit<16>>(fwd_packet_length_min_reg) fwd_packet_length_min_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 1 && hdr.ipv4.total_len < value) {
				value = hdr.ipv4.total_len;
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

	MathUnit<bit<16>>(MathOp_t.MUL, 1, 2) packet_length_mean_halve_unit;

	RegisterAction<bit<16>, bit<12>, bit<16>>(packet_length_mean_reg) packet_length_mean_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			value = packet_length_mean_halve_unit.execute(value + (bit<16>)hdr.ipv4.total_len);
			rv = value;
		}
	};


	action classify_flow_codeword_app_0(bit<2> class){
		meta.class_tree_app_0 = class;
	}

	action classify_flow_codeword_app_1(bit<2> class){
		meta.class_tree_app_1 = class;
	}

	action classify_flow_codeword_app_2(bit<2> class){
		meta.class_tree_app_2 = class;
	}

	action classify_flow_codeword_ddos_0(bit<1> class){
		meta.class_tree_ddos_0 = class;
	}

	action classify_flow_codeword_ddos_1(bit<1> class){
		meta.class_tree_ddos_1 = class;
	}

	action classify_flow_codeword_ddos_2(bit<1> class){
		meta.class_tree_ddos_2 = class;
	}

    
    action set_code_bwd_iat_max (bit<24> code) {
        meta.code_pk_bwd_iat_max = code;
    }
    
    action set_code_bwd_iat_mean (bit<40> code) {
        meta.code_pk_bwd_iat_mean = code;
    }
    
    action set_code_bwd_iat_min (bit<40> code) {
        meta.code_pk_bwd_iat_min = code;
    }
    
    action set_code_bwd_packet_length_max (bit<88> code) {
        meta.code_pk_bwd_packet_length_max = code;
    }
    
    action set_code_bwd_packet_length_mean (bit<56> code) {
        meta.code_pk_bwd_packet_length_mean = code;
    }
    
    action set_code_bwd_packet_length_min (bit<16> code) {
        meta.code_pk_bwd_packet_length_min = code;
    }
    
    action set_code_flow_iat_mean (bit<56> code) {
        meta.code_pk_flow_iat_mean = code;
    }
    
    action set_code_flow_iat_min (bit<48> code) {
        meta.code_pk_flow_iat_min = code;
    }
    
    action set_code_fwd_packet_length_max (bit<88> code) {
        meta.code_pk_fwd_packet_length_max = code;
    }
    
    action set_code_fwd_packet_length_mean (bit<56> code) {
        meta.code_pk_fwd_packet_length_mean = code;
    }
    
    action set_code_fwd_packet_length_min (bit<16> code) {
        meta.code_pk_fwd_packet_length_min = code;
    }
    
    action set_code_min_packet_length (bit<8> code) {
        meta.code_pk_min_packet_length = code;
    }
    
    action set_code_packet_length_mean (bit<80> code) {
        meta.code_pk_packet_length_mean = code;
    }



    @placement_priority(1)
    table get_classification_tree_app_0 {
        key = {
            meta.code_pk_bwd_iat_max[4:0] : ternary;
            meta.code_pk_bwd_iat_mean[18:8] : ternary;
            meta.code_pk_bwd_iat_min[6:0] : ternary;
            meta.code_pk_bwd_packet_length_max[87:64] : ternary;
            meta.code_pk_bwd_packet_length_mean[8:0] : ternary;
            meta.code_pk_bwd_packet_length_min[4:0] : ternary;
            meta.code_pk_flow_iat_mean[49:40] : ternary;
            meta.code_pk_flow_iat_min[15:9] : ternary;
            meta.code_pk_fwd_packet_length_max[75:64] : ternary;
            meta.code_pk_fwd_packet_length_mean[7:0] : ternary;
            meta.code_pk_fwd_packet_length_min[10:8] : ternary;
            meta.code_pk_min_packet_length[0:0] : ternary;
            meta.code_pk_packet_length_mean[11:0] : ternary;
        }
        actions = {
            classify_flow_codeword_app_0;        
        }
        size = 120;
    }
    
    @placement_priority(1)
    table get_classification_tree_app_1 {
        key = {
            meta.code_pk_bwd_iat_max[14:8] : ternary;
            meta.code_pk_bwd_iat_mean[35:24] : ternary;
            meta.code_pk_bwd_iat_min[18:8] : ternary;
            meta.code_pk_bwd_packet_length_max[24:0] : ternary;
            meta.code_pk_bwd_packet_length_mean[34:16] : ternary;
            meta.code_pk_bwd_packet_length_min[7:5] : ternary;
            meta.code_pk_flow_iat_mean[27:16] : ternary;
            meta.code_pk_flow_iat_min[8:0] : ternary;
            meta.code_pk_fwd_packet_length_max[28:0] : ternary;
            meta.code_pk_fwd_packet_length_mean[18:8] : ternary;
            meta.code_pk_fwd_packet_length_min[14:14] : ternary;
            meta.code_pk_min_packet_length[1:1] : ternary;
            meta.code_pk_packet_length_mean[60:48] : ternary;
        }
        actions = {
            classify_flow_codeword_app_1;        
        }
        size = 156;
    }
    
    @placement_priority(1)
    table get_classification_tree_app_2 {
        key = {
            meta.code_pk_bwd_iat_max[21:16] : ternary;
            meta.code_pk_bwd_iat_mean[23:19] : ternary;
            meta.code_pk_bwd_iat_min[21:19] : ternary;
            meta.code_pk_bwd_packet_length_max[45:32] : ternary;
            meta.code_pk_bwd_packet_length_mean[52:40] : ternary;
            meta.code_pk_bwd_packet_length_min[12:8] : ternary;
            meta.code_pk_flow_iat_mean[9:0] : ternary;
            meta.code_pk_flow_iat_min[23:16] : ternary;
            meta.code_pk_fwd_packet_length_max[59:40] : ternary;
            meta.code_pk_fwd_packet_length_mean[50:32] : ternary;
            meta.code_pk_fwd_packet_length_min[13:11] : ternary;
            meta.code_pk_min_packet_length[2:2] : ternary;
            meta.code_pk_packet_length_mean[41:24] : ternary;
        }
        actions = {
            classify_flow_codeword_app_2;        
        }
        size = 129;
    }
    
    @placement_priority(2)
    table get_classification_tree_ddos_0 {
        key = {
            meta.code_pk_bwd_iat_max[23:22] : ternary;
            meta.code_pk_bwd_iat_mean[39:36] : ternary;
            meta.code_pk_bwd_iat_min[30:24] : ternary;
            meta.code_pk_bwd_packet_length_max[31:25] : ternary;
            meta.code_pk_bwd_packet_length_mean[14:9] : ternary;
            meta.code_pk_flow_iat_mean[55:50] : ternary;
            meta.code_pk_flow_iat_min[30:24] : ternary;
            meta.code_pk_fwd_packet_length_max[86:80] : ternary;
            meta.code_pk_fwd_packet_length_mean[54:51] : ternary;
            meta.code_pk_fwd_packet_length_min[15:15] : ternary;
            meta.code_pk_packet_length_mean[72:64] : ternary;
        }
        actions = {
            classify_flow_codeword_ddos_0;        
        }
        size = 61;
    }
    
    @placement_priority(2)
    table get_classification_tree_ddos_1 {
        key = {
            meta.code_pk_bwd_iat_max[15:15] : ternary;
            meta.code_pk_bwd_iat_mean[1:0] : ternary;
            meta.code_pk_bwd_iat_min[34:32] : ternary;
            meta.code_pk_bwd_packet_length_max[60:58] : ternary;
            meta.code_pk_bwd_packet_length_mean[38:35] : ternary;
            meta.code_pk_bwd_packet_length_min[15:15] : ternary;
            meta.code_pk_flow_iat_mean[38:32] : ternary;
            meta.code_pk_flow_iat_min[45:42] : ternary;
            meta.code_pk_fwd_packet_length_max[37:32] : ternary;
            meta.code_pk_fwd_packet_length_mean[29:24] : ternary;
            meta.code_pk_min_packet_length[3:3] : ternary;
            meta.code_pk_packet_length_mean[23:16] : ternary;
        }
        actions = {
            classify_flow_codeword_ddos_1;        
        }
        size = 49;
    }
    
    @placement_priority(2)
    table get_classification_tree_ddos_2 {
        key = {
            meta.code_pk_bwd_iat_mean[3:2] : ternary;
            meta.code_pk_bwd_iat_min[37:35] : ternary;
            meta.code_pk_bwd_packet_length_max[57:48] : ternary;
            meta.code_pk_bwd_packet_length_mean[54:53] : ternary;
            meta.code_pk_bwd_packet_length_min[14:13] : ternary;
            meta.code_pk_flow_iat_mean[29:28] : ternary;
            meta.code_pk_flow_iat_min[41:32] : ternary;
            meta.code_pk_fwd_packet_length_max[63:60] : ternary;
            meta.code_pk_fwd_packet_length_mean[20:19] : ternary;
            meta.code_pk_fwd_packet_length_min[0:0] : ternary;
            meta.code_pk_min_packet_length[4:4] : ternary;
            meta.code_pk_packet_length_mean[44:42] : ternary;
        }
        actions = {
            classify_flow_codeword_ddos_2;        
        }
        size = 45;
    }
    
    @placement_priority(3)
    table table_0_bwd_iat_max {
        key = {
            meta.bwd_iat_max_val: range;
        }
        actions = {
            set_code_bwd_iat_max;        
        }
        size = 19;
    }   
    
    @placement_priority(3)
    table table_1_bwd_iat_mean {
        key = {
            meta.bwd_iat_mean_val: range;
        }
        actions = {
            set_code_bwd_iat_mean;        
        }
        size = 33;
    }   
    
    @placement_priority(3)
    table table_2_bwd_iat_min {
        key = {
            meta.bwd_iat_min_val: range;
        }
        actions = {
            set_code_bwd_iat_min;        
        }
        size = 30;
    }   
    
    @placement_priority(3)
    table table_3_bwd_packet_length_max {
        key = {
            meta.bwd_packet_length_max_val: range;
        }
        actions = {
            set_code_bwd_packet_length_max;        
        }
        size = 76;
    }   
    
    @placement_priority(3)
    table table_4_bwd_packet_length_mean {
        key = {
            meta.bwd_packet_length_mean_val: range;
        }
        actions = {
            set_code_bwd_packet_length_mean;        
        }
        size = 53;
    }   
    
    @placement_priority(3)
    table table_5_bwd_packet_length_min {
        key = {
            meta.bwd_packet_length_min_val: range;
        }
        actions = {
            set_code_bwd_packet_length_min;        
        }
        size = 11;
    }   
    
    @placement_priority(3)
    table table_6_flow_iat_mean {
        key = {
            meta.flow_iat_mean_val: range;
        }
        actions = {
            set_code_flow_iat_mean;        
        }
        size = 47;
    }   
    
    @placement_priority(3)
    table table_7_flow_iat_min {
        key = {
            meta.flow_iat_min_val: range;
        }
        actions = {
            set_code_flow_iat_min;        
        }
        size = 36;
    }   
    
    @placement_priority(3)
    table table_8_fwd_packet_length_max {
        key = {
            meta.fwd_packet_length_max_val: range;
        }
        actions = {
            set_code_fwd_packet_length_max;        
        }
        size = 72;
    }   
    
    @placement_priority(3)
    table table_9_fwd_packet_length_mean {
        key = {
            meta.fwd_packet_length_mean_val: range;
        }
        actions = {
            set_code_fwd_packet_length_mean;        
        }
        size = 46;
    }   
    
    @placement_priority(3)
    table table_10_fwd_packet_length_min {
        key = {
            meta.fwd_packet_length_min_val: range;
        }
        actions = {
            set_code_fwd_packet_length_min;        
        }
        size = 8;
    }   
    
    @placement_priority(3)
    table table_11_min_packet_length {
        key = {
            meta.min_packet_length_val: range;
        }
        actions = {
            set_code_min_packet_length;        
        }
        size = 3;
    }   
    
    @placement_priority(3)
    table table_12_packet_length_mean {
        key = {
            meta.packet_length_mean_val: range;
        }
        actions = {
            set_code_packet_length_mean;        
        }
        size = 55;
    }   
    	action set_classification_app(bit<2> winner) {
		ig_tm_md.qid = (bit<5>) winner;
	}

	table vote_app {
		key = {
			meta.class_tree_app_0 : exact;
			meta.class_tree_app_1 : exact;
			meta.class_tree_app_2 : exact;
		}
		actions = {
			set_classification_app;
		}
		size = 27;
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
			(1, 0, 2) : set_classification_app(0);
			(1, 1, 0) : set_classification_app(1);
			(1, 1, 1) : set_classification_app(1);
			(1, 1, 2) : set_classification_app(1);
			(1, 2, 0) : set_classification_app(0);
			(1, 2, 1) : set_classification_app(1);
			(1, 2, 2) : set_classification_app(2);
			(2, 0, 0) : set_classification_app(0);
			(2, 0, 1) : set_classification_app(0);
			(2, 0, 2) : set_classification_app(2);
			(2, 1, 0) : set_classification_app(0);
			(2, 1, 1) : set_classification_app(1);
			(2, 1, 2) : set_classification_app(2);
			(2, 2, 0) : set_classification_app(2);
			(2, 2, 1) : set_classification_app(2);
			(2, 2, 2) : set_classification_app(2);
		}
	}
	action set_classification_ddos(bit<1> winner) {
		ig_tm_md.ingress_cos = (bit<3>) winner;
	}

	table vote_ddos {
		key = {
			meta.class_tree_ddos_0 : exact;
			meta.class_tree_ddos_1 : exact;
			meta.class_tree_ddos_2 : exact;
		}
		actions = {
			set_classification_ddos;
		}
		size = 8;
		const entries = {
			(0, 0, 0) : set_classification_ddos(0);
			(0, 0, 1) : set_classification_ddos(0);
			(0, 1, 0) : set_classification_ddos(0);
			(0, 1, 1) : set_classification_ddos(1);
			(1, 0, 0) : set_classification_ddos(0);
			(1, 0, 1) : set_classification_ddos(1);
			(1, 1, 0) : set_classification_ddos(1);
			(1, 1, 1) : set_classification_ddos(1);
		}
	}


    apply {
        // First, not last: the vote actions write ig_tm_md (qid, ingress_cos), so p4c
        // orders a trailing assignment to ig_tm_md after the votes, in a stage of its own.
        ig_tm_md.ucast_egress_port = ig_intr_md.ingress_port;
        ig_tm_md.bypass_egress = 1w1;

        if (hdr.tcp.isValid()) {

			calc_flow_hash();
			calc_timestamp();

			meta.fwd = flow_orientation_action.execute(meta.flow_hash);

			meta.flow_current_iat = flow_last_arrival_time_action.execute(meta.flow_hash);
			meta.flow_iat_mean_val = flow_iat_mean_action.execute(meta.flow_hash);
			meta.flow_iat_min_val = flow_iat_min_action.execute(meta.flow_hash);
			meta.min_packet_length_val = min_packet_length_action.execute(meta.flow_hash);
			meta.packet_length_mean_val = packet_length_mean_action.execute(meta.flow_hash);

			meta.fwd_packet_length_max_val = fwd_packet_length_max_action.execute(meta.flow_hash);
			meta.fwd_packet_length_mean_val = fwd_packet_length_mean_action.execute(meta.flow_hash);
			meta.fwd_packet_length_min_val = fwd_packet_length_min_action.execute(meta.flow_hash);

			meta.bwd_current_iat = bwd_last_arrival_time_action.execute(meta.flow_hash);
			meta.bwd_iat_max_val = bwd_iat_max_action.execute(meta.flow_hash);
			meta.bwd_iat_mean_val = bwd_iat_mean_action.execute(meta.flow_hash);
			meta.bwd_iat_min_val = bwd_iat_min_action.execute(meta.flow_hash);
			meta.bwd_packet_length_max_val = bwd_packet_length_max_action.execute(meta.flow_hash);
			meta.bwd_packet_length_mean_val = bwd_packet_length_mean_action.execute(meta.flow_hash);
			meta.bwd_packet_length_min_val = bwd_packet_length_min_action.execute(meta.flow_hash);


			table_0_bwd_iat_max.apply();
			table_1_bwd_iat_mean.apply();
			table_2_bwd_iat_min.apply();
			table_3_bwd_packet_length_max.apply();
			table_4_bwd_packet_length_mean.apply();
			table_5_bwd_packet_length_min.apply();
			table_6_flow_iat_mean.apply();
			table_7_flow_iat_min.apply();
			table_8_fwd_packet_length_max.apply();
			table_9_fwd_packet_length_mean.apply();
			table_10_fwd_packet_length_min.apply();
			table_11_min_packet_length.apply();
			table_12_packet_length_mean.apply();

			get_classification_tree_app_0.apply();
			get_classification_tree_app_1.apply();
			get_classification_tree_app_2.apply();

			get_classification_tree_ddos_0.apply();
			get_classification_tree_ddos_1.apply();
			get_classification_tree_ddos_2.apply();
			vote_app.apply();
			vote_ddos.apply();




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
