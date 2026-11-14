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

@pa_no_overlay("ingress", "ig_md.fwd")
@pa_solitary("ingress", "ig_md.class_tree_app_0")
@pa_no_overlay("ingress", "ig_md.class_tree_app_0")
@pa_solitary("ingress", "ig_md.class_tree_ddos_0")
@pa_no_overlay("ingress", "ig_md.class_tree_ddos_0")
@pa_solitary("ingress", "ig_md.class_tree_ddos_1")
@pa_no_overlay("ingress", "ig_md.class_tree_ddos_1")
@pa_solitary("ingress", "ig_md.class_tree_ddos_2")
@pa_no_overlay("ingress", "ig_md.class_tree_ddos_2")
@pa_container_size("ingress", "ig_md.bwd_iat_max_val", 16)
@pa_container_size("ingress", "ig_md.bwd_iat_mean_val", 16)
@pa_container_size("ingress", "ig_md.bwd_iat_min_val", 16)
@pa_container_size("ingress", "ig_md.bwd_packet_length_max_val", 16)
@pa_container_size("ingress", "ig_md.bwd_packet_length_mean_val", 16)
@pa_container_size("ingress", "ig_md.bwd_packet_length_min_val", 16)
@pa_container_size("ingress", "ig_md.flow_iat_max_val", 16)
@pa_container_size("ingress", "ig_md.flow_iat_mean_val", 16)
@pa_container_size("ingress", "ig_md.flow_iat_min_val", 16)
@pa_container_size("ingress", "ig_md.fwd_iat_max_val", 16)
@pa_container_size("ingress", "ig_md.fwd_iat_mean_val", 16)
@pa_container_size("ingress", "ig_md.fwd_iat_min_val", 16)
@pa_container_size("ingress", "ig_md.fwd_packet_length_max_val", 16)
@pa_container_size("ingress", "ig_md.fwd_packet_length_mean_val", 16)
@pa_container_size("ingress", "ig_md.fwd_packet_length_min_val", 16)
@pa_container_size("ingress", "ig_md.min_packet_length_val", 16)
@pa_container_size("ingress", "ig_md.packet_length_mean_val", 16)
@pa_no_overlay("ingress", "ig_md.code_pk_bwd_iat_max")
@pa_container_size("ingress", "ig_md.code_pk_bwd_iat_max", 16)
@pa_no_overlay("ingress", "ig_md.code_pk_bwd_iat_mean")
@pa_container_size("ingress", "ig_md.code_pk_bwd_iat_mean", 32)
@pa_no_overlay("ingress", "ig_md.code_pk_bwd_iat_min")
@pa_container_size("ingress", "ig_md.code_pk_bwd_iat_min", 32)
@pa_no_overlay("ingress", "ig_md.code_pk_bwd_packet_length_max")
@pa_container_size("ingress", "ig_md.code_pk_bwd_packet_length_max", 32, 16)
@pa_no_overlay("ingress", "ig_md.code_pk_bwd_packet_length_mean")
@pa_container_size("ingress", "ig_md.code_pk_bwd_packet_length_mean", 32, 8)
@pa_no_overlay("ingress", "ig_md.code_pk_bwd_packet_length_min")
@pa_container_size("ingress", "ig_md.code_pk_bwd_packet_length_min", 16, 8)
@pa_no_overlay("ingress", "ig_md.code_pk_flow_iat_max")
@pa_container_size("ingress", "ig_md.code_pk_flow_iat_max", 32)
@pa_no_overlay("ingress", "ig_md.code_pk_flow_iat_mean")
@pa_container_size("ingress", "ig_md.code_pk_flow_iat_mean", 32)
@pa_no_overlay("ingress", "ig_md.code_pk_flow_iat_min")
@pa_container_size("ingress", "ig_md.code_pk_flow_iat_min", 32)
@pa_no_overlay("ingress", "ig_md.code_pk_fwd_iat_max")
@pa_container_size("ingress", "ig_md.code_pk_fwd_iat_max", 32)
@pa_no_overlay("ingress", "ig_md.code_pk_fwd_iat_mean")
@pa_container_size("ingress", "ig_md.code_pk_fwd_iat_mean", 32)
@pa_no_overlay("ingress", "ig_md.code_pk_fwd_iat_min")
@pa_container_size("ingress", "ig_md.code_pk_fwd_iat_min", 32)
@pa_no_overlay("ingress", "ig_md.code_pk_fwd_packet_length_max")
@pa_container_size("ingress", "ig_md.code_pk_fwd_packet_length_max", 32, 8)
@pa_no_overlay("ingress", "ig_md.code_pk_fwd_packet_length_mean")
@pa_container_size("ingress", "ig_md.code_pk_fwd_packet_length_mean", 32)
@pa_no_overlay("ingress", "ig_md.code_pk_fwd_packet_length_min")
@pa_container_size("ingress", "ig_md.code_pk_fwd_packet_length_min", 32)
@pa_no_overlay("ingress", "ig_md.code_pk_min_packet_length")
@pa_container_size("ingress", "ig_md.code_pk_min_packet_length", 16, 8)
@pa_no_overlay("ingress", "ig_md.code_pk_packet_length_mean")
@pa_container_size("ingress", "ig_md.code_pk_packet_length_mean", 32, 16)
struct metadata_t {
    bit<32> flow_hash;
    bit<1>  fwd;
    bit<16> now_pseudo_us;

	bit<16> bwd_current_iat;
	bit<16> flow_current_iat;
	bit<16> fwd_current_iat;
	bit<2> class_tree_app_0;
	bit<2> classification_app;
	bit<1> class_tree_ddos_0;
	bit<1> class_tree_ddos_1;
	bit<1> class_tree_ddos_2;
	bit<1> classification_ddos;
	bit<16> bwd_iat_max_val;
	bit<16> bwd_iat_mean_val;
	bit<16> bwd_iat_min_val;
	bit<16> bwd_packet_length_max_val;
	bit<16> bwd_packet_length_mean_val;
	bit<16> bwd_packet_length_min_val;
	bit<16> flow_iat_max_val;
	bit<16> flow_iat_mean_val;
	bit<16> flow_iat_min_val;
	bit<16> fwd_iat_max_val;
	bit<16> fwd_iat_mean_val;
	bit<16> fwd_iat_min_val;
	bit<16> fwd_packet_length_max_val;
	bit<16> fwd_packet_length_mean_val;
	bit<16> fwd_packet_length_min_val;
	bit<16> min_packet_length_val;
	bit<16> packet_length_mean_val;
	bit<16> code_pk_bwd_iat_max;
	bit<32> code_pk_bwd_iat_mean;
	bit<32> code_pk_bwd_iat_min;
	bit<48> code_pk_bwd_packet_length_max;
	bit<40> code_pk_bwd_packet_length_mean;
	bit<24> code_pk_bwd_packet_length_min;
	bit<32> code_pk_flow_iat_max;
	bit<32> code_pk_flow_iat_mean;
	bit<32> code_pk_flow_iat_min;
	bit<32> code_pk_fwd_iat_max;
	bit<32> code_pk_fwd_iat_mean;
	bit<32> code_pk_fwd_iat_min;
	bit<40> code_pk_fwd_packet_length_max;
	bit<32> code_pk_fwd_packet_length_mean;
	bit<32> code_pk_fwd_packet_length_min;
	bit<24> code_pk_min_packet_length;
	bit<48> code_pk_packet_length_mean;

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
	Register<bit<32>, bit<32>>(MAX_NUM_FLOWS) flow_forward_srcaddr_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) bwd_last_arrival_time_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) bwd_iat_max_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) bwd_iat_mean_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS, 65535) bwd_iat_min_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) bwd_packet_length_max_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) bwd_packet_length_mean_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS, 65535) bwd_packet_length_min_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) flow_last_arrival_time_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) flow_iat_max_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) flow_iat_mean_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS, 65535) flow_iat_min_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) fwd_last_arrival_time_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) fwd_iat_max_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) fwd_iat_mean_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS, 65535) fwd_iat_min_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) fwd_packet_length_max_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) fwd_packet_length_mean_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS, 65535) fwd_packet_length_min_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS, 65535) min_packet_length_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) packet_length_mean_reg;


	action calc_flow_hash() {
		meta.flow_hash = flow_hash_calc.get({
			hdr.ipv4.src_addr ^ hdr.ipv4.dst_addr,
			hdr.ipv4.protocol,
			hdr.tcp.src_port ^ hdr.tcp.dst_port
		}) & 0xFFF;
	}

	action calc_timestamp() {
		meta.now_pseudo_us = (bit<16>)(ig_prsr_md.global_tstamp >> 10);
	}

	RegisterAction<bit<32>, bit<32>, bit<1>>(flow_forward_srcaddr_reg) flow_orientation_action = {
		void apply(inout bit<32> value, out bit<1> rv) {
			if (value == 0) {
				value = hdr.ipv4.src_addr;
			}
			rv = (value == hdr.ipv4.src_addr) ? 1w1 : 1w0;
		}
	};

	RegisterAction<bit<16>, bit<32>, bit<16>>(bwd_last_arrival_time_reg) bwd_last_arrival_time_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 0) {
				rv = meta.now_pseudo_us - value;
				value = meta.now_pseudo_us;
			} else {
				rv = 0;
			}
		}
	};

	RegisterAction<bit<16>, bit<32>, bit<16>>(bwd_iat_max_reg) bwd_iat_max_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 0 && meta.bwd_current_iat > value) {
				value = meta.bwd_current_iat;
			}
			rv = value;
		}
	};

	MathUnit<bit<16>>(MathOp_t.MUL, 1, 2) bwd_iat_mean_halve_unit;

	RegisterAction<bit<16>, bit<32>, bit<16>>(bwd_iat_mean_reg) bwd_iat_mean_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 0) {
				value = bwd_iat_mean_halve_unit.execute(value + meta.bwd_current_iat);
			}
			rv = value;
		}
	};

	RegisterAction<bit<16>, bit<32>, bit<16>>(bwd_iat_min_reg) bwd_iat_min_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 0 && meta.bwd_current_iat < value) {
				value = meta.bwd_current_iat;
			}
			rv = value;
		}
	};

	RegisterAction<bit<16>, bit<32>, bit<16>>(bwd_packet_length_max_reg) bwd_packet_length_max_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 0 && hdr.ipv4.total_len > value) {
				value = hdr.ipv4.total_len;
			}
			rv = value;
		}
	};

	MathUnit<bit<16>>(MathOp_t.MUL, 1, 2) bwd_packet_length_mean_halve_unit;

	RegisterAction<bit<16>, bit<32>, bit<16>>(bwd_packet_length_mean_reg) bwd_packet_length_mean_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 0) {
				value = bwd_packet_length_mean_halve_unit.execute(value + (bit<16>)hdr.ipv4.total_len);
			}
			rv = value;
		}
	};

	RegisterAction<bit<16>, bit<32>, bit<16>>(bwd_packet_length_min_reg) bwd_packet_length_min_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 0 && hdr.ipv4.total_len < value) {
				value = hdr.ipv4.total_len;
			}
			rv = value;
		}
	};

	RegisterAction<bit<16>, bit<32>, bit<16>>(flow_last_arrival_time_reg) flow_last_arrival_time_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			rv = meta.now_pseudo_us - value;
			value = meta.now_pseudo_us;
		}
	};

	RegisterAction<bit<16>, bit<32>, bit<16>>(flow_iat_max_reg) flow_iat_max_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.flow_current_iat > value) {
				value = meta.flow_current_iat;
			}
			rv = value;
		}
	};

	MathUnit<bit<16>>(MathOp_t.MUL, 1, 2) flow_iat_mean_halve_unit;

	RegisterAction<bit<16>, bit<32>, bit<16>>(flow_iat_mean_reg) flow_iat_mean_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			value = flow_iat_mean_halve_unit.execute(value + meta.flow_current_iat);
			rv = value;
		}
	};

	RegisterAction<bit<16>, bit<32>, bit<16>>(flow_iat_min_reg) flow_iat_min_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.flow_current_iat < value) {
				value = meta.flow_current_iat;
			}
			rv = value;
		}
	};

	RegisterAction<bit<16>, bit<32>, bit<16>>(fwd_last_arrival_time_reg) fwd_last_arrival_time_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 1) {
				rv = meta.now_pseudo_us - value;
				value = meta.now_pseudo_us;
			} else {
				rv = 0;
			}
		}
	};

	RegisterAction<bit<16>, bit<32>, bit<16>>(fwd_iat_max_reg) fwd_iat_max_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 1 && meta.fwd_current_iat > value) {
				value = meta.fwd_current_iat;
			}
			rv = value;
		}
	};

	MathUnit<bit<16>>(MathOp_t.MUL, 1, 2) fwd_iat_mean_halve_unit;

	RegisterAction<bit<16>, bit<32>, bit<16>>(fwd_iat_mean_reg) fwd_iat_mean_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 1) {
				value = fwd_iat_mean_halve_unit.execute(value + meta.fwd_current_iat);
			}
			rv = value;
		}
	};

	RegisterAction<bit<16>, bit<32>, bit<16>>(fwd_iat_min_reg) fwd_iat_min_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 1 && meta.fwd_current_iat < value) {
				value = meta.fwd_current_iat;
			}
			rv = value;
		}
	};

	RegisterAction<bit<16>, bit<32>, bit<16>>(fwd_packet_length_max_reg) fwd_packet_length_max_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 1 && hdr.ipv4.total_len > value) {
				value = hdr.ipv4.total_len;
			}
			rv = value;
		}
	};

	MathUnit<bit<16>>(MathOp_t.MUL, 1, 2) fwd_packet_length_mean_halve_unit;

	RegisterAction<bit<16>, bit<32>, bit<16>>(fwd_packet_length_mean_reg) fwd_packet_length_mean_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 1) {
				value = fwd_packet_length_mean_halve_unit.execute(value + (bit<16>)hdr.ipv4.total_len);
			}
			rv = value;
		}
	};

	RegisterAction<bit<16>, bit<32>, bit<16>>(fwd_packet_length_min_reg) fwd_packet_length_min_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 1 && hdr.ipv4.total_len < value) {
				value = hdr.ipv4.total_len;
			}
			rv = value;
		}
	};

	RegisterAction<bit<16>, bit<32>, bit<16>>(min_packet_length_reg) min_packet_length_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (hdr.ipv4.total_len < value) {
				value = hdr.ipv4.total_len;
			}
			rv = value;
		}
	};

	MathUnit<bit<16>>(MathOp_t.MUL, 1, 2) packet_length_mean_halve_unit;

	RegisterAction<bit<16>, bit<32>, bit<16>>(packet_length_mean_reg) packet_length_mean_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			value = packet_length_mean_halve_unit.execute(value + (bit<16>)hdr.ipv4.total_len);
			rv = value;
		}
	};


	action classify_flow_codeword_app_0(bit<2> class){
		meta.class_tree_app_0 = class;
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

    
    action set_code_bwd_iat_max (bit<16> code) {
        meta.code_pk_bwd_iat_max = code;
    }
    
    action set_code_bwd_iat_mean (bit<32> code) {
        meta.code_pk_bwd_iat_mean = code;
    }
    
    action set_code_bwd_iat_min (bit<32> code) {
        meta.code_pk_bwd_iat_min = code;
    }
    
    action set_code_bwd_packet_length_max (bit<48> code) {
        meta.code_pk_bwd_packet_length_max = code;
    }
    
    action set_code_bwd_packet_length_mean (bit<40> code) {
        meta.code_pk_bwd_packet_length_mean = code;
    }
    
    action set_code_bwd_packet_length_min (bit<24> code) {
        meta.code_pk_bwd_packet_length_min = code;
    }
    
    action set_code_flow_iat_max (bit<32> code) {
        meta.code_pk_flow_iat_max = code;
    }
    
    action set_code_flow_iat_mean (bit<32> code) {
        meta.code_pk_flow_iat_mean = code;
    }
    
    action set_code_flow_iat_min (bit<32> code) {
        meta.code_pk_flow_iat_min = code;
    }
    
    action set_code_fwd_iat_max (bit<32> code) {
        meta.code_pk_fwd_iat_max = code;
    }
    
    action set_code_fwd_iat_mean (bit<32> code) {
        meta.code_pk_fwd_iat_mean = code;
    }
    
    action set_code_fwd_iat_min (bit<32> code) {
        meta.code_pk_fwd_iat_min = code;
    }
    
    action set_code_fwd_packet_length_max (bit<40> code) {
        meta.code_pk_fwd_packet_length_max = code;
    }
    
    action set_code_fwd_packet_length_mean (bit<32> code) {
        meta.code_pk_fwd_packet_length_mean = code;
    }
    
    action set_code_fwd_packet_length_min (bit<32> code) {
        meta.code_pk_fwd_packet_length_min = code;
    }
    
    action set_code_min_packet_length (bit<24> code) {
        meta.code_pk_min_packet_length = code;
    }
    
    action set_code_packet_length_mean (bit<48> code) {
        meta.code_pk_packet_length_mean = code;
    }



    @placement_priority(1)
    table get_classification_tree_app_0 {
        key = {
            meta.code_pk_bwd_iat_max[4:0] : ternary;
            meta.code_pk_bwd_iat_mean[15:8] : ternary;
            meta.code_pk_bwd_iat_min[18:16] : ternary;
            meta.code_pk_bwd_packet_length_max[9:0] : ternary;
            meta.code_pk_bwd_packet_length_mean[31:16] : ternary;
            meta.code_pk_bwd_packet_length_min[9:8] : ternary;
            meta.code_pk_flow_iat_max[4:0] : ternary;
            meta.code_pk_flow_iat_mean[13:8] : ternary;
            meta.code_pk_flow_iat_min[5:0] : ternary;
            meta.code_pk_fwd_iat_max[9:8] : ternary;
            meta.code_pk_fwd_iat_mean[5:0] : ternary;
            meta.code_pk_fwd_iat_min[14:8] : ternary;
            meta.code_pk_fwd_packet_length_max[12:0] : ternary;
            meta.code_pk_fwd_packet_length_mean[22:16] : ternary;
            meta.code_pk_fwd_packet_length_min[18:16] : ternary;
            meta.code_pk_min_packet_length[0:0] : ternary;
            meta.code_pk_packet_length_mean[16:8] : ternary;
        }
        actions = {
            classify_flow_codeword_app_0;        
        }
        size = 111;
    }
    
    @placement_priority(2)
    table get_classification_tree_ddos_0 {
        key = {
            meta.code_pk_bwd_iat_mean[0:0] : ternary;
            meta.code_pk_bwd_iat_min[12:8] : ternary;
            meta.code_pk_bwd_packet_length_max[21:16] : ternary;
            meta.code_pk_bwd_packet_length_mean[11:8] : ternary;
            meta.code_pk_bwd_packet_length_min[16:16] : ternary;
            meta.code_pk_flow_iat_max[18:16] : ternary;
            meta.code_pk_flow_iat_mean[1:0] : ternary;
            meta.code_pk_flow_iat_min[9:8] : ternary;
            meta.code_pk_fwd_iat_max[3:0] : ternary;
            meta.code_pk_fwd_iat_mean[9:8] : ternary;
            meta.code_pk_fwd_iat_min[3:0] : ternary;
            meta.code_pk_fwd_packet_length_max[21:16] : ternary;
            meta.code_pk_fwd_packet_length_mean[4:0] : ternary;
            meta.code_pk_fwd_packet_length_min[9:8] : ternary;
            meta.code_pk_packet_length_mean[33:24] : ternary;
        }
        actions = {
            classify_flow_codeword_ddos_0;        
        }
        size = 59;
    }
    
    @placement_priority(2)
    table get_classification_tree_ddos_1 {
        key = {
            meta.code_pk_bwd_iat_max[8:8] : ternary;
            meta.code_pk_bwd_iat_mean[17:16] : ternary;
            meta.code_pk_bwd_iat_min[3:0] : ternary;
            meta.code_pk_bwd_packet_length_max[26:24] : ternary;
            meta.code_pk_bwd_packet_length_mean[7:0] : ternary;
            meta.code_pk_flow_iat_max[8:8] : ternary;
            meta.code_pk_flow_iat_mean[16:16] : ternary;
            meta.code_pk_flow_iat_min[21:16] : ternary;
            meta.code_pk_fwd_iat_max[19:16] : ternary;
            meta.code_pk_fwd_iat_mean[19:16] : ternary;
            meta.code_pk_fwd_iat_min[20:16] : ternary;
            meta.code_pk_fwd_packet_length_max[29:24] : ternary;
            meta.code_pk_fwd_packet_length_mean[10:8] : ternary;
            meta.code_pk_fwd_packet_length_min[1:0] : ternary;
            meta.code_pk_min_packet_length[9:8] : ternary;
            meta.code_pk_packet_length_mean[6:0] : ternary;
        }
        actions = {
            classify_flow_codeword_ddos_1;        
        }
        size = 60;
    }
    
    @placement_priority(2)
    table get_classification_tree_ddos_2 {
        key = {
            meta.code_pk_bwd_iat_mean[27:24] : ternary;
            meta.code_pk_bwd_iat_min[25:24] : ternary;
            meta.code_pk_bwd_packet_length_max[40:32] : ternary;
            meta.code_pk_bwd_packet_length_mean[36:32] : ternary;
            meta.code_pk_bwd_packet_length_min[0:0] : ternary;
            meta.code_pk_flow_iat_max[26:24] : ternary;
            meta.code_pk_flow_iat_mean[29:24] : ternary;
            meta.code_pk_flow_iat_min[27:24] : ternary;
            meta.code_pk_fwd_iat_max[28:24] : ternary;
            meta.code_pk_fwd_iat_mean[25:24] : ternary;
            meta.code_pk_fwd_iat_min[29:24] : ternary;
            meta.code_pk_fwd_packet_length_max[35:32] : ternary;
            meta.code_pk_fwd_packet_length_mean[28:24] : ternary;
            meta.code_pk_fwd_packet_length_min[25:24] : ternary;
            meta.code_pk_min_packet_length[16:16] : ternary;
            meta.code_pk_packet_length_mean[42:40] : ternary;
        }
        actions = {
            classify_flow_codeword_ddos_2;        
        }
        size = 65;
    }
    
    @placement_priority(3)
    table table_0_bwd_iat_max {
        key = {
            meta.bwd_iat_max_val: range;
        }
        actions = {
            set_code_bwd_iat_max;        
        }
        size = 7;
    }   
    
    @placement_priority(3)
    table table_1_bwd_iat_mean {
        key = {
            meta.bwd_iat_mean_val: range;
        }
        actions = {
            set_code_bwd_iat_mean;        
        }
        size = 16;
    }   
    
    @placement_priority(3)
    table table_2_bwd_iat_min {
        key = {
            meta.bwd_iat_min_val: range;
        }
        actions = {
            set_code_bwd_iat_min;        
        }
        size = 11;
    }   
    
    @placement_priority(3)
    table table_3_bwd_packet_length_max {
        key = {
            meta.bwd_packet_length_max_val: range;
        }
        actions = {
            set_code_bwd_packet_length_max;        
        }
        size = 27;
    }   
    
    @placement_priority(3)
    table table_4_bwd_packet_length_mean {
        key = {
            meta.bwd_packet_length_mean_val: range;
        }
        actions = {
            set_code_bwd_packet_length_mean;        
        }
        size = 31;
    }   
    
    @placement_priority(3)
    table table_5_bwd_packet_length_min {
        key = {
            meta.bwd_packet_length_min_val: range;
        }
        actions = {
            set_code_bwd_packet_length_min;        
        }
        size = 3;
    }   
    
    @placement_priority(3)
    table table_6_flow_iat_max {
        key = {
            meta.flow_iat_max_val: range;
        }
        actions = {
            set_code_flow_iat_max;        
        }
        size = 13;
    }   
    
    @placement_priority(3)
    table table_7_flow_iat_mean {
        key = {
            meta.flow_iat_mean_val: range;
        }
        actions = {
            set_code_flow_iat_mean;        
        }
        size = 16;
    }   
    
    @placement_priority(3)
    table table_8_flow_iat_min {
        key = {
            meta.flow_iat_min_val: range;
        }
        actions = {
            set_code_flow_iat_min;        
        }
        size = 15;
    }   
    
    @placement_priority(3)
    table table_9_fwd_iat_max {
        key = {
            meta.fwd_iat_max_val: range;
        }
        actions = {
            set_code_fwd_iat_max;        
        }
        size = 16;
    }   
    
    @placement_priority(3)
    table table_10_fwd_iat_mean {
        key = {
            meta.fwd_iat_mean_val: range;
        }
        actions = {
            set_code_fwd_iat_mean;        
        }
        size = 14;
    }   
    
    @placement_priority(3)
    table table_11_fwd_iat_min {
        key = {
            meta.fwd_iat_min_val: range;
        }
        actions = {
            set_code_fwd_iat_min;        
        }
        size = 22;
    }   
    
    @placement_priority(3)
    table table_12_fwd_packet_length_max {
        key = {
            meta.fwd_packet_length_max_val: range;
        }
        actions = {
            set_code_fwd_packet_length_max;        
        }
        size = 29;
    }   
    
    @placement_priority(3)
    table table_13_fwd_packet_length_mean {
        key = {
            meta.fwd_packet_length_mean_val: range;
        }
        actions = {
            set_code_fwd_packet_length_mean;        
        }
        size = 18;
    }   
    
    @placement_priority(3)
    table table_14_fwd_packet_length_min {
        key = {
            meta.fwd_packet_length_min_val: range;
        }
        actions = {
            set_code_fwd_packet_length_min;        
        }
        size = 8;
    }   
    
    @placement_priority(3)
    table table_15_min_packet_length {
        key = {
            meta.min_packet_length_val: range;
        }
        actions = {
            set_code_min_packet_length;        
        }
        size = 3;
    }   
    
    @placement_priority(3)
    table table_16_packet_length_mean {
        key = {
            meta.packet_length_mean_val: range;
        }
        actions = {
            set_code_packet_length_mean;        
        }
        size = 27;
    }   
    	action set_classification_app(bit<2> winner) {
		meta.classification_app = winner;
	}

	table vote_app {
		key = {
			meta.class_tree_app_0 : exact;
		}
		actions = {
			set_classification_app;
		}
		size = 3;
		const entries = {
			(0) : set_classification_app(0);
			(1) : set_classification_app(1);
			(2) : set_classification_app(2);
		}
	}
	action set_classification_ddos(bit<1> winner) {
		meta.classification_ddos = winner;
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
        if (hdr.tcp.isValid()) {

			calc_flow_hash();
			calc_timestamp();

			meta.fwd = flow_orientation_action.execute(meta.flow_hash);

			meta.flow_current_iat = flow_last_arrival_time_action.execute(meta.flow_hash);
			meta.flow_iat_max_val = flow_iat_max_action.execute(meta.flow_hash);
			meta.flow_iat_mean_val = flow_iat_mean_action.execute(meta.flow_hash);
			meta.flow_iat_min_val = flow_iat_min_action.execute(meta.flow_hash);
			meta.min_packet_length_val = min_packet_length_action.execute(meta.flow_hash);
			meta.packet_length_mean_val = packet_length_mean_action.execute(meta.flow_hash);

			meta.fwd_current_iat = fwd_last_arrival_time_action.execute(meta.flow_hash);
			meta.fwd_iat_max_val = fwd_iat_max_action.execute(meta.flow_hash);
			meta.fwd_iat_mean_val = fwd_iat_mean_action.execute(meta.flow_hash);
			meta.fwd_iat_min_val = fwd_iat_min_action.execute(meta.flow_hash);
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
			table_6_flow_iat_max.apply();
			table_7_flow_iat_mean.apply();
			table_8_flow_iat_min.apply();
			table_9_fwd_iat_max.apply();
			table_10_fwd_iat_mean.apply();
			table_11_fwd_iat_min.apply();
			table_12_fwd_packet_length_max.apply();
			table_13_fwd_packet_length_mean.apply();
			table_14_fwd_packet_length_min.apply();
			table_15_min_packet_length.apply();
			table_16_packet_length_mean.apply();

			get_classification_tree_app_0.apply();

			get_classification_tree_ddos_0.apply();
			get_classification_tree_ddos_1.apply();
			get_classification_tree_ddos_2.apply();
			vote_app.apply();
			vote_ddos.apply();




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
