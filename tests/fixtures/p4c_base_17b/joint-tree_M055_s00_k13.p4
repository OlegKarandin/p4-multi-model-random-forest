/* -*- P4_16 -*- */
#include <core.p4>
// Tofino-1 TNA only: the resource model's constants (p4model/target.py) and the
// export widths (program.VOTE_EXPORT_BITS) are Tofino-1's.
#include <tna.p4>

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
@pa_container_size("ingress", "ig_md.bwd_iat_min_val", 16)
@pa_container_size("ingress", "ig_md.bwd_iat_total_val", 16)
@pa_container_size("ingress", "ig_md.bwd_packet_length_max_val", 16)
@pa_container_size("ingress", "ig_md.bwd_packet_length_min_val", 16)
@pa_container_size("ingress", "ig_md.bwd_packet_length_total_val", 16)
@pa_container_size("ingress", "ig_md.flow_duration_val", 16)
@pa_container_size("ingress", "ig_md.flow_iat_min_val", 16)
@pa_container_size("ingress", "ig_md.fwd_packet_length_max_val", 16)
@pa_container_size("ingress", "ig_md.fwd_packet_length_min_val", 16)
@pa_container_size("ingress", "ig_md.fwd_packet_length_total_val", 16)
@pa_container_size("ingress", "ig_md.min_packet_length_val", 16)
@pa_container_size("ingress", "ig_md.packet_length_total_val", 16)
@pa_no_overlay("ingress", "ig_md.code_pk_bwd_iat_max")
@pa_container_size("ingress", "ig_md.code_pk_bwd_iat_max", 16, 8)
@pa_no_overlay("ingress", "ig_md.code_pk_bwd_iat_min")
@pa_container_size("ingress", "ig_md.code_pk_bwd_iat_min", 32, 8)
@pa_no_overlay("ingress", "ig_md.code_pk_bwd_iat_total")
@pa_container_size("ingress", "ig_md.code_pk_bwd_iat_total", 32, 8)
@pa_no_overlay("ingress", "ig_md.code_pk_bwd_packet_length_max")
@pa_container_size("ingress", "ig_md.code_pk_bwd_packet_length_max", 32, 32, 16, 8)
@pa_no_overlay("ingress", "ig_md.code_pk_bwd_packet_length_min")
@pa_container_size("ingress", "ig_md.code_pk_bwd_packet_length_min", 16)
@pa_no_overlay("ingress", "ig_md.code_pk_bwd_packet_length_total")
@pa_container_size("ingress", "ig_md.code_pk_bwd_packet_length_total", 32, 16, 8)
@pa_no_overlay("ingress", "ig_md.code_pk_flow_duration")
@pa_container_size("ingress", "ig_md.code_pk_flow_duration", 32, 16, 8)
@pa_no_overlay("ingress", "ig_md.code_pk_flow_iat_min")
@pa_container_size("ingress", "ig_md.code_pk_flow_iat_min", 32, 16)
@pa_no_overlay("ingress", "ig_md.code_pk_fwd_packet_length_max")
@pa_container_size("ingress", "ig_md.code_pk_fwd_packet_length_max", 32, 32, 16, 8)
@pa_no_overlay("ingress", "ig_md.code_pk_fwd_packet_length_min")
@pa_container_size("ingress", "ig_md.code_pk_fwd_packet_length_min", 16)
@pa_no_overlay("ingress", "ig_md.code_pk_fwd_packet_length_total")
@pa_container_size("ingress", "ig_md.code_pk_fwd_packet_length_total", 32, 16, 8)
@pa_no_overlay("ingress", "ig_md.code_pk_min_packet_length")
@pa_container_size("ingress", "ig_md.code_pk_min_packet_length", 8)
@pa_no_overlay("ingress", "ig_md.code_pk_packet_length_total")
@pa_container_size("ingress", "ig_md.code_pk_packet_length_total", 32, 32, 16)
@pa_container_size("ingress", "ig_md.bwd_current_iat", 16)
@pa_container_size("ingress", "ig_md.flow_current_iat", 16)
@pa_solitary("ingress", "ig_md.now_pseudo_us")
@pa_solitary("ingress", "ig_md.now_coarse")
@pa_solitary("ingress", "ig_md.flow_hash")
struct metadata_t {
    bit<12> flow_hash;
    bit<1>  fwd;
    bit<32> now_pseudo_us;

	bit<32> now_coarse;
	bit<16> bwd_current_iat;
	bit<16> flow_current_iat;
	bit<2> class_tree_app_0;
	bit<2> class_tree_app_1;
	bit<2> class_tree_app_2;
	bit<1> class_tree_ddos_0;
	bit<1> class_tree_ddos_1;
	bit<1> class_tree_ddos_2;
	bit<16> bwd_iat_max_val;
	bit<16> bwd_iat_min_val;
	bit<16> bwd_iat_total_val;
	bit<16> bwd_packet_length_max_val;
	bit<16> bwd_packet_length_min_val;
	bit<16> bwd_packet_length_total_val;
	bit<16> flow_duration_val;
	bit<16> flow_iat_min_val;
	bit<16> fwd_packet_length_max_val;
	bit<16> fwd_packet_length_min_val;
	bit<16> fwd_packet_length_total_val;
	bit<16> min_packet_length_val;
	bit<16> packet_length_total_val;
	bit<24> code_pk_bwd_iat_max;
	bit<40> code_pk_bwd_iat_min;
	bit<40> code_pk_bwd_iat_total;
	bit<88> code_pk_bwd_packet_length_max;
	bit<16> code_pk_bwd_packet_length_min;
	bit<56> code_pk_bwd_packet_length_total;
	bit<56> code_pk_flow_duration;
	bit<48> code_pk_flow_iat_min;
	bit<88> code_pk_fwd_packet_length_max;
	bit<16> code_pk_fwd_packet_length_min;
	bit<56> code_pk_fwd_packet_length_total;
	bit<8> code_pk_min_packet_length;
	bit<80> code_pk_packet_length_total;

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
	Register<bit<32>, bit<12>>(MAX_NUM_FLOWS, 32w0xFFFFFFFF) bwd_last_arrival_time_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) bwd_iat_max_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS, 65535) bwd_iat_min_reg;
	Register<dur_pair_t, bit<12>>(MAX_NUM_FLOWS, {32w0xFFFFFFFF, 32w0}) bwd_iat_total_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) bwd_packet_length_max_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS, 65535) bwd_packet_length_min_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) bwd_packet_length_total_reg;
	Register<bit<32>, bit<12>>(MAX_NUM_FLOWS, 32w0xFFFFFFFF) flow_duration_reg;
	Register<bit<32>, bit<12>>(MAX_NUM_FLOWS, 32w0xFFFFFFFF) flow_last_arrival_time_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS, 65535) flow_iat_min_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) fwd_packet_length_max_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS, 65535) fwd_packet_length_min_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) fwd_packet_length_total_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS, 65535) min_packet_length_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) packet_length_total_reg;


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

	RegisterAction<bit<16>, bit<12>, bit<16>>(bwd_iat_max_reg) bwd_iat_max_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 0 && meta.bwd_current_iat > value) {
				value = meta.bwd_current_iat;
			}
			rv = value;
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

	RegisterAction<bit<16>, bit<12>, bit<16>>(bwd_packet_length_max_reg) bwd_packet_length_max_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 0 && hdr.ipv4.total_len > value) {
				value = hdr.ipv4.total_len;
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

	RegisterAction<bit<16>, bit<12>, bit<16>>(bwd_packet_length_total_reg) bwd_packet_length_total_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 0) {
				value = value |+| hdr.ipv4.total_len;
			}
			rv = value;
		}
	};

	RegisterAction<bit<32>, bit<12>, bit<32>>(flow_duration_reg) flow_duration_action = {
		void apply(inout bit<32> value, out bit<32> rv) {
			if (meta.now_coarse - value > 65535) {
				rv = 65535;
			} else {
				rv = meta.now_coarse |-| value;
			}
			value = min(value, meta.now_coarse);
		}
	};

	RegisterAction<bit<32>, bit<12>, bit<32>>(flow_last_arrival_time_reg) flow_last_arrival_time_action = {
		void apply(inout bit<32> value, out bit<32> rv) {
			if (meta.now_pseudo_us - value > 65535) {
				rv = 65535;
			} else {
				rv = meta.now_pseudo_us |-| value;
			}
			value = meta.now_pseudo_us - 1;
		}
	};

	RegisterAction<bit<16>, bit<12>, bit<16>>(flow_iat_min_reg) flow_iat_min_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.flow_current_iat != 0 && meta.flow_current_iat <= value) {
				value = meta.flow_current_iat - 1;
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

	RegisterAction<bit<16>, bit<12>, bit<16>>(fwd_packet_length_min_reg) fwd_packet_length_min_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 1 && hdr.ipv4.total_len < value) {
				value = hdr.ipv4.total_len;
			}
			rv = value;
		}
	};

	RegisterAction<bit<16>, bit<12>, bit<16>>(fwd_packet_length_total_reg) fwd_packet_length_total_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 1) {
				value = value |+| hdr.ipv4.total_len;
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
    
    action set_code_bwd_iat_min (bit<40> code) {
        meta.code_pk_bwd_iat_min = code;
    }
    
    action set_code_bwd_iat_total (bit<40> code) {
        meta.code_pk_bwd_iat_total = code;
    }
    
    action set_code_bwd_packet_length_max (bit<88> code) {
        meta.code_pk_bwd_packet_length_max = code;
    }
    
    action set_code_bwd_packet_length_min (bit<16> code) {
        meta.code_pk_bwd_packet_length_min = code;
    }
    
    action set_code_bwd_packet_length_total (bit<56> code) {
        meta.code_pk_bwd_packet_length_total = code;
    }
    
    action set_code_flow_duration (bit<56> code) {
        meta.code_pk_flow_duration = code;
    }
    
    action set_code_flow_iat_min (bit<48> code) {
        meta.code_pk_flow_iat_min = code;
    }
    
    action set_code_fwd_packet_length_max (bit<88> code) {
        meta.code_pk_fwd_packet_length_max = code;
    }
    
    action set_code_fwd_packet_length_min (bit<16> code) {
        meta.code_pk_fwd_packet_length_min = code;
    }
    
    action set_code_fwd_packet_length_total (bit<56> code) {
        meta.code_pk_fwd_packet_length_total = code;
    }
    
    action set_code_min_packet_length (bit<8> code) {
        meta.code_pk_min_packet_length = code;
    }
    
    action set_code_packet_length_total (bit<80> code) {
        meta.code_pk_packet_length_total = code;
    }



    @placement_priority(1)
    table get_classification_tree_app_0 {
        key = {
            meta.code_pk_bwd_iat_max[4:0] : ternary @name("code_pk_bwd_iat_max_4_0_");
            meta.code_pk_bwd_iat_min[14:8] : ternary @name("code_pk_bwd_iat_min_14_8_");
            meta.code_pk_bwd_iat_total[10:0] : ternary @name("code_pk_bwd_iat_total_10_0_");
            meta.code_pk_bwd_packet_length_max[71:48] : ternary @name("code_pk_bwd_packet_length_max_71_48_");
            meta.code_pk_bwd_packet_length_min[12:8] : ternary @name("code_pk_bwd_packet_length_min_12_8_");
            meta.code_pk_bwd_packet_length_total[8:0] : ternary @name("code_pk_bwd_packet_length_total_8_0_");
            meta.code_pk_flow_duration[25:16] : ternary @name("code_pk_flow_duration_25_16_");
            meta.code_pk_flow_iat_min[23:17] : ternary @name("code_pk_flow_iat_min_23_17_");
            meta.code_pk_fwd_packet_length_max[11:0] : ternary @name("code_pk_fwd_packet_length_max_11_0_");
            meta.code_pk_fwd_packet_length_min[10:8] : ternary @name("code_pk_fwd_packet_length_min_10_8_");
            meta.code_pk_fwd_packet_length_total[7:0] : ternary @name("code_pk_fwd_packet_length_total_7_0_");
            meta.code_pk_min_packet_length[0:0] : ternary @name("code_pk_min_packet_length_0_0_");
            meta.code_pk_packet_length_total[43:32] : ternary @name("code_pk_packet_length_total_43_32_");
        }
        actions = {
            classify_flow_codeword_app_0;        
        }
        size = 120;
    }
    
    @placement_priority(1)
    table get_classification_tree_app_1 {
        key = {
            meta.code_pk_bwd_iat_max[14:8] : ternary @name("code_pk_bwd_iat_max_14_8_");
            meta.code_pk_bwd_iat_min[26:16] : ternary @name("code_pk_bwd_iat_min_26_16_");
            meta.code_pk_bwd_iat_total[35:24] : ternary @name("code_pk_bwd_iat_total_35_24_");
            meta.code_pk_bwd_packet_length_max[24:0] : ternary @name("code_pk_bwd_packet_length_max_24_0_");
            meta.code_pk_bwd_packet_length_min[15:13] : ternary @name("code_pk_bwd_packet_length_min_15_13_");
            meta.code_pk_bwd_packet_length_total[34:16] : ternary @name("code_pk_bwd_packet_length_total_34_16_");
            meta.code_pk_flow_duration[43:32] : ternary @name("code_pk_flow_duration_43_32_");
            meta.code_pk_flow_iat_min[16:8] : ternary @name("code_pk_flow_iat_min_16_8_");
            meta.code_pk_fwd_packet_length_max[52:24] : ternary @name("code_pk_fwd_packet_length_max_52_24_");
            meta.code_pk_fwd_packet_length_min[14:14] : ternary @name("code_pk_fwd_packet_length_min_14_14_");
            meta.code_pk_fwd_packet_length_total[50:40] : ternary @name("code_pk_fwd_packet_length_total_50_40_");
            meta.code_pk_min_packet_length[1:1] : ternary @name("code_pk_min_packet_length_1_1_");
            meta.code_pk_packet_length_total[76:64] : ternary @name("code_pk_packet_length_total_76_64_");
        }
        actions = {
            classify_flow_codeword_app_1;        
        }
        size = 156;
    }
    
    @placement_priority(1)
    table get_classification_tree_app_2 {
        key = {
            meta.code_pk_bwd_iat_max[21:16] : ternary @name("code_pk_bwd_iat_max_21_16_");
            meta.code_pk_bwd_iat_min[29:27] : ternary @name("code_pk_bwd_iat_min_29_27_");
            meta.code_pk_bwd_iat_total[15:11] : ternary @name("code_pk_bwd_iat_total_15_11_");
            meta.code_pk_bwd_packet_length_max[45:32] : ternary @name("code_pk_bwd_packet_length_max_45_32_");
            meta.code_pk_bwd_packet_length_min[4:0] : ternary @name("code_pk_bwd_packet_length_min_4_0_");
            meta.code_pk_bwd_packet_length_total[52:40] : ternary @name("code_pk_bwd_packet_length_total_52_40_");
            meta.code_pk_flow_duration[9:0] : ternary @name("code_pk_flow_duration_9_0_");
            meta.code_pk_flow_iat_min[31:24] : ternary @name("code_pk_flow_iat_min_31_24_");
            meta.code_pk_fwd_packet_length_max[75:56] : ternary @name("code_pk_fwd_packet_length_max_75_56_");
            meta.code_pk_fwd_packet_length_min[13:11] : ternary @name("code_pk_fwd_packet_length_min_13_11_");
            meta.code_pk_fwd_packet_length_total[34:16] : ternary @name("code_pk_fwd_packet_length_total_34_16_");
            meta.code_pk_min_packet_length[2:2] : ternary @name("code_pk_min_packet_length_2_2_");
            meta.code_pk_packet_length_total[17:0] : ternary @name("code_pk_packet_length_total_17_0_");
        }
        actions = {
            classify_flow_codeword_app_2;        
        }
        size = 129;
    }
    
    @placement_priority(2)
    table get_classification_tree_ddos_0 {
        key = {
            meta.code_pk_bwd_iat_max[23:22] : ternary @name("code_pk_bwd_iat_max_23_22_");
            meta.code_pk_bwd_iat_min[38:32] : ternary @name("code_pk_bwd_iat_min_38_32_");
            meta.code_pk_bwd_iat_total[39:36] : ternary @name("code_pk_bwd_iat_total_39_36_");
            meta.code_pk_bwd_packet_length_max[31:25] : ternary @name("code_pk_bwd_packet_length_max_31_25_");
            meta.code_pk_bwd_packet_length_total[14:9] : ternary @name("code_pk_bwd_packet_length_total_14_9_");
            meta.code_pk_flow_duration[31:26] : ternary @name("code_pk_flow_duration_31_26_");
            meta.code_pk_flow_iat_min[6:0] : ternary @name("code_pk_flow_iat_min_6_0_");
            meta.code_pk_fwd_packet_length_max[22:16] : ternary @name("code_pk_fwd_packet_length_max_22_16_");
            meta.code_pk_fwd_packet_length_min[15:15] : ternary @name("code_pk_fwd_packet_length_min_15_15_");
            meta.code_pk_fwd_packet_length_total[38:35] : ternary @name("code_pk_fwd_packet_length_total_38_35_");
            meta.code_pk_packet_length_total[56:48] : ternary @name("code_pk_packet_length_total_56_48_");
        }
        actions = {
            classify_flow_codeword_ddos_0;        
        }
        size = 61;
    }
    
    @placement_priority(2)
    table get_classification_tree_ddos_1 {
        key = {
            meta.code_pk_bwd_iat_max[15:15] : ternary @name("code_pk_bwd_iat_max_15_15_");
            meta.code_pk_bwd_iat_min[2:0] : ternary @name("code_pk_bwd_iat_min_2_0_");
            meta.code_pk_bwd_iat_total[17:16] : ternary @name("code_pk_bwd_iat_total_17_16_");
            meta.code_pk_bwd_packet_length_max[84:82] : ternary @name("code_pk_bwd_packet_length_max_84_82_");
            meta.code_pk_bwd_packet_length_min[7:7] : ternary @name("code_pk_bwd_packet_length_min_7_7_");
            meta.code_pk_bwd_packet_length_total[38:35] : ternary @name("code_pk_bwd_packet_length_total_38_35_");
            meta.code_pk_flow_duration[54:48] : ternary @name("code_pk_flow_duration_54_48_");
            meta.code_pk_flow_iat_min[45:42] : ternary @name("code_pk_flow_iat_min_45_42_");
            meta.code_pk_fwd_packet_length_max[85:80] : ternary @name("code_pk_fwd_packet_length_max_85_80_");
            meta.code_pk_fwd_packet_length_total[13:8] : ternary @name("code_pk_fwd_packet_length_total_13_8_");
            meta.code_pk_min_packet_length[3:3] : ternary @name("code_pk_min_packet_length_3_3_");
            meta.code_pk_packet_length_total[31:24] : ternary @name("code_pk_packet_length_total_31_24_");
        }
        actions = {
            classify_flow_codeword_ddos_1;        
        }
        size = 49;
    }
    
    @placement_priority(2)
    table get_classification_tree_ddos_2 {
        key = {
            meta.code_pk_bwd_iat_min[5:3] : ternary @name("code_pk_bwd_iat_min_5_3_");
            meta.code_pk_bwd_iat_total[19:18] : ternary @name("code_pk_bwd_iat_total_19_18_");
            meta.code_pk_bwd_packet_length_max[81:72] : ternary @name("code_pk_bwd_packet_length_max_81_72_");
            meta.code_pk_bwd_packet_length_min[6:5] : ternary @name("code_pk_bwd_packet_length_min_6_5_");
            meta.code_pk_bwd_packet_length_total[54:53] : ternary @name("code_pk_bwd_packet_length_total_54_53_");
            meta.code_pk_flow_duration[45:44] : ternary @name("code_pk_flow_duration_45_44_");
            meta.code_pk_flow_iat_min[41:32] : ternary @name("code_pk_flow_iat_min_41_32_");
            meta.code_pk_fwd_packet_length_max[79:76] : ternary @name("code_pk_fwd_packet_length_max_79_76_");
            meta.code_pk_fwd_packet_length_min[0:0] : ternary @name("code_pk_fwd_packet_length_min_0_0_");
            meta.code_pk_fwd_packet_length_total[52:51] : ternary @name("code_pk_fwd_packet_length_total_52_51_");
            meta.code_pk_min_packet_length[4:4] : ternary @name("code_pk_min_packet_length_4_4_");
            meta.code_pk_packet_length_total[20:18] : ternary @name("code_pk_packet_length_total_20_18_");
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
    table table_1_bwd_iat_min {
        key = {
            meta.bwd_iat_min_val: range;
        }
        actions = {
            set_code_bwd_iat_min;        
        }
        size = 30;
    }   
    
    @placement_priority(3)
    table table_2_bwd_iat_total {
        key = {
            meta.bwd_iat_total_val: range;
        }
        actions = {
            set_code_bwd_iat_total;        
        }
        size = 33;
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
    table table_4_bwd_packet_length_min {
        key = {
            meta.bwd_packet_length_min_val: range;
        }
        actions = {
            set_code_bwd_packet_length_min;        
        }
        size = 11;
    }   
    
    @placement_priority(3)
    table table_5_bwd_packet_length_total {
        key = {
            meta.bwd_packet_length_total_val: range;
        }
        actions = {
            set_code_bwd_packet_length_total;        
        }
        size = 53;
    }   
    
    @placement_priority(3)
    table table_6_flow_duration {
        key = {
            meta.flow_duration_val: range;
        }
        actions = {
            set_code_flow_duration;        
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
    table table_9_fwd_packet_length_min {
        key = {
            meta.fwd_packet_length_min_val: range;
        }
        actions = {
            set_code_fwd_packet_length_min;        
        }
        size = 8;
    }   
    
    @placement_priority(3)
    table table_10_fwd_packet_length_total {
        key = {
            meta.fwd_packet_length_total_val: range;
        }
        actions = {
            set_code_fwd_packet_length_total;        
        }
        size = 46;
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
    table table_12_packet_length_total {
        key = {
            meta.packet_length_total_val: range;
        }
        actions = {
            set_code_packet_length_total;        
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
        // First, not last: the vote actions (and a one-tree task's tree action) write
        // ig_tm_md (qid, ingress_cos), so p4c orders a trailing assignment to ig_tm_md
        // after them, in a stage of its own.
        ig_tm_md.ucast_egress_port = ig_intr_md.ingress_port;
        ig_tm_md.bypass_egress = 1w1;

        if (hdr.tcp.isValid()) {

			calc_flow_hash();
			calc_timestamp();

			meta.fwd = flow_orientation_action.execute(meta.flow_hash);

			meta.flow_duration_val = (bit<16>) flow_duration_action.execute(meta.flow_hash);
			meta.flow_current_iat = (bit<16>) flow_last_arrival_time_action.execute(meta.flow_hash);
			meta.flow_iat_min_val = flow_iat_min_action.execute(meta.flow_hash) + 1;
			meta.min_packet_length_val = min_packet_length_action.execute(meta.flow_hash);
			meta.packet_length_total_val = packet_length_total_action.execute(meta.flow_hash);

			meta.fwd_packet_length_max_val = fwd_packet_length_max_action.execute(meta.flow_hash);
			meta.fwd_packet_length_min_val = fwd_packet_length_min_action.execute(meta.flow_hash);
			meta.fwd_packet_length_total_val = fwd_packet_length_total_action.execute(meta.flow_hash);

			meta.bwd_current_iat = (bit<16>) bwd_last_arrival_time_action.execute(meta.flow_hash);
			meta.bwd_iat_max_val = bwd_iat_max_action.execute(meta.flow_hash);
			meta.bwd_iat_min_val = bwd_iat_min_action.execute(meta.flow_hash) + 1;
			meta.bwd_iat_total_val = (bit<16>) bwd_iat_total_action.execute(meta.flow_hash);
			meta.bwd_packet_length_max_val = bwd_packet_length_max_action.execute(meta.flow_hash);
			meta.bwd_packet_length_min_val = bwd_packet_length_min_action.execute(meta.flow_hash);
			meta.bwd_packet_length_total_val = bwd_packet_length_total_action.execute(meta.flow_hash);


			table_0_bwd_iat_max.apply();
			table_1_bwd_iat_min.apply();
			table_2_bwd_iat_total.apply();
			table_3_bwd_packet_length_max.apply();
			table_4_bwd_packet_length_min.apply();
			table_5_bwd_packet_length_total.apply();
			table_6_flow_duration.apply();
			table_7_flow_iat_min.apply();
			table_8_fwd_packet_length_max.apply();
			table_9_fwd_packet_length_min.apply();
			table_10_fwd_packet_length_total.apply();
			table_11_min_packet_length.apply();
			table_12_packet_length_total.apply();

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
