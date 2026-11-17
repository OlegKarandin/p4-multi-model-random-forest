/* -*- P4_16 -*- */
#include <core.p4>
// Tofino-1 TNA only: the resource model's constants (p4model/target.py) and the
// export widths (program.VOTE_EXPORT_BITS) are Tofino-1's.
#include <tna.p4>

#include "p4_headers.p4"
#include "p4_util.p4"

const bit<32> MAX_NUM_FLOWS = 4096;

@pa_solitary("ingress", "ig_md.class_tree_app_0")
@pa_no_overlay("ingress", "ig_md.class_tree_app_0")
@pa_solitary("ingress", "ig_md.class_tree_app_1")
@pa_no_overlay("ingress", "ig_md.class_tree_app_1")
@pa_solitary("ingress", "ig_md.class_tree_app_2")
@pa_no_overlay("ingress", "ig_md.class_tree_app_2")
@pa_solitary("ingress", "ig_md.class_tree_ddos_0")
@pa_no_overlay("ingress", "ig_md.class_tree_ddos_0")
@pa_solitary("ingress", "ig_md.class_tree_ddos_1")
@pa_no_overlay("ingress", "ig_md.class_tree_ddos_1")
@pa_solitary("ingress", "ig_md.class_tree_ddos_2")
@pa_no_overlay("ingress", "ig_md.class_tree_ddos_2")
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
@pa_container_size("ingress", "ig_md.code_pk_bwd_iat_max", 32, 8)
@pa_container_size("ingress", "ig_md.code_pk_bwd_iat_min", 32, 16, 8)
@pa_container_size("ingress", "ig_md.code_pk_bwd_iat_total", 32, 32)
@pa_container_size("ingress", "ig_md.code_pk_bwd_packet_length_max", 32, 32, 32, 8)
@pa_container_size("ingress", "ig_md.code_pk_bwd_packet_length_min", 32, 8)
@pa_container_size("ingress", "ig_md.code_pk_bwd_packet_length_total", 32, 32, 16)
@pa_container_size("ingress", "ig_md.code_pk_flow_duration", 32, 32, 8)
@pa_container_size("ingress", "ig_md.code_pk_flow_iat_min", 32, 32)
@pa_container_size("ingress", "ig_md.code_pk_fwd_packet_length_max", 32, 32, 32)
@pa_container_size("ingress", "ig_md.code_pk_fwd_packet_length_min", 32, 8)
@pa_container_size("ingress", "ig_md.code_pk_fwd_packet_length_total", 32, 32, 8)
@pa_container_size("ingress", "ig_md.code_pk_min_packet_length", 32, 8)
@pa_container_size("ingress", "ig_md.code_pk_packet_length_total", 32, 32, 16, 8)
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
	bit<40> code_pk_bwd_iat_max;
	bit<56> code_pk_bwd_iat_min;
	bit<64> code_pk_bwd_iat_total;
	bit<104> code_pk_bwd_packet_length_max;
	bit<40> code_pk_bwd_packet_length_min;
	bit<80> code_pk_bwd_packet_length_total;
	bit<72> code_pk_flow_duration;
	bit<64> code_pk_flow_iat_min;
	bit<96> code_pk_fwd_packet_length_max;
	bit<40> code_pk_fwd_packet_length_min;
	bit<72> code_pk_fwd_packet_length_total;
	bit<40> code_pk_min_packet_length;
	bit<88> code_pk_packet_length_total;

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
			if (meta.fwd == 0 && hdr.ipv4.total_len <= value) {
				value = hdr.ipv4.total_len - 1;
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

    
    action set_code_bwd_iat_max (bit<40> code) {
        meta.code_pk_bwd_iat_max = code;
    }
    
    action set_code_bwd_iat_min (bit<56> code) {
        meta.code_pk_bwd_iat_min = code;
    }
    
    action set_code_bwd_iat_total (bit<64> code) {
        meta.code_pk_bwd_iat_total = code;
    }
    
    action set_code_bwd_packet_length_max (bit<104> code) {
        meta.code_pk_bwd_packet_length_max = code;
    }
    
    action set_code_bwd_packet_length_min (bit<40> code) {
        meta.code_pk_bwd_packet_length_min = code;
    }
    
    action set_code_bwd_packet_length_total (bit<80> code) {
        meta.code_pk_bwd_packet_length_total = code;
    }
    
    action set_code_flow_duration (bit<72> code) {
        meta.code_pk_flow_duration = code;
    }
    
    action set_code_flow_iat_min (bit<64> code) {
        meta.code_pk_flow_iat_min = code;
    }
    
    action set_code_fwd_packet_length_max (bit<96> code) {
        meta.code_pk_fwd_packet_length_max = code;
    }
    
    action set_code_fwd_packet_length_min (bit<40> code) {
        meta.code_pk_fwd_packet_length_min = code;
    }
    
    action set_code_fwd_packet_length_total (bit<72> code) {
        meta.code_pk_fwd_packet_length_total = code;
    }
    
    action set_code_min_packet_length (bit<40> code) {
        meta.code_pk_min_packet_length = code;
    }
    
    action set_code_packet_length_total (bit<88> code) {
        meta.code_pk_packet_length_total = code;
    }



    @placement_priority(1)
    table get_classification_tree_app_0 {
        key = {
            meta.code_pk_bwd_iat_max[4:0] : ternary @name("code_pk_bwd_iat_max_4_0_");
            meta.code_pk_bwd_iat_min[14:8] : ternary @name("code_pk_bwd_iat_min_14_8_");
            meta.code_pk_bwd_iat_total[31:16] : ternary @name("code_pk_bwd_iat_total_31_16_");
            meta.code_pk_bwd_packet_length_max[23:0] : ternary @name("code_pk_bwd_packet_length_max_23_0_");
            meta.code_pk_bwd_packet_length_min[28:24] : ternary @name("code_pk_bwd_packet_length_min_28_24_");
            meta.code_pk_bwd_packet_length_total[15:0] : ternary @name("code_pk_bwd_packet_length_total_15_0_");
            meta.code_pk_flow_duration[31:16] : ternary @name("code_pk_flow_duration_31_16_");
            meta.code_pk_flow_iat_min[6:0] : ternary @name("code_pk_flow_iat_min_6_0_");
            meta.code_pk_fwd_packet_length_max[55:40] : ternary @name("code_pk_fwd_packet_length_max_55_40_");
            meta.code_pk_fwd_packet_length_min[31:24] : ternary @name("code_pk_fwd_packet_length_min_31_24_");
            meta.code_pk_fwd_packet_length_total[7:0] : ternary @name("code_pk_fwd_packet_length_total_7_0_");
            meta.code_pk_min_packet_length[15:8] : ternary @name("code_pk_min_packet_length_15_8_");
            meta.code_pk_packet_length_total[47:32] : ternary @name("code_pk_packet_length_total_47_32_");
        }
        actions = {
            classify_flow_codeword_app_0;        
        }
        size = 600;
    }
    
    @placement_priority(1)
    table get_classification_tree_app_1 {
        key = {
            meta.code_pk_bwd_iat_max[14:8] : ternary @name("code_pk_bwd_iat_max_14_8_");
            meta.code_pk_bwd_iat_min[31:16] : ternary @name("code_pk_bwd_iat_min_31_16_");
            meta.code_pk_bwd_iat_total[15:0] : ternary @name("code_pk_bwd_iat_total_15_0_");
            meta.code_pk_bwd_packet_length_max[63:32] : ternary @name("code_pk_bwd_packet_length_max_63_32_");
            meta.code_pk_bwd_packet_length_min[7:0] : ternary @name("code_pk_bwd_packet_length_min_7_0_");
            meta.code_pk_bwd_packet_length_total[63:40] : ternary @name("code_pk_bwd_packet_length_total_63_40_");
            meta.code_pk_flow_duration[47:32] : ternary @name("code_pk_flow_duration_47_32_");
            meta.code_pk_flow_iat_min[31:16] : ternary @name("code_pk_flow_iat_min_31_16_");
            meta.code_pk_fwd_packet_length_max[28:0] : ternary @name("code_pk_fwd_packet_length_max_28_0_");
            meta.code_pk_fwd_packet_length_min[7:0] : ternary @name("code_pk_fwd_packet_length_min_7_0_");
            meta.code_pk_fwd_packet_length_total[63:48] : ternary @name("code_pk_fwd_packet_length_total_63_48_");
            meta.code_pk_min_packet_length[7:0] : ternary @name("code_pk_min_packet_length_7_0_");
            meta.code_pk_packet_length_total[76:64] : ternary @name("code_pk_packet_length_total_76_64_");
        }
        actions = {
            classify_flow_codeword_app_1;        
        }
        size = 1100;
    }
    
    @placement_priority(1)
    table get_classification_tree_app_2 {
        key = {
            meta.code_pk_bwd_iat_max[21:16] : ternary @name("code_pk_bwd_iat_max_21_16_");
            meta.code_pk_bwd_iat_min[7:0] : ternary @name("code_pk_bwd_iat_min_7_0_");
            meta.code_pk_bwd_iat_total[36:32] : ternary @name("code_pk_bwd_iat_total_36_32_");
            meta.code_pk_bwd_packet_length_max[101:88] : ternary @name("code_pk_bwd_packet_length_max_101_88_");
            meta.code_pk_bwd_packet_length_min[12:8] : ternary @name("code_pk_bwd_packet_length_min_12_8_");
            meta.code_pk_bwd_packet_length_total[28:16] : ternary @name("code_pk_bwd_packet_length_total_28_16_");
            meta.code_pk_flow_duration[15:0] : ternary @name("code_pk_flow_duration_15_0_");
            meta.code_pk_flow_iat_min[39:32] : ternary @name("code_pk_flow_iat_min_39_32_");
            meta.code_pk_fwd_packet_length_max[95:72] : ternary @name("code_pk_fwd_packet_length_max_95_72_");
            meta.code_pk_fwd_packet_length_min[23:16] : ternary @name("code_pk_fwd_packet_length_min_23_16_");
            meta.code_pk_fwd_packet_length_total[31:8] : ternary @name("code_pk_fwd_packet_length_total_31_8_");
            meta.code_pk_min_packet_length[31:24] : ternary @name("code_pk_min_packet_length_31_24_");
            meta.code_pk_packet_length_total[23:0] : ternary @name("code_pk_packet_length_total_23_0_");
        }
        actions = {
            classify_flow_codeword_app_2;        
        }
        size = 129;
    }
    
    @placement_priority(2)
    table get_classification_tree_ddos_0 {
        key = {
            meta.code_pk_bwd_iat_max[31:24] : ternary @name("code_pk_bwd_iat_max_31_24_");
            meta.code_pk_bwd_iat_min[38:32] : ternary @name("code_pk_bwd_iat_min_38_32_");
            meta.code_pk_bwd_iat_total[47:40] : ternary @name("code_pk_bwd_iat_total_47_40_");
            meta.code_pk_bwd_packet_length_max[70:64] : ternary @name("code_pk_bwd_packet_length_max_70_64_");
            meta.code_pk_bwd_packet_length_total[69:64] : ternary @name("code_pk_bwd_packet_length_total_69_64_");
            meta.code_pk_flow_duration[53:48] : ternary @name("code_pk_flow_duration_53_48_");
            meta.code_pk_flow_iat_min[46:40] : ternary @name("code_pk_flow_iat_min_46_40_");
            meta.code_pk_fwd_packet_length_max[62:56] : ternary @name("code_pk_fwd_packet_length_max_62_56_");
            meta.code_pk_fwd_packet_length_min[15:8] : ternary @name("code_pk_fwd_packet_length_min_15_8_");
            meta.code_pk_fwd_packet_length_total[39:32] : ternary @name("code_pk_fwd_packet_length_total_39_32_");
            meta.code_pk_packet_length_total[63:48] : ternary @name("code_pk_packet_length_total_63_48_");
        }
        actions = {
            classify_flow_codeword_ddos_0;        
        }
        size = 61;
    }
    
    @placement_priority(2)
    table get_classification_tree_ddos_1 {
        key = {
            meta.code_pk_bwd_iat_max[32:32] : ternary @name("code_pk_bwd_iat_max_32_32_");
            meta.code_pk_bwd_iat_min[42:40] : ternary @name("code_pk_bwd_iat_min_42_40_");
            meta.code_pk_bwd_iat_total[49:48] : ternary @name("code_pk_bwd_iat_total_49_48_");
            meta.code_pk_bwd_packet_length_max[31:24] : ternary @name("code_pk_bwd_packet_length_max_31_24_");
            meta.code_pk_bwd_packet_length_min[16:16] : ternary @name("code_pk_bwd_packet_length_min_16_16_");
            meta.code_pk_bwd_packet_length_total[39:32] : ternary @name("code_pk_bwd_packet_length_total_39_32_");
            meta.code_pk_flow_duration[62:56] : ternary @name("code_pk_flow_duration_62_56_");
            meta.code_pk_flow_iat_min[15:8] : ternary @name("code_pk_flow_iat_min_15_8_");
            meta.code_pk_fwd_packet_length_max[37:32] : ternary @name("code_pk_fwd_packet_length_max_37_32_");
            meta.code_pk_fwd_packet_length_total[45:40] : ternary @name("code_pk_fwd_packet_length_total_45_40_");
            meta.code_pk_min_packet_length[16:16] : ternary @name("code_pk_min_packet_length_16_16_");
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
            meta.code_pk_bwd_iat_min[50:48] : ternary @name("code_pk_bwd_iat_min_50_48_");
            meta.code_pk_bwd_iat_total[57:56] : ternary @name("code_pk_bwd_iat_total_57_56_");
            meta.code_pk_bwd_packet_length_max[87:72] : ternary @name("code_pk_bwd_packet_length_max_87_72_");
            meta.code_pk_bwd_packet_length_min[33:32] : ternary @name("code_pk_bwd_packet_length_min_33_32_");
            meta.code_pk_bwd_packet_length_total[73:72] : ternary @name("code_pk_bwd_packet_length_total_73_72_");
            meta.code_pk_flow_duration[65:64] : ternary @name("code_pk_flow_duration_65_64_");
            meta.code_pk_flow_iat_min[57:48] : ternary @name("code_pk_flow_iat_min_57_48_");
            meta.code_pk_fwd_packet_length_max[67:64] : ternary @name("code_pk_fwd_packet_length_max_67_64_");
            meta.code_pk_fwd_packet_length_min[32:32] : ternary @name("code_pk_fwd_packet_length_min_32_32_");
            meta.code_pk_fwd_packet_length_total[65:64] : ternary @name("code_pk_fwd_packet_length_total_65_64_");
            meta.code_pk_min_packet_length[32:32] : ternary @name("code_pk_min_packet_length_32_32_");
            meta.code_pk_packet_length_total[82:80] : ternary @name("code_pk_packet_length_total_82_80_");
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
			meta.bwd_packet_length_min_val = bwd_packet_length_min_action.execute(meta.flow_hash) + 1;
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
