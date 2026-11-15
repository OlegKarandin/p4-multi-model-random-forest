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
@pa_solitary("ingress", "ig_md.class_tree_ddos_3")
@pa_solitary("ingress", "ig_md.class_tree_ddos_4")
@pa_container_size("ingress", "ig_md.bwd_iat_max_val", 16)
@pa_container_size("ingress", "ig_md.bwd_packet_length_max_val", 16)
@pa_container_size("ingress", "ig_md.bwd_packet_length_total_val", 16)
@pa_container_size("ingress", "ig_md.flow_iat_max_val", 16)
@pa_container_size("ingress", "ig_md.fwd_packet_length_max_val", 16)
@pa_container_size("ingress", "ig_md.fwd_packet_length_total_val", 16)
@pa_container_size("ingress", "ig_md.packet_length_total_val", 16)
@pa_no_overlay("ingress", "ig_md.code_pk_bwd_iat_max")
@pa_container_size("ingress", "ig_md.code_pk_bwd_iat_max", 32, 32, 16, 8)
@pa_no_overlay("ingress", "ig_md.code_pk_bwd_packet_length_max")
@pa_container_size("ingress", "ig_md.code_pk_bwd_packet_length_max", 32, 32, 32, 16)
@pa_no_overlay("ingress", "ig_md.code_pk_bwd_packet_length_total")
@pa_container_size("ingress", "ig_md.code_pk_bwd_packet_length_total", 32, 32, 32, 32, 32)
@pa_no_overlay("ingress", "ig_md.code_pk_flow_iat_max")
@pa_container_size("ingress", "ig_md.code_pk_flow_iat_max", 32, 32, 32, 8)
@pa_no_overlay("ingress", "ig_md.code_pk_fwd_packet_length_max")
@pa_container_size("ingress", "ig_md.code_pk_fwd_packet_length_max", 32, 32, 32, 32, 32)
@pa_no_overlay("ingress", "ig_md.code_pk_fwd_packet_length_total")
@pa_container_size("ingress", "ig_md.code_pk_fwd_packet_length_total", 32, 32, 32, 32, 32)
@pa_no_overlay("ingress", "ig_md.code_pk_packet_length_total")
@pa_container_size("ingress", "ig_md.code_pk_packet_length_total", 32, 32, 32, 32, 32)
@pa_container_size("ingress", "ig_md.bwd_current_iat", 16)
@pa_container_size("ingress", "ig_md.flow_current_iat", 16)
@pa_solitary("ingress", "ig_md.now_pseudo_us")
@pa_solitary("ingress", "ig_md.flow_hash")
struct metadata_t {
    bit<12> flow_hash;
    bit<1>  fwd;
    bit<32> now_pseudo_us;

	bit<16> bwd_current_iat;
	bit<16> flow_current_iat;
	bit<2> class_tree_app_0;
	bit<2> class_tree_app_1;
	bit<2> class_tree_app_2;
	bit<1> class_tree_ddos_0;
	bit<1> class_tree_ddos_1;
	bit<1> class_tree_ddos_2;
	bit<1> class_tree_ddos_3;
	bit<1> class_tree_ddos_4;
	bit<16> bwd_iat_max_val;
	bit<16> bwd_packet_length_max_val;
	bit<16> bwd_packet_length_total_val;
	bit<16> flow_iat_max_val;
	bit<16> fwd_packet_length_max_val;
	bit<16> fwd_packet_length_total_val;
	bit<16> packet_length_total_val;
	bit<88> code_pk_bwd_iat_max;
	bit<112> code_pk_bwd_packet_length_max;
	bit<160> code_pk_bwd_packet_length_total;
	bit<104> code_pk_flow_iat_max;
	bit<160> code_pk_fwd_packet_length_max;
	bit<160> code_pk_fwd_packet_length_total;
	bit<160> code_pk_packet_length_total;

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
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) bwd_packet_length_max_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) bwd_packet_length_total_reg;
	Register<bit<32>, bit<12>>(MAX_NUM_FLOWS, 32w0xFFFFFFFF) flow_last_arrival_time_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) flow_iat_max_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) fwd_packet_length_max_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) fwd_packet_length_total_reg;
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

	RegisterAction<bit<16>, bit<12>, bit<16>>(bwd_packet_length_max_reg) bwd_packet_length_max_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 0 && hdr.ipv4.total_len > value) {
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

	RegisterAction<bit<16>, bit<12>, bit<16>>(flow_iat_max_reg) flow_iat_max_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.flow_current_iat > value) {
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

	RegisterAction<bit<16>, bit<12>, bit<16>>(fwd_packet_length_total_reg) fwd_packet_length_total_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 1) {
				value = value |+| hdr.ipv4.total_len;
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

	action classify_flow_codeword_ddos_3(bit<1> class){
		meta.class_tree_ddos_3 = class;
	}

	action classify_flow_codeword_ddos_4(bit<1> class){
		meta.class_tree_ddos_4 = class;
	}

    
    action set_code_bwd_iat_max (bit<88> code) {
        meta.code_pk_bwd_iat_max = code;
    }
    
    action set_code_bwd_packet_length_max (bit<112> code) {
        meta.code_pk_bwd_packet_length_max = code;
    }
    
    action set_code_bwd_packet_length_total (bit<160> code) {
        meta.code_pk_bwd_packet_length_total = code;
    }
    
    action set_code_flow_iat_max (bit<104> code) {
        meta.code_pk_flow_iat_max = code;
    }
    
    action set_code_fwd_packet_length_max (bit<160> code) {
        meta.code_pk_fwd_packet_length_max = code;
    }
    
    action set_code_fwd_packet_length_total (bit<160> code) {
        meta.code_pk_fwd_packet_length_total = code;
    }
    
    action set_code_packet_length_total (bit<160> code) {
        meta.code_pk_packet_length_total = code;
    }



    @placement_priority(1)
    table get_classification_tree_app_0 {
        key = {
            meta.code_pk_bwd_iat_max[15:0] : ternary @name("code_pk_bwd_iat_max_15_0_");
            meta.code_pk_bwd_packet_length_max[95:80] : ternary @name("code_pk_bwd_packet_length_max_95_80_");
            meta.code_pk_bwd_packet_length_total[20:0] : ternary @name("code_pk_bwd_packet_length_total_20_0_");
            meta.code_pk_flow_iat_max[38:24] : ternary @name("code_pk_flow_iat_max_38_24_");
            meta.code_pk_fwd_packet_length_max[31:0] : ternary @name("code_pk_fwd_packet_length_max_31_0_");
            meta.code_pk_fwd_packet_length_total[63:40] : ternary @name("code_pk_fwd_packet_length_total_63_40_");
            meta.code_pk_packet_length_total[20:0] : ternary @name("code_pk_packet_length_total_20_0_");
        }
        actions = {
            classify_flow_codeword_app_0;        
        }
        size = 132;
    }
    
    @placement_priority(1)
    table get_classification_tree_app_1 {
        key = {
            meta.code_pk_bwd_iat_max[31:16] : ternary @name("code_pk_bwd_iat_max_31_16_");
            meta.code_pk_bwd_packet_length_max[20:0] : ternary @name("code_pk_bwd_packet_length_max_20_0_");
            meta.code_pk_bwd_packet_length_total[55:24] : ternary @name("code_pk_bwd_packet_length_total_55_24_");
            meta.code_pk_flow_iat_max[15:0] : ternary @name("code_pk_flow_iat_max_15_0_");
            meta.code_pk_fwd_packet_length_max[62:32] : ternary @name("code_pk_fwd_packet_length_max_62_32_");
            meta.code_pk_fwd_packet_length_total[84:64] : ternary @name("code_pk_fwd_packet_length_total_84_64_");
            meta.code_pk_packet_length_total[119:88] : ternary @name("code_pk_packet_length_total_119_88_");
        }
        actions = {
            classify_flow_codeword_app_1;        
        }
        size = 157;
    }
    
    @placement_priority(1)
    table get_classification_tree_app_2 {
        key = {
            meta.code_pk_bwd_iat_max[47:32] : ternary @name("code_pk_bwd_iat_max_47_32_");
            meta.code_pk_bwd_packet_length_max[61:40] : ternary @name("code_pk_bwd_packet_length_max_61_40_");
            meta.code_pk_bwd_packet_length_total[84:64] : ternary @name("code_pk_bwd_packet_length_total_84_64_");
            meta.code_pk_flow_iat_max[63:48] : ternary @name("code_pk_flow_iat_max_63_48_");
            meta.code_pk_fwd_packet_length_max[92:64] : ternary @name("code_pk_fwd_packet_length_max_92_64_");
            meta.code_pk_fwd_packet_length_total[39:0] : ternary @name("code_pk_fwd_packet_length_total_39_0_");
            meta.code_pk_packet_length_total[53:24] : ternary @name("code_pk_packet_length_total_53_24_");
        }
        actions = {
            classify_flow_codeword_app_2;        
        }
        size = 161;
    }
    
    @placement_priority(2)
    table get_classification_tree_ddos_0 {
        key = {
            meta.code_pk_bwd_iat_max[51:48] : ternary @name("code_pk_bwd_iat_max_51_48_");
            meta.code_pk_bwd_packet_length_max[39:24] : ternary @name("code_pk_bwd_packet_length_max_39_24_");
            meta.code_pk_bwd_packet_length_total[103:96] : ternary @name("code_pk_bwd_packet_length_total_103_96_");
            meta.code_pk_flow_iat_max[44:40] : ternary @name("code_pk_flow_iat_max_44_40_");
            meta.code_pk_fwd_packet_length_max[110:104] : ternary @name("code_pk_fwd_packet_length_max_110_104_");
            meta.code_pk_fwd_packet_length_total[103:96] : ternary @name("code_pk_fwd_packet_length_total_103_96_");
            meta.code_pk_packet_length_total[127:120] : ternary @name("code_pk_packet_length_total_127_120_");
        }
        actions = {
            classify_flow_codeword_ddos_0;        
        }
        size = 45;
    }
    
    @placement_priority(2)
    table get_classification_tree_ddos_1 {
        key = {
            meta.code_pk_bwd_iat_max[63:56] : ternary @name("code_pk_bwd_iat_max_63_56_");
            meta.code_pk_bwd_packet_length_max[71:64] : ternary @name("code_pk_bwd_packet_length_max_71_64_");
            meta.code_pk_bwd_packet_length_total[119:104] : ternary @name("code_pk_bwd_packet_length_total_119_104_");
            meta.code_pk_flow_iat_max[79:64] : ternary @name("code_pk_flow_iat_max_79_64_");
            meta.code_pk_fwd_packet_length_max[117:112] : ternary @name("code_pk_fwd_packet_length_max_117_112_");
            meta.code_pk_fwd_packet_length_total[92:88] : ternary @name("code_pk_fwd_packet_length_total_92_88_");
            meta.code_pk_packet_length_total[61:56] : ternary @name("code_pk_packet_length_total_61_56_");
        }
        actions = {
            classify_flow_codeword_ddos_1;        
        }
        size = 45;
    }
    
    @placement_priority(2)
    table get_classification_tree_ddos_2 {
        key = {
            meta.code_pk_bwd_iat_max[67:64] : ternary @name("code_pk_bwd_iat_max_67_64_");
            meta.code_pk_bwd_packet_length_max[79:72] : ternary @name("code_pk_bwd_packet_length_max_79_72_");
            meta.code_pk_bwd_packet_length_total[62:56] : ternary @name("code_pk_bwd_packet_length_total_62_56_");
            meta.code_pk_flow_iat_max[95:80] : ternary @name("code_pk_flow_iat_max_95_80_");
            meta.code_pk_fwd_packet_length_max[100:96] : ternary @name("code_pk_fwd_packet_length_max_100_96_");
            meta.code_pk_fwd_packet_length_total[119:104] : ternary @name("code_pk_fwd_packet_length_total_119_104_");
            meta.code_pk_packet_length_total[69:64] : ternary @name("code_pk_packet_length_total_69_64_");
        }
        actions = {
            classify_flow_codeword_ddos_2;        
        }
        size = 48;
    }
    
    @placement_priority(2)
    table get_classification_tree_ddos_3 {
        key = {
            meta.code_pk_bwd_iat_max[72:72] : ternary @name("code_pk_bwd_iat_max_72_72_");
            meta.code_pk_bwd_packet_length_max[100:96] : ternary @name("code_pk_bwd_packet_length_max_100_96_");
            meta.code_pk_bwd_packet_length_total[135:120] : ternary @name("code_pk_bwd_packet_length_total_135_120_");
            meta.code_pk_flow_iat_max[103:96] : ternary @name("code_pk_flow_iat_max_103_96_");
            meta.code_pk_fwd_packet_length_max[125:120] : ternary @name("code_pk_fwd_packet_length_max_125_120_");
            meta.code_pk_fwd_packet_length_total[134:128] : ternary @name("code_pk_fwd_packet_length_total_134_128_");
            meta.code_pk_packet_length_total[81:72] : ternary @name("code_pk_packet_length_total_81_72_");
        }
        actions = {
            classify_flow_codeword_ddos_3;        
        }
        size = 48;
    }
    
    @placement_priority(2)
    table get_classification_tree_ddos_4 {
        key = {
            meta.code_pk_bwd_iat_max[84:80] : ternary @name("code_pk_bwd_iat_max_84_80_");
            meta.code_pk_bwd_packet_length_max[109:104] : ternary @name("code_pk_bwd_packet_length_max_109_104_");
            meta.code_pk_bwd_packet_length_total[95:88] : ternary @name("code_pk_bwd_packet_length_total_95_88_");
            meta.code_pk_flow_iat_max[22:16] : ternary @name("code_pk_flow_iat_max_22_16_");
            meta.code_pk_fwd_packet_length_max[139:128] : ternary @name("code_pk_fwd_packet_length_max_139_128_");
            meta.code_pk_fwd_packet_length_total[125:120] : ternary @name("code_pk_fwd_packet_length_total_125_120_");
            meta.code_pk_packet_length_total[135:128] : ternary @name("code_pk_packet_length_total_135_128_");
        }
        actions = {
            classify_flow_codeword_ddos_4;        
        }
        size = 53;
    }
    
    @placement_priority(3)
    table table_0_bwd_iat_max {
        key = {
            meta.bwd_iat_max_val: range;
        }
        actions = {
            set_code_bwd_iat_max;        
        }
        size = 41;
    }   
    
    @placement_priority(3)
    table table_1_bwd_packet_length_max {
        key = {
            meta.bwd_packet_length_max_val: range;
        }
        actions = {
            set_code_bwd_packet_length_max;        
        }
        size = 78;
    }   
    
    @placement_priority(3)
    table table_2_bwd_packet_length_total {
        key = {
            meta.bwd_packet_length_total_val: range;
        }
        actions = {
            set_code_bwd_packet_length_total;        
        }
        size = 90;
    }   
    
    @placement_priority(3)
    table table_3_flow_iat_max {
        key = {
            meta.flow_iat_max_val: range;
        }
        actions = {
            set_code_flow_iat_max;        
        }
        size = 73;
    }   
    
    @placement_priority(3)
    table table_4_fwd_packet_length_max {
        key = {
            meta.fwd_packet_length_max_val: range;
        }
        actions = {
            set_code_fwd_packet_length_max;        
        }
        size = 103;
    }   
    
    @placement_priority(3)
    table table_5_fwd_packet_length_total {
        key = {
            meta.fwd_packet_length_total_val: range;
        }
        actions = {
            set_code_fwd_packet_length_total;        
        }
        size = 86;
    }   
    
    @placement_priority(3)
    table table_6_packet_length_total {
        key = {
            meta.packet_length_total_val: range;
        }
        actions = {
            set_code_packet_length_total;        
        }
        size = 92;
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
			meta.class_tree_ddos_3 : exact;
			meta.class_tree_ddos_4 : exact;
		}
		actions = {
			set_classification_ddos;
		}
		size = 32;
		const entries = {
			(0, 0, 0, 0, 0) : set_classification_ddos(0);
			(0, 0, 0, 0, 1) : set_classification_ddos(0);
			(0, 0, 0, 1, 0) : set_classification_ddos(0);
			(0, 0, 0, 1, 1) : set_classification_ddos(0);
			(0, 0, 1, 0, 0) : set_classification_ddos(0);
			(0, 0, 1, 0, 1) : set_classification_ddos(0);
			(0, 0, 1, 1, 0) : set_classification_ddos(0);
			(0, 0, 1, 1, 1) : set_classification_ddos(1);
			(0, 1, 0, 0, 0) : set_classification_ddos(0);
			(0, 1, 0, 0, 1) : set_classification_ddos(0);
			(0, 1, 0, 1, 0) : set_classification_ddos(0);
			(0, 1, 0, 1, 1) : set_classification_ddos(1);
			(0, 1, 1, 0, 0) : set_classification_ddos(0);
			(0, 1, 1, 0, 1) : set_classification_ddos(1);
			(0, 1, 1, 1, 0) : set_classification_ddos(1);
			(0, 1, 1, 1, 1) : set_classification_ddos(1);
			(1, 0, 0, 0, 0) : set_classification_ddos(0);
			(1, 0, 0, 0, 1) : set_classification_ddos(0);
			(1, 0, 0, 1, 0) : set_classification_ddos(0);
			(1, 0, 0, 1, 1) : set_classification_ddos(1);
			(1, 0, 1, 0, 0) : set_classification_ddos(0);
			(1, 0, 1, 0, 1) : set_classification_ddos(1);
			(1, 0, 1, 1, 0) : set_classification_ddos(1);
			(1, 0, 1, 1, 1) : set_classification_ddos(1);
			(1, 1, 0, 0, 0) : set_classification_ddos(0);
			(1, 1, 0, 0, 1) : set_classification_ddos(1);
			(1, 1, 0, 1, 0) : set_classification_ddos(1);
			(1, 1, 0, 1, 1) : set_classification_ddos(1);
			(1, 1, 1, 0, 0) : set_classification_ddos(1);
			(1, 1, 1, 0, 1) : set_classification_ddos(1);
			(1, 1, 1, 1, 0) : set_classification_ddos(1);
			(1, 1, 1, 1, 1) : set_classification_ddos(1);
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

			meta.flow_current_iat = (bit<16>) flow_last_arrival_time_action.execute(meta.flow_hash);
			meta.flow_iat_max_val = flow_iat_max_action.execute(meta.flow_hash);
			meta.packet_length_total_val = packet_length_total_action.execute(meta.flow_hash);

			meta.fwd_packet_length_max_val = fwd_packet_length_max_action.execute(meta.flow_hash);
			meta.fwd_packet_length_total_val = fwd_packet_length_total_action.execute(meta.flow_hash);

			meta.bwd_current_iat = (bit<16>) bwd_last_arrival_time_action.execute(meta.flow_hash);
			meta.bwd_iat_max_val = bwd_iat_max_action.execute(meta.flow_hash);
			meta.bwd_packet_length_max_val = bwd_packet_length_max_action.execute(meta.flow_hash);
			meta.bwd_packet_length_total_val = bwd_packet_length_total_action.execute(meta.flow_hash);


			table_0_bwd_iat_max.apply();
			table_1_bwd_packet_length_max.apply();
			table_2_bwd_packet_length_total.apply();
			table_3_flow_iat_max.apply();
			table_4_fwd_packet_length_max.apply();
			table_5_fwd_packet_length_total.apply();
			table_6_packet_length_total.apply();

			get_classification_tree_app_0.apply();
			get_classification_tree_app_1.apply();
			get_classification_tree_app_2.apply();

			get_classification_tree_ddos_0.apply();
			get_classification_tree_ddos_1.apply();
			get_classification_tree_ddos_2.apply();
			get_classification_tree_ddos_3.apply();
			get_classification_tree_ddos_4.apply();
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
