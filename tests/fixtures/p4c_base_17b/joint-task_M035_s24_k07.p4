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
@pa_container_size("ingress", "ig_md.bwd_iat_mean_val", 16)
@pa_container_size("ingress", "ig_md.bwd_packet_length_max_val", 16)
@pa_container_size("ingress", "ig_md.bwd_packet_length_mean_val", 16)
@pa_container_size("ingress", "ig_md.bwd_packet_length_min_val", 16)
@pa_container_size("ingress", "ig_md.fwd_packet_length_max_val", 16)
@pa_container_size("ingress", "ig_md.fwd_packet_length_mean_val", 16)
@pa_container_size("ingress", "ig_md.packet_length_mean_val", 16)
@pa_no_overlay("ingress", "ig_md.code_tk_app_bwd_iat_mean")
@pa_container_size("ingress", "ig_md.code_tk_app_bwd_iat_mean", 32)
@pa_no_overlay("ingress", "ig_md.code_tk_ddos_bwd_iat_mean")
@pa_container_size("ingress", "ig_md.code_tk_ddos_bwd_iat_mean", 16)
@pa_no_overlay("ingress", "ig_md.code_tk_app_bwd_packet_length_max")
@pa_container_size("ingress", "ig_md.code_tk_app_bwd_packet_length_max", 32)
@pa_no_overlay("ingress", "ig_md.code_tk_ddos_bwd_packet_length_max")
@pa_container_size("ingress", "ig_md.code_tk_ddos_bwd_packet_length_max", 16)
@pa_no_overlay("ingress", "ig_md.code_tk_app_bwd_packet_length_mean")
@pa_container_size("ingress", "ig_md.code_tk_app_bwd_packet_length_mean", 32)
@pa_no_overlay("ingress", "ig_md.code_tk_ddos_bwd_packet_length_mean")
@pa_container_size("ingress", "ig_md.code_tk_ddos_bwd_packet_length_mean", 32)
@pa_no_overlay("ingress", "ig_md.code_tk_app_bwd_packet_length_min")
@pa_container_size("ingress", "ig_md.code_tk_app_bwd_packet_length_min", 8)
@pa_no_overlay("ingress", "ig_md.code_tk_ddos_bwd_packet_length_min")
@pa_container_size("ingress", "ig_md.code_tk_ddos_bwd_packet_length_min", 8)
@pa_no_overlay("ingress", "ig_md.code_tk_app_fwd_packet_length_max")
@pa_container_size("ingress", "ig_md.code_tk_app_fwd_packet_length_max", 32)
@pa_no_overlay("ingress", "ig_md.code_tk_ddos_fwd_packet_length_max")
@pa_container_size("ingress", "ig_md.code_tk_ddos_fwd_packet_length_max", 32)
@pa_no_overlay("ingress", "ig_md.code_tk_app_fwd_packet_length_mean")
@pa_container_size("ingress", "ig_md.code_tk_app_fwd_packet_length_mean", 32)
@pa_no_overlay("ingress", "ig_md.code_tk_ddos_fwd_packet_length_mean")
@pa_container_size("ingress", "ig_md.code_tk_ddos_fwd_packet_length_mean", 32)
@pa_no_overlay("ingress", "ig_md.code_tk_app_packet_length_mean")
@pa_container_size("ingress", "ig_md.code_tk_app_packet_length_mean", 32)
@pa_no_overlay("ingress", "ig_md.code_tk_ddos_packet_length_mean")
@pa_container_size("ingress", "ig_md.code_tk_ddos_packet_length_mean", 32)
@pa_container_size("ingress", "ig_md.bwd_current_iat", 16)
@pa_solitary("ingress", "ig_md.now_pseudo_us")
@pa_solitary("ingress", "ig_md.flow_hash")
struct metadata_t {
    bit<12> flow_hash;
    bit<1>  fwd;
    bit<16> now_pseudo_us;

	bit<16> bwd_current_iat;
	bit<2> class_tree_app_0;
	bit<2> class_tree_app_1;
	bit<2> class_tree_app_2;
	bit<1> class_tree_ddos_0;
	bit<1> class_tree_ddos_1;
	bit<1> class_tree_ddos_2;
	bit<16> bwd_iat_mean_val;
	bit<16> bwd_packet_length_max_val;
	bit<16> bwd_packet_length_mean_val;
	bit<16> bwd_packet_length_min_val;
	bit<16> fwd_packet_length_max_val;
	bit<16> fwd_packet_length_mean_val;
	bit<16> packet_length_mean_val;
	bit<32> code_tk_app_bwd_iat_mean;
	bit<16> code_tk_ddos_bwd_iat_mean;
	bit<32> code_tk_app_bwd_packet_length_max;
	bit<16> code_tk_ddos_bwd_packet_length_max;
	bit<32> code_tk_app_bwd_packet_length_mean;
	bit<32> code_tk_ddos_bwd_packet_length_mean;
	bit<8> code_tk_app_bwd_packet_length_min;
	bit<8> code_tk_ddos_bwd_packet_length_min;
	bit<32> code_tk_app_fwd_packet_length_max;
	bit<32> code_tk_ddos_fwd_packet_length_max;
	bit<32> code_tk_app_fwd_packet_length_mean;
	bit<32> code_tk_ddos_fwd_packet_length_mean;
	bit<32> code_tk_app_packet_length_mean;
	bit<32> code_tk_ddos_packet_length_mean;

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
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) bwd_iat_mean_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) bwd_packet_length_max_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) bwd_packet_length_mean_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS, 65535) bwd_packet_length_min_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) fwd_packet_length_max_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) fwd_packet_length_mean_reg;
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
		meta.now_pseudo_us = ig_prsr_md.global_tstamp[25:10];
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

	MathUnit<bit<16>>(MathOp_t.MUL, 1, 2) bwd_iat_mean_halve_unit;

	RegisterAction<bit<16>, bit<12>, bit<16>>(bwd_iat_mean_reg) bwd_iat_mean_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 0) {
				value = bwd_iat_mean_halve_unit.execute(value + meta.bwd_current_iat);
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

    
    action set_code_bwd_iat_mean (bit<32> code_app, bit<16> code_ddos) {
        meta.code_tk_app_bwd_iat_mean = code_app;
        meta.code_tk_ddos_bwd_iat_mean = code_ddos;
    }
    
    action set_code_bwd_packet_length_max (bit<32> code_app, bit<16> code_ddos) {
        meta.code_tk_app_bwd_packet_length_max = code_app;
        meta.code_tk_ddos_bwd_packet_length_max = code_ddos;
    }
    
    action set_code_bwd_packet_length_mean (bit<32> code_app, bit<32> code_ddos) {
        meta.code_tk_app_bwd_packet_length_mean = code_app;
        meta.code_tk_ddos_bwd_packet_length_mean = code_ddos;
    }
    
    action set_code_bwd_packet_length_min (bit<8> code_app, bit<8> code_ddos) {
        meta.code_tk_app_bwd_packet_length_min = code_app;
        meta.code_tk_ddos_bwd_packet_length_min = code_ddos;
    }
    
    action set_code_fwd_packet_length_max (bit<32> code_app, bit<32> code_ddos) {
        meta.code_tk_app_fwd_packet_length_max = code_app;
        meta.code_tk_ddos_fwd_packet_length_max = code_ddos;
    }
    
    action set_code_fwd_packet_length_mean (bit<32> code_app, bit<32> code_ddos) {
        meta.code_tk_app_fwd_packet_length_mean = code_app;
        meta.code_tk_ddos_fwd_packet_length_mean = code_ddos;
    }
    
    action set_code_packet_length_mean (bit<32> code_app, bit<32> code_ddos) {
        meta.code_tk_app_packet_length_mean = code_app;
        meta.code_tk_ddos_packet_length_mean = code_ddos;
    }



    @placement_priority(1)
    table get_classification_tree_app_0 {
        key = {
            meta.code_tk_app_bwd_iat_mean[16:0] : ternary @name("code_tk_app_bwd_iat_mean_16_0_");
            meta.code_tk_app_bwd_packet_length_max[26:0] : ternary @name("code_tk_app_bwd_packet_length_max_26_0_");
            meta.code_tk_app_bwd_packet_length_mean[17:0] : ternary @name("code_tk_app_bwd_packet_length_mean_17_0_");
            meta.code_tk_app_bwd_packet_length_min[4:0] : ternary @name("code_tk_app_bwd_packet_length_min_4_0_");
            meta.code_tk_app_fwd_packet_length_max[30:0] : ternary @name("code_tk_app_fwd_packet_length_max_30_0_");
            meta.code_tk_app_fwd_packet_length_mean[26:0] : ternary @name("code_tk_app_fwd_packet_length_mean_26_0_");
            meta.code_tk_app_packet_length_mean[21:0] : ternary @name("code_tk_app_packet_length_mean_21_0_");
        }
        actions = {
            classify_flow_codeword_app_0;        
        }
        size = 59;
    }
    
    @placement_priority(1)
    table get_classification_tree_app_1 {
        key = {
            meta.code_tk_app_bwd_iat_mean[16:0] : ternary @name("code_tk_app_bwd_iat_mean_16_0_");
            meta.code_tk_app_bwd_packet_length_max[26:0] : ternary @name("code_tk_app_bwd_packet_length_max_26_0_");
            meta.code_tk_app_bwd_packet_length_mean[17:0] : ternary @name("code_tk_app_bwd_packet_length_mean_17_0_");
            meta.code_tk_app_bwd_packet_length_min[4:0] : ternary @name("code_tk_app_bwd_packet_length_min_4_0_");
            meta.code_tk_app_fwd_packet_length_max[30:0] : ternary @name("code_tk_app_fwd_packet_length_max_30_0_");
            meta.code_tk_app_fwd_packet_length_mean[26:0] : ternary @name("code_tk_app_fwd_packet_length_mean_26_0_");
            meta.code_tk_app_packet_length_mean[21:0] : ternary @name("code_tk_app_packet_length_mean_21_0_");
        }
        actions = {
            classify_flow_codeword_app_1;        
        }
        size = 50;
    }
    
    @placement_priority(1)
    table get_classification_tree_app_2 {
        key = {
            meta.code_tk_app_bwd_iat_mean[16:0] : ternary @name("code_tk_app_bwd_iat_mean_16_0_");
            meta.code_tk_app_bwd_packet_length_max[26:0] : ternary @name("code_tk_app_bwd_packet_length_max_26_0_");
            meta.code_tk_app_bwd_packet_length_mean[17:0] : ternary @name("code_tk_app_bwd_packet_length_mean_17_0_");
            meta.code_tk_app_bwd_packet_length_min[4:0] : ternary @name("code_tk_app_bwd_packet_length_min_4_0_");
            meta.code_tk_app_fwd_packet_length_max[30:0] : ternary @name("code_tk_app_fwd_packet_length_max_30_0_");
            meta.code_tk_app_fwd_packet_length_mean[26:0] : ternary @name("code_tk_app_fwd_packet_length_mean_26_0_");
            meta.code_tk_app_packet_length_mean[21:0] : ternary @name("code_tk_app_packet_length_mean_21_0_");
        }
        actions = {
            classify_flow_codeword_app_2;        
        }
        size = 65;
    }
    
    @placement_priority(2)
    table get_classification_tree_ddos_0 {
        key = {
            meta.code_tk_ddos_bwd_iat_mean[9:0] : ternary @name("code_tk_ddos_bwd_iat_mean_9_0_");
            meta.code_tk_ddos_bwd_packet_length_max[10:0] : ternary @name("code_tk_ddos_bwd_packet_length_max_10_0_");
            meta.code_tk_ddos_bwd_packet_length_mean[20:0] : ternary @name("code_tk_ddos_bwd_packet_length_mean_20_0_");
            meta.code_tk_ddos_fwd_packet_length_max[19:0] : ternary @name("code_tk_ddos_fwd_packet_length_max_19_0_");
            meta.code_tk_ddos_fwd_packet_length_mean[22:0] : ternary @name("code_tk_ddos_fwd_packet_length_mean_22_0_");
            meta.code_tk_ddos_packet_length_mean[28:0] : ternary @name("code_tk_ddos_packet_length_mean_28_0_");
        }
        actions = {
            classify_flow_codeword_ddos_0;        
        }
        size = 46;
    }
    
    @placement_priority(2)
    table get_classification_tree_ddos_1 {
        key = {
            meta.code_tk_ddos_bwd_iat_mean[9:0] : ternary @name("code_tk_ddos_bwd_iat_mean_9_0_");
            meta.code_tk_ddos_bwd_packet_length_max[10:0] : ternary @name("code_tk_ddos_bwd_packet_length_max_10_0_");
            meta.code_tk_ddos_bwd_packet_length_mean[20:0] : ternary @name("code_tk_ddos_bwd_packet_length_mean_20_0_");
            meta.code_tk_ddos_bwd_packet_length_min[1:0] : ternary @name("code_tk_ddos_bwd_packet_length_min_1_0_");
            meta.code_tk_ddos_fwd_packet_length_max[19:0] : ternary @name("code_tk_ddos_fwd_packet_length_max_19_0_");
            meta.code_tk_ddos_fwd_packet_length_mean[22:0] : ternary @name("code_tk_ddos_fwd_packet_length_mean_22_0_");
            meta.code_tk_ddos_packet_length_mean[28:0] : ternary @name("code_tk_ddos_packet_length_mean_28_0_");
        }
        actions = {
            classify_flow_codeword_ddos_1;        
        }
        size = 42;
    }
    
    @placement_priority(2)
    table get_classification_tree_ddos_2 {
        key = {
            meta.code_tk_ddos_bwd_iat_mean[9:0] : ternary @name("code_tk_ddos_bwd_iat_mean_9_0_");
            meta.code_tk_ddos_bwd_packet_length_max[10:0] : ternary @name("code_tk_ddos_bwd_packet_length_max_10_0_");
            meta.code_tk_ddos_bwd_packet_length_mean[20:0] : ternary @name("code_tk_ddos_bwd_packet_length_mean_20_0_");
            meta.code_tk_ddos_bwd_packet_length_min[1:0] : ternary @name("code_tk_ddos_bwd_packet_length_min_1_0_");
            meta.code_tk_ddos_fwd_packet_length_max[19:0] : ternary @name("code_tk_ddos_fwd_packet_length_max_19_0_");
            meta.code_tk_ddos_fwd_packet_length_mean[22:0] : ternary @name("code_tk_ddos_fwd_packet_length_mean_22_0_");
            meta.code_tk_ddos_packet_length_mean[28:0] : ternary @name("code_tk_ddos_packet_length_mean_28_0_");
        }
        actions = {
            classify_flow_codeword_ddos_2;        
        }
        size = 54;
    }
    
    @placement_priority(3)
    table table_0_bwd_iat_mean {
        key = {
            meta.bwd_iat_mean_val: range;
        }
        actions = {
            set_code_bwd_iat_mean;        
        }
        size = 27;
    }   
    
    @placement_priority(3)
    table table_1_bwd_packet_length_max {
        key = {
            meta.bwd_packet_length_max_val: range;
        }
        actions = {
            set_code_bwd_packet_length_max;        
        }
        size = 39;
    }   
    
    @placement_priority(3)
    table table_2_bwd_packet_length_mean {
        key = {
            meta.bwd_packet_length_mean_val: range;
        }
        actions = {
            set_code_bwd_packet_length_mean;        
        }
        size = 39;
    }   
    
    @placement_priority(3)
    table table_3_bwd_packet_length_min {
        key = {
            meta.bwd_packet_length_min_val: range;
        }
        actions = {
            set_code_bwd_packet_length_min;        
        }
        size = 7;
    }   
    
    @placement_priority(3)
    table table_4_fwd_packet_length_max {
        key = {
            meta.fwd_packet_length_max_val: range;
        }
        actions = {
            set_code_fwd_packet_length_max;        
        }
        size = 52;
    }   
    
    @placement_priority(3)
    table table_5_fwd_packet_length_mean {
        key = {
            meta.fwd_packet_length_mean_val: range;
        }
        actions = {
            set_code_fwd_packet_length_mean;        
        }
        size = 48;
    }   
    
    @placement_priority(3)
    table table_6_packet_length_mean {
        key = {
            meta.packet_length_mean_val: range;
        }
        actions = {
            set_code_packet_length_mean;        
        }
        size = 51;
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

			meta.packet_length_mean_val = packet_length_mean_action.execute(meta.flow_hash);

			meta.fwd_packet_length_max_val = fwd_packet_length_max_action.execute(meta.flow_hash);
			meta.fwd_packet_length_mean_val = fwd_packet_length_mean_action.execute(meta.flow_hash);

			meta.bwd_current_iat = bwd_last_arrival_time_action.execute(meta.flow_hash);
			meta.bwd_iat_mean_val = bwd_iat_mean_action.execute(meta.flow_hash);
			meta.bwd_packet_length_max_val = bwd_packet_length_max_action.execute(meta.flow_hash);
			meta.bwd_packet_length_mean_val = bwd_packet_length_mean_action.execute(meta.flow_hash);
			meta.bwd_packet_length_min_val = bwd_packet_length_min_action.execute(meta.flow_hash);


			table_0_bwd_iat_mean.apply();
			table_1_bwd_packet_length_max.apply();
			table_2_bwd_packet_length_mean.apply();
			table_3_bwd_packet_length_min.apply();
			table_4_fwd_packet_length_max.apply();
			table_5_fwd_packet_length_mean.apply();
			table_6_packet_length_mean.apply();

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
