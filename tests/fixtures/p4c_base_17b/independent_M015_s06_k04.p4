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
@pa_solitary("ingress", "ig_md.class_tree_ddos_0")
@pa_container_size("ingress", "ig_md.bwd_iat_mean_val", 16)
@pa_container_size("ingress", "ig_md.fwd_iat_min_val", 16)
@pa_container_size("ingress", "ig_md.min_packet_length_val", 16)
@pa_container_size("ingress", "ig_md.packet_length_mean_val", 16)
@pa_container_size("ingress", "ig_md.bwd_iat_min_val", 16)
@pa_container_size("ingress", "ig_md.fwd_iat_mean_val", 16)
@pa_no_overlay("ingress", "ig_md.code_bwd_iat_mean")
@pa_container_size("ingress", "ig_md.code_bwd_iat_mean", 32)
@pa_no_overlay("ingress", "ig_md.code_app_fwd_iat_min")
@pa_container_size("ingress", "ig_md.code_app_fwd_iat_min", 8, 32)
@pa_no_overlay("ingress", "ig_md.code_min_packet_length")
@pa_container_size("ingress", "ig_md.code_min_packet_length", 8)
@pa_no_overlay("ingress", "ig_md.code_app_packet_length_mean")
@pa_container_size("ingress", "ig_md.code_app_packet_length_mean", 32, 32)
@pa_no_overlay("ingress", "ig_md.code_bwd_iat_min")
@pa_container_size("ingress", "ig_md.code_bwd_iat_min", 8)
@pa_no_overlay("ingress", "ig_md.code_fwd_iat_mean")
@pa_container_size("ingress", "ig_md.code_fwd_iat_mean", 8)
@pa_no_overlay("ingress", "ig_md.code_ddos_fwd_iat_min")
@pa_container_size("ingress", "ig_md.code_ddos_fwd_iat_min", 16)
@pa_no_overlay("ingress", "ig_md.code_ddos_packet_length_mean")
@pa_container_size("ingress", "ig_md.code_ddos_packet_length_mean", 8)
@pa_container_size("ingress", "ig_md.bwd_current_iat", 16)
@pa_container_size("ingress", "ig_md.fwd_current_iat", 16)
struct metadata_t {
    bit<32> flow_hash;
    bit<1>  fwd;
    bit<16> now_pseudo_us;

	bit<16> bwd_current_iat;
	bit<16> fwd_current_iat;
	bit<2> class_tree_app_0;
	bit<1> class_tree_ddos_0;
	bit<16> bwd_iat_mean_val;
	bit<16> fwd_iat_min_val;
	bit<16> min_packet_length_val;
	bit<16> packet_length_mean_val;
	bit<16> bwd_iat_min_val;
	bit<16> fwd_iat_mean_val;
	bit<32> code_bwd_iat_mean;
	bit<40> code_app_fwd_iat_min;
	bit<8> code_min_packet_length;
	bit<64> code_app_packet_length_mean;
	bit<8> code_bwd_iat_min;
	bit<8> code_fwd_iat_mean;
	bit<16> code_ddos_fwd_iat_min;
	bit<8> code_ddos_packet_length_mean;

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
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) bwd_iat_mean_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) fwd_last_arrival_time_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS, 65535) fwd_iat_min_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS, 65535) min_packet_length_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) packet_length_mean_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS, 65535) bwd_iat_min_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) fwd_iat_mean_reg;


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

	MathUnit<bit<16>>(MathOp_t.MUL, 1, 2) bwd_iat_mean_halve_unit;

	RegisterAction<bit<16>, bit<32>, bit<16>>(bwd_iat_mean_reg) bwd_iat_mean_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 0) {
				value = bwd_iat_mean_halve_unit.execute(value + meta.bwd_current_iat);
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

	RegisterAction<bit<16>, bit<32>, bit<16>>(fwd_iat_min_reg) fwd_iat_min_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 1 && meta.fwd_current_iat < value) {
				value = meta.fwd_current_iat;
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

	RegisterAction<bit<16>, bit<32>, bit<16>>(bwd_iat_min_reg) bwd_iat_min_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 0 && meta.bwd_current_iat < value) {
				value = meta.bwd_current_iat;
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


	action classify_flow_codeword_app_0(bit<2> class){
		meta.class_tree_app_0 = class;
	}

	action classify_flow_codeword_ddos_0(bit<1> class){
		meta.class_tree_ddos_0 = class;
	}

    
    action set_code_bwd_iat_mean (bit<32> code) {
        meta.code_bwd_iat_mean = code;
    }
    
    action set_code_app_fwd_iat_min (bit<40> code) {
        meta.code_app_fwd_iat_min = code;
    }
    
    action set_code_min_packet_length (bit<8> code) {
        meta.code_min_packet_length = code;
    }
    
    action set_code_app_packet_length_mean (bit<64> code) {
        meta.code_app_packet_length_mean = code;
    }
    
    action set_code_bwd_iat_min (bit<8> code) {
        meta.code_bwd_iat_min = code;
    }
    
    action set_code_fwd_iat_mean (bit<8> code) {
        meta.code_fwd_iat_mean = code;
    }
    
    action set_code_ddos_fwd_iat_min (bit<16> code) {
        meta.code_ddos_fwd_iat_min = code;
    }
    
    action set_code_ddos_packet_length_mean (bit<8> code) {
        meta.code_ddos_packet_length_mean = code;
    }



    @placement_priority(1)
    table get_classification_tree_app_0 {
        key = {
            meta.code_bwd_iat_mean[29:0] : ternary;
            meta.code_app_fwd_iat_min[32:0] : ternary;
            meta.code_min_packet_length[0:0] : ternary;
            meta.code_app_packet_length_mean[60:0] : ternary;
        }
        actions = {
            classify_flow_codeword_app_0;        
        }
        size = 136;
    }
    
    @placement_priority(2)
    table get_classification_tree_ddos_0 {
        key = {
            meta.code_bwd_iat_min[4:0] : ternary;
            meta.code_fwd_iat_mean[1:0] : ternary;
            meta.code_ddos_fwd_iat_min[9:0] : ternary;
            meta.code_ddos_packet_length_mean[5:0] : ternary;
        }
        actions = {
            classify_flow_codeword_ddos_0;        
        }
        size = 24;
    }
    
    @placement_priority(3)
    table table_0_bwd_iat_mean {
        key = {
            meta.bwd_iat_mean_val: range;
        }
        actions = {
            set_code_bwd_iat_mean;        
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
    table table_3_app_packet_length_mean {
        key = {
            meta.packet_length_mean_val: range;
        }
        actions = {
            set_code_app_packet_length_mean;        
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
    table table_5_fwd_iat_mean {
        key = {
            meta.fwd_iat_mean_val: range;
        }
        actions = {
            set_code_fwd_iat_mean;        
        }
        size = 3;
    }   
    
    @placement_priority(3)
    table table_6_ddos_fwd_iat_min {
        key = {
            meta.fwd_iat_min_val: range;
        }
        actions = {
            set_code_ddos_fwd_iat_min;        
        }
        size = 11;
    }   
    
    @placement_priority(3)
    table table_7_ddos_packet_length_mean {
        key = {
            meta.packet_length_mean_val: range;
        }
        actions = {
            set_code_ddos_packet_length_mean;        
        }
        size = 7;
    }   
    	action set_classification_app(bit<2> winner) {
		ig_tm_md.qid = (bit<5>) winner;
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
		ig_tm_md.ingress_cos = (bit<3>) winner;
	}

	table vote_ddos {
		key = {
			meta.class_tree_ddos_0 : exact;
		}
		actions = {
			set_classification_ddos;
		}
		size = 2;
		const entries = {
			(0) : set_classification_ddos(0);
			(1) : set_classification_ddos(1);
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

			meta.min_packet_length_val = min_packet_length_action.execute(meta.flow_hash);
			meta.packet_length_mean_val = packet_length_mean_action.execute(meta.flow_hash);

			meta.fwd_current_iat = fwd_last_arrival_time_action.execute(meta.flow_hash);
			meta.fwd_iat_min_val = fwd_iat_min_action.execute(meta.flow_hash);
			meta.fwd_iat_mean_val = fwd_iat_mean_action.execute(meta.flow_hash);

			meta.bwd_current_iat = bwd_last_arrival_time_action.execute(meta.flow_hash);
			meta.bwd_iat_mean_val = bwd_iat_mean_action.execute(meta.flow_hash);
			meta.bwd_iat_min_val = bwd_iat_min_action.execute(meta.flow_hash);


			table_0_bwd_iat_mean.apply();
			table_1_app_fwd_iat_min.apply();
			table_2_min_packet_length.apply();
			table_3_app_packet_length_mean.apply();
			table_4_bwd_iat_min.apply();
			table_5_fwd_iat_mean.apply();
			table_6_ddos_fwd_iat_min.apply();
			table_7_ddos_packet_length_mean.apply();

			get_classification_tree_app_0.apply();

			get_classification_tree_ddos_0.apply();
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
