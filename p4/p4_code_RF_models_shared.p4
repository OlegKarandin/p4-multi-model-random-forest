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

@pa_container_size("ingress", "ig_md.flow_iat_max_val", 16)
@pa_container_size("ingress", "ig_md.fwd_packet_length_max_val", 16)
struct metadata_t {
    bit<32> flow_hash;
    bit<1>  fwd;
    bit<16> now_pseudo_us;
    bit<16> current_iat;

	bit<2> class_tree_app_0;
	bit<2> class_tree_app_1;
	bit<2> classification_app;
	bit<1> class_tree_ddos_0;
	bit<1> classification_ddos;
	bit<16> flow_iat_max_val;
	bit<16> fwd_packet_length_max_val;
	bit<1> code_app_flow_iat_max;
	bit<1> code_ddos_flow_iat_max;
	bit<1> code_fwd_packet_length_max;

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
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) flow_last_arrival_time_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) flow_iat_max_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) fwd_packet_length_max_reg;


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

	RegisterAction<bit<16>, bit<32>, bit<16>>(flow_last_arrival_time_reg) flow_last_arrival_time_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			rv = meta.now_pseudo_us - value;
			value = meta.now_pseudo_us;
		}
	};

	RegisterAction<bit<16>, bit<32>, bit<16>>(flow_iat_max_reg) flow_iat_max_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.current_iat > value) {
				value = meta.current_iat;
			}
			rv = value;
		}
	};

	RegisterAction<bit<16>, bit<32>, bit<16>>(fwd_packet_length_max_reg) fwd_packet_length_max_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (hdr.ipv4.total_len > value) {
				value = hdr.ipv4.total_len;
			}
			rv = value;
		}
	};


	action classify_flow_codeword_app_0(bit<2> class){
		meta.class_tree_app_0 = class;
	}

	action classify_flow_codeword_app_1(bit<2> class){
		meta.class_tree_app_1 = class;
	}

	action classify_flow_codeword_ddos_0(bit<1> class){
		meta.class_tree_ddos_0 = class;
	}

    
    action set_code_app_flow_iat_max (bit<1> code) {
        meta.code_app_flow_iat_max = code;
    }
    
    action set_code_ddos_flow_iat_max (bit<1> code) {
        meta.code_ddos_flow_iat_max = code;
    }
    
    action set_code_fwd_packet_length_max (bit<1> code) {
        meta.code_fwd_packet_length_max = code;
    }



    table get_classification_tree_app_0 {
        key = {
            meta.code_app_flow_iat_max : ternary;
        }
        actions = {
            classify_flow_codeword_app_0;        
        }
        size = 5;
    }
    
    table get_classification_tree_app_1 {
        key = {
            meta.code_app_flow_iat_max : ternary;
        }
        actions = {
            classify_flow_codeword_app_1;        
        }
        size = 5;
    }
    
    table get_classification_tree_ddos_0 {
        key = {
            meta.code_ddos_flow_iat_max : ternary;
            meta.code_fwd_packet_length_max : ternary;
        }
        actions = {
            classify_flow_codeword_ddos_0;        
        }
        size = 5;
    }
    
    table table_0_app_flow_iat_max {
        key = {
            meta.flow_iat_max_val: range;
        }
        actions = {
            set_code_app_flow_iat_max;        
        }
        size = 2;
    }   
    
    table table_1_ddos_flow_iat_max {
        key = {
            meta.flow_iat_max_val: range;
        }
        actions = {
            set_code_ddos_flow_iat_max;        
        }
        size = 2;
    }   
    
    table table_2_fwd_packet_length_max {
        key = {
            meta.fwd_packet_length_max_val: range;
        }
        actions = {
            set_code_fwd_packet_length_max;        
        }
        size = 2;
    }   
    	action set_classification_app(bit<2> winner) {
		meta.classification_app = winner;
	}

	table vote_app {
		key = {
			meta.class_tree_app_0 : exact;
			meta.class_tree_app_1 : exact;
		}
		actions = {
			set_classification_app;
		}
		size = 9;
		const entries = {
			(0, 0) : set_classification_app(0);
			(0, 1) : set_classification_app(0);
			(0, 2) : set_classification_app(0);
			(1, 0) : set_classification_app(0);
			(1, 1) : set_classification_app(1);
			(1, 2) : set_classification_app(1);
			(2, 0) : set_classification_app(0);
			(2, 1) : set_classification_app(1);
			(2, 2) : set_classification_app(2);
		}
	}
	action set_classification_ddos(bit<1> winner) {
		meta.classification_ddos = winner;
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
        if (hdr.tcp.isValid()) {

			calc_flow_hash();
			calc_timestamp();

			meta.fwd = flow_orientation_action.execute(meta.flow_hash);

			meta.current_iat = flow_last_arrival_time_action.execute(meta.flow_hash);
			meta.flow_iat_max_val = flow_iat_max_action.execute(meta.flow_hash);

			if (meta.fwd == 1) {
				meta.fwd_packet_length_max_val = fwd_packet_length_max_action.execute(meta.flow_hash);
			}


			table_0_app_flow_iat_max.apply();
			table_1_ddos_flow_iat_max.apply();
			table_2_fwd_packet_length_max.apply();

			get_classification_tree_app_0.apply();
			get_classification_tree_app_1.apply();

			get_classification_tree_ddos_0.apply();
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
