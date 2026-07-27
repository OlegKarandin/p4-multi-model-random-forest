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

struct metadata_t {
    bit<32> flow_hash;
    bit<32> flow_hash_self;
    bit<32> flow_hash_other;
    bit<1>  fwd;
    bit<16> now_pseudo_us;
    bit<16> current_iat;

	bit<2> class_tree_app_0;
	bit<1> class_tree_app_0_is_set;
	bit<2> class_tree_app_1;
	bit<1> class_tree_app_1_is_set;
	bit<2> class_tree_app_2;
	bit<1> class_tree_app_2_is_set;
	bit<2> classification_app;
	bit<16> flow_iat_max_val;
	bit<25> code_flow_iat_max;
	bit<16> flow_iat_mean_val;
	bit<27> code_flow_iat_mean;
	bit<16> fwd_iat_max_val;
	bit<36> code_fwd_iat_max;
	bit<16> fwd_packet_length_max_val;
	bit<57> code_fwd_packet_length_max;

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

	Hash<bit<32>>(HashAlgorithm_t.CRC32) flow_hash_calc_self;
	Hash<bit<32>>(HashAlgorithm_t.CRC32) flow_hash_calc_other;
	Register<bit<1>, bit<32>>(MAX_NUM_FLOWS) flows_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) flow_last_arrival_time_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) flow_iat_max_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) flow_iat_mean_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) fwd_last_arrival_time_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) fwd_iat_max_reg;
	Register<bit<16>, bit<32>>(MAX_NUM_FLOWS) fwd_packet_length_max_reg;


	action calc_flow_hash_self() {
		meta.flow_hash_self = flow_hash_calc_self.get({
			hdr.ipv4.src_addr,
			hdr.ipv4.dst_addr,
			hdr.ipv4.protocol,
			hdr.tcp.src_port,
			hdr.tcp.dst_port
		}) & 0xFFF;
	}

	action calc_flow_hash_other() {
		meta.flow_hash_other = flow_hash_calc_other.get({
			hdr.ipv4.dst_addr,
			hdr.ipv4.src_addr,
			hdr.ipv4.protocol,
			hdr.tcp.dst_port,
			hdr.tcp.src_port
		}) & 0xFFF;
	}

	action calc_timestamp() {
		meta.now_pseudo_us = (bit<16>)(ig_prsr_md.global_tstamp >> 10);
	}

	RegisterAction<bit<1>, bit<32>, bit<1>>(flows_reg) flows_test_other = {
		void apply(inout bit<1> value, out bit<1> rv) {
			rv = value;
		}
	};

	RegisterAction<bit<1>, bit<32>, bit<1>>(flows_reg) flows_set_self = {
		void apply(inout bit<1> value, out bit<1> rv) {
			value = 1;
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
			if (meta.current_iat > value) {
				value = meta.current_iat;
			}
			rv = value;
		}
	};

	MathUnit<bit<16>>(MathOp_t.MUL, 1, 2) flow_iat_mean_halve_unit;

	RegisterAction<bit<16>, bit<32>, bit<16>>(flow_iat_mean_reg) flow_iat_mean_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			value = flow_iat_mean_halve_unit.execute(value + meta.current_iat);
			rv = value;
		}
	};

	RegisterAction<bit<16>, bit<32>, bit<16>>(fwd_last_arrival_time_reg) fwd_last_arrival_time_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			rv = meta.now_pseudo_us - value;
			value = meta.now_pseudo_us;
		}
	};

	RegisterAction<bit<16>, bit<32>, bit<16>>(fwd_iat_max_reg) fwd_iat_max_action = {
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

	action classify_flow_codeword_app_2(bit<2> class){
		meta.class_tree_app_2 = class;
	}

    
    action set_code_flow_iat_max (bit<25> code) {
        meta.code_flow_iat_max = code;
    }
    
    action set_code_flow_iat_mean (bit<27> code) {
        meta.code_flow_iat_mean = code;
    }
    
    action set_code_fwd_iat_max (bit<36> code) {
        meta.code_fwd_iat_max = code;
    }
    
    action set_code_fwd_packet_length_max (bit<57> code) {
        meta.code_fwd_packet_length_max = code;
    }



    table get_classification_tree_app_0 {
        key = {
            meta.code_flow_iat_max : ternary;
            meta.code_flow_iat_mean : ternary;
            meta.code_fwd_iat_max : ternary;
            meta.code_fwd_packet_length_max : ternary;
        }
        actions = {
            classify_flow_codeword_app_0;        
        }
        size = 400;
    }
    
    table get_classification_tree_app_1 {
        key = {
            meta.code_flow_iat_max : ternary;
            meta.code_flow_iat_mean : ternary;
            meta.code_fwd_iat_max : ternary;
            meta.code_fwd_packet_length_max : ternary;
        }
        actions = {
            classify_flow_codeword_app_1;        
        }
        size = 400;
    }
    
    table get_classification_tree_app_2 {
        key = {
            meta.code_flow_iat_max : ternary;
            meta.code_flow_iat_mean : ternary;
            meta.code_fwd_iat_max : ternary;
            meta.code_fwd_packet_length_max : ternary;
        }
        actions = {
            classify_flow_codeword_app_2;        
        }
        size = 400;
    }
    
    table table_0_flow_iat_max {
        key = {
            meta.flow_iat_max_val: range;
        }
        actions = {
            set_code_flow_iat_max;        
        }
        size = 200;
    }   
    
    table table_1_flow_iat_mean {
        key = {
            meta.flow_iat_mean_val: range;
        }
        actions = {
            set_code_flow_iat_mean;        
        }
        size = 200;
    }   
    
    table table_2_fwd_iat_max {
        key = {
            meta.fwd_iat_max_val: range;
        }
        actions = {
            set_code_fwd_iat_max;        
        }
        size = 200;
    }   
    
    table table_3_fwd_packet_length_max {
        key = {
            meta.fwd_packet_length_max_val: range;
        }
        actions = {
            set_code_fwd_packet_length_max;        
        }
        size = 200;
    }   
    


    action set_classification_app(bit<2> winner) {
        meta.classification_app = winner;
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
        size = 32;
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
            (1, 0, 2) : set_classification_app(1);
            (1, 1, 0) : set_classification_app(1);
            (1, 1, 1) : set_classification_app(1);
            (1, 1, 2) : set_classification_app(1);
            (1, 2, 0) : set_classification_app(1);
            (1, 2, 1) : set_classification_app(1);
            (1, 2, 2) : set_classification_app(2);
            (2, 0, 0) : set_classification_app(0);
            (2, 0, 1) : set_classification_app(2);
            (2, 0, 2) : set_classification_app(2);
            (2, 1, 0) : set_classification_app(2);
            (2, 1, 1) : set_classification_app(1);
            (2, 1, 2) : set_classification_app(2);
            (2, 2, 0) : set_classification_app(2);
            (2, 2, 1) : set_classification_app(2);
            (2, 2, 2) : set_classification_app(2);
        }
    }

    apply {
        if (hdr.tcp.isValid()) {

			calc_flow_hash_self();
			calc_flow_hash_other();
			calc_timestamp();

			bit<1> other_seen = flows_test_other.execute(meta.flow_hash_other);

			if (other_seen == 1) {
				meta.fwd = 0;
				meta.flow_hash = meta.flow_hash_other;
			} else {
				flows_set_self.execute(meta.flow_hash_self);
				meta.fwd = 1;
				meta.flow_hash = meta.flow_hash_self;
			}

			meta.current_iat = flow_last_arrival_time_action.execute(meta.flow_hash);
			meta.flow_iat_max_val = flow_iat_max_action.execute(meta.flow_hash);
			meta.flow_iat_mean_val = flow_iat_mean_action.execute(meta.flow_hash);

			if (meta.fwd == 1) {
				meta.current_iat = fwd_last_arrival_time_action.execute(meta.flow_hash);
				meta.fwd_iat_max_val = fwd_iat_max_action.execute(meta.flow_hash);
				meta.fwd_packet_length_max_val = fwd_packet_length_max_action.execute(meta.flow_hash);
			}


			table_0_flow_iat_max.apply();
			table_1_flow_iat_mean.apply();
			table_2_fwd_iat_max.apply();
			table_3_fwd_packet_length_max.apply();

			get_classification_tree_app_0.apply();
			get_classification_tree_app_1.apply();
			get_classification_tree_app_2.apply();



			vote_app.apply();


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
