/* -*- P4_16 -*- */
#include <core.p4>
// Tofino-1 TNA only: the resource model's constants (p4model/target.py) and the
// export widths (program.VOTE_EXPORT_BITS) are Tofino-1's.
#include <tna.p4>

#include "p4_headers.p4"
#include "p4_util.p4"

const bit<32> MAX_NUM_FLOWS = 4096;

@pa_solitary("ingress", "ig_md.class_tree_ddos_0")
@pa_no_overlay("ingress", "ig_md.class_tree_ddos_0")
@pa_solitary("ingress", "ig_md.class_tree_ddos_1")
@pa_no_overlay("ingress", "ig_md.class_tree_ddos_1")
@pa_solitary("ingress", "ig_md.class_tree_ddos_2")
@pa_no_overlay("ingress", "ig_md.class_tree_ddos_2")
@pa_container_size("ingress", "ig_md.bwd_iat_total_val", 16)
@pa_container_size("ingress", "ig_md.bwd_packet_length_max_val", 16)
@pa_container_size("ingress", "ig_md.bwd_packet_length_min_val", 16)
@pa_container_size("ingress", "ig_md.bwd_packet_length_total_val", 16)
@pa_container_size("ingress", "ig_md.min_packet_length_val", 16)
@pa_container_size("ingress", "ig_md.packet_length_total_val", 16)
@pa_container_size("ingress", "ig_md.code_pk_bwd_iat_total", 32, 8)
@pa_container_size("ingress", "ig_md.code_pk_bwd_packet_length_max", 32, 8)
@pa_container_size("ingress", "ig_md.code_pk_bwd_packet_length_min", 16, 8)
@pa_container_size("ingress", "ig_md.code_pk_bwd_packet_length_total", 32, 8)
@pa_container_size("ingress", "ig_md.code_pk_min_packet_length", 32)
@pa_container_size("ingress", "ig_md.code_pk_packet_length_total", 32, 16)
@pa_solitary("ingress", "ig_md.now_coarse")
@pa_solitary("ingress", "ig_md.flow_hash")
struct metadata_t {
    bit<12> flow_hash;
    bit<1>  fwd;
    bit<32> now_pseudo_us;

	bit<32> now_coarse;
	bit<1> class_tree_ddos_0;
	bit<1> class_tree_ddos_1;
	bit<1> class_tree_ddos_2;
	bit<16> bwd_iat_total_val;
	bit<16> bwd_packet_length_max_val;
	bit<16> bwd_packet_length_min_val;
	bit<16> bwd_packet_length_total_val;
	bit<16> min_packet_length_val;
	bit<16> packet_length_total_val;
	bit<40> code_pk_bwd_iat_total;
	bit<40> code_pk_bwd_packet_length_max;
	bit<24> code_pk_bwd_packet_length_min;
	bit<40> code_pk_bwd_packet_length_total;
	bit<32> code_pk_min_packet_length;
	bit<48> code_pk_packet_length_total;

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
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) bwd_packet_length_max_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS, 65535) bwd_packet_length_min_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) bwd_packet_length_total_reg;
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
		ig_tm_md.qid = (bit<5>) class;
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

    
    action set_code_bwd_iat_total (bit<40> code) {
        meta.code_pk_bwd_iat_total = code;
    }
    
    action set_code_bwd_packet_length_max (bit<40> code) {
        meta.code_pk_bwd_packet_length_max = code;
    }
    
    action set_code_bwd_packet_length_min (bit<24> code) {
        meta.code_pk_bwd_packet_length_min = code;
    }
    
    action set_code_bwd_packet_length_total (bit<40> code) {
        meta.code_pk_bwd_packet_length_total = code;
    }
    
    action set_code_min_packet_length (bit<32> code) {
        meta.code_pk_min_packet_length = code;
    }
    
    action set_code_packet_length_total (bit<48> code) {
        meta.code_pk_packet_length_total = code;
    }



    @placement_priority(1)
    table get_classification_tree_app_0 {
        key = {
            meta.code_pk_bwd_iat_total[15:0] : ternary @name("code_pk_bwd_iat_total_15_0_");
            meta.code_pk_bwd_packet_length_max[31:16] : ternary @name("code_pk_bwd_packet_length_max_31_16_");
            meta.code_pk_bwd_packet_length_min[2:0] : ternary @name("code_pk_bwd_packet_length_min_2_0_");
            meta.code_pk_bwd_packet_length_total[23:8] : ternary @name("code_pk_bwd_packet_length_total_23_8_");
            meta.code_pk_min_packet_length[31:24] : ternary @name("code_pk_min_packet_length_31_24_");
            meta.code_pk_packet_length_total[23:0] : ternary @name("code_pk_packet_length_total_23_0_");
        }
        actions = {
            classify_flow_codeword_app_0;        
        }
        size = 69;
    }
    
    @placement_priority(2)
    table get_classification_tree_ddos_0 {
        key = {
            meta.code_pk_bwd_iat_total[18:16] : ternary @name("code_pk_bwd_iat_total_18_16_");
            meta.code_pk_bwd_packet_length_max[5:0] : ternary @name("code_pk_bwd_packet_length_max_5_0_");
            meta.code_pk_bwd_packet_length_min[8:8] : ternary @name("code_pk_bwd_packet_length_min_8_8_");
            meta.code_pk_bwd_packet_length_total[31:24] : ternary @name("code_pk_bwd_packet_length_total_31_24_");
            meta.code_pk_min_packet_length[7:0] : ternary @name("code_pk_min_packet_length_7_0_");
            meta.code_pk_packet_length_total[31:24] : ternary @name("code_pk_packet_length_total_31_24_");
        }
        actions = {
            classify_flow_codeword_ddos_0;        
        }
        size = 23;
    }
    
    @placement_priority(2)
    table get_classification_tree_ddos_1 {
        key = {
            meta.code_pk_bwd_iat_total[25:24] : ternary @name("code_pk_bwd_iat_total_25_24_");
            meta.code_pk_bwd_packet_length_max[12:8] : ternary @name("code_pk_bwd_packet_length_max_12_8_");
            meta.code_pk_bwd_packet_length_total[2:0] : ternary @name("code_pk_bwd_packet_length_total_2_0_");
            meta.code_pk_min_packet_length[16:16] : ternary @name("code_pk_min_packet_length_16_16_");
            meta.code_pk_packet_length_total[38:32] : ternary @name("code_pk_packet_length_total_38_32_");
        }
        actions = {
            classify_flow_codeword_ddos_1;        
        }
        size = 19;
    }
    
    @placement_priority(2)
    table get_classification_tree_ddos_2 {
        key = {
            meta.code_pk_bwd_iat_total[39:32] : ternary @name("code_pk_bwd_iat_total_39_32_");
            meta.code_pk_bwd_packet_length_max[39:32] : ternary @name("code_pk_bwd_packet_length_max_39_32_");
            meta.code_pk_bwd_packet_length_min[23:16] : ternary @name("code_pk_bwd_packet_length_min_23_16_");
            meta.code_pk_bwd_packet_length_total[39:32] : ternary @name("code_pk_bwd_packet_length_total_39_32_");
            meta.code_pk_min_packet_length[8:8] : ternary @name("code_pk_min_packet_length_8_8_");
            meta.code_pk_packet_length_total[47:40] : ternary @name("code_pk_packet_length_total_47_40_");
        }
        actions = {
            classify_flow_codeword_ddos_2;        
        }
        size = 25;
    }
    
    @placement_priority(3)
    table table_0_bwd_iat_total {
        key = {
            meta.bwd_iat_total_val: range;
        }
        actions = {
            set_code_bwd_iat_total;        
        }
        size = 16;
    }   
    
    @placement_priority(3)
    table table_1_bwd_packet_length_max {
        key = {
            meta.bwd_packet_length_max_val: range;
        }
        actions = {
            set_code_bwd_packet_length_max;        
        }
        size = 29;
    }   
    
    @placement_priority(3)
    table table_2_bwd_packet_length_min {
        key = {
            meta.bwd_packet_length_min_val: range;
        }
        actions = {
            set_code_bwd_packet_length_min;        
        }
        size = 4;
    }   
    
    @placement_priority(3)
    table table_3_bwd_packet_length_total {
        key = {
            meta.bwd_packet_length_total_val: range;
        }
        actions = {
            set_code_bwd_packet_length_total;        
        }
        size = 24;
    }   
    
    @placement_priority(3)
    table table_4_min_packet_length {
        key = {
            meta.min_packet_length_val: range;
        }
        actions = {
            set_code_min_packet_length;        
        }
        size = 4;
    }   
    
    @placement_priority(3)
    table table_5_packet_length_total {
        key = {
            meta.packet_length_total_val: range;
        }
        actions = {
            set_code_packet_length_total;        
        }
        size = 34;
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

			meta.min_packet_length_val = min_packet_length_action.execute(meta.flow_hash);
			meta.packet_length_total_val = packet_length_total_action.execute(meta.flow_hash);

			meta.bwd_iat_total_val = (bit<16>) bwd_iat_total_action.execute(meta.flow_hash);
			meta.bwd_packet_length_max_val = bwd_packet_length_max_action.execute(meta.flow_hash);
			meta.bwd_packet_length_min_val = bwd_packet_length_min_action.execute(meta.flow_hash) + 1;
			meta.bwd_packet_length_total_val = bwd_packet_length_total_action.execute(meta.flow_hash);


			table_0_bwd_iat_total.apply();
			table_1_bwd_packet_length_max.apply();
			table_2_bwd_packet_length_min.apply();
			table_3_bwd_packet_length_total.apply();
			table_4_min_packet_length.apply();
			table_5_packet_length_total.apply();

			get_classification_tree_app_0.apply();

			get_classification_tree_ddos_0.apply();
			get_classification_tree_ddos_1.apply();
			get_classification_tree_ddos_2.apply();
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
