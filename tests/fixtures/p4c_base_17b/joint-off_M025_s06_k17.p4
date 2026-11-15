/* -*- P4_16 -*- */
#include <core.p4>
// Tofino-1 TNA only: the resource model's constants (p4model/target.py) and the
// export widths (program.VOTE_EXPORT_BITS) are Tofino-1's.
#include <tna.p4>

#include "p4_headers.p4"
#include "p4_util.p4"

const bit<32> MAX_NUM_FLOWS = 4096;

@pa_container_size("ingress", "ig_md.bwd_iat_max_val", 16)
@pa_container_size("ingress", "ig_md.bwd_iat_mean_val", 16)
@pa_container_size("ingress", "ig_md.bwd_iat_min_val", 16)
@pa_container_size("ingress", "ig_md.bwd_packet_length_max_val", 16)
@pa_container_size("ingress", "ig_md.bwd_packet_length_mean_val", 16)
@pa_container_size("ingress", "ig_md.bwd_packet_length_min_val", 16)
@pa_container_size("ingress", "ig_md.flow_iat_max_val", 16)
@pa_container_size("ingress", "ig_md.flow_iat_mean_val", 16)
@pa_container_size("ingress", "ig_md.flow_iat_min_val", 16)
@pa_container_size("ingress", "ig_md.fwd_iat_mean_val", 16)
@pa_container_size("ingress", "ig_md.fwd_iat_min_val", 16)
@pa_container_size("ingress", "ig_md.fwd_packet_length_max_val", 16)
@pa_container_size("ingress", "ig_md.fwd_packet_length_mean_val", 16)
@pa_container_size("ingress", "ig_md.fwd_packet_length_min_val", 16)
@pa_container_size("ingress", "ig_md.min_packet_length_val", 16)
@pa_container_size("ingress", "ig_md.packet_length_mean_val", 16)
@pa_no_overlay("ingress", "ig_md.code_bwd_iat_max")
@pa_container_size("ingress", "ig_md.code_bwd_iat_max", 8)
@pa_no_overlay("ingress", "ig_md.code_bwd_iat_mean")
@pa_container_size("ingress", "ig_md.code_bwd_iat_mean", 8)
@pa_no_overlay("ingress", "ig_md.code_bwd_iat_min")
@pa_container_size("ingress", "ig_md.code_bwd_iat_min", 8)
@pa_no_overlay("ingress", "ig_md.code_bwd_packet_length_max")
@pa_container_size("ingress", "ig_md.code_bwd_packet_length_max", 8)
@pa_no_overlay("ingress", "ig_md.code_bwd_packet_length_mean")
@pa_container_size("ingress", "ig_md.code_bwd_packet_length_mean", 16)
@pa_no_overlay("ingress", "ig_md.code_bwd_packet_length_min")
@pa_container_size("ingress", "ig_md.code_bwd_packet_length_min", 8)
@pa_no_overlay("ingress", "ig_md.code_flow_iat_max")
@pa_container_size("ingress", "ig_md.code_flow_iat_max", 8)
@pa_no_overlay("ingress", "ig_md.code_flow_iat_mean")
@pa_container_size("ingress", "ig_md.code_flow_iat_mean", 8)
@pa_no_overlay("ingress", "ig_md.code_flow_iat_min")
@pa_container_size("ingress", "ig_md.code_flow_iat_min", 8)
@pa_no_overlay("ingress", "ig_md.code_fwd_iat_mean")
@pa_container_size("ingress", "ig_md.code_fwd_iat_mean", 8)
@pa_no_overlay("ingress", "ig_md.code_fwd_iat_min")
@pa_container_size("ingress", "ig_md.code_fwd_iat_min", 8)
@pa_no_overlay("ingress", "ig_md.code_fwd_packet_length_max")
@pa_container_size("ingress", "ig_md.code_fwd_packet_length_max", 16)
@pa_no_overlay("ingress", "ig_md.code_fwd_packet_length_mean")
@pa_container_size("ingress", "ig_md.code_fwd_packet_length_mean", 8)
@pa_no_overlay("ingress", "ig_md.code_fwd_packet_length_min")
@pa_container_size("ingress", "ig_md.code_fwd_packet_length_min", 8)
@pa_no_overlay("ingress", "ig_md.code_min_packet_length")
@pa_container_size("ingress", "ig_md.code_min_packet_length", 8)
@pa_no_overlay("ingress", "ig_md.code_packet_length_mean")
@pa_container_size("ingress", "ig_md.code_packet_length_mean", 8)
@pa_container_size("ingress", "ig_md.bwd_current_iat", 16)
@pa_container_size("ingress", "ig_md.flow_current_iat", 16)
@pa_container_size("ingress", "ig_md.fwd_current_iat", 16)
@pa_solitary("ingress", "ig_md.now_pseudo_us")
@pa_solitary("ingress", "ig_md.flow_hash")
struct metadata_t {
    bit<12> flow_hash;
    bit<1>  fwd;
    bit<32> now_pseudo_us;

	bit<16> bwd_current_iat;
	bit<16> flow_current_iat;
	bit<16> fwd_current_iat;
	bit<16> bwd_iat_max_val;
	bit<16> bwd_iat_mean_val;
	bit<16> bwd_iat_min_val;
	bit<16> bwd_packet_length_max_val;
	bit<16> bwd_packet_length_mean_val;
	bit<16> bwd_packet_length_min_val;
	bit<16> flow_iat_max_val;
	bit<16> flow_iat_mean_val;
	bit<16> flow_iat_min_val;
	bit<16> fwd_iat_mean_val;
	bit<16> fwd_iat_min_val;
	bit<16> fwd_packet_length_max_val;
	bit<16> fwd_packet_length_mean_val;
	bit<16> fwd_packet_length_min_val;
	bit<16> min_packet_length_val;
	bit<16> packet_length_mean_val;
	bit<8> code_bwd_iat_max;
	bit<8> code_bwd_iat_mean;
	bit<8> code_bwd_iat_min;
	bit<8> code_bwd_packet_length_max;
	bit<16> code_bwd_packet_length_mean;
	bit<8> code_bwd_packet_length_min;
	bit<8> code_flow_iat_max;
	bit<8> code_flow_iat_mean;
	bit<8> code_flow_iat_min;
	bit<8> code_fwd_iat_mean;
	bit<8> code_fwd_iat_min;
	bit<16> code_fwd_packet_length_max;
	bit<8> code_fwd_packet_length_mean;
	bit<8> code_fwd_packet_length_min;
	bit<8> code_min_packet_length;
	bit<8> code_packet_length_mean;

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
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) bwd_iat_mean_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS, 65535) bwd_iat_min_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) bwd_packet_length_max_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) bwd_packet_length_mean_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS, 65535) bwd_packet_length_min_reg;
	Register<bit<32>, bit<12>>(MAX_NUM_FLOWS, 32w0xFFFFFFFF) flow_last_arrival_time_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) flow_iat_max_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) flow_iat_mean_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS, 65535) flow_iat_min_reg;
	Register<bit<32>, bit<12>>(MAX_NUM_FLOWS, 32w0xFFFFFFFF) fwd_last_arrival_time_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) fwd_iat_mean_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS, 65535) fwd_iat_min_reg;
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

	MathUnit<bit<16>>(MathOp_t.MUL, 1, 2) bwd_iat_mean_halve_unit;

	RegisterAction<bit<16>, bit<12>, bit<16>>(bwd_iat_mean_reg) bwd_iat_mean_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.bwd_current_iat != 0) {
				value = bwd_iat_mean_halve_unit.execute(value + meta.bwd_current_iat);
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

	MathUnit<bit<16>>(MathOp_t.MUL, 1, 2) flow_iat_mean_halve_unit;

	RegisterAction<bit<16>, bit<12>, bit<16>>(flow_iat_mean_reg) flow_iat_mean_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.flow_current_iat != 0) {
				value = flow_iat_mean_halve_unit.execute(value + meta.flow_current_iat);
			}
			rv = value;
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

	RegisterAction<bit<32>, bit<12>, bit<32>>(fwd_last_arrival_time_reg) fwd_last_arrival_time_action = {
		void apply(inout bit<32> value, out bit<32> rv) {
			if (meta.fwd == 1) {
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

	MathUnit<bit<16>>(MathOp_t.MUL, 1, 2) fwd_iat_mean_halve_unit;

	RegisterAction<bit<16>, bit<12>, bit<16>>(fwd_iat_mean_reg) fwd_iat_mean_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd_current_iat != 0) {
				value = fwd_iat_mean_halve_unit.execute(value + meta.fwd_current_iat);
			}
			rv = value;
		}
	};

	RegisterAction<bit<16>, bit<12>, bit<16>>(fwd_iat_min_reg) fwd_iat_min_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd_current_iat != 0 && meta.fwd_current_iat <= value) {
				value = meta.fwd_current_iat - 1;
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
		ig_tm_md.qid = (bit<5>) class;
	}

	action classify_flow_codeword_ddos_0(bit<1> class){
		ig_tm_md.ingress_cos = (bit<3>) class;
	}

    
    action set_code_bwd_iat_max (bit<8> code) {
        meta.code_bwd_iat_max = code;
    }
    
    action set_code_bwd_iat_mean (bit<8> code) {
        meta.code_bwd_iat_mean = code;
    }
    
    action set_code_bwd_iat_min (bit<8> code) {
        meta.code_bwd_iat_min = code;
    }
    
    action set_code_bwd_packet_length_max (bit<8> code) {
        meta.code_bwd_packet_length_max = code;
    }
    
    action set_code_bwd_packet_length_mean (bit<16> code) {
        meta.code_bwd_packet_length_mean = code;
    }
    
    action set_code_bwd_packet_length_min (bit<8> code) {
        meta.code_bwd_packet_length_min = code;
    }
    
    action set_code_flow_iat_max (bit<8> code) {
        meta.code_flow_iat_max = code;
    }
    
    action set_code_flow_iat_mean (bit<8> code) {
        meta.code_flow_iat_mean = code;
    }
    
    action set_code_flow_iat_min (bit<8> code) {
        meta.code_flow_iat_min = code;
    }
    
    action set_code_fwd_iat_mean (bit<8> code) {
        meta.code_fwd_iat_mean = code;
    }
    
    action set_code_fwd_iat_min (bit<8> code) {
        meta.code_fwd_iat_min = code;
    }
    
    action set_code_fwd_packet_length_max (bit<16> code) {
        meta.code_fwd_packet_length_max = code;
    }
    
    action set_code_fwd_packet_length_mean (bit<8> code) {
        meta.code_fwd_packet_length_mean = code;
    }
    
    action set_code_fwd_packet_length_min (bit<8> code) {
        meta.code_fwd_packet_length_min = code;
    }
    
    action set_code_min_packet_length (bit<8> code) {
        meta.code_min_packet_length = code;
    }
    
    action set_code_packet_length_mean (bit<8> code) {
        meta.code_packet_length_mean = code;
    }



    @placement_priority(1)
    table get_classification_tree_app_0 {
        key = {
            meta.code_bwd_iat_max[2:0] : ternary @name("code_bwd_iat_max_2_0_");
            meta.code_bwd_iat_mean[1:0] : ternary @name("code_bwd_iat_mean_1_0_");
            meta.code_bwd_iat_min[3:0] : ternary @name("code_bwd_iat_min_3_0_");
            meta.code_bwd_packet_length_max[7:0] : ternary @name("code_bwd_packet_length_max_7_0_");
            meta.code_bwd_packet_length_mean[9:0] : ternary @name("code_bwd_packet_length_mean_9_0_");
            meta.code_bwd_packet_length_min[2:0] : ternary @name("code_bwd_packet_length_min_2_0_");
            meta.code_flow_iat_max[0:0] : ternary @name("code_flow_iat_max_0_0_");
            meta.code_flow_iat_mean[1:0] : ternary @name("code_flow_iat_mean_1_0_");
            meta.code_flow_iat_min[3:0] : ternary @name("code_flow_iat_min_3_0_");
            meta.code_fwd_iat_mean[3:0] : ternary @name("code_fwd_iat_mean_3_0_");
            meta.code_fwd_iat_min[3:0] : ternary @name("code_fwd_iat_min_3_0_");
            meta.code_fwd_packet_length_max[11:0] : ternary @name("code_fwd_packet_length_max_11_0_");
            meta.code_fwd_packet_length_mean[3:0] : ternary @name("code_fwd_packet_length_mean_3_0_");
            meta.code_fwd_packet_length_min[1:0] : ternary @name("code_fwd_packet_length_min_1_0_");
            meta.code_packet_length_mean[5:0] : ternary @name("code_packet_length_mean_5_0_");
        }
        actions = {
            classify_flow_codeword_app_0;        
        }
        size = 54;
    }
    
    @placement_priority(2)
    table get_classification_tree_ddos_0 {
        key = {
            meta.code_bwd_iat_mean[1:0] : ternary @name("code_bwd_iat_mean_1_0_");
            meta.code_bwd_iat_min[3:0] : ternary @name("code_bwd_iat_min_3_0_");
            meta.code_bwd_packet_length_max[7:0] : ternary @name("code_bwd_packet_length_max_7_0_");
            meta.code_bwd_packet_length_mean[9:0] : ternary @name("code_bwd_packet_length_mean_9_0_");
            meta.code_bwd_packet_length_min[2:0] : ternary @name("code_bwd_packet_length_min_2_0_");
            meta.code_flow_iat_mean[1:0] : ternary @name("code_flow_iat_mean_1_0_");
            meta.code_flow_iat_min[3:0] : ternary @name("code_flow_iat_min_3_0_");
            meta.code_fwd_iat_mean[3:0] : ternary @name("code_fwd_iat_mean_3_0_");
            meta.code_fwd_iat_min[3:0] : ternary @name("code_fwd_iat_min_3_0_");
            meta.code_fwd_packet_length_max[11:0] : ternary @name("code_fwd_packet_length_max_11_0_");
            meta.code_fwd_packet_length_mean[3:0] : ternary @name("code_fwd_packet_length_mean_3_0_");
            meta.code_min_packet_length[0:0] : ternary @name("code_min_packet_length_0_0_");
            meta.code_packet_length_mean[5:0] : ternary @name("code_packet_length_mean_5_0_");
        }
        actions = {
            classify_flow_codeword_ddos_0;        
        }
        size = 22;
    }
    
    @placement_priority(3)
    table table_0_bwd_iat_max {
        key = {
            meta.bwd_iat_max_val: range;
        }
        actions = {
            set_code_bwd_iat_max;        
        }
        size = 4;
    }   
    
    @placement_priority(3)
    table table_1_bwd_iat_mean {
        key = {
            meta.bwd_iat_mean_val: range;
        }
        actions = {
            set_code_bwd_iat_mean;        
        }
        size = 3;
    }   
    
    @placement_priority(3)
    table table_2_bwd_iat_min {
        key = {
            meta.bwd_iat_min_val: range;
        }
        actions = {
            set_code_bwd_iat_min;        
        }
        size = 5;
    }   
    
    @placement_priority(3)
    table table_3_bwd_packet_length_max {
        key = {
            meta.bwd_packet_length_max_val: range;
        }
        actions = {
            set_code_bwd_packet_length_max;        
        }
        size = 9;
    }   
    
    @placement_priority(3)
    table table_4_bwd_packet_length_mean {
        key = {
            meta.bwd_packet_length_mean_val: range;
        }
        actions = {
            set_code_bwd_packet_length_mean;        
        }
        size = 11;
    }   
    
    @placement_priority(3)
    table table_5_bwd_packet_length_min {
        key = {
            meta.bwd_packet_length_min_val: range;
        }
        actions = {
            set_code_bwd_packet_length_min;        
        }
        size = 4;
    }   
    
    @placement_priority(3)
    table table_6_flow_iat_max {
        key = {
            meta.flow_iat_max_val: range;
        }
        actions = {
            set_code_flow_iat_max;        
        }
        size = 2;
    }   
    
    @placement_priority(3)
    table table_7_flow_iat_mean {
        key = {
            meta.flow_iat_mean_val: range;
        }
        actions = {
            set_code_flow_iat_mean;        
        }
        size = 3;
    }   
    
    @placement_priority(3)
    table table_8_flow_iat_min {
        key = {
            meta.flow_iat_min_val: range;
        }
        actions = {
            set_code_flow_iat_min;        
        }
        size = 5;
    }   
    
    @placement_priority(3)
    table table_9_fwd_iat_mean {
        key = {
            meta.fwd_iat_mean_val: range;
        }
        actions = {
            set_code_fwd_iat_mean;        
        }
        size = 5;
    }   
    
    @placement_priority(3)
    table table_10_fwd_iat_min {
        key = {
            meta.fwd_iat_min_val: range;
        }
        actions = {
            set_code_fwd_iat_min;        
        }
        size = 5;
    }   
    
    @placement_priority(3)
    table table_11_fwd_packet_length_max {
        key = {
            meta.fwd_packet_length_max_val: range;
        }
        actions = {
            set_code_fwd_packet_length_max;        
        }
        size = 13;
    }   
    
    @placement_priority(3)
    table table_12_fwd_packet_length_mean {
        key = {
            meta.fwd_packet_length_mean_val: range;
        }
        actions = {
            set_code_fwd_packet_length_mean;        
        }
        size = 5;
    }   
    
    @placement_priority(3)
    table table_13_fwd_packet_length_min {
        key = {
            meta.fwd_packet_length_min_val: range;
        }
        actions = {
            set_code_fwd_packet_length_min;        
        }
        size = 3;
    }   
    
    @placement_priority(3)
    table table_14_min_packet_length {
        key = {
            meta.min_packet_length_val: range;
        }
        actions = {
            set_code_min_packet_length;        
        }
        size = 2;
    }   
    
    @placement_priority(3)
    table table_15_packet_length_mean {
        key = {
            meta.packet_length_mean_val: range;
        }
        actions = {
            set_code_packet_length_mean;        
        }
        size = 7;
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
			meta.flow_iat_mean_val = flow_iat_mean_action.execute(meta.flow_hash);
			meta.flow_iat_min_val = flow_iat_min_action.execute(meta.flow_hash) + 1;
			meta.min_packet_length_val = min_packet_length_action.execute(meta.flow_hash);
			meta.packet_length_mean_val = packet_length_mean_action.execute(meta.flow_hash);

			meta.fwd_current_iat = (bit<16>) fwd_last_arrival_time_action.execute(meta.flow_hash);
			meta.fwd_iat_mean_val = fwd_iat_mean_action.execute(meta.flow_hash);
			meta.fwd_iat_min_val = fwd_iat_min_action.execute(meta.flow_hash) + 1;
			meta.fwd_packet_length_max_val = fwd_packet_length_max_action.execute(meta.flow_hash);
			meta.fwd_packet_length_mean_val = fwd_packet_length_mean_action.execute(meta.flow_hash);
			meta.fwd_packet_length_min_val = fwd_packet_length_min_action.execute(meta.flow_hash);

			meta.bwd_current_iat = (bit<16>) bwd_last_arrival_time_action.execute(meta.flow_hash);
			meta.bwd_iat_max_val = bwd_iat_max_action.execute(meta.flow_hash);
			meta.bwd_iat_mean_val = bwd_iat_mean_action.execute(meta.flow_hash);
			meta.bwd_iat_min_val = bwd_iat_min_action.execute(meta.flow_hash) + 1;
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
			table_9_fwd_iat_mean.apply();
			table_10_fwd_iat_min.apply();
			table_11_fwd_packet_length_max.apply();
			table_12_fwd_packet_length_mean.apply();
			table_13_fwd_packet_length_min.apply();
			table_14_min_packet_length.apply();
			table_15_packet_length_mean.apply();

			get_classification_tree_app_0.apply();

			get_classification_tree_ddos_0.apply();


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
