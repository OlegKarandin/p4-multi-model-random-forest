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
@pa_container_size("ingress", "ig_md.bwd_packet_length_max_val", 16)
@pa_container_size("ingress", "ig_md.bwd_packet_length_mean_val", 16)
@pa_container_size("ingress", "ig_md.fwd_packet_length_max_val", 16)
@pa_container_size("ingress", "ig_md.fwd_packet_length_mean_val", 16)
@pa_no_overlay("ingress", "ig_md.code_tk_app_bwd_packet_length_max")
@pa_container_size("ingress", "ig_md.code_tk_app_bwd_packet_length_max", 8, 32, 32, 32, 32)
@pa_no_overlay("ingress", "ig_md.code_tk_ddos_bwd_packet_length_max")
@pa_container_size("ingress", "ig_md.code_tk_ddos_bwd_packet_length_max", 16)
@pa_no_overlay("ingress", "ig_md.code_tk_app_bwd_packet_length_mean")
@pa_container_size("ingress", "ig_md.code_tk_app_bwd_packet_length_mean", 8, 32, 32, 32)
@pa_no_overlay("ingress", "ig_md.code_tk_ddos_bwd_packet_length_mean")
@pa_container_size("ingress", "ig_md.code_tk_ddos_bwd_packet_length_mean", 32)
@pa_no_overlay("ingress", "ig_md.code_tk_app_fwd_packet_length_max")
@pa_container_size("ingress", "ig_md.code_tk_app_fwd_packet_length_max", 32, 32, 32, 32, 32)
@pa_no_overlay("ingress", "ig_md.code_tk_ddos_fwd_packet_length_max")
@pa_container_size("ingress", "ig_md.code_tk_ddos_fwd_packet_length_max", 32)
@pa_no_overlay("ingress", "ig_md.code_tk_app_fwd_packet_length_mean")
@pa_container_size("ingress", "ig_md.code_tk_app_fwd_packet_length_mean", 8, 32, 32, 32)
@pa_no_overlay("ingress", "ig_md.code_tk_ddos_fwd_packet_length_mean")
@pa_container_size("ingress", "ig_md.code_tk_ddos_fwd_packet_length_mean", 32)
struct metadata_t {
    bit<12> flow_hash;
    bit<1>  fwd;
    bit<16> now_pseudo_us;

	bit<2> class_tree_app_0;
	bit<2> class_tree_app_1;
	bit<2> class_tree_app_2;
	bit<1> class_tree_ddos_0;
	bit<1> class_tree_ddos_1;
	bit<1> class_tree_ddos_2;
	bit<16> bwd_packet_length_max_val;
	bit<16> bwd_packet_length_mean_val;
	bit<16> fwd_packet_length_max_val;
	bit<16> fwd_packet_length_mean_val;
	bit<136> code_tk_app_bwd_packet_length_max;
	bit<16> code_tk_ddos_bwd_packet_length_max;
	bit<104> code_tk_app_bwd_packet_length_mean;
	bit<32> code_tk_ddos_bwd_packet_length_mean;
	bit<160> code_tk_app_fwd_packet_length_max;
	bit<32> code_tk_ddos_fwd_packet_length_max;
	bit<104> code_tk_app_fwd_packet_length_mean;
	bit<32> code_tk_ddos_fwd_packet_length_mean;

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
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) bwd_packet_length_max_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) bwd_packet_length_mean_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) fwd_packet_length_max_reg;
	Register<bit<16>, bit<12>>(MAX_NUM_FLOWS) fwd_packet_length_mean_reg;


	action calc_flow_hash() {
		meta.flow_hash = flow_hash_calc.get({
			hdr.ipv4.src_addr,
			hdr.ipv4.dst_addr,
			hdr.ipv4.protocol,
			hdr.tcp.src_port,
			hdr.tcp.dst_port
		});
	}

	RegisterAction<bit<32>, bit<12>, bit<1>>(flow_forward_srcaddr_reg) flow_orientation_action = {
		void apply(inout bit<32> value, out bit<1> rv) {
			if (value == 0) {
				value = hdr.ipv4.src_addr;
			}
			rv = (value == hdr.ipv4.src_addr) ? 1w1 : 1w0;
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

    
    action set_code_bwd_packet_length_max (bit<136> code_app, bit<16> code_ddos) {
        meta.code_tk_app_bwd_packet_length_max = code_app;
        meta.code_tk_ddos_bwd_packet_length_max = code_ddos;
    }
    
    action set_code_bwd_packet_length_mean (bit<104> code_app, bit<32> code_ddos) {
        meta.code_tk_app_bwd_packet_length_mean = code_app;
        meta.code_tk_ddos_bwd_packet_length_mean = code_ddos;
    }
    
    action set_code_fwd_packet_length_max (bit<160> code_app, bit<32> code_ddos) {
        meta.code_tk_app_fwd_packet_length_max = code_app;
        meta.code_tk_ddos_fwd_packet_length_max = code_ddos;
    }
    
    action set_code_fwd_packet_length_mean (bit<104> code_app, bit<32> code_ddos) {
        meta.code_tk_app_fwd_packet_length_mean = code_app;
        meta.code_tk_ddos_fwd_packet_length_mean = code_ddos;
    }



    @placement_priority(1)
    table get_classification_tree_app_0 {
        key = {
            meta.code_tk_app_bwd_packet_length_max[63:0] : ternary @name("code_tk_app_bwd_packet_length_max_63_0_");
            meta.code_tk_app_bwd_packet_length_max[127:64] : ternary @name("code_tk_app_bwd_packet_length_max_127_64_");
            meta.code_tk_app_bwd_packet_length_max[129:128] : ternary @name("code_tk_app_bwd_packet_length_max_129_128_");
            meta.code_tk_app_bwd_packet_length_mean[63:0] : ternary @name("code_tk_app_bwd_packet_length_mean_63_0_");
            meta.code_tk_app_bwd_packet_length_mean[99:64] : ternary @name("code_tk_app_bwd_packet_length_mean_99_64_");
            meta.code_tk_app_fwd_packet_length_max[63:0] : ternary @name("code_tk_app_fwd_packet_length_max_63_0_");
            meta.code_tk_app_fwd_packet_length_max[127:64] : ternary @name("code_tk_app_fwd_packet_length_max_127_64_");
            meta.code_tk_app_fwd_packet_length_max[148:128] : ternary @name("code_tk_app_fwd_packet_length_max_148_128_");
            meta.code_tk_app_fwd_packet_length_mean[63:0] : ternary @name("code_tk_app_fwd_packet_length_mean_63_0_");
            meta.code_tk_app_fwd_packet_length_mean[102:64] : ternary @name("code_tk_app_fwd_packet_length_mean_102_64_");
        }
        actions = {
            classify_flow_codeword_app_0;        
        }
        size = 192;
    }
    
    @placement_priority(1)
    table get_classification_tree_app_1 {
        key = {
            meta.code_tk_app_bwd_packet_length_max[63:0] : ternary @name("code_tk_app_bwd_packet_length_max_63_0_");
            meta.code_tk_app_bwd_packet_length_max[127:64] : ternary @name("code_tk_app_bwd_packet_length_max_127_64_");
            meta.code_tk_app_bwd_packet_length_max[129:128] : ternary @name("code_tk_app_bwd_packet_length_max_129_128_");
            meta.code_tk_app_bwd_packet_length_mean[63:0] : ternary @name("code_tk_app_bwd_packet_length_mean_63_0_");
            meta.code_tk_app_bwd_packet_length_mean[99:64] : ternary @name("code_tk_app_bwd_packet_length_mean_99_64_");
            meta.code_tk_app_fwd_packet_length_max[63:0] : ternary @name("code_tk_app_fwd_packet_length_max_63_0_");
            meta.code_tk_app_fwd_packet_length_max[127:64] : ternary @name("code_tk_app_fwd_packet_length_max_127_64_");
            meta.code_tk_app_fwd_packet_length_max[148:128] : ternary @name("code_tk_app_fwd_packet_length_max_148_128_");
            meta.code_tk_app_fwd_packet_length_mean[63:0] : ternary @name("code_tk_app_fwd_packet_length_mean_63_0_");
            meta.code_tk_app_fwd_packet_length_mean[102:64] : ternary @name("code_tk_app_fwd_packet_length_mean_102_64_");
        }
        actions = {
            classify_flow_codeword_app_1;        
        }
        size = 202;
    }
    
    @placement_priority(1)
    table get_classification_tree_app_2 {
        key = {
            meta.code_tk_app_bwd_packet_length_max[63:0] : ternary @name("code_tk_app_bwd_packet_length_max_63_0_");
            meta.code_tk_app_bwd_packet_length_max[127:64] : ternary @name("code_tk_app_bwd_packet_length_max_127_64_");
            meta.code_tk_app_bwd_packet_length_max[129:128] : ternary @name("code_tk_app_bwd_packet_length_max_129_128_");
            meta.code_tk_app_bwd_packet_length_mean[63:0] : ternary @name("code_tk_app_bwd_packet_length_mean_63_0_");
            meta.code_tk_app_bwd_packet_length_mean[99:64] : ternary @name("code_tk_app_bwd_packet_length_mean_99_64_");
            meta.code_tk_app_fwd_packet_length_max[63:0] : ternary @name("code_tk_app_fwd_packet_length_max_63_0_");
            meta.code_tk_app_fwd_packet_length_max[127:64] : ternary @name("code_tk_app_fwd_packet_length_max_127_64_");
            meta.code_tk_app_fwd_packet_length_max[148:128] : ternary @name("code_tk_app_fwd_packet_length_max_148_128_");
            meta.code_tk_app_fwd_packet_length_mean[63:0] : ternary @name("code_tk_app_fwd_packet_length_mean_63_0_");
            meta.code_tk_app_fwd_packet_length_mean[102:64] : ternary @name("code_tk_app_fwd_packet_length_mean_102_64_");
        }
        actions = {
            classify_flow_codeword_app_2;        
        }
        size = 193;
    }
    
    @placement_priority(2)
    table get_classification_tree_ddos_0 {
        key = {
            meta.code_tk_ddos_bwd_packet_length_max[12:0] : ternary @name("code_tk_ddos_bwd_packet_length_max_12_0_");
            meta.code_tk_ddos_bwd_packet_length_mean[22:0] : ternary @name("code_tk_ddos_bwd_packet_length_mean_22_0_");
            meta.code_tk_ddos_fwd_packet_length_max[21:0] : ternary @name("code_tk_ddos_fwd_packet_length_max_21_0_");
            meta.code_tk_ddos_fwd_packet_length_mean[24:0] : ternary @name("code_tk_ddos_fwd_packet_length_mean_24_0_");
        }
        actions = {
            classify_flow_codeword_ddos_0;        
        }
        size = 39;
    }
    
    @placement_priority(2)
    table get_classification_tree_ddos_1 {
        key = {
            meta.code_tk_ddos_bwd_packet_length_max[12:0] : ternary @name("code_tk_ddos_bwd_packet_length_max_12_0_");
            meta.code_tk_ddos_bwd_packet_length_mean[22:0] : ternary @name("code_tk_ddos_bwd_packet_length_mean_22_0_");
            meta.code_tk_ddos_fwd_packet_length_max[21:0] : ternary @name("code_tk_ddos_fwd_packet_length_max_21_0_");
            meta.code_tk_ddos_fwd_packet_length_mean[24:0] : ternary @name("code_tk_ddos_fwd_packet_length_mean_24_0_");
        }
        actions = {
            classify_flow_codeword_ddos_1;        
        }
        size = 36;
    }
    
    @placement_priority(2)
    table get_classification_tree_ddos_2 {
        key = {
            meta.code_tk_ddos_bwd_packet_length_max[12:0] : ternary @name("code_tk_ddos_bwd_packet_length_max_12_0_");
            meta.code_tk_ddos_bwd_packet_length_mean[22:0] : ternary @name("code_tk_ddos_bwd_packet_length_mean_22_0_");
            meta.code_tk_ddos_fwd_packet_length_max[21:0] : ternary @name("code_tk_ddos_fwd_packet_length_max_21_0_");
            meta.code_tk_ddos_fwd_packet_length_mean[24:0] : ternary @name("code_tk_ddos_fwd_packet_length_mean_24_0_");
        }
        actions = {
            classify_flow_codeword_ddos_2;        
        }
        size = 32;
    }
    
    @placement_priority(3)
    table table_0_bwd_packet_length_max {
        key = {
            meta.bwd_packet_length_max_val: range;
        }
        actions = {
            set_code_bwd_packet_length_max;        
        }
        size = 143;
    }   
    
    @placement_priority(3)
    table table_1_bwd_packet_length_mean {
        key = {
            meta.bwd_packet_length_mean_val: range;
        }
        actions = {
            set_code_bwd_packet_length_mean;        
        }
        size = 121;
    }   
    
    @placement_priority(3)
    table table_2_fwd_packet_length_max {
        key = {
            meta.fwd_packet_length_max_val: range;
        }
        actions = {
            set_code_fwd_packet_length_max;        
        }
        size = 170;
    }   
    
    @placement_priority(3)
    table table_3_fwd_packet_length_mean {
        key = {
            meta.fwd_packet_length_mean_val: range;
        }
        actions = {
            set_code_fwd_packet_length_mean;        
        }
        size = 117;
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

			meta.fwd = flow_orientation_action.execute(meta.flow_hash);


			meta.fwd_packet_length_max_val = fwd_packet_length_max_action.execute(meta.flow_hash);
			meta.fwd_packet_length_mean_val = fwd_packet_length_mean_action.execute(meta.flow_hash);

			meta.bwd_packet_length_max_val = bwd_packet_length_max_action.execute(meta.flow_hash);
			meta.bwd_packet_length_mean_val = bwd_packet_length_mean_action.execute(meta.flow_hash);


			table_0_bwd_packet_length_max.apply();
			table_1_bwd_packet_length_mean.apply();
			table_2_fwd_packet_length_max.apply();
			table_3_fwd_packet_length_mean.apply();

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
