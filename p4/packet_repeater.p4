/* -*- P4_16 -*- */
#include <core.p4>
#include <v1model.p4>


const bit<16> TYPE_IPV4 = 0x800;
const bit<16> TYPE_IPV6 = 0x86DD;

const bit<32> MAX_NUM_FLOWS = 4096;
const bit<1> HASH_BASE = 0;
/*************************************************************************
*********************** H E A D E R S  ***********************************
*************************************************************************/

struct metadata {
    bit<32> flow_hash;
    bit<2> classification_app;
    bit<1> classification_ddos;
 
	bit<2> class_tree_app_0;
	bit<1> class_tree_app_0_is_set;
	bit<2> class_tree_app_1;
	bit<1> class_tree_app_1_is_set;
	bit<2> class_tree_app_2;
	bit<1> class_tree_app_2_is_set;
	bit<1> class_tree_ddos_0;
	bit<1> class_tree_ddos_0_is_set;
	bit<97> codeword;


    //GENERAL FLOW FEATURES
    bit<16> flow_duration;
    bit<16> flow_packet_count;
    bit<16> flow_packet_length_total;
    bit<16> flow_packet_length_max;
    bit<16> flow_packet_length_min;
    bit<16> flow_packet_length_mean;
    bit<16> flow_iat_max;
    bit<16> flow_iat_min;
    bit<16> flow_iat_mean;
    //FWD SUBFLOW FEATURES
    bit<16> fwd_packet_count;
    bit<16> fwd_packet_length_total;
    bit<16> fwd_packet_length_max;
    bit<16> fwd_packet_length_min;
    bit<16> fwd_packet_length_mean;
    bit<16> fwd_iat_total;
    bit<16> fwd_iat_max;
    bit<16> fwd_iat_min;
    bit<16> fwd_iat_mean;
    bit<16> fwd_header_length;
    bit<16> fwd_act_data_pkt;
    bit<16> fwd_min_segment_size;
    //BWD SUBFLOW FEATURES
    bit<16> bwd_packet_count;
    bit<16> bwd_packet_length_total;
    bit<16> bwd_packet_length_max;
    bit<16> bwd_packet_length_min;
    bit<16> bwd_packet_length_mean;
    bit<16> bwd_iat_total;
    bit<16> bwd_iat_max;
    bit<16> bwd_iat_min;
    bit<16> bwd_iat_mean;
    bit<16> bwd_header_length;
}


header ethernet_t {
    bit<48> dstMac;
    bit<48> srcMac;
    bit<16> etherType;
}

header ipv4_t {
    bit<4>    version;
    bit<4>    ihl;
    bit<8>    diffserv;
    bit<16>   totalLen;
    bit<16>   identification;
    bit<3>    flags;
    bit<13>   fragOffset;
    bit<8>    ttl;
    bit<8>    protocol;
    bit<16>   hdrChecksum;
    bit<32>   srcAddr;
    bit<32>   dstAddr;
}

header ipv6_t {
    bit<4> version;
    bit<8> trafficClass;
    bit<20> flowLabel;
    bit<16> payloadLength;
    bit<8> nextHeader;
    bit<8> hopLimit;
    bit<128> srcAddr;
    bit<128> dstAddr;    
}

header tcp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<32> seqNo;
    bit<32> ackNo;
    bit<4>  dataOffset;
    bit<3>  res;
    bit<3>  ecn;
    bit<6>  ctrl;
    bit<16> window;
    bit<16> checksum;
    bit<16> urgentPtr;
}

header udp_t {
    bit<16> src_port;
    bit<16> dst_port;
    bit<16> udp_total_len;
    bit<16> checksum;
}

header classification_t {
    bit<8> class;
}

struct headers {
    ethernet_t eth;
    ipv4_t     ipv4;
    tcp_t      tcp;
    udp_t      udp;
    ipv6_t     ipv6;
    classification_t classification;
}

struct hash_fields_t {
    bit<32> srcAddr;
    bit<32> dstAddr;
    bit<8> protocol;
    bit<16> srcPort;
    bit<16> dstPort;
}

register <bit<1>> (MAX_NUM_FLOWS) flows;
register <bit<1>> (MAX_NUM_FLOWS) flow_classifications;
register <bit<8>> (MAX_NUM_FLOWS) next_classification;

//GENERAL FLOW FEATURES
register <bit<16>> (MAX_NUM_FLOWS) flow_duration;
register <bit<16>> (MAX_NUM_FLOWS) flow_packet_count;
register <bit<16>> (MAX_NUM_FLOWS) flow_packet_length_total;
register <bit<16>> (MAX_NUM_FLOWS) flow_packet_length_max;
register <bit<16>> (MAX_NUM_FLOWS) flow_packet_length_min;
register <bit<16>> (MAX_NUM_FLOWS) flow_packet_length_mean;
register <bit<16>> (MAX_NUM_FLOWS) flow_iat_max;
register <bit<16>> (MAX_NUM_FLOWS) flow_iat_min;
register <bit<16>> (MAX_NUM_FLOWS) flow_iat_mean;
//FWD SUBFLOW FEATURES
register <bit<16>> (MAX_NUM_FLOWS) fwd_packet_count;
register <bit<16>> (MAX_NUM_FLOWS) fwd_packet_length_total;
register <bit<16>> (MAX_NUM_FLOWS) fwd_packet_length_max;
register <bit<16>> (MAX_NUM_FLOWS) fwd_packet_length_min;
register <bit<16>> (MAX_NUM_FLOWS) fwd_packet_length_mean;
register <bit<16>> (MAX_NUM_FLOWS) fwd_iat_total;
register <bit<16>> (MAX_NUM_FLOWS) fwd_iat_max;
register <bit<16>> (MAX_NUM_FLOWS) fwd_iat_min;
register <bit<16>> (MAX_NUM_FLOWS) fwd_iat_mean;
register <bit<16>> (MAX_NUM_FLOWS) fwd_header_length;
register <bit<16>> (MAX_NUM_FLOWS) fwd_act_data_pkt;
register <bit<16>> (MAX_NUM_FLOWS) fwd_min_segment_size;
//BWD SUBFLOW FEATURES
register <bit<16>> (MAX_NUM_FLOWS) bwd_packet_count;
register <bit<16>> (MAX_NUM_FLOWS) bwd_packet_length_total;
register <bit<16>> (MAX_NUM_FLOWS) bwd_packet_length_max;
register <bit<16>> (MAX_NUM_FLOWS) bwd_packet_length_min;
register <bit<16>> (MAX_NUM_FLOWS) bwd_packet_length_mean;
register <bit<16>> (MAX_NUM_FLOWS) bwd_iat_total;
register <bit<16>> (MAX_NUM_FLOWS) bwd_iat_max;
register <bit<16>> (MAX_NUM_FLOWS) bwd_iat_min;
register <bit<16>> (MAX_NUM_FLOWS) bwd_iat_mean;
register <bit<16>> (MAX_NUM_FLOWS) bwd_header_length;
//IAT AUXILIARS
register <bit<16>> (MAX_NUM_FLOWS) flow_last_arrival_time;
register <bit<16>> (MAX_NUM_FLOWS) fwd_last_arrival_time;
register <bit<16>> (MAX_NUM_FLOWS) bwd_last_arrival_time;
//EWMA AUXILIARS
register <bit<8>> (1) flow_next_bit_shift;
register <bit<8>> (1) fwd_next_bit_shift;
register <bit<8>> (1) bwd_next_bit_shift;

/*************************************************************************
*********************** P A R S E R  ***********************************
*************************************************************************/

parser MyParser(packet_in packet, out headers hdr, inout metadata meta, inout standard_metadata_t standard_metadata) {

state start {
    transition parse_ethernet;
}

state parse_ethernet {
    packet.extract(hdr.eth);
    transition select(hdr.eth.etherType) {
        TYPE_IPV4: parse_ipv4;
        TYPE_IPV6: parse_ipv6;
        default: accept;
    }
}

state parse_ipv4 {
    packet.extract(hdr.ipv4);
    transition select(hdr.ipv4.protocol) {
        6: parse_tcp;
        17: parse_udp;
        default: accept;
    }
}

    state parse_ipv6 {
        packet.extract(hdr.ipv6);
        transition select(hdr.ipv6.nextHeader) {
            6: parse_tcp;
            17: parse_udp;
            default: accept;
        }
    }

    state parse_tcp {
        packet.extract(hdr.tcp);
        transition accept;
    }

    state parse_udp {
        packet.extract(hdr.udp);
        transition accept;

    }

}

/*************************************************************************
************   C H E C K S U M    V E R I F I C A T I O N   *************
*************************************************************************/

control MyVerifyChecksum(inout headers hdr, inout metadata meta) {
    apply {  }
}


/*************************************************************************
**************  I N G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control MyIngress(inout headers hdr,
                  inout metadata meta,
                  inout standard_metadata_t standard_metadata) {
    
    //Usually p4 switch uses 9 bits for defining ports. We can also use 32 bits.
    //This is important when we write complex programs, to save resources.
    action forward(bit<9> egress_port){
        standard_metadata.egress_spec = egress_port;
    }

    /***
    action drop(){
        mark_to_drop(standard_metadata);
    }
    ***/

    action get_flow_hash_fwd(){        
        hash_fields_t hash_fields;
        hash_fields.srcAddr = hdr.ipv4.srcAddr;
        hash_fields.dstAddr = hdr.ipv4.dstAddr;
        hash_fields.protocol = hdr.ipv4.protocol;
        hash_fields.srcPort = hdr.tcp.srcPort;
        hash_fields.dstPort = hdr.tcp.dstPort;

        hash(meta.flow_hash, HashAlgorithm.crc32, HASH_BASE, hash_fields, MAX_NUM_FLOWS);
    }

    action get_flow_hash_bwd(){        
        hash_fields_t hash_fields;
        hash_fields.srcAddr = hdr.ipv4.dstAddr;
        hash_fields.dstAddr = hdr.ipv4.srcAddr;
        hash_fields.protocol = hdr.ipv4.protocol;
        hash_fields.srcPort = hdr.tcp.dstPort;
        hash_fields.dstPort = hdr.tcp.srcPort;

        hash(meta.flow_hash, HashAlgorithm.crc32, HASH_BASE, hash_fields, MAX_NUM_FLOWS);
    }


    action update_current_flow_features(){
        //GENERAL FLOW FEATURES
        flow_duration.read(meta.flow_duration, meta.flow_hash);
        flow_packet_count.read(meta.flow_packet_count, meta.flow_hash);
        flow_packet_length_total.read(meta.flow_packet_length_total, meta.flow_hash);
        flow_packet_length_max.read(meta.flow_packet_length_max, meta.flow_hash);
        flow_packet_length_min.read(meta.flow_packet_length_min, meta.flow_hash);
        flow_packet_length_mean.read(meta.flow_packet_length_mean, meta.flow_hash);
        flow_iat_max.read(meta.flow_iat_max, meta.flow_hash);
        flow_iat_min.read(meta.flow_iat_min, meta.flow_hash);
        flow_iat_mean.read(meta.flow_iat_mean, meta.flow_hash);
        //FWD SUBFLOW FEATURES
        fwd_packet_count.read(meta.fwd_packet_count, meta.flow_hash);
        fwd_packet_length_total.read(meta.fwd_packet_length_total, meta.flow_hash);
        fwd_packet_length_max.read(meta.fwd_packet_length_max, meta.flow_hash);
        fwd_packet_length_min.read(meta.fwd_packet_length_min, meta.flow_hash);
        fwd_packet_length_mean.read(meta.fwd_packet_length_mean, meta.flow_hash);
        fwd_iat_total.read(meta.fwd_iat_total, meta.flow_hash);
        fwd_iat_max.read(meta.fwd_iat_max, meta.flow_hash);
        fwd_iat_min.read(meta.fwd_iat_min, meta.flow_hash);
        fwd_iat_mean.read(meta.fwd_iat_mean, meta.flow_hash);
        fwd_header_length.read(meta.fwd_header_length, meta.flow_hash);
        fwd_act_data_pkt.read(meta.fwd_act_data_pkt, meta.flow_hash);
        fwd_min_segment_size.read(meta.fwd_min_segment_size, meta.flow_hash);
        //BWD SUBFLOW FEATURES
        bwd_packet_count.read(meta.bwd_packet_count, meta.flow_hash);
        bwd_packet_length_total.read(meta.bwd_packet_length_total, meta.flow_hash);
        bwd_packet_length_max.read(meta.bwd_packet_length_max, meta.flow_hash);
        bwd_packet_length_min.read(meta.bwd_packet_length_min, meta.flow_hash);
        bwd_packet_length_mean.read(meta.bwd_packet_length_mean, meta.flow_hash);
        bwd_iat_total.read(meta.bwd_iat_total, meta.flow_hash);
        bwd_iat_max.read(meta.bwd_iat_max, meta.flow_hash);
        bwd_iat_min.read(meta.bwd_iat_min, meta.flow_hash);
        bwd_iat_mean.read(meta.bwd_iat_mean, meta.flow_hash);
        bwd_header_length.read(meta.bwd_header_length, meta.flow_hash);
        
    }

   
	action classify_flow_codeword_app(bit<2> tree, bit<2> class){
		if (tree == 0){
			meta.class_tree_app_0 = class;
		}
		if (tree == 1){
			meta.class_tree_app_1 = class;
		}
		if (tree == 2){
			meta.class_tree_app_2 = class;
		}
	}

	action classify_flow_codeword_ddos(bit<1> tree, bit<1> class){
		if (tree == 0){
			meta.class_tree_ddos_0 = class;
		}
	}
    
    action set_code_flow_iat_mean (bit<31> code) {
        meta.codeword[96:66] = code;
    }
    
    action set_code_flow_packet_length_mean (bit<19> code) {
        meta.codeword[65:47] = code;
    }
    
    action set_code_fwd_packet_length_max (bit<22> code) {
        meta.codeword[46:25] = code;
    }
    
    action set_code_fwd_packet_length_mean (bit<25> code) {
        meta.codeword[24:0] = code;
    }


    
    //Classification tables: 1 table per tree. Parse codeword and obtain tree's classification

    //Feature tables: In feature-based encoding we define 1 table per feature

    
    table get_classification_tree_app_0 {
        key = {
            meta.codeword: ternary;
        }
        actions = {
            classify_flow_codeword_app;        
        }
        size = 400;
    }   
    
    table get_classification_tree_app_1 {
        key = {
            meta.codeword: ternary;
        }
        actions = {
            classify_flow_codeword_app;        
        }
        size = 400;
    }   
    
    table get_classification_tree_app_2 {
        key = {
            meta.codeword: ternary;
        }
        actions = {
            classify_flow_codeword_app;        
        }
        size = 400;
    }   
    
    table get_classification_tree_ddos_0 {
        key = {
            meta.codeword: ternary;
        }
        actions = {
            classify_flow_codeword_ddos;        
        }
        size = 400;
    }   
    
    table table_0_flow_iat_mean {
        key = {
            meta.flow_iat_mean: range;
        }
        actions = {
            set_code_flow_iat_mean;        
        }
        size = 200;
    }   
    
    table table_1_flow_packet_length_mean {
        key = {
            meta.flow_packet_length_mean: range;
        }
        actions = {
            set_code_flow_packet_length_mean;        
        }
        size = 200;
    }   
    
    table table_2_fwd_packet_length_max {
        key = {
            meta.fwd_packet_length_max: range;
        }
        actions = {
            set_code_fwd_packet_length_max;        
        }
        size = 200;
    }   
    
    table table_3_fwd_packet_length_mean {
        key = {
            meta.fwd_packet_length_mean: range;
        }
        actions = {
            set_code_fwd_packet_length_mean;        
        }
        size = 200;
    }   
    
    

    //Define table    
    table repeater {
        key = {
            standard_metadata.ingress_port: exact;
        }
        actions = {
            //Define actions
            forward;
            NoAction;
        }
        size = 2;
        default_action = NoAction;
    }
    

     //Define what we want to do with the defined tables,actions
    apply {        

            bit<1> flow_exists; // '1': New Flow, '0': Existing Flow  
            bit<1> fwd = 0; // '1': Packet belongs to Fwd Subflow
            bit<1> bwd = 0; // '1': Packet belongs to Bwd Subflow

            get_flow_hash_fwd();
            //Check if FlowID is new                   
            flows.read(flow_exists, meta.flow_hash);
            //If FlowID is new, check opposite direction (bwd)
            if (flow_exists == 0){
                get_flow_hash_bwd();
                flows.read(flow_exists, meta.flow_hash);
                if (flow_exists == 1){
                    //Packet belongs to existing Bwd Subflow
                    bwd = 1;
                }else{
                    //New Flow (Fwd Subflow)
                    get_flow_hash_fwd();
                    flows.write(meta.flow_hash, 1);
                }
                
            }else{
                //Packet belongs to existing Fwd Subflow
                fwd = 1;
            }

            ///// FEATURE EXTRACTION            
            bit<16> current_feature;

            // Packet Count            
            flow_packet_count.read(current_feature, meta.flow_hash);
            current_feature = current_feature +1;
            flow_packet_count.write(meta.flow_hash, current_feature);
            if (fwd == 1){
                fwd_packet_count.read(current_feature, meta.flow_hash);
                current_feature = current_feature + 1;
                fwd_packet_count.write(meta.flow_hash, current_feature);
            }
            if (bwd == 1){
                bwd_packet_count.read(current_feature, meta.flow_hash);
                current_feature = current_feature + 1;
                bwd_packet_count.write(meta.flow_hash, current_feature);
            }
            // Packet Length
            bit<16> packet_length = hdr.ipv4.totalLen;
            // Packet Length Total            
            flow_packet_length_total.read(current_feature, meta.flow_hash);
            current_feature = current_feature + packet_length;
            flow_packet_length_total.write(meta.flow_hash, current_feature);
            if (fwd == 1){
                fwd_packet_length_total.read(current_feature, meta.flow_hash);
                current_feature = current_feature + packet_length;
                fwd_packet_length_total.write(meta.flow_hash, current_feature);
            }
            if (bwd == 1){
                bwd_packet_length_total.read(current_feature, meta.flow_hash);
                current_feature = current_feature + packet_length;
                bwd_packet_length_total.write(meta.flow_hash, current_feature);
            }
            // Packet Length Max
            flow_packet_length_max.read(current_feature, meta.flow_hash);
            if (current_feature < packet_length){
                flow_packet_length_max.write(meta.flow_hash, packet_length);
            }
            if (fwd == 1){
                fwd_packet_length_max.read(current_feature, meta.flow_hash);
                if (current_feature < packet_length){
                    fwd_packet_length_max.write(meta.flow_hash, packet_length);
                }
            }
            if (bwd == 1){
                bwd_packet_length_max.read(current_feature, meta.flow_hash);
                if (current_feature < packet_length){
                    bwd_packet_length_max.write(meta.flow_hash, packet_length);
                }
            }
            // Packet Length Min
            flow_packet_length_min.read(current_feature, meta.flow_hash);
            if (current_feature > packet_length){
                flow_packet_length_min.write(meta.flow_hash, packet_length);
            }
            if (fwd == 1){
                fwd_packet_length_min.read(current_feature, meta.flow_hash);
                if (current_feature > packet_length){
                    fwd_packet_length_min.write(meta.flow_hash, packet_length);
                }
            }
            if (bwd == 1){
                bwd_packet_length_min.read(current_feature, meta.flow_hash);
                if (current_feature > packet_length){
                    bwd_packet_length_min.write(meta.flow_hash, packet_length);
                }
            }
            // Packet Length Mean
            ////Check if current packet count is power of 2
            bit<16> packet_count;
            bit<16> packet_length_total;
            bit<8> bit_shift;
            bit<16> estimated_mean;
            bit<16> packet_length_mean;

            flow_packet_count.read(packet_count, meta.flow_hash);
            flow_packet_length_total.read(packet_length_total, meta.flow_hash);
            flow_packet_length_mean.read(packet_length_mean, meta.flow_hash);

            if (packet_count == 1){
                flow_next_bit_shift.write(0,1);
            }
            bool flow_isPowerOf2 = (packet_count & (packet_count - 1)) == 0;
            bool fwd_isPowerOf2;
            bool bwd_isPowerOf2;

            if (flow_isPowerOf2){
                flow_next_bit_shift.read(bit_shift, 0);
                estimated_mean = packet_length_total >> bit_shift;
            }else{
                estimated_mean = (packet_length >> 1) + (packet_length_mean >> 1);                
            }
            flow_packet_length_mean.write(meta.flow_hash, estimated_mean);

            if (fwd == 1){
                fwd_packet_count.read(packet_count, meta.flow_hash);
                fwd_packet_length_total.read(packet_length_total, meta.flow_hash);
                fwd_packet_length_mean.read(packet_length_mean, meta.flow_hash);

                if (packet_count == 1){
                    fwd_next_bit_shift.write(0,1);
                }
                fwd_isPowerOf2 = (packet_count & (packet_count - 1)) == 0;

                packet_length_total = 433;

                if (fwd_isPowerOf2){
                    fwd_next_bit_shift.read(bit_shift, 0);
                    estimated_mean = packet_length_total >> bit_shift;
                }else{
                    estimated_mean = (packet_length >> 1) + (packet_length_mean >> 1);                    
                }
                fwd_packet_length_mean.write(meta.flow_hash, estimated_mean);
            }
            if (bwd == 1){
                bwd_packet_count.read(packet_count, meta.flow_hash);
                bwd_packet_length_total.read(packet_length_total, meta.flow_hash);
                bwd_packet_length_mean.read(packet_length_mean, meta.flow_hash);

                if (packet_count == 1){
                    bwd_next_bit_shift.write(0,1);
                }
                bwd_isPowerOf2 = (packet_count & (packet_count - 1)) == 0;

                packet_length_total = 433;

                if (bwd_isPowerOf2){
                    bwd_next_bit_shift.read(bit_shift, 0);
                    estimated_mean = packet_length_total >> bit_shift;
                }else{
                    estimated_mean = (packet_length >> 1) + (packet_length_mean >> 1);                    
                }
                bwd_packet_length_mean.write(meta.flow_hash, estimated_mean);
            }

            bit<16> current_iat;
            //Flow Duration
            bit<16> flow_last_arrival;
            flow_last_arrival_time.read(flow_last_arrival, meta.flow_hash);
            flow_duration.read(current_feature, meta.flow_hash);
            current_iat = standard_metadata.ingress_global_timestamp[15:0] - flow_last_arrival;
            current_feature = current_feature + current_iat;
            flow_duration.write(meta.flow_hash, current_feature);
            //Flow IAT Max
            flow_iat_max.read(current_feature, meta.flow_hash);
            if (current_iat > current_feature){
                flow_iat_max.write(meta.flow_hash, current_iat);
            }
            //Flow IAT Min
            flow_iat_min.read(current_feature, meta.flow_hash);
            if (current_iat < current_feature){
                flow_iat_min.write(meta.flow_hash, current_iat);
            }            
            //Flow IAT Mean
            bit<16> iat_total;
            bit<16> iat_mean;

            flow_packet_count.read(packet_count, meta.flow_hash);
            flow_duration.read(iat_total, meta.flow_hash);
            flow_iat_mean.read(iat_mean, meta.flow_hash);

            if (flow_isPowerOf2){
                flow_next_bit_shift.read(bit_shift, 0);
                estimated_mean = iat_total >> bit_shift;
            }else{
                estimated_mean = (current_iat >> 1) + (iat_mean >> 1);
            }
            flow_iat_mean.write(meta.flow_hash, estimated_mean);

            //Update Arrival Time of Flow Last Packet
            flow_last_arrival_time.write(meta.flow_hash, standard_metadata.ingress_global_timestamp[15:0]);

            if (fwd == 1){
                //Fwd IAT Total
                bit<16> fwd_last_arrival;
                fwd_last_arrival_time.read(fwd_last_arrival, meta.flow_hash);
                fwd_iat_total.read(current_feature, meta.flow_hash);
                current_iat = standard_metadata.ingress_global_timestamp[15:0] - fwd_last_arrival;
                current_feature = current_feature + current_iat;
                fwd_iat_total.write(meta.flow_hash, current_feature);
                //Fwd IAT Max
                fwd_iat_max.read(current_feature, meta.flow_hash);
                if (current_iat > current_feature){
                    fwd_iat_max.write(meta.flow_hash, current_feature);
                }
                //FWD IAT Min
                fwd_iat_min.read(current_feature, meta.flow_hash);
                if (current_iat < current_feature){
                    fwd_iat_min.write(meta.flow_hash, current_feature);
                }
                //Fwd IAT Mean
                fwd_packet_count.read(packet_count, meta.flow_hash);
                fwd_iat_total.read(iat_total, meta.flow_hash);
                fwd_iat_mean.read(iat_mean, meta.flow_hash);
                fwd_isPowerOf2 = (packet_count & (packet_count - 1)) == 0;

                if (fwd_isPowerOf2){
                    fwd_next_bit_shift.read(bit_shift, 0);
                    estimated_mean = iat_total >> bit_shift;
                }else{
                    estimated_mean = (current_iat >> 1) + (iat_mean >> 1);
                }
                fwd_iat_mean.write(meta.flow_hash, estimated_mean);
                fwd_last_arrival_time.write(meta.flow_hash, standard_metadata.ingress_global_timestamp[15:0]);

            }
            if (bwd == 1){
                //Bwd IAT Total
                bit<16> bwd_last_arrival;
                bwd_last_arrival_time.read(bwd_last_arrival, meta.flow_hash);
                bwd_iat_total.read(current_feature, meta.flow_hash);
                current_iat = standard_metadata.ingress_global_timestamp[15:0] - bwd_last_arrival;
                current_feature = current_feature + current_iat;
                bwd_iat_total.write(meta.flow_hash, current_feature);
                //Bwd IAT Max
                bwd_iat_max.read(current_feature, meta.flow_hash);
                if (current_iat > current_feature){
                    bwd_iat_max.write(meta.flow_hash, current_feature);
                }
                //Bwd IAT Min
                bwd_iat_min.read(current_feature, meta.flow_hash);
                if (current_iat < current_feature){
                    bwd_iat_min.write(meta.flow_hash, current_feature);
                }
                //Bwd IAT Mean
                bwd_packet_count.read(packet_count, meta.flow_hash);
                bwd_iat_total.read(iat_total, meta.flow_hash);
                bwd_iat_mean.read(iat_mean, meta.flow_hash);
                bwd_isPowerOf2 = (packet_count & (packet_count - 1)) == 0;

                if (bwd_isPowerOf2){
                    bwd_next_bit_shift.read(bit_shift, 0);
                    estimated_mean = iat_total >> bit_shift;
                }else{
                    estimated_mean = (current_iat >> 1) + (iat_mean >> 1);
                }
                bwd_iat_mean.write(meta.flow_hash, estimated_mean);
                bwd_last_arrival_time.write(meta.flow_hash, standard_metadata.ingress_global_timestamp[15:0]);
            }            

            // Fwd-Bwd Header Length
            bit<16> header_length;
            header_length[3:0] = hdr.ipv4.ihl;
            if (fwd == 1){
                fwd_header_length.read(current_feature, meta.flow_hash);
                current_feature = current_feature + header_length;
                fwd_header_length.write(meta.flow_hash, current_feature);
            }
            if (bwd == 1){
                bwd_header_length.read(current_feature, meta.flow_hash);
                current_feature = current_feature + header_length;
                bwd_header_length.write(meta.flow_hash, current_feature);
            }

            //Update bit shifts
            if (flow_isPowerOf2){
                flow_next_bit_shift.read(bit_shift, 0);
                bit_shift =  bit_shift + 1;
                flow_next_bit_shift.write(0, bit_shift);
            }
            if (fwd == 1){
                fwd_packet_count.read(packet_count, meta.flow_hash);
                fwd_isPowerOf2 = (packet_count & (packet_count - 1)) == 0;

                if (fwd_isPowerOf2){
                    fwd_next_bit_shift.read(bit_shift, 0);
                    bit_shift =  bit_shift + 1;
                    fwd_next_bit_shift.write(0, bit_shift);
                }
            }
            if (bwd == 1){
                bwd_packet_count.read(packet_count, meta.flow_hash);
                bwd_isPowerOf2 = (packet_count & (packet_count - 1)) == 0;
                if (bwd_isPowerOf2){
                    bwd_next_bit_shift.read(bit_shift, 0);
                    bit_shift =  bit_shift + 1;
                    bwd_next_bit_shift.write(0, bit_shift);
                }            
            }
            update_current_flow_features();


			table_0_flow_iat_mean.apply();
			table_1_flow_packet_length_mean.apply();
			table_2_fwd_packet_length_max.apply();
			table_3_fwd_packet_length_mean.apply();

			get_classification_tree_app_0.apply();
			get_classification_tree_app_1.apply();
			get_classification_tree_app_2.apply();

			get_classification_tree_ddos_0.apply();



			if ((meta.class_tree_app_0 == 0) && (meta.class_tree_app_1 == 0) && (meta.class_tree_app_2 == 0)) {
				meta.classification_app = 0;
			}
			if ((meta.class_tree_app_0 == 0) && (meta.class_tree_app_1 == 0) && (meta.class_tree_app_2 == 1)) {
				meta.classification_app = 0;
			}
			if ((meta.class_tree_app_0 == 0) && (meta.class_tree_app_1 == 0) && (meta.class_tree_app_2 == 2)) {
				meta.classification_app = 0;
			}
			if ((meta.class_tree_app_0 == 0) && (meta.class_tree_app_1 == 1) && (meta.class_tree_app_2 == 0)) {
				meta.classification_app = 0;
			}
			if ((meta.class_tree_app_0 == 0) && (meta.class_tree_app_1 == 1) && (meta.class_tree_app_2 == 1)) {
				meta.classification_app = 1;
			}
			if ((meta.class_tree_app_0 == 0) && (meta.class_tree_app_1 == 1) && (meta.class_tree_app_2 == 2)) {
				meta.classification_app = 2;
			}
			if ((meta.class_tree_app_0 == 0) && (meta.class_tree_app_1 == 2) && (meta.class_tree_app_2 == 0)) {
				meta.classification_app = 0;
			}
			if ((meta.class_tree_app_0 == 0) && (meta.class_tree_app_1 == 2) && (meta.class_tree_app_2 == 1)) {
				meta.classification_app = 0;
			}
			if ((meta.class_tree_app_0 == 0) && (meta.class_tree_app_1 == 2) && (meta.class_tree_app_2 == 2)) {
				meta.classification_app = 2;
			}
			if ((meta.class_tree_app_0 == 1) && (meta.class_tree_app_1 == 0) && (meta.class_tree_app_2 == 0)) {
				meta.classification_app = 0;
			}
			if ((meta.class_tree_app_0 == 1) && (meta.class_tree_app_1 == 0) && (meta.class_tree_app_2 == 1)) {
				meta.classification_app = 1;
			}
			if ((meta.class_tree_app_0 == 1) && (meta.class_tree_app_1 == 0) && (meta.class_tree_app_2 == 2)) {
				meta.classification_app = 0;
			}
			if ((meta.class_tree_app_0 == 1) && (meta.class_tree_app_1 == 1) && (meta.class_tree_app_2 == 0)) {
				meta.classification_app = 1;
			}
			if ((meta.class_tree_app_0 == 1) && (meta.class_tree_app_1 == 1) && (meta.class_tree_app_2 == 1)) {
				meta.classification_app = 1;
			}
			if ((meta.class_tree_app_0 == 1) && (meta.class_tree_app_1 == 1) && (meta.class_tree_app_2 == 2)) {
				meta.classification_app = 1;
			}
			if ((meta.class_tree_app_0 == 1) && (meta.class_tree_app_1 == 2) && (meta.class_tree_app_2 == 0)) {
				meta.classification_app = 1;
			}
			if ((meta.class_tree_app_0 == 1) && (meta.class_tree_app_1 == 2) && (meta.class_tree_app_2 == 1)) {
				meta.classification_app = 1;
			}
			if ((meta.class_tree_app_0 == 1) && (meta.class_tree_app_1 == 2) && (meta.class_tree_app_2 == 2)) {
				meta.classification_app = 2;
			}
			if ((meta.class_tree_app_0 == 2) && (meta.class_tree_app_1 == 0) && (meta.class_tree_app_2 == 0)) {
				meta.classification_app = 0;
			}
			if ((meta.class_tree_app_0 == 2) && (meta.class_tree_app_1 == 0) && (meta.class_tree_app_2 == 1)) {
				meta.classification_app = 1;
			}
			if ((meta.class_tree_app_0 == 2) && (meta.class_tree_app_1 == 0) && (meta.class_tree_app_2 == 2)) {
				meta.classification_app = 2;
			}
			if ((meta.class_tree_app_0 == 2) && (meta.class_tree_app_1 == 1) && (meta.class_tree_app_2 == 0)) {
				meta.classification_app = 1;
			}
			if ((meta.class_tree_app_0 == 2) && (meta.class_tree_app_1 == 1) && (meta.class_tree_app_2 == 1)) {
				meta.classification_app = 1;
			}
			if ((meta.class_tree_app_0 == 2) && (meta.class_tree_app_1 == 1) && (meta.class_tree_app_2 == 2)) {
				meta.classification_app = 2;
			}
			if ((meta.class_tree_app_0 == 2) && (meta.class_tree_app_1 == 2) && (meta.class_tree_app_2 == 0)) {
				meta.classification_app = 2;
			}
			if ((meta.class_tree_app_0 == 2) && (meta.class_tree_app_1 == 2) && (meta.class_tree_app_2 == 1)) {
				meta.classification_app = 2;
			}
			if ((meta.class_tree_app_0 == 2) && (meta.class_tree_app_1 == 2) && (meta.class_tree_app_2 == 2)) {
				meta.classification_app = 2;
			}

			if ((meta.class_tree_ddos_0 == 0)) {
				meta.classification_ddos = 0;
			}
			if ((meta.class_tree_ddos_0 == 1)) {
				meta.classification_ddos = 1;
			}


            
            //Additional: Forwarding Tasks
            //repeater.apply();

    }
}

/*************************************************************************
****************  E G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control MyEgress(inout headers hdr,
                 inout metadata meta,
                 inout standard_metadata_t standard_metadata) {
    apply{

    }
}

/*************************************************************************
*************   C H E C K S U M    C O M P U T A T I O N   **************
*************************************************************************/

control MyComputeChecksum(inout headers  hdr, inout metadata meta) {
    apply { }
}

/*************************************************************************
***********************  D E P A R S E R  *******************************
*************************************************************************/

control MyDeparser(packet_out packet, in headers hdr) {
    apply {
        //Serialize Headers
        packet.emit(hdr.eth);
        packet.emit(hdr.ipv4);
        packet.emit(hdr.ipv6);
        packet.emit(hdr.tcp);
        packet.emit(hdr.udp);
        packet.emit(hdr.classification);
    }
}

/*************************************************************************
***********************  S W I T C H  *******************************
*************************************************************************/

V1Switch(
MyParser(),
MyVerifyChecksum(),
MyIngress(),
MyEgress(),
MyComputeChecksum(),
MyDeparser()
) main;