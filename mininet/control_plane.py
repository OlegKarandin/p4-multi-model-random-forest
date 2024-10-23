#!/usr/bin/env python3
import os
import json
from p4utils.utils.helper import load_topo
from p4utils.utils.sswitch_p4runtime_API import SimpleSwitchP4RuntimeAPI

ENTRIES_FILE_PATH = "../p4/table_entries.json"


topo = load_topo('topology.json')
controllers = {}

for switch, data in topo.get_p4rtswitches().items():
    controllers[switch] = SimpleSwitchP4RuntimeAPI(data['device_id'], data['grpc_port'],
                                                  p4rt_path=data['p4rt_path'],
                                                  json_path=data['json_path'])

controller = controllers['s1']                        


with open(ENTRIES_FILE_PATH, "r") as json_file:
    table_entries = json.loads(json_file.read())

for table_entry in table_entries:
    controller.table_add(table_entry["table_name"],
                         table_entry["action"], 
                         table_entry["key"], 
                         table_entry["action_params"],
                         prio=1)
    
#Remove all the entries that are in the table
controller.table_clear('repeater')
#By doing it this way (with match-action tables) we avoid modifying the p4 program. We 
#just modify the control plane
#If the ingress port is 1, we want the packet to be redirected to port 2
controller.table_add('repeater', 'forward', ['1'], ['2'])
#If the ingress port is 2, we want the packet to be redirected to port 1
controller.table_add('repeater', 'forward', ['2'], ['1'])