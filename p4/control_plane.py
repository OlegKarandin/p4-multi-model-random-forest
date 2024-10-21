#!/usr/bin/env python3
import os
import json
from p4utils.utils.helper import load_topo
from p4utils.utils.sswitch_p4runtime_API import SimpleSwitchP4RuntimeAPI

ENTRIES_FILE_PATH = "table_entries.json"


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