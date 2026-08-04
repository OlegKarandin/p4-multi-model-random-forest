"""Deploys table_entries.json (see build_p4_script.get_table_entries) to a real running Tofino
pipeline via the real bf_rt Python API. Run via:
    ~/open-p4studio/run_bfshell.sh -b p4/deploy_table_entries.py
(assumes bfrt is already injected into the execution namespace by bfshell, matching every
p4/tofino_spike/t12_experiments/rm8_*.py script's own established convention -- do not import it).

NOT live-verified against a running switch as of this writing: doing so requires installing this
project's own compiled program into the SDE's runtime catalog first (~/open-p4studio/install/share/
tofinopd/<PROGRAM_NAME>), which this project has not yet established a repeatable process for (RM-8's
own program was pre-installed before this investigation began). Every field/method name below is
confirmed against this SDE's real bfrtcli.py source and this project's own real compiled bf-rt.json
schema, not guessed -- but the full script has not been exercised end-to-end.

What was confirmed directly in the real driver source:
  * bfrtcli._create_add_with_action / _create_set_default_with_action generate, per table+action, an
    `add_with_<action>` / `set_default_with_<action>` method taking NAMED kwargs per field.
  * bfrtcli._make_core_method_strs exposes a RANGE key field as <name>_start / <name>_end, and a
    TERNARY key field as <name> / <name>_mask -- exactly the two key_fields spec shapes
    get_table_entries writes.
  * set_default_with_<action> is built from data fields ONLY: no key fields and no MATCH_PRIORITY,
    which is why an is_default_action record carries an empty key_fields and a null priority.
  * The string values get_table_entries writes are fine as-is: bfrtTable.parse_str_input ->
    _Field.parse_input -> _parse_int does `int(value, 0)` for str input, so "0x5" parses as 5 and
    "100" as 100. (Passing plain ints would work identically -- see rm8_insert_test.py.)
"""
import json

PROGRAM_NAME = "p4_code_RF_models"  # must match the program name used to launch bf_switchd -p <PROGRAM_NAME>
CONTROL_BLOCK = "SwitchIngress"     # matches this project's real control block name (resources/p4_template.p4)
ENTRIES_FILE_PATH = "p4/table_entries.json"

with open(ENTRIES_FILE_PATH) as f:
    table_entries = json.load(f)

program = getattr(bfrt, PROGRAM_NAME)  # noqa: F821 -- bfrt is injected by bfshell
pipe = program.pipe

installed = 0
defaults_set = 0
for entry in table_entries:
    table = getattr(getattr(pipe, CONTROL_BLOCK), entry["table_name"])
    action_name = entry["action"]

    if entry.get("is_default_action"):
        set_default_method = getattr(table, "set_default_with_" + action_name)
        set_default_method(**entry["action_params"])
        defaults_set += 1
        continue

    kwargs = {}
    for field_name, spec in entry["key_fields"].items():
        if "value" in spec:
            # ternary key field -> <name>= / <name>_mask=
            kwargs[field_name] = spec["value"]
            kwargs[field_name + "_mask"] = spec["mask"]
        else:
            # range key field -> <name>_start= / <name>_end=
            kwargs[field_name + "_start"] = spec["start"]
            kwargs[field_name + "_end"] = spec["end"]
    if entry.get("priority") is not None:
        kwargs["MATCH_PRIORITY"] = entry["priority"]
    kwargs.update(entry["action_params"])

    add_method = getattr(table, "add_with_" + action_name)
    add_method(**kwargs)
    installed += 1

print("DEPLOY_DONE: %d entries installed, %d default actions set" % (installed, defaults_set))
