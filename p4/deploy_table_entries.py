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
import keyword

PROGRAM_NAME = "p4_code_RF_models"  # must match the program name used to launch bf_switchd -p <PROGRAM_NAME>
CONTROL_BLOCK = "SwitchIngress"     # matches this project's real control block name (resources/p4_template.p4)
ENTRIES_FILE_PATH = "p4/table_entries.json"


def _range_entry_count(lo, hi, nibble_widths=(4, 4, 4, 4)):
    # Duplicated from src/p4gen/evaluation.py:range_entry_count rather than imported: this
    # script runs inside bfshell's embedded Python, which does not have this project's normal
    # dependency stack (sklearn etc., pulled in transitively by evaluation.py's own imports).
    # Exact port of expand_range() (bf-drivers/src/pipe_mgr/pipe_mgr_entry_format.c) -- computes
    # the real number of physical TCAM rows a range key [lo, hi] needs once installed.
    n = len(nibble_widths)
    start_vals, end_vals = [], []
    shift = 0
    for w in nibble_widths:
        start_vals.append(1 << shift)
        end_vals.append((1 << (w + shift)) - 1)
        shift += w

    if hi < lo:
        raise ValueError("hi < lo")

    range_start, end, count = lo, hi, 0
    while True:
        if range_start == 0:
            start_nibble = n - 1
        else:
            zeroes = (range_start & -range_start).bit_length() - 1
            cum, start_nibble = 0, n - 1
            for j in range(n):
                cum += nibble_widths[j]
                if cum > zeroes:
                    start_nibble = j
                    break

        range_end = None
        for i in range(start_nibble + 1, 0, -1):
            candidate = range_start | end_vals[i - 1]
            while (candidate >= range_start and candidate > end and
                   candidate >= start_vals[i - 1]):
                candidate -= start_vals[i - 1]
            if candidate >= range_start and candidate <= end:
                range_end = candidate
                break

        count += 1
        range_start = range_end + 1
        if range_end >= end:
            break

    return count


def _row_cost(entry):
    # Non-zero only for range-keyed entries (feature codeword-bit tables); ternary
    # classification entries and is_default_action records always cost 0 here, so the sort
    # below leaves their relative order untouched (Python's sort is stable) and only reorders
    # within each range table's own entries.
    for spec in entry.get("key_fields", {}).values():
        if "start" in spec:
            return _range_entry_count(int(spec["start"]), int(spec["end"]))
    return 0


with open(ENTRIES_FILE_PATH) as f:
    table_entries = json.load(f)

# Real hardware (reviews/t12_tcam_model_experiment_plan.md Section 13-14, ".superpowers/sdd/
# task-4c-report.md") confirmed a TCAM block's real usable capacity depends on insertion order:
# installing wide/multi-row range entries first and narrow/single-row entries last reaches the
# full nominal 512-row capacity exactly; the reverse order loses up to 6 rows to fragmentation
# (pipe_mgr_tcam_find_next_free requires multi-row entries to land in mutually contiguous free
# rows). Sorting by descending per-entry row cost before installing reproduces the order that
# reached full capacity in both tested configs.
table_entries.sort(key=lambda e: -_row_cost(e))


def _data_kwargs(action_params):
    # Every classify_flow_codeword_* P4 action has a single parameter literally named
    # "class" (see build_p4_script.generate_P4_actions). "class" is a Python keyword, so
    # bfrtcli's own generated add_with_<action>/set_default_with_<action> methods cannot
    # have a parameter literally named "class" (that would be a SyntaxError in the exec'd
    # def). bfrtcli's validate_p4_entity_name/add_prefix_to_entry (see bfrtcli.py) detects
    # the keyword collision and renames the data field to "data_<name>" (names_prefix=
    # "data_" for data fields in _make_core_method_strs), so the real method signature has
    # data_class=None, not class=None. Apply the same rename here before passing kwargs.
    return {("data_" + k) if keyword.iskeyword(k) else k: v
            for k, v in action_params.items()}

program = getattr(bfrt, PROGRAM_NAME)  # noqa: F821 -- bfrt is injected by bfshell
pipe = program.pipe

installed = 0
defaults_set = 0
for entry in table_entries:
    table = getattr(getattr(pipe, CONTROL_BLOCK), entry["table_name"])
    action_name = entry["action"]

    if entry.get("is_default_action"):
        set_default_method = getattr(table, "set_default_with_" + action_name)
        set_default_method(**_data_kwargs(entry["action_params"]))
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
    kwargs.update(_data_kwargs(entry["action_params"]))

    add_method = getattr(table, "add_with_" + action_name)
    add_method(**kwargs)
    installed += 1

print("DEPLOY_DONE: %d entries installed, %d default actions set" % (installed, defaults_set))
