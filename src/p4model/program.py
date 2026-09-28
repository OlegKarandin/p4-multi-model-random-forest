"""These describe the program *this* generator emits, not the chip it targets;
a reader modelling a different P4 program changes this file, not target.py."""
import math

# Width of every per-feature value field the range-matching tables key on
# (build_p4_script.py:775 emits "bit<16> <feature>_val" for each selected
# feature). 16 bits is the project's decided feature precision; one range
# table keys on exactly one such field, hence 2 crossbar bytes per table.
FEATURE_VALUE_BIT_WIDTH = 16
RANGE_TABLE_KEY_BYTES = math.ceil(FEATURE_VALUE_BIT_WIDTH / 8)

# Every per-flow register in this design is indexed by meta.flow_hash, so the
# hash occupies whole stages ahead of any register touch -- THREE of them, not
# the one this constant used to claim. Measured over all 121 range tables in
# the 19 real compiles of results/compiler_calibration/ (see
# reviews/p4_tofino_reference.md §4.6): every committed placement opens with
# a metadata-init table at stage 0, tbl_calc_flow_hash$precompute at stage 1
# and tbl_calc_flow_hash at stage 2, so the first RegisterAction in any
# feature's chain lands at stage 3. `real_stage - level` had a floor of
# exactly +2 in every one of the 19 rows under the old value of 1; at 3 the
# floor is 0, i.e. levels now name the earliest stage the compiler really
# uses. (Values above the floor are tables the packer legitimately pushed
# later, which is placement, not origin.)
FLOW_HASH_LEVEL = 3

# The vote_app/vote_ddos tables read every tree's class and so always sit one
# stage past the last classification table. Measured: exactly 1 in all 19
# compiles, with no exceptions and no scaling. stage_depth is the quantity
# checked against TOFINO_PIPELINE_STAGES, so leaving this out understated the
# depth of every design by one whole stage.
VOTE_EPILOGUE_STAGES = 1

# flow_forward_srcaddr_reg backs flow_orientation_action, which resolves
# meta.fwd. generate_P4_registers_and_apply emits its .execute() call
# UNCONDITIONALLY into the apply block (build_p4_script.py:2092), not just
# when a gated feature is selected, so it always claims one stateful ALU in
# the first register stage and every fwd-/bwd-gated register waits a stage on
# it. It lives outside FEATURE_REGISTER_CATALOG (no feature owns it), so the
# schedule has to add it by hand.
ORIENTATION_REGISTER = "flow_forward_srcaddr"

# The order generate_P4_registers_and_apply lays the RegisterAction .execute()
# call sites down in: the unconditional ones straight into the apply block,
# then `if (meta.fwd == 1) { ... }`, then `if (meta.fwd == 0) { ... }`
# (build_p4_script.py's three _execute_lines calls, in exactly this order).
# It is a real ordering, not a presentation choice: p4c's table placer walks
# the control block with a work-list CURSOR, so it cannot begin the second
# gated block before the first one is fully placed.
REGISTER_BLOCK_ORDER = (None, "fwd", "bwd")

# The two classification tasks this program serves, spelled exactly as the
# generator spells them in table names (get_classification_tree_app_3,
# table_7_ddos_flow_iat_min, vote_app) and as accuracy_metrics takes them.
# evaluation._pool_inputs labels every table with one of these (its
# range_task / ternary_task lists), usage.assemble_usage reads the labels to
# decide which range tables a tree waits for, and p4_artifact_replay recovers
# the same labels from a compiled program's table names.
APP_TASK = "app"
DDOS_TASK = "ddos"
TASKS = (APP_TASK, DDOS_TASK)

# range_task's label for a range table whose code_<feature> field EVERY task's
# trees key on, so every tree waits for it: all of them under 'joint' (one
# merged interval set), and under 'disjoint' a feature both models split
# identically -- build_p4_script._resolve_disjoint_feature_plan then emits one
# un-prefixed table (table_1_bwd_iat_mean) instead of an _app_/_ddos_ pair.
# Never a ternary_task value: every classification tree belongs to one task.
SHARED_TASK = "shared"
