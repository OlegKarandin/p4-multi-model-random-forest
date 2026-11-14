"""Decision 3.8: no metadata field of a generated program is read before it is written."""
import os

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from p4gen import build_p4_script as bps
from p4gen.feature_registers import FEATURE_REGISTER_CATALOG
from p4gen.program_checks import _analyse, metadata_reads_before_writes
from p4model.program import (DISJOINT_ENCODING, JOINT_ENCODING, PER_TASK_ENCODING,
                             PER_TREE_ENCODING)

FEATURES = sorted(FEATURE_REGISTER_CATALOG)          # 18 features


def _forest(n_classes, seed, names):
  rng = np.random.RandomState(seed)
  X = rng.randint(0, 2000, size=(600, len(names)))
  y = ((X[:, seed % len(names)] + X[:, (seed + 1) % len(names)] // 3)
       // (2600 // n_classes)).clip(0, n_classes - 1)
  return bps.dt_thresholds_float_to_int(RandomForestClassifier(
      n_estimators=3, max_depth=4, random_state=seed).fit(X, y))


def _program(tmp_path, encoding, k):
  names = FEATURES[:k]
  app, ddos = _forest(3, 2, names), _forest(2, 1, names)
  if encoding == JOINT_ENCODING:
    joint = bps.get_joint_feature_intervals(app, names, ddos, names)
    intervals = (joint, joint)
  else:
    intervals = (bps.get_feature_intervals(app, names), bps.get_feature_intervals(ddos, names))
  path = bps.generate_P4_code(
      3, 2, app, ddos, feature_intervals_app=intervals[0], feature_intervals_ddos=intervals[1],
      output_dir=str(tmp_path) + os.sep, output_filename="design.p4",
      selected_features_app=names, selected_features_ddos=names, encoding=encoding)
  with open(path, encoding="utf-8") as handle:
    return handle.read()


@pytest.mark.parametrize("encoding", [JOINT_ENCODING, DISJOINT_ENCODING, PER_TASK_ENCODING,
                                      PER_TREE_ENCODING])
@pytest.mark.parametrize("k", [2, 8, 18])
def test_generated_programs_read_no_metadata_before_it_is_written(tmp_path, encoding, k):
  violations, stats = _analyse(_program(tmp_path, encoding, k))
  assert violations == []
  # Non-vacuity: the walk really executed registers and applied every declared table
  # (range + tree + vote tables).
  assert stats["registers_executed"] >= 1
  assert stats["tables_applied"] == stats["tables_declared"] >= 3


_MINI = """
control Ingress(inout header_t hdr, inout metadata_t meta, in intr ig_intr_md) {
	Register<bit<16>, bit<32>>(8) r_reg;
	action calc_flow_hash() {
		meta.flow_hash = h.get({ hdr.ipv4.src_addr }) & 0xFFF;
	}
	RegisterAction<bit<16>, bit<32>, bit<16>>(r_reg) r_action = {
		void apply(inout bit<16> value, out bit<16> rv) {
			if (meta.fwd == 0) { value = meta.now_pseudo_us; }
			rv = value;
		}
	};
	action set_code(bit<8> code) { meta.code_x = code; }
	table t_range {
		key = { meta.x_val: range; }
		actions = { set_code; }
		size = 4;
	}
	table t_tree {
		key = { meta.code_x[3:0] : ternary; }
		actions = { set_code; }
		size = 4;
	}
	apply {
		if (hdr.tcp.isValid()) {
%s
		}
		ig_tm_md.qid = 1;
	}
}
"""

_ORDER = (
    "\t\t\tcalc_flow_hash();\n"
    "\t\t\tmeta.fwd = 1;\n"
    "\t\t\tmeta.now_pseudo_us = 5;\n"
    "\t\t\tmeta.x_val = r_action.execute(meta.flow_hash);\n"
    "\t\t\tt_range.apply();\n"
    "\t\t\tt_tree.apply();\n")


def test_an_ordered_hand_made_program_is_clean():
  assert metadata_reads_before_writes(_MINI % _ORDER) == []


def test_a_val_read_before_its_register_executes_is_named():
  text = _MINI % ("\t\t\tcalc_flow_hash();\n\t\t\tmeta.fwd = 1;\n\t\t\tmeta.now_pseudo_us = 5;\n"
                  "\t\t\tt_range.apply();\n"
                  "\t\t\tmeta.x_val = r_action.execute(meta.flow_hash);\n")
  assert metadata_reads_before_writes(text) == ["x_val"]


def test_register_body_reads_count_where_the_register_runs():
  text = _MINI % ("\t\t\tcalc_flow_hash();\n"
                  "\t\t\tmeta.x_val = r_action.execute(meta.flow_hash);\n")
  assert metadata_reads_before_writes(text) == ["fwd", "now_pseudo_us"]


def test_the_execute_index_and_a_code_slice_key_are_reads():
  text = _MINI % ("\t\t\tmeta.fwd = 1;\n\t\t\tmeta.now_pseudo_us = 5;\n"
                  "\t\t\tmeta.x_val = r_action.execute(meta.flow_hash);\n"
                  "\t\t\tt_tree.apply();\n")
  assert metadata_reads_before_writes(text) == ["flow_hash", "code_x"]


def test_a_table_write_counts_from_that_tables_apply():
  text = _MINI % ("\t\t\tcalc_flow_hash();\n\t\t\tmeta.fwd = 1;\n\t\t\tmeta.now_pseudo_us = 5;\n"
                  "\t\t\tmeta.x_val = r_action.execute(meta.flow_hash);\n"
                  "\t\t\tt_tree.apply();\n\t\t\tt_range.apply();\n")
  assert metadata_reads_before_writes(text) == ["code_x"]



_PRELUDE = ("\t\t\tcalc_flow_hash();\n\t\t\tmeta.fwd = 1;\n\t\t\tmeta.now_pseudo_us = 5;\n")


def _with(lines):
  return _MINI % (_PRELUDE + lines)


def test_a_write_in_one_if_branch_does_not_count_afterwards():
  text = _with("\t\t\tif (meta.fwd == 1) { meta.a = 1; }\n\t\t\tmeta.b = meta.a;\n")
  assert metadata_reads_before_writes(text) == ["a"]


def test_a_write_in_both_branches_counts_afterwards():
  text = _with("\t\t\tif (meta.fwd == 1) { meta.a = 1; } else { meta.a = 2; }\n"
               "\t\t\tmeta.b = meta.a;\n")
  assert metadata_reads_before_writes(text) == []


def test_an_else_if_chain_needs_every_branch_to_write():
  text = _with("\t\t\tif (meta.fwd == 1) { meta.a = 1; } "
               "else if (meta.fwd == 0) { meta.a = 2; }\n\t\t\tmeta.b = meta.a;\n")
  assert metadata_reads_before_writes(text) == ["a"]


def test_unknown_calls_raise_instead_of_passing():
  with pytest.raises(ValueError, match="unknown register action"):
    metadata_reads_before_writes(_with("\t\t\tmeta.x_val = nope.execute(1);\n"))
  with pytest.raises(ValueError, match="unknown table"):
    metadata_reads_before_writes(_with("\t\t\tt_missing.apply();\n"))
  with pytest.raises(ValueError, match="unknown action"):
    metadata_reads_before_writes(_with("\t\t\tno_such_action();\n"))
  with pytest.raises(ValueError, match=r"\.apply\(\)"):
    metadata_reads_before_writes(_with("\t\t\tif (t_range.apply().hit) { }\n"))
