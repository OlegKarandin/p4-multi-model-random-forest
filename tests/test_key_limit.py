"""Decision 6a (Task 32): the key-size rule on a lone ternary key is p4c's own two
checks -- at most target.MAX_TERNARY_KEY_BITS = 524 LIVE bits
(phv/table_phv_constraints.cpp:88) and a placement in one stage by the ported
crossbar (12 groups, at most 12 blocks per 512-row word, tables.word_blocks) --
read by every site that refuses a key. Pinned on the five keys of probe 6a
(results/encoding_evidence_2026_11/model8/impl/probe_6a_precise.md, p4c 9.13.4,
5 compiles): every site accepts / refuses each as p4c did, at p4c's blocks."""
import pytest

from p4gen import evaluation as ev
from p4model import lanes, packing, tables, target
from p4model.errors import CrossbarKeyTooWide
from src.training import align_budget


def test_the_limits_are_p4cs():
  assert target.MAX_TERNARY_KEY_BITS == 524
  assert target.TCAM_ROWS_PER_STAGE == 12
  assert target.TERNARY_CROSSBAR_STAGE_BYTES == 66


# (name, live bits per field, p4c verdict: blocks per word, or None = refused)
PROBE_KEYS = [
    ("i: 65 x 8 + one 4-bit (66 B, 524 b)", (8,) * 65 + (4,), 12),
    ("ii: 64 x 8 + two 6-bit (66 B, 524 b)", (8,) * 64 + (6, 6), None),
    ("iii: 66 x 8 (66 B, 528 b)", (8,) * 66, None),
    ("i-b: 16 x 32 + one 12-bit (66 B, 524 b)", (32,) * 16 + (12,), 12),
    ("ii-b: 15 x 32 + two 22-bit (66 B, 524 b)", (32,) * 15 + (22, 22), None),
]
PROBE_IDS = ["i", "ii", "iii", "i-b", "ii-b"]


def _intervals(widths):
  """One feature per field, `w` code bits (w + 1 intervals) each."""
  return {"F%d" % i: [(j, j) for j in range(w + 1)] for i, w in enumerate(widths)}


def _training(widths):
  """tables.ternary_matching_resource_usage, the joint / disjoint / per-task
  training refusal; the per-word price of a one-word tree."""
  _, blocks, _, _ = tables.ternary_matching_resource_usage(
      {0: {"0" * sum(widths): 0}}, _intervals(widths))
  return blocks


def _lane_price(widths):
  lanes._table_blocks.cache_clear()
  return lanes.table_blocks(widths)


def _alignment(widths):
  """Threshold alignment's tree-key price (align_budget.port_price)."""
  lanes._table_blocks.cache_clear()
  return align_budget.port_price(widths)


def _stage_test(widths):
  """packing's classification pool, which the replay of a program text reaches."""
  key = tuple((i, tuple(lanes.layout(w))) for i, w in enumerate(widths))
  byte_width = tables.codeword_fields_to_bytes_from_bits(widths)
  packing._alone_blocks.cache_clear()
  plan = packing.crossbar_stages_needed([(12, byte_width)], key_field_bits=[tuple(widths)],
                                        key_slices=[key])
  return plan.table_blocks[0]


def _replay(widths):
  """The replay's per-tree refusal of a program text (p4_replay.check_trees' key
  part: the bit check on tree_key_bits, then _field_price's crossbar fit). The
  PHV count check_trees runs after it is left out: 66 one-byte code fields
  would exhaust the 8-bit containers, a refusal of its own."""
  from p4gen import p4_replay
  name = "get_classification_tree_app_0"
  fields = ["code_f%d" % i for i in range(len(widths))]
  program = p4_replay.Program(
      tables={name: fields}, widths={f: -(-w // 8) for f, w in zip(fields, widths)},
      bits=dict(zip(fields, widths)), sizes={name: 1})
  packing._alone_blocks.cache_clear()
  tables.check_key_bits(p4_replay.tree_key_bits(program, name))
  return p4_replay._field_price(program, name)


SITES = [_training, _lane_price, _alignment, _replay]


@pytest.mark.parametrize("site", SITES, ids=lambda site: site.__name__)
@pytest.mark.parametrize("name,widths,p4c", PROBE_KEYS, ids=PROBE_IDS)
def test_every_site_decides_the_probe_keys_as_p4c(site, name, widths, p4c):
  if p4c is None:
    with pytest.raises(CrossbarKeyTooWide):
      site(widths)
  else:
    assert site(widths) == p4c, name


@pytest.mark.parametrize("name,widths,p4c", [k for k in PROBE_KEYS if k[2] is not None],
                         ids=["i", "i-b"])
def test_the_stage_simulation_places_the_compiling_probe_keys(name, widths, p4c):
  assert _stage_test(widths) == p4c, name


def test_the_bit_check_names_the_live_bits():
  with pytest.raises(CrossbarKeyTooWide) as info:
    tables.check_key_bits(528)
  assert info.value.args[1] == 528 and info.value.magnitude == 4
  tables.check_key_bits(524)


def test_a_word_over_a_column_is_refused():
  assert tables.word_blocks(12) == 12
  assert tables.word_blocks(13) is None
  assert tables.word_blocks(None) is None


def test_the_crossbar_refusal_of_probe_ii_is_the_ports_13_blocks_per_word():
  """Probe (ii) fits the 12 groups (66 whole bytes = 12 x 5 + 6 midbytes) but
  needs a 13th block per word for the version bits: refused by word_blocks,
  not by any byte constant."""
  from p4model import xbar
  widths = (8,) * 64 + (6, 6)
  assert sum(widths) == target.MAX_TERNARY_KEY_BITS
  assert xbar.alone_blocks(lanes.key_fields(widths)) == 13


def _packed_training(widths):
  """evaluation._packed_tree_entries, the per-tree training bit check."""
  return ev._packed_tree_entries({0: {"0": 0}}, sum(widths), False)


@pytest.mark.parametrize("name,widths,p4c", PROBE_KEYS, ids=PROBE_IDS)
def test_the_packed_bit_check(name, widths, p4c):
  if sum(widths) > target.MAX_TERNARY_KEY_BITS:
    with pytest.raises(CrossbarKeyTooWide):
      _packed_training(widths)
  else:
    _packed_training(widths)


def test_a_range_key_counts_four_crossbar_byte_slots():
  """Decision 6f: RANGE_LO + RANGE_HI per byte of the 16-bit key, the unit
  xbar.range_key uses; the specs and the stage fields carry the same 4."""
  intervals = {"Flow IAT Max": [(0, 10), (11, 20)]}
  (fields,) = tables.range_key_fields_for(intervals)
  assert {size for _, size in fields} == {4}
  _, _, specs = tables.range_matching_resource_usage(intervals)
  assert [width for _, width in specs] == [4]
