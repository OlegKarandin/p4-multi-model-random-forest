"""p4model/lanes.py: the crossbar lane model (audit
reviews/model_audit_2026-09-27.md Sec 7.2-7.5), ported from the audit's scratch
scripts width_layout.py / fastlane.py (its stage simulation, from
stage_sim_fast.py, was deleted 2026-11-13: xbar.py prices shared stages).

Every number pinned here was OBSERVED in a real p4c compile, not derived:

  * w019 / w027 -- results/tcam_phv_slice_sweep.csv (field widths (19, 64) and
    (27, 56), real_blocks 2 and 3); the tables.tail_is_isolatable docstring
    walks through why they differ.
  * REAL_DESIGN_KEYS -- every distinct classification-tree key (field bit
    widths) in the real design compiles results/compiler_calibration_v6,
    results/compiler_calibration_extra, results/tcam_margin_screen and the
    audit's arm-D recompiles (reviews/model_audit_scratch/
    priority_exp_ddos_first_noovct). Embedded because those trees are
    gitignored.

The port was also checked against the scratch code itself (not committed; the
scratch directory is untracked): layout for w in 0..329,
key_bytes (since deleted) and standalone on 1679 keys -- 0 differences, 99 read as None.
"""
import pytest

from p4model import lanes
from p4model.tables import codeword_to_blocks


REAL_DESIGN_KEYS = (
    (1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4),
    (1, 1, 1, 1, 1, 2, 5, 6, 6, 6, 6, 7, 8, 8, 9, 10, 10),
    (1, 1, 1, 1, 2, 2, 2, 2, 2, 3),
    (1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5, 5, 7, 8),
    (1, 1, 1, 1, 2, 2, 3, 3, 6, 6, 9),
    (1, 1, 1, 2, 2, 2, 2, 2, 4, 4, 4, 5),
    (1, 1, 1, 2, 2, 3, 3),
    (1, 1, 1, 2, 3, 3, 3, 3, 4, 8),
    (1, 1, 2, 2, 2, 3, 3, 3, 5, 6),
    (1, 1, 2, 2, 2, 3, 4, 4, 4, 5, 5, 6, 7, 7),
    (1, 2, 2, 2, 4, 5, 5, 6, 8, 8, 11, 12, 13, 17),
    (1, 2, 2, 6, 7, 8, 9, 10, 13, 13, 14, 14, 15, 15, 18, 22),
    (1, 2, 3, 4, 5, 6, 7, 7, 7, 8, 9, 10, 14),
    (1, 2, 6, 6, 8, 8, 9, 9, 10, 10),
    (1, 3, 3, 4, 4, 6, 7, 8, 8, 8, 10, 11, 11),
    (1, 3, 5, 16, 16, 17, 19, 21, 22, 26, 27, 29, 32),
    (1, 3, 7, 9, 11, 12, 14, 16, 17, 19, 19, 22, 25),
    (1, 5, 5, 6, 13, 14, 15, 22, 22, 28, 30, 33, 34, 38),
    (1, 5, 9, 14, 15, 16, 17, 18, 18, 19, 19, 21, 22),
    (1, 6, 15, 19, 22, 23, 25, 27, 30, 32),
    (1, 10, 10, 11, 14, 15, 18, 22, 23, 25, 25, 28, 30),
    (2, 2, 2, 4, 4, 5, 5, 5, 7, 7, 8, 9, 10, 12),
    (2, 3, 4, 5, 6, 7, 7, 7, 8, 13, 15, 20, 24, 24, 28, 28),
    (2, 3, 5, 6, 8, 9, 11, 12, 12, 19, 19, 22, 23, 28, 30),
    (2, 5, 7, 7, 9, 10, 10, 12, 21, 22, 22, 23, 25),
    (2, 5, 7, 8, 9, 10, 10, 11, 11, 19, 20, 20, 22, 24, 27),
    (2, 5, 9, 10, 12, 14, 17, 20, 23, 23, 24, 26, 26, 32),
    (2, 17, 20, 20, 24, 31),
    (3, 4, 4, 9, 9, 10, 10, 13, 16, 17, 18, 18),
    (3, 4, 5, 5, 6, 6, 9, 11, 12, 14, 16, 17),
    (4, 8, 9, 13, 24, 25, 27, 30, 32, 46),
    (4, 8, 10, 12, 19, 20, 23, 23, 30, 39),
    (4, 17, 19, 29, 31, 33, 37, 44),
    (5, 12, 12, 12, 12, 13, 15, 16, 17, 22),
    (6, 6, 19, 23, 28, 37, 37, 40, 49, 58),
    (7, 17, 31, 33, 34, 36, 39, 46),
    (8, 9, 13, 17, 17),
    (9, 36, 48, 51, 53, 53, 56),
    (15, 23),
    (17, 18, 23, 24, 28, 29, 30, 30),
    (18, 18, 18, 19, 19, 21, 24, 27),
    (19, 20, 21, 21, 23, 23, 28),
    (23, 33, 36, 38),
    (26, 30, 31, 32, 32),
    (26, 57, 60, 65, 74),
    (27, 28, 29, 32),
    (27, 52),
    (29,),
    (33, 42, 48, 49, 54, 66),
    (35, 85, 99, 101),
    (37, 46, 61),
    (37, 49),
    (39, 45),
    (41, 42, 42, 52),
    (46,),
    (46, 46, 48, 64),
    (47, 56, 66, 69, 72),
    (52, 62, 63, 65),
    (54, 56),
    (60, 73, 76, 86, 88),
    (62, 67, 76, 86, 86),
    (64, 69),
    (70, 104, 111),
    (79, 138),
    (88, 90, 108, 120),
    (93, 156),
    (98, 106, 126),
    (140, 144),
    (146,),
    (179, 204),
    (188,),
    (188, 214),
    (321,),
    (477,),
)

# Real design keys where the lane price is BELOW what p4c charged -- the
# accepted "greedy crossbar miss" class (reviews/campaign_2026_10_lane_findings.md
# Sec 5): p4c parks a whole byte on the midbyte and leaves no nibble for the
# version bits. (key widths) -> (lane price, p4c observed). The ladder priced
# (27, 52) right by luck.
LANE_BELOW_P4C_REAL_KEYS = {
    (27, 52): (2, 3),   # independent_low_sd5's ddos trees
}

# The two real keys p4c charged one block less than the plain width rule gives:
# they were compiled under the relaxed H16 + remainder pin the model used to
# choose (M150_k7_s11, M50_k8_s13; spec 2026-09-29 Sec 1.2). Since decision 4.1
# the pin is the plain one, which prices them one higher (unpinned, p4c charged
# the plain price too: +1 block per tree). (key widths) -> (plain price, p4c
# charged under the old relaxed pin).
PLAIN_ABOVE_RELAXED_COMPILES = {
    (19, 20, 21, 21, 23, 23, 28): (5, 4),
    (18, 18, 18, 19, 19, 21, 24, 27): (6, 5),
}


def _key_bytes(widths):
  """A key's byte list as this model lays it out: the width rule."""
  return lanes.bytes_from_layout(list(widths), lanes.layout)


def _price(widths):
  return lanes.standalone(_key_bytes(widths))


# ---------------------------------------------------------------- pinned probes

def test_w019_costs_two_blocks():
  assert _price((19, 64)) == 2


def test_w027_costs_three_blocks():
  assert _price((27, 56)) == 3


# -------------------------------------------------- against the production price

def test_real_design_key_set_is_the_audited_74():
  assert len(set(REAL_DESIGN_KEYS)) == 74


@pytest.mark.parametrize('widths', REAL_DESIGN_KEYS)
def test_lane_standalone_is_never_below_the_headline_ladder(widths):
  """The lane model never prices a real key cheaper than the ladder, except
  the listed greedy-miss keys, where p4c itself charged the ladder's price."""
  price = _price(widths)
  assert price is not None
  if widths in LANE_BELOW_P4C_REAL_KEYS:
    lane, observed = LANE_BELOW_P4C_REAL_KEYS[widths]
    assert price == lane < observed == codeword_to_blocks(widths)
    return
  assert price >= codeword_to_blocks(widths)
  if widths in PLAIN_ABOVE_RELAXED_COMPILES:
    assert (price, codeword_to_blocks(widths)) == PLAIN_ABOVE_RELAXED_COMPILES[widths]


@pytest.mark.parametrize('widths', REAL_DESIGN_KEYS)
def test_lane_standalone_equals_production_on_real_keys(widths):
  """On every real design key but the listed greedy misses the lane price and
  the ladder agree."""
  if widths in LANE_BELOW_P4C_REAL_KEYS or widths in PLAIN_ABOVE_RELAXED_COMPILES:
    pytest.skip('listed miss, asserted above')
  assert _price(widths) == codeword_to_blocks(widths)


# Synthetic probe keys (results/tcam_table_scoreboard.csv) where the lane price
# is BELOW codeword_to_blocks -- and equal to what p4c actually charged. These
# are production over-predictions, not lane under-predictions; they are why the
# "never below" guarantee above is stated over real design keys only.
LANE_BEATS_PRODUCTION_PROBES = (
    # (source, id, widths, p4c observed blocks = lane price, production)
    ('tcam_ledger_divergence_sweep', 'div04_n3_g5', (47, 51, 71), 4, 5),
    ('tcam_ledger_divergence_sweep', 'div06_n4_g5', (23, 27, 48, 69), 4, 5),
    ('tcam_ledger_divergence_sweep', 'div12_n5_g11', (62, 85, 89, 93, 94), 10, 11),
    ('tcam_field_count_sweep', 't28_n4', (16, 16, 24, 28), 2, 3),
    ('tcam_field_count_sweep', 't28_n5', (8, 16, 16, 16, 28), 2, 3),
    ('tcam_field_count_sweep', 't51_n3', (16, 16, 51), 2, 3),
    ('tcam_field_count_sweep', 't51_n4', (8, 8, 16, 51), 2, 3),
    ('tcam_field_count_sweep', 't51_n5', (8, 8, 8, 8, 51), 2, 3),
    ('tcam_field_count_sweep', 't28_n3_alt', (16, 28, 40), 2, 3),
)


@pytest.mark.parametrize('source,ident,widths,observed,production',
                         LANE_BEATS_PRODUCTION_PROBES,
                         ids=[p[1] for p in LANE_BEATS_PRODUCTION_PROBES])
def test_lane_below_production_only_where_p4c_agrees(source, ident, widths, observed, production):
  assert codeword_to_blocks(widths) == production
  assert _price(widths) == observed < production


# ---------------------------------------------------------------- layout rule

@pytest.mark.parametrize('w,expected', [
    (0, []),
    (3, [('B', 3)]),
    (8, [('B', 8)]),
    (11, [('H', 11)]),
    (19, [('W', 19)]),
    (32, [('W', 32)]),
    (33, [('B', 8), ('W', 25)]),
    (41, [('H', 16), ('W', 25)]),
    (57, [('W', 32), ('W', 25)]),
    (65, [('B', 8), ('W', 32), ('W', 25)]),
    (204, [('H', 16)] + [('W', 32)] * 5 + [('W', 28)]),
])
def test_width_layout(w, expected):
  assert lanes.layout(w) == expected
  assert sum(bits for _, bits in lanes.layout(w)) == max(w, 0)


# The layout rule BEFORE 2026-10-04's fill-low fix, kept verbatim so the pin
# identity below compares against what campaign_2026_10 emitted.
def _pre_fill_low_layout(w):
  if w <= 0:
    return []
  if w <= 8:
    return [('B', w)]
  if w <= 16:
    return [('H', w)]
  if w <= 32:
    return [('W', w)]
  top = 24 + ((w - 1) % 8) + 1
  low = w - top
  out = []
  rem = low % 32
  if rem == 8:
    out.append(('B', 8))
  elif rem == 16:
    out.append(('H', 16))
  elif rem == 24:
    out.append(('W', 24))
  out += [('W', 32)] * (low // 32)
  out.append(('W', top))
  return out


@pytest.mark.parametrize('w,expected', [
    (49, [('W', 32), ('W', 17)]),
    (51, [('W', 32), ('W', 19)]),
    (56, [('W', 32), ('W', 24)]),
    (81, [('W', 32), ('W', 32), ('W', 17)]),
    (88, [('W', 32), ('W', 32), ('W', 24)]),
    (113, [('W', 32), ('W', 32), ('W', 32), ('W', 17)]),
    (120, [('W', 32), ('W', 32), ('W', 32), ('W', 24)]),
])
def test_pinned_24_remainder_fields_fill_the_low_container_first(w, expected):
  """Pins fix container SIZES; p4c fills the LOW container first (measured
  7,479/7,479 fields over 32 bits, reviews/campaign_2026_10_lane_findings.md
  Sec 4). A 51-bit field is 32|19, not 24|27."""
  assert lanes.layout(w) == expected


@pytest.mark.parametrize('w,expected', [
    (19, [32]), (7, [8]), (33, [8, 32]), (46, [16, 32]), (56, [32, 32]), (0, []),
])
def test_container_sizes_plain_values(w, expected):
  """The plain pin rule's values, low container first (interim review T6)."""
  assert lanes.container_sizes(w) == expected


def test_fill_low_fix_leaves_every_pin_unchanged():
  """Per FIELD, the pins (container sizes, low slice first) do not move, for
  every width 0..329."""
  failures = []
  for w in range(0, 330):
    sizes = lanes.container_sizes(w)
    if (sizes != [lanes.CONTAINER_BITS[k] for k, _ in _pre_fill_low_layout(w)]
        or sorted(sizes) != sorted(lanes.CONTAINER_BITS[k] for k, _ in lanes.layout(w))
        or sum(bits for _, bits in lanes.layout(w)) != max(w, 0)):
      failures.append(w)
  assert not failures, 'widths whose pins moved: %r' % failures


def test_byte_list_marks_nibble_only_bytes():
  assert lanes.bytes_from_widths((12,)) == [('H', 0, False), ('H', 1, True)]


# ---------------------------------------------------------------- None, not 99

def test_unpriceable_key_is_none():
  assert lanes.standalone(_key_bytes((600,))) is None


# ------------------------------------------------ table_blocks (spec 2026-10-04)

from p4model.errors import CrossbarKeyTooWide
from p4model.tables import codeword_bits_to_blocks


@pytest.mark.parametrize('widths', REAL_DESIGN_KEYS)
def test_table_blocks_is_the_crossbar_ports_price_of_the_pinned_layout(widths):
  # Since 2026-11-13 the per-table price is p4c's ternary crossbar allocation
  # (xbar.py) of the pinned layout, no longer the lane count.
  from p4model import xbar
  assert lanes.table_blocks(widths) == xbar.alone_blocks(lanes.key_fields(widths))


@pytest.mark.parametrize('widths', REAL_DESIGN_KEYS)
def test_table_blocks_equals_p4c_on_real_keys(widths):
  # The ladder is p4c's charge on every real design key; on the one key where
  # the lane count was below p4c (LANE_BELOW_P4C_REAL_KEYS, the greedy miss)
  # the port gives p4c's number.
  if widths in LANE_BELOW_P4C_REAL_KEYS:
    assert lanes.table_blocks(widths) == LANE_BELOW_P4C_REAL_KEYS[widths][1]
  elif widths in PLAIN_ABOVE_RELAXED_COMPILES:
    assert lanes.table_blocks(widths) == PLAIN_ABOVE_RELAXED_COMPILES[widths][0]
  else:
    assert lanes.table_blocks(widths) == codeword_to_blocks(widths)


def test_table_blocks_pins_the_two_known_ladder_disagreements_off_the_compiled_set():
  # Lane-higher: three 49-56-bit fields (spec Sec I.2, review decision 6).
  assert codeword_to_blocks((54, 55, 56)) == 4
  assert lanes.table_blocks((54, 55, 56)) == 5
  # Lane-lower: the dsp41 probe key.
  assert lanes.table_blocks((84, 84)) == 4


def test_table_blocks_ignores_field_order():
  # 3: p4c's charge on independent_low_sd5's ddos trees (the lane count said 2)
  assert lanes.table_blocks((52, 27)) == lanes.table_blocks((27, 52)) == 3


def test_table_blocks_of_the_empty_key_is_the_ladders_floor():
  assert lanes.table_blocks(()) == codeword_bits_to_blocks(0)
  assert lanes.table_blocks((0, 0)) == codeword_bits_to_blocks(0)


def test_table_blocks_rejects_a_key_over_the_crossbar_before_enumerating(monkeypatch):
  def boom(*_):
    raise AssertionError('the crossbar must not run for a 66-byte key')
  monkeypatch.setattr(lanes.xbar, 'alone_blocks', boom)
  lanes._table_blocks.cache_clear()
  with pytest.raises(CrossbarKeyTooWide) as info:
    lanes.table_blocks((8,) * 66)
  assert info.value.args[1] == 66


def test_table_blocks_raises_when_the_crossbar_cannot_fit_the_key(monkeypatch):
  monkeypatch.setattr(lanes.xbar, 'alone_blocks', lambda _fields: None)
  lanes._table_blocks.cache_clear()
  with pytest.raises(CrossbarKeyTooWide) as info:
    lanes.table_blocks((27, 52))
  assert info.value.args[1] == 11
  lanes._table_blocks.cache_clear()


def test_table_blocks_memo_returns_identical_results():
  lanes._table_blocks.cache_clear()
  first = [lanes.table_blocks(w) for w in REAL_DESIGN_KEYS]
  second = [lanes.table_blocks(w) for w in REAL_DESIGN_KEYS]
  assert first == second
  info = lanes._table_blocks.cache_info()
  assert info.hits >= len(REAL_DESIGN_KEYS) and info.maxsize == 65536


def test_table_blocks_can_rise_when_one_field_loses_a_bit():
  """p4c's allocator is greedy, so its price is not monotone in the key: a
  5-bit field is a whole byte, a 4-bit one a single nibble, and with the
  nibble p4c's midbyte choice leaves this key one block dearer, under every
  container-numbering tie (2026-11-13). The lane count (monotone, the B3
  measurement) said 4 for both. Threshold alignment prices with the ladder
  codeword_to_blocks, which stays monotone."""
  from p4model import xbar
  assert lanes.table_blocks((5, 19, 24, 53, 62)) == 4
  assert lanes.table_blocks((4, 19, 24, 53, 62)) == 5
  for tie in xbar.TIES:
    for widths, price in (((5, 19, 24, 53, 62), 4), ((4, 19, 24, 53, 62), 5)):
      key = xbar.field_key(lanes.key_fields(widths), {})
      crossbar = xbar.StageXbar(tie=tie)
      assert crossbar.allocate(key) and xbar.blocks_per_row(key) == price
  assert (codeword_to_blocks((4, 19, 24, 53, 62))
          <= codeword_to_blocks((5, 19, 24, 53, 62)))


# ---- lane cap: only 3 of the 12 groups start on any one lane (2026-11-11) ----

# get_classification_tree_ddos_2 of heldout7's joint-tree_M035_s08_k16: 9 of its
# 14 key bytes are byte 3 of a 32-bit container. p4c: 5 blocks (groups 0, 1, 2,
# 7, 10; the three that start on lane 3 are 2, 7 and 10).
_LANE3_HEAVY = ([('W', 3, True)] * 7 + [('W', 3, False)] * 2 + [('W', 2, True)]
                + [('H', 1, True)] + [('B', 0, True)] * 2 + [('B', 0, False)])


def test_three_groups_start_on_each_lane():
  starts = [start % 4 for start in lanes.GROUP_START]
  assert sorted(starts) == [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3
  assert lanes.GROUPS_PER_START_LANE == {0: 3, 1: 3, 2: 3, 3: 3}


def test_standalone_never_takes_more_extra_slots_in_a_lane_than_groups_start_there():
  # 4 groups would need 4 extras on lane 3 (4 + 4 + the lane-3 midbyte = 9);
  # the chip offers 4 + 3 + 1 = 8, so the key needs a fifth group.
  assert lanes.standalone(_LANE3_HEAVY) == 5


def test_lane_cap_leaves_a_balanced_key_alone():
  # 10 bytes, lanes [3, 3, 2, 2]: one group starting on lane 0, one on lane 1.
  key = [('W', k, False) for k in (0, 1, 2, 3, 0, 1, 2, 3, 0, 1)]
  assert lanes.standalone(key) == 2
