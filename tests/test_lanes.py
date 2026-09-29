"""src/p4model/lanes.py: the crossbar lane model (audit
reviews/model_audit_2026-09-27.md Sec 7.2-7.5), ported from the audit's scratch
scripts width_layout.py / fastlane.py / stage_sim_fast.py.

Every number pinned here was OBSERVED in a real p4c compile, not derived:

  * w019 / w027 -- results/tcam_phv_slice_sweep.csv (field widths (19, 64) and
    (27, 56), real_blocks 2 and 3); the tables.tail_is_isolatable docstring
    walks through why they differ.
  * dsp41 -- results/tcam_discount_scan.csv / crowded_stages.csv: a 41-byte
    spacer key (328,) placed first (crossbar groups 0-7, 8 blocks) and the
    (84, 84) key second in the same stage, which p4c charged 7 blocks although
    it costs 5 alone (4 groups + 3 whole midbytes -> max(4, 2*3 + 1) = 7).
  * REAL_DESIGN_KEYS -- every distinct classification-tree key (field bit
    widths) in the real design compiles results/compiler_calibration_v6,
    results/compiler_calibration_extra, results/tcam_margin_screen and the
    audit's arm-D recompiles (reviews/model_audit_scratch/
    priority_exp_ddos_first_noovct). Embedded because those trees are
    gitignored.

The port was also checked against the scratch code itself (not committed; the
scratch directory is untracked): layout/relaxed_layout for w in 0..329,
key_bytes and standalone on 1679 keys, stage_prices on 1611 random 2-4 key
stages in random order, price_with_supply on 2000 random supplies -- 0
differences, 99 read as None.
"""
import pytest

from src.p4model import lanes
from src.p4model.tables import codeword_to_blocks
from src.p4model.target import TERNARY_CROSSBAR_BYTE_GROUPS_PER_STAGE


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


def _price(widths):
  return lanes.standalone(lanes.key_bytes(widths))


# ---------------------------------------------------------------- pinned probes

def test_w019_costs_two_blocks():
  assert _price((19, 64)) == 2


def test_w027_costs_three_blocks():
  assert _price((27, 56)) == 3


def test_dsp41_second_key_pays_seven_beside_a_41_byte_spacer():
  spacer, key = lanes.key_bytes((328,)), lanes.key_bytes((84, 84))
  assert lanes.standalone(key) == 5 == codeword_to_blocks((84, 84))
  assert lanes.stage_prices([spacer, key], (0, 1)) == {0: 8, 1: 7}


# -------------------------------------------------- against the production price

def test_real_design_key_set_is_the_audited_74():
  assert len(set(REAL_DESIGN_KEYS)) == 74


@pytest.mark.parametrize('widths', REAL_DESIGN_KEYS)
def test_lane_standalone_is_never_below_the_headline_ladder(widths):
  """The lane model must never price a real key CHEAPER than production."""
  price = _price(widths)
  assert price is not None
  assert price >= codeword_to_blocks(widths)


@pytest.mark.parametrize('widths', REAL_DESIGN_KEYS)
def test_lane_standalone_equals_production_on_real_keys(widths):
  """Audit invariant 1: the first key in a stage keeps codeword_to_blocks,
  which is safe because the two agree on all 74 real design keys."""
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


def test_relaxed_layout_splits_17_to_32_bit_fields_into_halves():
  assert lanes.relaxed_layout(19) == [('H', 16), ('B', 3)]
  assert lanes.relaxed_layout(28) == [('H', 16), ('H', 12)]
  assert lanes.relaxed_layout(16) == lanes.layout(16)
  assert lanes.relaxed_layout(33) == lanes.layout(33)


@pytest.mark.parametrize('widths,plain,relaxed', [
    ((19, 20, 21, 21, 23, 23, 28), 5, 4),       # M150_k7_s11
    ((18, 18, 18, 19, 19, 21, 24, 27), 6, 5),   # M50_k8_s13
])
def test_layout_fallback_rule(widths, plain, relaxed):
  """Audit Sec 7.4: the width layout, unless its lane price exceeds
  codeword_to_blocks -- then the relaxed layout."""
  assert lanes.standalone(lanes.bytes_from_widths(widths)) == plain
  chosen = lanes.key_bytes(widths)
  assert chosen == lanes.bytes_from_layout(widths, lanes.relaxed_layout)
  assert lanes.standalone(chosen) == relaxed == codeword_to_blocks(widths)


def test_fallback_keeps_the_width_layout_when_it_already_matches_production():
  assert lanes.key_bytes((19, 64)) == lanes.bytes_from_widths((19, 64))


# ------------------------------------------------ key_layout (spec 2026-09-29)
# The generator pins every tree-key code_* field to the layout this model
# prices the key with (@pa_container_size), so the layout DECISION must be one
# function both sides call -- key_layout -- not a byte list the generator
# would have to reverse-engineer.

@pytest.mark.parametrize('widths', REAL_DESIGN_KEYS + (
    (19, 20, 21, 21, 23, 23, 28), (18, 18, 18, 19, 19, 21, 24, 27)))
def test_key_bytes_is_key_layout_applied(widths):
  lay = lanes.key_layout(widths)
  assert lay is lanes.layout or lay is lanes.relaxed_layout
  assert lanes.key_bytes(widths) == lanes.bytes_from_layout(widths, lay)


@pytest.mark.parametrize('widths,relaxed', [
    ((18, 18, 18, 19, 19, 21, 24, 27), True),    # M50_k8_s13 ddos key
    ((7, 17, 31, 33, 34, 36, 39, 46), False),    # M50_k8_s13 app key
    ((19, 20, 21, 21, 23, 23, 28), True),        # M150_k7_s11 ddos key
    ((19, 64), False),
])
def test_key_layout_picks_relaxed_only_where_key_bytes_does(widths, relaxed):
  assert lanes.key_layout(widths) is (
      lanes.relaxed_layout if relaxed else lanes.layout)


def test_container_sizes_are_container_widths_low_slice_first():
  assert lanes.container_sizes(19, lanes.relaxed_layout) == [16, 8]
  assert lanes.container_sizes(28, lanes.relaxed_layout) == [16, 16]
  assert lanes.container_sizes(19) == [32]
  assert lanes.container_sizes(7) == [8]
  assert lanes.container_sizes(33) == [8, 32]    # B8 low, W25 top
  assert lanes.container_sizes(46) == [16, 32]   # H16 low, W30 top
  assert lanes.container_sizes(56) == [32, 32]   # W24 low, W32 top
  assert lanes.container_sizes(0) == []
  assert lanes.container_sizes(0, lanes.relaxed_layout) == []


def test_byte_list_marks_nibble_only_bytes():
  assert lanes.bytes_from_widths((12,)) == [('H', 0, False), ('H', 1, True)]


# ---------------------------------------------------------------- None, not 99

def test_unpriceable_key_is_none():
  assert lanes.standalone(lanes.key_bytes((600,))) is None


def test_no_price_propagates_to_every_later_key():
  huge, small = lanes.key_bytes((600,)), lanes.key_bytes((8,))
  assert lanes.stage_prices([huge, small]) == {0: None, 1: None}
  full = lanes.key_bytes((477,))
  prices = lanes.stage_prices([full, lanes.key_bytes((200,)), small])
  assert prices[0] == codeword_to_blocks((477,))
  assert prices[1] is None and prices[2] is None


def test_price_with_supply_on_an_exhausted_crossbar_is_none():
  assert lanes.price_with_supply(lanes.key_bytes((8,)), {}, set()) is None


# ---------------------------------------------------------------- simulation

def test_first_key_price_is_its_standalone_price():
  for widths in REAL_DESIGN_KEYS[:10]:
    kb = lanes.key_bytes(widths)
    assert lanes.stage_prices([kb, lanes.key_bytes((8,))])[0] == lanes.standalone(kb)


def test_order_decides_who_pays():
  spacer, key = lanes.key_bytes((328,)), lanes.key_bytes((84, 84))
  prices = lanes.stage_prices([spacer, key], (1, 0))
  assert prices[1] == 5          # (84, 84) first: its standalone price


def test_first_key_occupancy_uses_fullest_lane_midbytes():
  """(84, 84) at 5 blocks takes (5 - 1) // 2 = 2 whole midbytes."""
  key = lanes.key_bytes((84, 84))
  free, free_mids = lanes.first_key_occupancy(key, 5)
  assert len(free_mids) == TERNARY_CROSSBAR_BYTE_GROUPS_PER_STAGE - 2
  assert all(free[g] == set(range(5)) for g in range(5, 12))


def test_no_module_level_mode_flags_remain():
  for name in ('FILL', 'FIRST_MID'):
    assert not hasattr(lanes, name)


def test_later_key_occupancy_consumes_whole_groups_and_low_midbytes_only():
  # stage_prices' later-key bookkeeping, split out for packing's simulation:
  # a key priced at 5 consumes the first 5 groups with a free slot, wholly,
  # and the lowest (5 - 1) // 2 = 2 free midbytes; its inputs are untouched.
  free = {g: set(range(5)) for g in range(12)}
  free[0] = set()
  mids = {1, 2, 4, 5}
  new_free, new_mids = lanes.later_key_occupancy(free, mids, 5)
  assert [g for g in range(12) if new_free[g]] == [6, 7, 8, 9, 10, 11]
  assert new_mids == {4, 5}
  assert free[1] == set(range(5)) and mids == {1, 2, 4, 5}
