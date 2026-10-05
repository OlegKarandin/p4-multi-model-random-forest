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
    the ladder prices it at 5 alone (the lane model: 4 under the pinned fill-low layout; the dsp probes
    were compiled WITHOUT pins, where p4c splits each 84-bit field W24-low and charges 5).
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

# Real design keys where the lane price is BELOW what p4c charged -- the
# accepted "greedy crossbar miss" class (reviews/campaign_2026_10_lane_findings.md
# Sec 5): p4c parks a whole byte on the midbyte and leaves no nibble for the
# version bits. (key widths) -> (lane price, p4c observed). The ladder priced
# (27, 52) right by luck.
LANE_BELOW_P4C_REAL_KEYS = {
    (27, 52): (2, 3),   # independent_low_sd5's ddos trees
}


def _price(widths):
  return lanes.standalone(lanes.key_bytes(widths))


# ---------------------------------------------------------------- pinned probes

def test_w019_costs_two_blocks():
  assert _price((19, 64)) == 2


def test_w027_costs_three_blocks():
  assert _price((27, 56)) == 3


def test_dsp41_second_key_pays_seven_beside_a_41_byte_spacer():
  """The (84, 84) key costs 4 alone in the lane model under the fill-low
  layout (ladder: 5). The only compiles of it, dsp01-dsp10 in
  results/tcam_discount_scan.csv, are PRE-PIN and show 5, so the standalone
  value is unverified on a pinned program (spec 2026-10-04 decision 6). The
  crowded-stage price p4c charged, 7, is unchanged."""
  spacer, key = lanes.key_bytes((328,)), lanes.key_bytes((84, 84))
  assert lanes.standalone(key) == 4
  assert codeword_to_blocks((84, 84)) == 5
  assert lanes.stage_prices([spacer, key], (0, 1)) == {0: 8, 1: 7}


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


@pytest.mark.parametrize('widths', REAL_DESIGN_KEYS)
def test_lane_standalone_equals_production_on_real_keys(widths):
  """On every real design key but the listed greedy misses the lane price and
  the ladder agree."""
  if widths in LANE_BELOW_P4C_REAL_KEYS:
    pytest.skip('listed greedy miss, asserted above')
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
  assert lanes.relaxed_layout(w) == expected


@pytest.mark.parametrize('w', range(0, 330))
def test_fill_low_fix_leaves_every_pin_unchanged(w):
  """Per FIELD, the pins (container sizes, low slice first) do not move. This
  does NOT make the pins of a whole KEY unchanged: see
  test_fill_low_fix_can_flip_a_keys_plain_vs_relaxed_pin."""
  assert lanes.container_sizes(w) == [
      lanes.CONTAINER_BITS[k] for k, _ in _pre_fill_low_layout(w)]
  assert sorted(lanes.container_sizes(w)) == sorted(
      lanes.CONTAINER_BITS[k] for k, _ in lanes.layout(w))
  assert sum(bits for _, bits in lanes.layout(w)) == max(w, 0)


def _key_choice(widths, monkeypatch, old):
  """('relaxed'|'plain', container_sizes of each field) of key_layout(widths),
  under the pre-fill-low layout when `old`, else the current one."""
  with monkeypatch.context() as m:
    if old:
      m.setattr(lanes, 'layout', _pre_fill_low_layout)
    lanes._table_blocks.cache_clear()
    lay = lanes.key_layout(widths)
    name = 'relaxed' if lay is lanes.relaxed_layout else 'plain'
    sizes = [lanes.container_sizes(w, lay) for w in widths]
  lanes._table_blocks.cache_clear()
  return name, sizes


def test_fill_low_fix_can_flip_a_keys_plain_vs_relaxed_pin(monkeypatch):
  """Per-key pin identity is NOT structural. container_sizes(w) is identical
  per field (test above), but key_layout compares the plain lane price with
  the ladder, and the fill-low fix lowers the plain price of keys holding a
  24-remainder field, so a key mixing one with 17-32-bit fields can now pin
  plain W32 where the old code pinned relaxed H16 + B8. It held on all 7,681
  campaign_2026_10 keys (tests/test_campaign_pins.py); a new campaign can pin
  plain W32 where the old code pinned relaxed H16 on such keys, which no
  compile has tested."""
  key = (3, 21, 49)
  old, old_sizes = _key_choice(key, monkeypatch, old=True)
  new, new_sizes = _key_choice(key, monkeypatch, old=False)
  assert (old, new) == ('relaxed', 'plain')
  assert old_sizes[1] == [16, 8]
  assert new_sizes[1] == [32]
  assert old_sizes[0] == new_sizes[0] and old_sizes[2] == new_sizes[2]
  key2 = (22, 30, 50, 54)
  assert _key_choice(key2, monkeypatch, old=True)[0] == 'relaxed'
  assert _key_choice(key2, monkeypatch, old=False)[0] == 'plain'
  # A campaign-style key without a flip is unchanged.
  stable = (19, 64)
  assert (_key_choice(stable, monkeypatch, old=True)
          == _key_choice(stable, monkeypatch, old=False))


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
  assert prices[1] == 4          # (84, 84) first: its standalone price (fill-low layout)
  assert prices[0] == 8          # the spacer behind it still pays 8


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


# ------------------------------------------------ table_blocks (spec 2026-10-04)

from src.p4model.errors import CrossbarKeyTooWide
from src.p4model.tables import codeword_bits_to_blocks


@pytest.mark.parametrize('widths', REAL_DESIGN_KEYS)
def test_table_blocks_is_the_lane_price_of_the_pinned_layout(widths):
  assert lanes.table_blocks(widths) == _price(widths)


@pytest.mark.parametrize('widths', REAL_DESIGN_KEYS)
def test_table_blocks_equals_the_ladder_on_real_keys_except_greedy_misses(widths):
  if widths in LANE_BELOW_P4C_REAL_KEYS:
    assert lanes.table_blocks(widths) == LANE_BELOW_P4C_REAL_KEYS[widths][0]
  else:
    assert lanes.table_blocks(widths) == codeword_to_blocks(widths)


def test_table_blocks_pins_the_two_known_ladder_disagreements_off_the_compiled_set():
  # Lane-higher: three 49-56-bit fields (spec Sec I.2, review decision 6).
  assert codeword_to_blocks((54, 55, 56)) == 4
  assert lanes.table_blocks((54, 55, 56)) == 5
  # Lane-lower: the dsp41 probe key.
  assert lanes.table_blocks((84, 84)) == 4


def test_table_blocks_ignores_field_order():
  assert lanes.table_blocks((52, 27)) == lanes.table_blocks((27, 52)) == 2


def test_table_blocks_of_the_empty_key_is_the_ladders_floor():
  assert lanes.table_blocks(()) == codeword_bits_to_blocks(0)
  assert lanes.table_blocks((0, 0)) == codeword_bits_to_blocks(0)


def test_table_blocks_rejects_a_key_over_the_crossbar_before_enumerating(monkeypatch):
  def boom(*_):
    raise AssertionError('standalone must not run for a 65-byte key')
  monkeypatch.setattr(lanes, 'standalone', boom)
  lanes._table_blocks.cache_clear()
  with pytest.raises(CrossbarKeyTooWide) as info:
    lanes.table_blocks((8,) * 65)
  assert info.value.args[1] == 65


def test_table_blocks_raises_when_no_lane_layout_fits(monkeypatch):
  monkeypatch.setattr(lanes, 'standalone', lambda _bytes: None)
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


def test_table_blocks_never_rises_when_one_field_loses_a_bit():
  """Monotonicity guard (the B3 measurement: 0 increases in ~503k steps)."""
  import random
  rng = random.Random('table-blocks-monotone')
  pool = [w for key in REAL_DESIGN_KEYS for w in key] + list(range(49, 57)) * 4
  checked = 0
  while checked < 3000:
    key = tuple(sorted(rng.choice(pool) for _ in range(rng.randint(1, 8))))
    if sum(-(-w // 8) for w in key) > 64:
      continue
    try:
      base = lanes.table_blocks(key)
    except CrossbarKeyTooWide:
      continue
    for i, w in enumerate(key):
      if w <= 1:
        continue
      smaller = key[:i] + (w - 1,) + key[i + 1:]
      assert lanes.table_blocks(smaller) <= base, (key, smaller)
      checked += 1
