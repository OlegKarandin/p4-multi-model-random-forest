"""The switch's unit for a feature value: the IAT features in 1.024 us ticks (Task 34).

One statement of the conversion, shared by the control plane's range-entry writer
(p4gen.build_p4_script._range_table_entries, which re-exports these names) and the
model's range-row guard (p4model.tables: range_entries / range_rows_over count the
rows of the entries the control plane really installs)."""
from p4model.catalog import FEATURE_REGISTER_CATALOG
from p4model.names import normalise_feature_name

# The top of a 16-bit range key (= build_p4_script.INFINITE).
KEY_TOP = (1 << 16) - 1


def iat_ticks(threshold_us):
  '''An IAT threshold of the trees (integer microseconds, the CSVs' unit) in the
  switch's IAT unit (Task 34; user decision 2026-11-15):
  T = floor((1000 t + 500) / 1024) + 1, capped at KEY_TOP.

  The switch's clock ticks every 1.024 us (calc_timestamp: global_tstamp[41:10]),
  and an IAT is the difference of two floored clock readings: a gap of g ns reads
  floor(g / 1024) or one more, with mean g / 1024, so the tree's `x_us <= t` is
  matched best by centring on t + 0.5 us -- the ROUND rule floor((1000 t + 500) /
  1024) (Task 29, impl/probe_timestamp.md: agreement with software 99.98 / 99.97%
  app, 100.00 / 99.99% ddos on 200 DEV designs, 1 design with two thresholds on one
  tick; the floor rule collapsed 60). The + 1 is C1f's: the switch holds a real gap
  of g ticks as g + 1, 0 meaning "no IAT" (build_p4_script._REGISTER_ACTION_BODIES
  ["iat_delta"]). Accuracy (p4gen.switch_semantics) is computed in the software unit.'''
  return min((1000 * int(threshold_us) + 500) // 1024 + 1, KEY_TOP)


def is_iat_feature(feature, catalog=None):
  '''Whether a feature's value is an inter-arrival time: its catalog entry has a
  last-arrival ("iat_delta") register -- the IAT Max / Min / Mean features.'''
  if catalog is None:
    catalog = FEATURE_REGISTER_CATALOG
  entry = catalog.get(normalise_feature_name(feature), {})
  return any(reg["body"] == "iat_delta" for reg in entry.get("registers", ()))


def switch_range(feature, interval):
  '''A range table interval [lo, hi] of the trees' units in the switch's: the
  identity for every feature but an IAT one, whose bounds are thresholds in
  microseconds (hi = t, lo = previous t + 1) and go through iat_ticks; 0 and
  KEY_TOP stay. Two thresholds on one tick leave an empty interval (lo > hi),
  which the control plane does not install.'''
  lo, hi = interval
  if not is_iat_feature(feature):
    return lo, hi
  return (0 if lo == 0 else iat_ticks(lo - 1) + 1,
          KEY_TOP if hi >= KEY_TOP else iat_ticks(hi))


def installed_intervals(feature, intervals):
  '''The [lo, hi] bounds the control plane installs for a feature's intervals, in
  order: switch_range of each, empty ones (collapsed onto one tick) left out.'''
  return [(lo, hi) for lo, hi in (switch_range(feature, interval) for interval in intervals)
          if lo <= hi]
