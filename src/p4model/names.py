"""Canonical spelling for feature names, shared by the register catalog, the P4
identifier emitter, and every crossbar field id.

Deliberately import-free apart from `re`: it sits at the bottom of p4model's
dependency graph, and feature_registers.py used to need a function-local import
to reach it without a cycle."""
import re


_IDENT_RE = re.compile(r'[^0-9a-z]+')


def normalise_feature_name(name):
  """Canonical form for both FEATURE_REGISTER_CATALOG keys and P4 identifiers.

  Dataset columns arrive dot-separated ('Flow.IAT.Max' -- dataset.py renames
  every column with .replace(' ', '.')), older fixtures arrive space- or
  underscore-separated. Every run of non-alphanumeric characters collapses to a
  single '_' so all three spellings land on one key, and that key is a legal P4
  identifier. Leading/trailing separators are stripped so 'Flow.IAT.Max.' cannot
  become a distinct key."""
  return _IDENT_RE.sub('_', name.lower()).strip('_')
