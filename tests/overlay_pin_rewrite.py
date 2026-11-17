"""The 2026-11-17 overlay-pin change as a text rewrite (Handoff 6, fairness steps 5-6:
results/encoding_evidence_2026_11/model8/fairness/step6_avoid_extra_stage.md).

The generator stopped writing @pa_no_overlay on every code_* field and started writing it on
every class_tree_* field, on the line after that field's @pa_solitary. Nothing else in a
program changed. `current_pins(text)` applies exactly that edit to a text of the generator
before it, so a committed fixture text (an older generator's program, which check_program now
refuses for its pins) can still reach the check a test is about. Checked against the
generator itself: for every frozen-hash case of the test suite, current_pins(text at
10788cb) == the current generator's text.
"""
import re

_CODE_NO_OVERLAY = re.compile(r'^@pa_no_overlay\("ingress", "ig_md\.code_\w+"\)\r?\n', re.M)
_CLASS_SOLITARY = re.compile(r'^@pa_solitary\("ingress", "ig_md\.(class_tree_\w+)"\)(\r?\n)', re.M)


def current_pins(text):
  text = _CODE_NO_OVERLAY.sub("", text)
  return _CLASS_SOLITARY.sub(
      lambda m: m.group(0) + '@pa_no_overlay("ingress", "ig_md.%s")%s' % (m.group(1), m.group(2)),
      text)
