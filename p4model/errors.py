"""Exceptions raised when a design does not fit within Tofino 1's TCAM limits."""


class CodewordTooLong(RuntimeError):
  """Codeword exceeds MAX_CODEWORD_LENGTH. args = (message, codeword_length)."""


class CrossbarKeyTooWide(RuntimeError):
  """One table's match key exceeds the per-stage ternary crossbar byte budget;
  the compiler rejects such a table outright rather than splitting it.
  args = (message, byte_width)."""


class ActionDataTooWide(RuntimeError):
  """A packed code field needs more bus bytes than the action data bus carries
  (target.ACTION_DATA_BUS_BYTES = 128): the range table that writes it would put
  more than 8 action-RAM words on the bus (packed.layout_for; spec 2026-11-11
  Sec 3.2/6). args = (message, field bytes)."""


class PhvContainersExhausted(RuntimeError):
  """The design needs more PHV containers than the ingress has: either a packed code
  field finds no container of its canonical size left (packed.layout_for, decision
  4.2: never another size), or the program's total over some size exceeds
  target.PHV_INGRESS_CONTAINERS (phv.check, decision 8.3). args = (message,
  containers short); the message names the size(s)."""


class NoSplitTree(RuntimeError):
  """A tree with no threshold has no key field, so its table would have no key
  (spec 9.4). args = (message, tree label)."""
