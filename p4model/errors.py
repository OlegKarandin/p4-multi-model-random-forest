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
  """No PHV container of any size is left for a packed code field: its budgets
  (packed.budgets, from target.PHV_INGRESS_CONTAINERS) are used up in 32, 16 and 8
  bits alike. args = (message, containers short)."""


class NoSplitTree(RuntimeError):
  """A tree with no threshold has no key field, so its table would have no key
  (spec 9.4). args = (message, tree label)."""
