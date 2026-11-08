"""Exceptions raised when a design does not fit within Tofino 1's TCAM limits."""


class CodewordTooLong(RuntimeError):
  """Codeword exceeds MAX_CODEWORD_LENGTH. args = (message, codeword_length)."""


class CrossbarKeyTooWide(RuntimeError):
  """One table's match key exceeds the per-stage ternary crossbar byte budget;
  the compiler rejects such a table outright rather than splitting it.
  args = (message, byte_width)."""


class ActionDataTooWide(RuntimeError):
  """One range table's action writes more container bytes than the action data
  bus carries (target.PER_TREE_MAX_ACTION_BYTES; spec 9.3). args = (message,
  action_bytes)."""


class NoSplitTree(RuntimeError):
  """A tree with no threshold has no key field, so its table would have no key
  (spec 9.4). args = (message, tree label)."""
