"""Exceptions raised when a design does not fit within Tofino 1's TCAM limits.

Every one is a REFUSAL (p4model.refusals): it names the constraint it violates
(`constraint`, one of refusals.CONSTRAINTS) and how far over it the design is
(`magnitude`, in that constraint's natural unit, at least 1), so training's
objective, the replay and the verifier report one name per refusal source
(decisions 8.2, 9.2)."""


class ModelRefusal(RuntimeError):
  """A design the model refuses to price. Subclasses set `constraint`."""
  constraint = None

  @property
  def magnitude(self):
    return 1


def _over(args, limit=0):
  """args[1] - limit floored at 1, or 1 when the exception carries no args[1]."""
  return max(1, int(args[1]) - limit) if len(args) > 1 else 1


class CrossbarKeyTooWide(ModelRefusal):
  """One table's match key exceeds target.MAX_KEY_BYTES crossbar bytes;
  the compiler rejects such a table outright rather than splitting it.
  args = (message, byte_width). Also raised for a key of <= MAX_KEY_BYTES bytes
  with no crossbar allocation (xbar.alone_blocks), whose overflow would read <= 0,
  hence the floor of 1. magnitude: bytes over target.MAX_KEY_BYTES (read through
  the module so a test can move it)."""
  constraint = 'key_bytes_over'

  @property
  def magnitude(self):
    from p4model import target
    return _over(self.args, target.MAX_KEY_BYTES)


class ActionDataTooWide(ModelRefusal):
  """A packed code field needs more bus bytes than the action data bus carries
  (target.ACTION_DATA_BUS_BYTES = 128): the range table that writes it would put
  more than 8 action-RAM words on the bus (packed.check_bus_bytes; spec
  2026-11-11 Sec 3.2/6). args = (message, field bytes). magnitude: bytes over."""
  constraint = 'bus_bytes_over'

  @property
  def magnitude(self):
    from p4model import target
    return _over(self.args, target.ACTION_DATA_BUS_BYTES)


class PhvContainersExhausted(ModelRefusal):
  """The design needs more PHV containers than the ingress has: either a packed code
  field finds no container of its canonical size left (packed.layout_for, decision
  4.2: never another size), or the program's total over some size exceeds
  target.PHV_INGRESS_CONTAINERS (phv.check, decision 8.3). args = (message,
  containers short); the message names the size(s). magnitude: containers short."""
  constraint = 'phv_containers_short'

  @property
  def magnitude(self):
    return _over(self.args)


class RangeEntriesDoNotFit(ModelRefusal):
  """The real range entries of some range table need more TCAM rows than the blocks
  p4c commits for its declared size can hold (target.TCAM_USABLE_ROWS_PER_BLOCK rows
  each): the control plane could not install them (decision 6d,
  tables.range_deployment_overflow). args = (message, rows_over), rows_over the rows
  short summed over the tables. Training / generator only: a program text has no
  entries. magnitude: rows over."""
  constraint = 'range_rows_over'

  @property
  def magnitude(self):
    return _over(self.args)


class PlacementFailed(ModelRefusal):
  """The placement simulation reached a state p4c's placer cannot reach, so the design
  has no placement the model can price (decision 7.4, S29/S30): the classification pool
  never placed a tree (packing._place_in_order's horizon), or a stage the range and
  classification pools share broke its table cap or column packing
  (usage.check_shared_stages). A refusal, not a crash: training marks the trial
  infeasible and the replay reports it. args = (message,). magnitude 1."""
  constraint = 'placement_failed'


class NoSplitTree(ModelRefusal):
  """A tree with no threshold has no key field, so its table would have no key
  (spec 9.4). args = (message, tree label). magnitude 1."""
  constraint = 'no_split_tree'


class ProgramShapeRefused(ModelRefusal):
  """A program text breaks an assumption the replay prices by (decision 8.2,
  p4gen.program_checks.check_program): a pin the generator emits is missing, the
  registers run in gated blocks, a table carries an @stage pin, or the apply order /
  prologue / vote tables are not the generator's. `reason` names it:
  missing_pin:<pin>:<field>, gated_register_block, stage_pin or
  unexpected_shape:<what>. args = (reason,). magnitude 1."""
  constraint = 'program_shape'

  def __init__(self, reason):
    super().__init__(reason)
    self.reason = reason
