"""One NAMED constraint per refusal source, in natural units (decision 9.2).

Every refusal the model knows is one of CONSTRAINTS. Seven are exceptions
(p4model.errors, each exposing `constraint` and `magnitude`); two are numbers the
model computes and compares with a limit (stages_over, blocks_over). Training's
objective records each trial's violation under these names
(src.training.early_stopping.constraint_values returns one value per name, in this
order), and the replay (p4gen.p4_replay.model_breakdown) reports the same name for
the same design, so training, --rescore, --model-archive and outside users share
one implementation (decision 8.2).

  key_bytes_over        crossbar bytes over target.MAX_KEY_BYTES (CrossbarKeyTooWide)
  phv_containers_short  PHV containers short (PhvContainersExhausted)
  bus_bytes_over        action-data bus bytes over (ActionDataTooWide)
  range_rows_over       TCAM rows the real range entries miss (RangeEntriesDoNotFit;
                        training / generator only: a program text has no entries)
  stages_over           pipeline stages over target.TOFINO_PIPELINE_STAGES
  blocks_over           TCAM blocks over the experiment's budget M
  no_split_tree         a tree with no split (NoSplitTree), 1
  placement_failed      no placement the model can price (PlacementFailed), 1
  program_shape         the program text breaks a replay assumption
                        (ProgramShapeRefused, program_checks.check_program), 1

Magnitudes are at least 1 for a violated constraint and 0 for a satisfied one."""
from p4model import target
from p4model.errors import (ActionDataTooWide, CrossbarKeyTooWide, NoSplitTree,
                            PhvContainersExhausted, PlacementFailed, ProgramShapeRefused,
                            RangeEntriesDoNotFit)

CONSTRAINTS = ('key_bytes_over', 'phv_containers_short', 'bus_bytes_over', 'range_rows_over',
               'stages_over', 'blocks_over', 'no_split_tree', 'placement_failed',
               'program_shape')

# The refusals raised as exceptions; `except REFUSAL_ERRORS` catches every one.
REFUSAL_ERRORS = (CrossbarKeyTooWide, PhvContainersExhausted, ActionDataTooWide,
                  RangeEntriesDoNotFit, NoSplitTree, PlacementFailed, ProgramShapeRefused)

# The two refusals that are a model number over a limit, not an exception: the
# replay still reports the numbers for these.
NUMERIC = ('stages_over', 'blocks_over')

assert sorted(CONSTRAINTS) == sorted({e.constraint for e in REFUSAL_ERRORS} | set(NUMERIC))


def of(exc):
  """(constraint, magnitude) of a refusal exception."""
  return exc.constraint, exc.magnitude


def reason_of(exc):
  """The refusal's detailed name: ProgramShapeRefused's reason, else the constraint."""
  return getattr(exc, 'reason', None) or exc.constraint


def stages_over(stage_depth):
  """Stages over target.TOFINO_PIPELINE_STAGES (0 when the design fits)."""
  return max(0, int(stage_depth) - target.TOFINO_PIPELINE_STAGES)


def blocks_over(blocks, max_blocks):
  """Blocks over the budget max_blocks (0 when it fits or there is no budget:
  max_blocks None or inf)."""
  if max_blocks is None or max_blocks == float('inf'):
    return 0
  return max(0, blocks - max_blocks)


def violations(constraint=None, magnitude=0):
  """{name: magnitude} for every constraint: 0 for each, `magnitude` for
  `constraint` (None: every constraint satisfied)."""
  if constraint is not None and constraint not in CONSTRAINTS:
    raise ValueError('unknown constraint %r' % (constraint,))
  return {name: (magnitude if name == constraint else 0.0) for name in CONSTRAINTS}
