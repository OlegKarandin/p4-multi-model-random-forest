"""PHV containers the generated program takes, counted the same way in training and
in the replay of a program text, and refused when over the ingress budget
(design review decision 8.3, inventory V5; decision 4.2 for the packed layout).

ONE counting function, `count`, fed by two thin adapters: `from_pool` (the pool
`evaluation._pool_inputs` builds, key "phv_fields") and `from_program` (a parsed
`p4_replay.Program`). `containers_used` picks the adapter. What a program takes:

  * the template fields (target.PHV_TEMPLATE_FIELDS);
  * one 16-bit container per raw `<feature>_val` field;
  * one 16-bit container per IAT scratch field in use (`<chain>_current_iat`,
    build_p4_script.iat_field_name) -- this program's own fields, each pinned to
    one 16-bit container by the generator since gate G3 (2026-11-15:
    @pa_container_size(<chain>_current_iat, 16), which program_checks requires);
  * one 16-bit container per call-site constant (`n_constants`: C1f's Min read
    `<f>_val = <f>_action.execute(...) + 1`, build_p4_script.call_site_constants):
    p4c keeps the constant 1 in a PHV container that a hidden stage-0 `$precompute`
    table sets (Task 33, prog.bfa `add H3, <min salu>, H4`; +1 p4c container per
    Min feature, 3 Min features +3);
  * two 32-bit containers when the program computes the coarse clock of the
    Task 38 durations (COARSE_CLOCK_CONTAINERS: meta.now_coarse and its source
    ig_intr_md.ingress_mac_tstamp[47:16]);
  * one 8-bit container per `class_tree_<task>_<i>` field (a task with one tree
    has none since review A2: class_fields);
  * each code field's pinned containers: lanes.container_sizes for joint /
    disjoint / per-task fields (`@pa_container_size` of the generator), the
    packed layout's containers for a `code_pk_<f>` field.

The budgets are the ingress counts (target.PHV_INGRESS_CONTAINERS); the template
counts were measured for decision 6c in Task 31 on the 160 Task 17b base designs
compiled with the Task 31 generator (target.py has the evidence). Nothing is fitted
here. NOT counted yet: the 32-bit clock of V1o (Task 34; p4c splits now_pseudo_us
over B + H + H, +1..+3 containers on Task 29's 3 designs, not one fixed shape), left
to the template re-measure (target.PHV_TEMPLATE_FIELDS), which also has to take out
the fine clock's share of the template for a program whose durations left no
last-arrival register (Task 38: calc_timestamp then writes now_coarse only)."""
from p4model import lanes
from p4model.errors import PhvContainersExhausted
from p4model.target import PHV_INGRESS_CONTAINERS, PHV_TEMPLATE_FIELDS

SIZES = (8, 16, 32)
_CODE = 'code_'
# Task 38 (P3h): the coarse clock of the duration registers. p4c keeps the 32-bit
# meta.now_coarse (pinned @pa_solitary) in a W container of its own and the slice it
# copies, ig_intr_md.ingress_mac_tstamp[47:16], in another W (its low 16 bits go to a
# tagalong container, which the ingress count does not include): +2 W on all 3 Task 37
# compiles that have the field (phv_allocation_summary: W0 / W1, W8 / W9, and W10 /
# W11 for the one-clock global_tstamp form), whether or not the fine clock exists.
COARSE_CLOCK_FIELD = 'now_coarse'
COARSE_CLOCK_CONTAINERS = (32, 32)


def count(n_vals, n_scratch, n_trees, code_containers, n_constants=0, coarse_clock=False):
  """{8: n, 16: n, 32: n}: containers the program takes. n_vals value fields,
  n_scratch IAT scratch fields and n_constants call-site constants (16 bits each),
  n_trees class fields (8 bits each), the coarse clock's COARSE_CLOCK_CONTAINERS when
  `coarse_clock`, and `code_containers`, one tuple of container sizes per code
  field."""
  used = {size: len(PHV_TEMPLATE_FIELDS[size]) for size in SIZES}
  used[16] += n_vals + n_scratch + n_constants
  used[8] += n_trees
  for size in (COARSE_CLOCK_CONTAINERS if coarse_clock else ()):
    used[size] += 1
  for containers in code_containers:
    for size in containers:
      used[size] += 1
  return used


def class_fields(tree_counts):
  """The class_tree_<task>_<i> fields a program declares, from its trees per task:
  one per tree of a task with more than one tree, none for a one-tree task (its
  tree action writes the export field, review A2)."""
  return sum(n for n in tree_counts if n > 1)


def budgets(n_vals, n_scratch, n_trees, n_constants=0, coarse_clock=False):
  """Containers left for code fields, per size, once the program's own fields
  (everything but the code fields) are counted."""
  own = count(n_vals, n_scratch, n_trees, (), n_constants, coarse_clock)
  return {size: PHV_INGRESS_CONTAINERS[size] - own[size] for size in SIZES}


def from_pool(pool):
  """`count` over the pool's "phv_fields" (evaluation._pool_inputs); a pool built
  before C1f has no "constants" (0), one built before Task 38 no "coarse_clock"."""
  fields = pool['phv_fields']
  return count(fields['vals'], fields['scratch'], fields['trees'], fields['code_containers'],
               fields.get('constants', 0), fields.get('coarse_clock', False))


def from_program(program):
  """`count` over a parsed generated program (p4_replay.Program): its value fields
  and class fields by name in the declarations, its IAT scratch fields, its
  call-site constants (Program.call_constants), the coarse clock when the metadata
  declares now_coarse, and each code field's pin
  (Program.containers for a packed field, Program.code_containers for any other); a
  code field with no pin -- one no tree keys -- takes the containers the width rule
  would pin it to."""
  code = []
  for name, bits in program.bits.items():
    if not name.startswith(_CODE):
      continue
    pin = program.containers.get(name) or program.code_containers.get(name)
    code.append(tuple(pin) if pin else tuple(lanes.container_sizes(int(bits), lanes.layout)))
  vals = sum(1 for name in program.bits if name.endswith('_val') and not name.startswith(_CODE))
  return count(vals, len(program.scratch_fields), len(program.class_fields), code,
               len(getattr(program, 'call_constants', ())),
               COARSE_CLOCK_FIELD in getattr(program, 'declared', {}))


def containers_used(source):
  """Containers a design takes, {8: n, 16: n, 32: n}: `source` is the training pool
  (a dict) or a parsed program, and the two agree on the same design."""
  return from_pool(source) if isinstance(source, dict) else from_program(source)


def check(used):
  """Raises PhvContainersExhausted(message, containers short) when `used` exceeds
  the ingress budget of some size; the message names every size and its shortfall."""
  short = {size: used[size] - PHV_INGRESS_CONTAINERS[size] for size in SIZES
           if used[size] > PHV_INGRESS_CONTAINERS[size]}
  if short:
    raise PhvContainersExhausted(
        'PHV containers over the ingress budget: %s' % ', '.join(
            '%d-bit needs %d of %d (short by %d)'
            % (size, used[size], PHV_INGRESS_CONTAINERS[size], by)
            for size, by in short.items()),
        sum(short.values()))
