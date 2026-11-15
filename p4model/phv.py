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
  * one 8-bit container per `class_tree_<task>_<i>` field (a task with one tree
    has none since review A2: class_fields);
  * each code field's pinned containers: lanes.container_sizes for joint /
    disjoint / per-task fields (`@pa_container_size` of the generator), the
    packed layout's containers for a `code_pk_<f>` field.

The budgets are the ingress counts (target.PHV_INGRESS_CONTAINERS); the template
counts were measured for decision 6c in Task 23 on the 160 Task 17b base designs
compiled with the final G3 pins (target.py has the evidence). Nothing is fitted
here."""
from p4model import lanes
from p4model.errors import PhvContainersExhausted
from p4model.target import PHV_INGRESS_CONTAINERS, PHV_TEMPLATE_FIELDS

SIZES = (8, 16, 32)
_CODE = 'code_'


def count(n_vals, n_scratch, n_trees, code_containers):
  """{8: n, 16: n, 32: n}: containers the program takes. n_vals value fields and
  n_scratch IAT scratch fields (16 bits each), n_trees class fields (8 bits each),
  and `code_containers`, one tuple of container sizes per code field."""
  used = {size: len(PHV_TEMPLATE_FIELDS[size]) for size in SIZES}
  used[16] += n_vals + n_scratch
  used[8] += n_trees
  for containers in code_containers:
    for size in containers:
      used[size] += 1
  return used


def class_fields(tree_counts):
  """The class_tree_<task>_<i> fields a program declares, from its trees per task:
  one per tree of a task with more than one tree, none for a one-tree task (its
  tree action writes the export field, review A2)."""
  return sum(n for n in tree_counts if n > 1)


def budgets(n_vals, n_scratch, n_trees):
  """Containers left for code fields, per size, once the program's own fields
  (everything but the code fields) are counted."""
  own = count(n_vals, n_scratch, n_trees, ())
  return {size: PHV_INGRESS_CONTAINERS[size] - own[size] for size in SIZES}


def from_pool(pool):
  """`count` over the pool's "phv_fields" (evaluation._pool_inputs)."""
  fields = pool['phv_fields']
  return count(fields['vals'], fields['scratch'], fields['trees'], fields['code_containers'])


def from_program(program):
  """`count` over a parsed generated program (p4_replay.Program): its value fields
  and class fields by name in the declarations, its IAT scratch fields, and each
  code field's pin (Program.containers for a packed field, Program.code_containers
  for any other); a code field with no pin -- one no tree keys -- takes the
  containers the width rule would pin it to."""
  code = []
  for name, bits in program.bits.items():
    if not name.startswith(_CODE):
      continue
    pin = program.containers.get(name) or program.code_containers.get(name)
    code.append(tuple(pin) if pin else tuple(lanes.container_sizes(int(bits), lanes.layout)))
  vals = sum(1 for name in program.bits if name.endswith('_val') and not name.startswith(_CODE))
  return count(vals, len(program.scratch_fields), len(program.class_fields), code)


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
