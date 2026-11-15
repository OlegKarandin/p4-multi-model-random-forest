"""Static checks on a generated P4 program's text (decision 3.8).

Without `@pa_auto_init_metadata`, p4c marks every metadata field that MAY be read before
it is written `@pa_no_init`: its value is undefined and the field may be overlaid. So no
field of ours should be read before something in the program writes it.
`metadata_reads_before_writes` walks the ingress control's apply block in program order
and names the `meta.<field>` reads that precede every write of that field.

CAVEAT -- a green result does NOT mean p4c reports no no-init fields. A table's write is
counted here at the table's apply, hit OR miss; p4c does not do that: it marks fields
written only by a table action `@pa_no_init` because of the miss path. The Task 4 compile
of three programs (joint, per_task, per_tree; `-Xp4c=-Tauto_init_metadata:1`) listed
class_tree_* and code_* fields of ours as no_init while this check was clean
(.superpowers/sdd/impl_plan/task-4-report.md).

Plain text analysis: the generator's output is regular enough that regexes do. Path
aware: a field is written after an `if` only if both branches write it (an `if` without
`else` contributes nothing). What counts, in apply order:
  read   a `meta.<f>` on the right-hand side of an assignment, in an `if` condition, in the
         `.execute(...)` index, in the body of a register action or action at the point it
         runs, or in a table's key (a code-field slice reads its whole field);
  write  an assignment target (`meta.<f> = ...`, also `meta.<f>[a:b] = ...`), a register
         `.execute` assigned to a field, and the fields assigned by the actions a table
         lists, counted at that table's apply.
Intrinsic metadata (`ig_intr_md`, `ig_prsr_md`, `ig_tm_md`, `ig_dprsr_md`) and headers are
not `meta.` fields and are exempt.

A form the walk does not understand raises ValueError instead of passing silently: an
`.execute(` on an unknown register action, an `.apply()` on an unknown table or inside a
condition (`if (t.apply().hit)`), a bare call of an unknown action.

check_program (decision 8.2) checks a parsed program's SHAPE -- the pins, priorities,
declared widths and apply order the replay assumes -- and refuses one it does not know
with a named reason (ProgramShapeRefused); p4_replay.model_breakdown calls it first.
"""
import re

from p4model import lanes
from p4model.errors import ProgramShapeRefused

_META = re.compile(r"\bmeta\.(\w+)")
_ASSIGN = re.compile(r"^meta\.(\w+)\s*(?:\[[^\]]*\])?\s*=(?!=)\s*(.*)$", re.S)
_EXECUTE = re.compile(r"\b(\w+)\.execute\s*\(")
_ACTION = re.compile(r"\baction\s+(\w+)\s*\([^)]*\)\s*\{")
_REGISTER_ACTION = re.compile(r"RegisterAction<[^\n]*?>\s*\(\s*\w+\s*\)\s+(\w+)\s*=\s*\{")
_MATH_UNIT = re.compile(r"MathUnit<[^\n]*?>\s*\([^\n]*\)\s+(\w+)\s*;")
_TABLE = re.compile(r"\btable\s+(\w+)\s*\{")
_CONTROL = re.compile(r"\bcontrol\s+\w+\s*\(([^)]*)\)\s*\{")
_APPLY_CALL = re.compile(r"^(\w+)\.apply\s*\(\s*\)$")
_ANY_APPLY = re.compile(r"\.apply\s*\(")
_ACTION_CALL = re.compile(r"^(\w+)\s*\(\s*\)$")
_IF = re.compile(r"if\b\s*\(")
_ELSE = re.compile(r"else\b")


def _matching(text, open_index, opener="{", closer="}"):
  """Index of the `closer` matching the `opener` at open_index."""
  depth = 0
  for index in range(open_index, len(text)):
    if text[index] == opener:
      depth += 1
    elif text[index] == closer:
      depth -= 1
      if depth == 0:
        return index
  raise ValueError("unbalanced %s%s in the program text" % (opener, closer))


def _strip_comments(text):
  text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
  return re.sub(r"//[^\n]*", "", text)


def _bodies(text, pattern):
  """{name: body text} for every `<pattern> {...}` block (pattern's group 1 is the name)."""
  found = {}
  for match in pattern.finditer(text):
    open_index = match.end() - 1
    found[match.group(1)] = text[open_index + 1:_matching(text, open_index)]
  return found


def _parse(text):
  """The block's nodes: ("stmt", text) or ("if", cond, then_nodes, else_nodes)."""
  nodes, pos = [], 0
  while True:
    while pos < len(text) and text[pos] in " \t\r\n;":
      pos += 1
    if pos >= len(text):
      return nodes
    node, pos = _parse_one(text, pos)
    nodes.append(node)


def _parse_one(text, pos):
  head = _IF.match(text, pos)
  if head:
    paren = head.end() - 1
    close = _matching(text, paren, "(", ")")
    cond = text[paren + 1:close]
    then_nodes, pos = _parse_branch(text, close + 1)
    else_nodes = []
    while pos < len(text) and text[pos] in " \t\r\n":
      pos += 1
    after = _ELSE.match(text, pos)
    if after:
      else_nodes, pos = _parse_branch(text, after.end())
    return ("if", cond, then_nodes, else_nodes), pos
  # A statement runs to the next `;` outside any (...) or {...} (a hash call carries braces).
  depth, index = 0, pos
  while index < len(text):
    char = text[index]
    if char in "({":
      depth += 1
    elif char in ")}":
      depth -= 1
    elif char == ";" and depth == 0:
      break
    index += 1
  return ("stmt", text[pos:index].strip()), index + 1


def _parse_branch(text, pos):
  while pos < len(text) and text[pos] in " \t\r\n":
    pos += 1
  if text.startswith("{", pos):
    close = _matching(text, pos)
    return _parse(text[pos + 1:close]), close + 1
  node, pos = _parse_one(text, pos)           # `else if ...` or a braceless branch
  return [node], pos


def _apply_body(register_body):
  """The inside of the register action's `void apply(...) { ... }`."""
  signature = re.search(r"\bapply\s*\(", register_body)
  if signature is None:
    raise ValueError("a register action without an apply body")
  close = _matching(register_body, signature.end() - 1, "(", ")")
  brace = register_body.index("{", close)
  return register_body[brace + 1:_matching(register_body, brace)]


class _Walk:
  def __init__(self, text):
    self.actions = {name: _parse(body) for name, body in _bodies(text, _ACTION).items()}
    self.register_actions = {name: _parse(_apply_body(body))
                             for name, body in _bodies(text, _REGISTER_ACTION).items()}
    self.units = set(_MATH_UNIT.findall(text))
    self.tables = {}
    for match in _TABLE.finditer(text):
      open_index = match.end() - 1
      block = text[open_index + 1:_matching(text, open_index)]
      key = re.search(r"\bkey\s*=\s*\{([^}]*)\}", block)
      listed = re.search(r"\bactions\s*=\s*\{([^}]*)\}", block)
      names = re.findall(r"\w+", listed.group(1)) if listed else []
      names += re.findall(r"\bdefault_action\s*=\s*(\w+)", block)
      self.tables[match.group(1)] = (
          _META.findall(key.group(1)) if key else [], names)
    self.violations = []
    self.executed = set()
    self.applied = set()

  def _read(self, field, written):
    if field not in written and field not in self.violations:
      self.violations.append(field)

  def _reads_of(self, expression, written):
    """Every meta field the expression reads, including those read by the body of each
    register action it executes."""
    if _ANY_APPLY.search(expression):
      raise ValueError("an .apply() inside an expression: %r" % expression.strip())
    for field in _META.findall(expression):
      self._read(field, written)
    for name in _EXECUTE.findall(expression):
      if name in self.register_actions:
        self.executed.add(name)
        self._run(self.register_actions[name], set(written))
      elif name not in self.units:
        raise ValueError("`.execute(` on unknown register action %r" % name)

  def _run(self, nodes, written):
    """Walk the nodes, updating `written` in place."""
    for node in nodes:
      if node[0] == "if":
        _, cond, then_nodes, else_nodes = node
        self._reads_of(cond, written)
        after_then, after_else = set(written), set(written)
        self._run(then_nodes, after_then)
        self._run(else_nodes, after_else)
        written |= after_then & after_else
      else:
        self._statement(node[1], written)

  def _statement(self, statement, written):
    applied = _APPLY_CALL.match(statement)
    if applied:
      name = applied.group(1)
      if name not in self.tables:
        raise ValueError("`.apply()` on unknown table %r" % name)
      keys, actions = self.tables[name]
      self.applied.add(name)
      for field in keys:
        self._read(field, written)
      union = set(written)
      for action in actions:
        if action not in self.actions:
          raise ValueError("table %r lists unknown action %r" % (name, action))
        branch = set(written)
        self._run(self.actions[action], branch)
        union |= branch
      written |= union                    # a table's write counts hit or miss (see docstring)
      return
    called = _ACTION_CALL.match(statement)
    if called:
      if called.group(1) not in self.actions:
        raise ValueError("call of unknown action %r" % called.group(1))
      self._run(self.actions[called.group(1)], written)
      return
    assign = _ASSIGN.match(statement)
    if assign:
      self._reads_of(assign.group(2), written)
      written.add(assign.group(1))
      return
    self._reads_of(statement, written)


def _analyse(text):
  """(violations, stats): stats counts what the walk actually saw, so a test can tell a
  clean program from a walk that matched nothing."""
  text = _strip_comments(text)
  walk = _Walk(text)
  for match in _CONTROL.finditer(text):
    if not re.search(r"\binout\s+\w+\s+meta\b", match.group(1)):
      continue
    open_index = match.end() - 1
    control = text[open_index + 1:_matching(text, open_index)]
    apply = re.search(r"(?<![\w.])apply\s*\{", control)
    if apply is None:
      raise ValueError("the ingress control has no apply block")
    start = apply.end() - 1
    walk._run(_parse(control[start + 1:_matching(control, start)]), set())
    stats = {"registers_executed": len(walk.executed),
             "tables_applied": len(walk.applied),
             "tables_declared": len(walk.tables)}
    return walk.violations, stats
  raise ValueError("no control with an `inout <type> meta` parameter")


def metadata_reads_before_writes(text):
  """Metadata field names (without `meta.`) read in the ingress apply block, in a register
  action body or action it runs, or in a table key, before any statement, action or table
  earlier in the apply order writes them on every path. First-read order, no repeats;
  empty when the program is clean. Raises ValueError on a form it cannot follow."""
  return _analyse(text)[0]


# ------------------------------------------------------------------ program shape (8.2)


def apply_steps(text):
  """The ingress apply block as a flat tuple of (kind, name, depth, under_fwd) steps in
  program order: kind 'apply' (a `<table>.apply()`), 'execute' (a RegisterAction's
  `.execute(...)`, name = the action), 'call' (a bare `<action>()`) or 'other'; depth =
  the number of enclosing `if`s; under_fwd = some enclosing `if` condition reads
  meta.fwd. Empty when the text has no ingress control (a control with an
  `inout <type> meta` parameter) or a form the parser cannot follow -- check_program
  refuses that as a shape it does not know. Read by p4_replay.parse_program_text into
  Program.apply_steps."""
  try:
    text = _strip_comments(text)
    for match in _CONTROL.finditer(text):
      if not re.search(r"\binout\s+\w+\s+meta\b", match.group(1)):
        continue
      open_index = match.end() - 1
      control = text[open_index + 1:_matching(text, open_index)]
      apply = re.search(r"(?<![\w.])apply\s*\{", control)
      if apply is None:
        return ()
      start = apply.end() - 1
      nodes = _parse(control[start + 1:_matching(control, start)])
      steps = []
      _flatten(nodes, 0, False, steps)
      return tuple(steps)
  except ValueError:
    return ()
  return ()


_HASH_DECL = re.compile(r"\bHash<\s*bit<\s*(\d+)\s*>\s*>\s*\([^)]*\)\s*(\w+)\s*;")
_SYMMETRIC = re.compile(r'@symmetric\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)')
_FLOW_HASH_ACTION = re.compile(r"\baction\s+calc_flow_hash\s*\(\s*\)\s*\{")


def flow_hash_shape(text):
  """(output bits, @symmetric pairs, operators) of the program's flow hash: the bit
  width of the Hash instance calc_flow_hash calls (None when there is none), the
  sorted (a, b) pairs of every @symmetric annotation in the text, and the
  operators `&` / `^` found in calc_flow_hash's body (the pre-A1 generator XORed
  the inputs and masked the 32-bit result `& 0xFFF`). Read by
  p4_replay.parse_program_text into Program.flow_hash; check_program judges it."""
  text = _strip_comments(text)
  bodies = [text[m.end():_matching(text, m.end() - 1)]
            for m in _FLOW_HASH_ACTION.finditer(text)]
  body = bodies[0] if bodies else ""
  called = re.search(r"\b(\w+)\.get\s*\(", body)
  bits = None
  for match in _HASH_DECL.finditer(text):
    if called and match.group(2) == called.group(1):
      bits = int(match.group(1))
  pairs = tuple(sorted(tuple(pair) for pair in _SYMMETRIC.findall(text)))
  operators = "".join(sorted(set(re.findall(r"[&^]", body))))
  return bits, pairs, operators


def _flatten(nodes, depth, under_fwd, steps):
  for node in nodes:
    if node[0] == "if":
      _, cond, then_nodes, else_nodes = node
      fwd = under_fwd or re.search(r"\bmeta\.fwd\b", cond) is not None
      _flatten(then_nodes, depth + 1, fwd, steps)
      _flatten(else_nodes, depth + 1, fwd, steps)
      continue
    statement = node[1]
    applied = _APPLY_CALL.match(statement)
    if applied:
      steps.append(("apply", applied.group(1), depth, under_fwd))
      continue
    executed = _EXECUTE.findall(statement)
    if executed:
      steps.extend(("execute", name, depth, under_fwd) for name in executed)
      continue
    called = _ACTION_CALL.match(statement)
    steps.append(("call" if called else "other",
                  called.group(1) if called else statement, depth, under_fwd))


_TREE_TABLE = re.compile(r"^get_classification_tree_(\w+)_(\d+)$")
_RANGE_TABLE = re.compile(r"^table_\d+_\w+$")
_VOTE_TABLE = re.compile(r"^vote_(\w+)$")
_PACKED = "code_pk_"
_ORIENTATION_ACTION = "flow_orientation_action"
_PROLOGUE_CALL = "calc_flow_hash"
# The action writing meta.now_pseudo_us, and that field (its pa_solitary pin, A1b).
_TIMESTAMP_CALL = "calc_timestamp"
_TIMESTAMP_FIELD = "now_pseudo_us"
# The template's two forwarding assignments, the apply block's first steps (Task 17b):
# the vote actions write ig_tm_md too (decision 1f), so p4c orders these AFTER the votes
# when they come last and sometimes spends a stage on them, which the model does not price
# (7 k=1 under-prices in the Task 17 compile experiment).
_EGRESS_ASSIGNMENTS = ("ig_tm_md.ucast_egress_port = ig_intr_md.ingress_port",
                       "ig_tm_md.bypass_egress = 1w1")


def _pins(program):
  """{(pin, field): args} from Program.pragmas."""
  return {(pin, field): args for pin, field, args in program.pragmas}


def _require(pins, pin, field):
  if (pin, field) not in pins:
    raise ProgramShapeRefused("missing_pin:%s:%s" % (pin, field))
  return pins[(pin, field)]


def _check_pins(program, pins):
  """A1, A7-A10 and the code-field pins of decisions 4.1 / 4.3, with the pin set
  of gate G3 (2026-11-15; build_p4_script.generate_p4_program documents each pin
  and its evidence): every pin the generator emits is present, and no other --

    class_tree_*        pa_solitary                    (3.2)
    <f>_val             pa_container_size 16           (3.4)
    <chain>_current_iat pa_container_size 16           (G3: IAT scratch fields)
    code_* / code_pk_*  pa_no_overlay, pa_container_size (3.6 / 3.7; none of
                        the latter for a field with no bits)
    now_pseudo_us       pa_solitary, in a program that computes it
                        (calc_timestamp; review A1b, 2026-11-15)

  Pins dropped at G3 (pa_no_overlay on fwd and class_tree_*, pa_solitary on
  code_*) are refused as unexpected_shape:pin:<pin>:<field>, like any pin this
  generator does not write: such a text is an older generator's program, whose
  PHV p4c allocates under other constraints. Every pinned code field is
  declared at its containers' full width, and (joint / disjoint / per-task, not
  the packed code_pk_ fields) its split is lanes.container_sizes of its live
  width, the layout training prices the key with."""
  expected = set()

  def need(pin, field):
    expected.add((pin, field))
    return _require(pins, pin, field)

  for field in program.class_fields:
    need("pa_solitary", field)
  for field in program.scratch_fields:
    if tuple(need("pa_container_size", field)) != (16,):
      raise ProgramShapeRefused("unexpected_shape:scratch_container:%s" % field)
  if any(kind == "call" and name == _TIMESTAMP_CALL for kind, name, _, _ in program.apply_steps):
    if tuple(need("pa_solitary", _TIMESTAMP_FIELD)) != ():
      raise ProgramShapeRefused("unexpected_shape:pin:pa_solitary:%s" % _TIMESTAMP_FIELD)
  for field in sorted(program.declared):
    bits = program.declared[field]
    if field.endswith("_val") and not field.startswith("code_"):
      if bits != 16:
        raise ProgramShapeRefused("unexpected_shape:val_width:%s" % field)
      if tuple(need("pa_container_size", field)) != (16,):
        raise ProgramShapeRefused("unexpected_shape:val_container:%s" % field)
    elif field.startswith("code_"):
      need("pa_no_overlay", field)
      if bits == 0:
        continue
      sizes = need("pa_container_size", field)
      if sum(sizes) != bits:
        raise ProgramShapeRefused("unexpected_shape:code_width:%s" % field)
      if not field.startswith(_PACKED) and (
          list(sizes) != lanes.container_sizes(program.bits.get(field, bits))):
        raise ProgramShapeRefused("unexpected_shape:code_container:%s" % field)
  for pin, field, _ in program.pragmas:
    if (pin, field) not in expected:
      raise ProgramShapeRefused("unexpected_shape:pin:%s:%s" % (pin, field))


def _check_priorities(program):
  """A4 / A5, the relations the model uses (not the generator's numbers, which
  depend on tasks that may have no trees): every tree carries its task's
  @placement_priority, one per task with trees and distinct across those tasks;
  every range table carries one shared priority above every tree priority
  (program.range_table_priority)."""
  task_priority = {}
  for name in program.tables:
    tree = _TREE_TABLE.match(name)
    if not tree:
      continue
    if name not in program.priorities:
      raise ProgramShapeRefused("missing_pin:placement_priority:%s" % name)
    if task_priority.setdefault(tree.group(1), program.priorities[name]) != (
        program.priorities[name]):
      raise ProgramShapeRefused("unexpected_shape:tree_priority:%s" % name)
  if len(set(task_priority.values())) != len(task_priority):
    raise ProgramShapeRefused("unexpected_shape:tree_priority:%s" % ",".join(
        "%s=%d" % item for item in sorted(task_priority.items())))
  range_priority = None
  for name in program.tables:
    if not _RANGE_TABLE.match(name):
      continue
    if name not in program.priorities:
      raise ProgramShapeRefused("missing_pin:placement_priority:%s" % name)
    if range_priority is None:
      range_priority = program.priorities[name]
    if (program.priorities[name] != range_priority
        or range_priority <= max(task_priority.values(), default=0)):
      raise ProgramShapeRefused("unexpected_shape:range_priority:%s" % name)


def _check_apply(program):
  """A11-A14: registers unconditional and executed once each, in the catalog's
  call-site order for the range tables' features; the prologue (calc_flow_hash)
  before the first register, the orientation register first; then range tables,
  trees, and one exact-match vote table per task with more than one tree (none
  for a one-tree task, review A2) after every tree, all at one
  nesting depth. The forwarding assignments to ig_tm_md come first, before
  anything else and only there (a trailing copy is a stage the model does not
  price)."""
  steps = program.apply_steps
  if not steps:
    raise ProgramShapeRefused("unexpected_shape:no_ingress_apply")
  head = tuple((kind, " ".join(name.split()), depth)
               for kind, name, depth, _ in steps[:len(_EGRESS_ASSIGNMENTS)])
  if (head != tuple(("other", text, 0) for text in _EGRESS_ASSIGNMENTS)
      or any(step[0] == "other" and step[1].startswith("ig_tm_md.")
             for step in steps[len(_EGRESS_ASSIGNMENTS):])):
    raise ProgramShapeRefused("unexpected_shape:egress_assignment")
  executes = [step for step in steps if step[0] == "execute"]
  applies = [step for step in steps if step[0] == "apply"]
  if not executes:
    raise ProgramShapeRefused("unexpected_shape:no_registers")
  depth = executes[0][2]
  if any(step[2] != depth for step in executes + applies):
    raise ProgramShapeRefused("unexpected_shape:nesting")
  if len({step[1] for step in executes}) != len(executes):
    raise ProgramShapeRefused("unexpected_shape:register_repeated")
  first = steps.index(executes[0])
  if not any(step[0] == "call" and step[1] == _PROLOGUE_CALL for step in steps[:first]):
    raise ProgramShapeRefused("unexpected_shape:prologue")
  if executes[0][1] != _ORIENTATION_ACTION:
    raise ProgramShapeRefused("unexpected_shape:orientation_register")
  # Lazy: p4model.registers needs nothing from here, but keep this module's
  # import cost at the stdlib for metadata_reads_before_writes' callers.
  from p4model.registers import register_call_order
  features = []
  for name, keys in program.tables.items():
    if _RANGE_TABLE.match(name):
      if len(keys) != 1 or not keys[0].endswith("_val"):
        raise ProgramShapeRefused("unexpected_shape:range_key:%s" % name)
      if keys[0][:-len("_val")] not in features:
        features.append(keys[0][:-len("_val")])
  if tuple(program.register_calls) != tuple(register_call_order(features)):
    raise ProgramShapeRefused("unexpected_shape:register_calls")
  rank, last = None, -1
  trees, votes = {}, []
  for kind, name, _, _ in steps:
    if kind == "execute":
      rank = 0
    elif kind == "apply" and _RANGE_TABLE.match(name):
      rank = 1
    elif kind == "apply" and _TREE_TABLE.match(name):
      rank = 2
      trees.setdefault(_TREE_TABLE.match(name).group(1), []).append(name)
    elif kind == "apply" and _VOTE_TABLE.match(name):
      rank = 3
      votes.append(_VOTE_TABLE.match(name).group(1))
    elif kind == "apply":
      raise ProgramShapeRefused("unexpected_shape:table:%s" % name)
    else:
      continue
    if rank < last:
      raise ProgramShapeRefused("unexpected_shape:apply_order")
    last = rank
  # One vote table per task with MORE THAN ONE tree; a one-tree task has none
  # (review A2: its tree action writes the export field), and the model prices
  # a vote stage exactly for the tasks that have one (usage.pipeline_depth).
  if sorted(votes) != sorted(task for task, names in trees.items() if len(names) > 1):
    raise ProgramShapeRefused("unexpected_shape:vote_tables")
  for task in votes:
    keys = program.exact_keys.get("vote_" + task, [])
    if sorted(keys) != sorted("class_tree_%s_%s" % (task, _TREE_TABLE.match(tree).group(2))
                              for tree in trees[task]):
      raise ProgramShapeRefused("unexpected_shape:vote_key:vote_" + task)


def _check_flow_hash(program):
  """Review A1 (2026-11-15, Task 31 review Important 1): the replay prices the
  prologue at program.FLOW_HASH_LEVEL = 1 stage, which holds only for the
  generator's hash -- Hash<bit<FLOW_INDEX_BITS>> called by calc_flow_hash on the
  raw header fields, both FLOW_HASH_SYMMETRIC_PAIRS on it, no `&` mask and no `^`
  XOR of the inputs. An older generator's hash (XOR inputs, a 32-bit CRC masked
  `& 0xFFF`) costs p4c 3 stages, so its text is refused, never priced 2 stages
  low."""
  from p4model.program import FLOW_HASH_SYMMETRIC_PAIRS, FLOW_INDEX_BITS
  bits, pairs, operators = tuple(program.flow_hash) or (None, (), "")
  if (bits != FLOW_INDEX_BITS or operators
      or tuple(pairs) != tuple(sorted(FLOW_HASH_SYMMETRIC_PAIRS))):
    raise ProgramShapeRefused("unexpected_shape:flow_hash")


def _check_key_names(program):
  """Review B1 / Task 31 review minor 8: every sliced key `meta.<f>[hi:lo]` carries
  @name("<bf_rt name>"), the name the control plane's entry writer uses
  (program.bfrt_key_name). A pre-B1 text prices the same but its entries would
  not match the driver's names, so it is refused as an older generator's."""
  from p4model.program import bfrt_key_name
  for (table, field), runs in sorted(program.key_slices.items()):
    for hi, lo in runs:
      expression = "meta.%s[%d:%d]" % (field, hi, lo)
      if program.key_names.get((table, field, hi, lo)) != bfrt_key_name(expression):
        raise ProgramShapeRefused("unexpected_shape:key_name:%s:%s" % (table, expression))


def check_program(program):
  """Refuse a parsed program (p4_replay.Program) that breaks an assumption the replay
  prices by (decision 8.2; inventory_model Sec 6, A1, A5, A7, A8, A10-A14, A17, and
  the pins of decisions 4.1 / 4.3), raising ProgramShapeRefused(reason):

    stage_pin                    an @stage line (A17)
    gated_register_block         a RegisterAction executed inside `if (meta.fwd ...)`
                                 (every program before decisions 1a / 7.1 option B)
    missing_pin:<pin>:<field>    a pin the generator emits is absent: pa_solitary on
                                 class_tree_* and (with calc_timestamp) on
                                 now_pseudo_us, pa_container_size 16 on *_val and on
                                 every <chain>_current_iat, pa_no_overlay and
                                 pa_container_size on code_*, placement_priority on
                                 every tree and range table
    unexpected_shape:<what>      a declared width, priority, key or apply order other
                                 than the generator's, or (pin:<pin>:<field>) a PHV pin
                                 the generator does not write (the pins dropped at
                                 gate G3 among them); flow_hash: a flow hash other
                                 than the generator's (review A1); key_name:<table>:
                                 <slice>: a sliced key without its bf_rt @name (B1)

  A program the generator writes passes; an archived program of an older generator
  is refused, never priced as if it were current."""
  if program.stage_pins:
    raise ProgramShapeRefused("stage_pin")
  if program.gated_register_blocks or any(
      kind == "execute" and under_fwd for kind, _, _, under_fwd in program.apply_steps):
    raise ProgramShapeRefused("gated_register_block")
  _check_flow_hash(program)
  _check_key_names(program)
  _check_pins(program, _pins(program))
  _check_priorities(program)
  _check_apply(program)
