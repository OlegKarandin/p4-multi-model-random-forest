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
"""
import re

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
