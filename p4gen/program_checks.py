"""Static checks on a generated P4 program's text (decision 3.8).

Without `@pa_auto_init_metadata`, p4c marks every metadata field that MAY be read before
it is written `@pa_no_init`: its value is undefined and the field may be overlaid. So no
field of ours may be read before something in the program writes it.
`metadata_reads_before_writes` walks the ingress control's apply block in program order
and names the `meta.<field>` reads that precede every write of that field.

Plain text analysis: the generator's output is regular enough that regexes over it do.
What counts, in apply order:
  read   a `meta.<f>` on the right-hand side of an assignment, in an `if` condition, in the
         `.execute(...)` index, in the body of a register action or action at the point it
         runs, or in a table's key (a code-field slice reads its whole field);
  write  an assignment target (`meta.<f> = ...`, also `meta.<f>[a:b] = ...`), a register
         `.execute` assigned to a field, and the fields assigned by the actions a table
         lists, counted at that table's apply (a table write counts as a write at the
         table's apply, hit or miss; a generous reading the range tables' full coverage
         justifies).
Intrinsic metadata (`ig_intr_md`, `ig_prsr_md`, `ig_tm_md`, `ig_dprsr_md`) and headers are
not `meta.` fields and are exempt.
"""
import re

_META = re.compile(r"\bmeta\.(\w+)")
_ASSIGN = re.compile(r"^meta\.(\w+)\s*(?:\[[^\]]*\])?\s*=(?!=)\s*(.*)$", re.S)
_EXECUTE = re.compile(r"\b(\w+)\.execute\s*\(")
_ACTION = re.compile(r"\baction\s+(\w+)\s*\([^)]*\)\s*\{")
_REGISTER_ACTION = re.compile(r"RegisterAction<[^\n]*?>\s*\(\s*\w+\s*\)\s+(\w+)\s*=\s*\{")
_TABLE = re.compile(r"\btable\s+(\w+)\s*\{")
_CONTROL = re.compile(r"\bcontrol\s+\w+\s*\(([^)]*)\)\s*\{")
_APPLY_CALL = re.compile(r"^(\w+)\.apply\s*\(\s*\)$")
_ACTION_CALL = re.compile(r"^(\w+)\s*\(\s*\)$")


def _matching(text, open_index):
  """Index of the `}` closing the `{` at open_index."""
  depth = 0
  for index in range(open_index, len(text)):
    if text[index] == "{":
      depth += 1
    elif text[index] == "}":
      depth -= 1
      if depth == 0:
        return index
  raise ValueError("unbalanced braces in the program text")


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


def _statements(body):
  """The body's statements and conditions, in order: split at `;`, `{` and `}`."""
  return [part.strip() for part in re.split(r"[;{}]", body) if part.strip()]


class _Walk:
  def __init__(self, text):
    self.actions = _bodies(text, _ACTION)
    self.register_actions = _bodies(text, _REGISTER_ACTION)
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
    self.written = set()
    self.violations = []

  def _read(self, field):
    if field not in self.written and field not in self.violations:
      self.violations.append(field)

  def _reads_of(self, expression):
    """Every meta field the expression reads, including those read by the body of each
    register action it executes."""
    for field in _META.findall(expression):
      self._read(field)
    for name in _EXECUTE.findall(expression):
      if name in self.register_actions:
        self._run(self.register_actions[name])

  def _run(self, body):
    for statement in _statements(body):
      self._statement(statement)

  def _statement(self, statement):
    applied = _APPLY_CALL.match(statement)
    if applied and applied.group(1) in self.tables:
      keys, actions = self.tables[applied.group(1)]
      for field in keys:
        self._read(field)
      for name in actions:
        self._write_only(self.actions.get(name, ""))
      return
    called = _ACTION_CALL.match(statement)
    if called and called.group(1) in self.actions:
      self._run(self.actions[called.group(1)])
      return
    assign = _ASSIGN.match(statement)
    if assign:
      self._reads_of(assign.group(2))
      self.written.add(assign.group(1))
      return
    self._reads_of(statement)

  def _write_only(self, action_body):
    """A table's action: its own reads (action data is not metadata) run at the table
    too, then its writes land."""
    for statement in _statements(action_body):
      assign = _ASSIGN.match(statement)
      if assign:
        self._reads_of(assign.group(2))
        self.written.add(assign.group(1))
      else:
        self._reads_of(statement)


def metadata_reads_before_writes(text):
  """Metadata field names (without `meta.`) read in the ingress apply block, in a register
  action body or action it runs, or in a table key, before any statement, action or table
  earlier in the apply order writes them. First-read order, no repeats; empty when the
  program is clean."""
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
    walk._run(control[start + 1:_matching(control, start)])
    return walk.violations
  raise ValueError("no control with an `inout <type> meta` parameter")
