"""p4model and p4gen are installable on their own (pyproject.toml: distribution
"p4rf"). These tests pin the properties that make that true, each in a fresh
interpreter so that imports already made by the test session cannot hide a
violation:

  * neither package imports the repository-only `src` package (training code);
  * p4model needs nothing outside the standard library;
  * p4gen finds its P4 templates from its own location, not from the current
    directory (they used to be read from a cwd-relative "resources/").
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(code, cwd):
    """Runs `code` in a fresh interpreter whose only access to this checkout is
    PYTHONPATH, started from `cwd`; returns the CompletedProcess."""
    env = dict(os.environ, PYTHONPATH=REPO)
    return subprocess.run([sys.executable, '-c', code], cwd=cwd, env=env,
                          capture_output=True, text=True)


def test_the_packages_do_not_import_the_repository_only_src_package(tmp_path):
    # sys.modules[name] = None makes `import name` raise ImportError.
    code = ("import sys; sys.modules['src'] = None\n"
            "import p4model, p4model.usage, p4model.packing, p4model.lanes, p4model.tables\n"
            "import p4gen.build_p4_script, p4gen.evaluation, p4gen.p4_compile, p4gen.p4_replay\n")
    result = _run(code, tmp_path)

    assert result.returncode == 0, result.stderr[-800:]


def test_p4model_needs_only_the_standard_library(tmp_path):
    code = ("import sys\n"
            "for name in ('numpy', 'sklearn', 'scipy', 'pandas'):\n"
            "    sys.modules[name] = None\n"
            "import p4model, p4model.usage, p4model.packing, p4model.lanes, p4model.tables\n")
    result = _run(code, tmp_path)

    assert result.returncode == 0, result.stderr[-800:]


def test_p4gen_finds_its_templates_from_any_directory(tmp_path):
    code = ("import os, p4gen.build_p4_script as b, p4gen.p4_compile as c\n"
            "assert os.path.isabs(b.PATH), b.PATH\n"
            "assert os.path.isfile(b.PATH_P4_CODE_TEMPLATE_INPUT), b.PATH_P4_CODE_TEMPLATE_INPUT\n"
            "for name in ('p4_template.p4', 'p4_headers.p4', 'p4_util.p4'):\n"
            "    assert os.path.isfile(os.path.join(c.DEFAULT_INCLUDE_PATH, name)), name\n")
    result = _run(code, tmp_path)

    assert result.returncode == 0, result.stderr[-800:]
