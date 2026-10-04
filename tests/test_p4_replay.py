"""Program-level replay in the library (spec 2026-09-29 §5.3, §6.6 --rescore).
The c1 fixture holds already-parsed programs (tests/fixtures/c1_replay_designs.json),
so these run without results/."""
import json
import os

import pytest

from src.p4gen import p4_replay as rp

_C1 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "fixtures", "c1_replay_designs.json")


def _designs():
    with open(_C1, encoding="utf-8") as handle:
        return sorted(json.load(handle)["designs"].items())


def _program(design):
    return rp.Program(design["tables"], design["widths"], design["bits"], design["sizes"])


@pytest.mark.parametrize("name,design", _designs(), ids=[n for n, _ in _designs()])
def test_breakdown_totals_equal_the_scripts_replay(name, design):
    from scripts.p4_artifact_replay import replay_program
    depth, blocks = replay_program(name, design["tables"], design["widths"],
                                   design["bits"], design["sizes"])
    breakdown = rp.model_breakdown(_program(design), row_id=name)
    assert (breakdown["stage_depth"], breakdown["blocks"]) == (depth, blocks)
    assert sum(t["blocks"] for t in breakdown["tables"]) == blocks


@pytest.mark.parametrize("name,design", _designs(), ids=[n for n, _ in _designs()])
def test_breakdown_names_every_priced_table_once(name, design):
    breakdown = rp.model_breakdown(_program(design), row_id=name)
    names = [t["table"] for t in breakdown["tables"]]
    assert len(names) == len(set(names))
    assert set(names) == set(rp.declared_prices(_program(design)))
    for entry in breakdown["tables"]:
        assert entry["kind"] == ("range" if entry["table"].startswith("table_") else "ternary")


def test_parse_program_text_reads_keys_widths_and_sizes():
    text = """
    struct metadata_t {
        bit<12> code_a;
        bit<16> a_val;
    }
    table table_0_a {
        key = {
            meta.a_val : range;
        }
        size = 9;
    }
    table get_classification_tree_app_0 {
        key = {
            meta.code_a : ternary;
        }
        size = 40;
    }
    """
    program = rp.parse_program_text(text)
    assert program.tables == {"table_0_a": ["a_val"],
                              "get_classification_tree_app_0": ["code_a"]}
    assert program.bits == {"code_a": 12, "a_val": 16}
    assert program.widths == {"code_a": 2, "a_val": 2}
    assert program.sizes == {"table_0_a": 9, "get_classification_tree_app_0": 40}


def test_declared_prices_charge_classification_tables_with_table_blocks():
    from src.p4gen.p4_replay import Program, declared_prices
    program = Program(
        tables={'get_classification_tree_ddos_0': ['code_a', 'code_b']},
        widths={'code_a': 4, 'code_b': 7},
        bits={'code_a': 27, 'code_b': 52},
        sizes={'get_classification_tree_ddos_0': 600})       # 2 words
    assert declared_prices(program) == {'get_classification_tree_ddos_0': 4}
