"""函数式入口、稠密缓存适配与 CLI 验收。"""

import sys as _sys
from pathlib import Path as _Path
_T = str(_Path(__file__).resolve().parent)
if _T not in _sys.path:
    _sys.path.insert(0, _T)

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from _contract_util import SRC  # noqa: F401

import operator_paged_attention as opa
from operator_paged_attention import paged_attention

REPO = Path(__file__).resolve().parents[1]


class FunctionalApiTest(unittest.TestCase):
    def _raw_pages(self):
        # page 0: 4 slots, 2 valid；page 1: 4 slots, 1 valid
        pages = {
            0: {
                "session": "s", "valid_tokens": 2,
                "keys": [[[0.1, 0.2]], [[0.3, 0.4]], [[9e3, 9e3]], [[9e3, 9e3]]],
                "values": [[[1.0, 0.0]], [[0.0, 1.0]], [[9e3, 9e3]], [[9e3, 9e3]]],
            },
            1: {
                "session": "s", "valid_tokens": 1,
                "keys": [[[0.5, 0.6]], [[9e3, 9e3]], [[9e3, 9e3]], [[9e3, 9e3]]],
                "values": [[[1.0, 1.0]], [[9e3, 9e3]], [[9e3, 9e3]], [[9e3, 9e3]]],
            },
        }
        return pages

    def test_keyword_functional_call(self):
        res = paged_attention(
            queries=[[0.2, -0.1]],
            page_table=[0, 1],
            pages=self._raw_pages(),
            page_size=4, kv_length=5, position=4,
        )
        self.assertEqual(res.witnesses[0].visible_positions, (0, 1, 4))
        self.assertTrue(all(abs(x) < 100 for x in res.rows[0].flat_output()))

    def test_q_and_block_table_aliases(self):
        res = paged_attention(
            q=[0.2, -0.1], block_table=[0], pages={
                0: {"valid_tokens": 1,
                    "keys": [[[0.1, 0.2]]], "values": [[[1.0, 0.0]]]}
            }, page_size=4, kv_length=1, position=0,
        )
        self.assertEqual(res.witnesses[0].visible_positions, (0,))

    def test_positional_form(self):
        res = paged_attention(
            [[0.2, -0.1]], [0], self._raw_pages(),
            page_size=4, kv_length=2, position=1,
        )
        self.assertEqual(res.witnesses[0].visible_positions, (0, 1))

    def test_dense_key_value_cache_form(self):
        key_cache = [
            [[[0.1, 0.2]], [[0.3, 0.4]], [[0.0, 0.0]], [[0.0, 0.0]]],
        ]
        value_cache = [
            [[[1.0, 0.0]], [[0.0, 1.0]], [[0.0, 0.0]], [[0.0, 0.0]]],
        ]
        res = paged_attention(
            q=[0.1, 0.1],
            key_cache=key_cache, value_cache=value_cache,
            page_table=[0], page_size=4, kv_length=2, position=1,
            valid_tokens={0: 2},
        )
        self.assertEqual(res.witnesses[0].visible_positions, (0, 1))

    def test_unknown_option_raises(self):
        with self.assertRaises(TypeError):
            paged_attention(q=[0.1], page_table=[0], pages={}, bogus_option=123)

    def test_public_aliases_present(self):
        for name in [
            "PagedAttention", "PagedAttentionOperator", "paged_attention",
            "PageTable", "BlockTable", "PagedKVCache", "AttentionRequest",
            "AttentionResult", "Witness", "ResultLedger", "LeaseExpiredError",
            "PageReclaimedError", "FullyMaskedRowError", "solve",
        ]:
            self.assertTrue(hasattr(opa, name), msg=name)


class CliAndSolveTest(unittest.TestCase):
    def test_solve_default_fixture(self):
        report = opa.solve()
        self.assertIn("requests", report)
        self.assertTrue(report["reclamation_demo"][
            "incomplete_request_failed_explicitly"])

    def test_cli_emits_json(self):
        from operator_paged_attention.__main__ import main
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["--data", str(REPO / "data" / "input.json"),
                         "--no-reclamation-demo"])
        self.assertEqual(code, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["page_size"], 4)
        self.assertNotIn("reclamation_demo", payload)


if __name__ == "__main__":
    unittest.main()
