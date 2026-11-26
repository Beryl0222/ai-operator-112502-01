"""资料夹具上的可见区间、物理坐标与 witness 解释契约。"""

import sys as _sys
from pathlib import Path as _Path
_T = str(_Path(__file__).resolve().parent)
if _T not in _sys.path:
    _sys.path.insert(0, _T)

import json
import unittest
from pathlib import Path

from _contract_util import SRC  # noqa: F401  (确保 src 已入 sys.path)

from operator_paged_attention import (
    Witness,
    build_fixture_scenario,
    load_fixture,
    solve_fixture,
)

REPO = Path(__file__).resolve().parents[1]


class FixtureVisibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = load_fixture(REPO / "data" / "input.json")
        cls.scenario = build_fixture_scenario(cls.payload)
        cls.op = cls.scenario.operator

    def test_fixture_loads_two_requests(self):
        self.assertEqual(
            [r["request_id"] for r in self.payload["requests"]],
            ["chat-a", "chat-b"],
        )
        self.assertEqual(self.payload["page_size"], 4)

    def test_chat_a_prefill_visibility_with_window_five(self):
        spec = self.payload["requests"][0]
        self.assertEqual(spec["request_id"], "chat-a")
        qs = [self.scenario.query_at("chat-a", p) for p in range(6)]
        result = self.op.prefill(
            request_id="chat-a", session_id="chat-a",
            page_table=self.scenario.tables["chat-a"], kv_length=6,
            queries=qs, window=5, causal=True,
        )
        expected = [
            (0,),
            (0, 1),
            (0, 1, 2),
            (0, 1, 2, 3),
            (0, 1, 2, 3, 4),
            (1, 2, 3, 4, 5),  # W=5 inclusive: p=5 -> lower=1
        ]
        self.assertEqual([w.visible_positions for w in result.witnesses], expected)
        for w in result.witnesses:
            self.assertEqual(len(w.mask), 6)
            self.assertEqual(sum(w.mask), len(w.visible_positions))

    def test_chat_b_unlimited_window_prefill(self):
        qs = [self.scenario.query_at("chat-b", p) for p in range(3)]
        result = self.op.prefill(
            request_id="chat-b", session_id="chat-b",
            page_table=self.scenario.tables["chat-b"], kv_length=3,
            queries=qs, window=0, causal=True,
        )
        self.assertEqual(
            [w.visible_positions for w in result.witnesses],
            [(0,), (0, 1), (0, 1, 2)],
        )

    def test_decode_visibility_sets(self):
        # chat-a positions 6/7
        table = self.scenario.tables["chat-a"]
        for pos, expected in [(6, (2, 3, 4, 5, 6)), (7, (3, 4, 5, 6, 7))]:
            k, v = self.scenario.token_kv_at("chat-a", pos)
            self.op.cache.append_token(table, pos, k, v, session_id="chat-a")
            q = self.scenario.query_at("chat-a", pos)
            res = self.op.decode(
                request_id="chat-a", session_id="chat-a", page_table=table,
                kv_length=pos + 1, query=q, position=pos, window=5,
            )
            self.assertEqual(res.witnesses[0].visible_positions, expected)

    def test_witness_physical_coordinates_cross_page(self):
        qs = [self.scenario.query_at("chat-a", 5)]
        result = self.op.prefill(
            request_id="chat-a", session_id="chat-a",
            page_table=self.scenario.tables["chat-a"], kv_length=6,
            queries=qs, query_positions=[5], window=5,
        )
        w: Witness = result.witnesses[0]
        coords = {v.position: (v.page_id, v.slot) for v in w.visible}
        # 位置 3 在 page 0 slot 3；位置 4/5 跨页到 page 1 slot 0/1
        self.assertEqual(coords[3], (0, 3))
        self.assertEqual(coords[4], (1, 0))
        self.assertEqual(coords[5], (1, 1))

    def test_witness_weights_and_normalizer(self):
        qs = [self.scenario.query_at("chat-b", 2)]
        result = self.op.prefill(
            request_id="chat-b", session_id="chat-b",
            page_table=self.scenario.tables["chat-b"], kv_length=3,
            queries=qs, query_positions=[2],
        )
        w = result.witnesses[0]
        self.assertEqual(len(w.normalizers), self.scenario.num_heads)
        for h in range(self.scenario.num_heads):
            weights = [v.weights[h] for v in w.visible]
            self.assertAlmostEqual(sum(weights), 1.0, places=12)
            self.assertTrue(all(x >= 0.0 for x in weights))
            self.assertGreater(w.normalizers[h], 0.0)

    def test_witness_explain_and_json(self):
        report = solve_fixture(self.payload)
        text = report["requests"]["chat-a"]["last_decode_explanation"]
        self.assertIn("visible positions", text)
        self.assertIn("page=", text)
        # 整个报告必须 JSON 可序列化（证据可落盘/可查询）
        json.dumps(report, ensure_ascii=False)

    def test_solve_report_visibility(self):
        report = solve_fixture(self.payload, include_reclamation_demo=False)
        a = report["requests"]["chat-a"]
        self.assertTrue(a["decode_deterministic_bitwise"])
        self.assertLess(a["prefill_fast_max_abs_diff"], 1e-9)
        self.assertEqual(
            a["decode"][0]["visible_positions"], [[2, 3, 4, 5, 6]]
        )


if __name__ == "__main__":
    unittest.main()
