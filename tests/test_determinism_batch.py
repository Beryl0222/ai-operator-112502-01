"""解码确定性、严格误差约束与混合元数据批处理。"""

import sys as _sys
from pathlib import Path as _Path
_T = str(_Path(__file__).resolve().parent)
if _T not in _sys.path:
    _sys.path.insert(0, _T)

import random
import unittest

from _contract_util import SRC, brute_force_row  # noqa: F401

from operator_paged_attention import PagedAttentionOperator, PagedKVCache


def _random_cache(seed, n_req=3):
    rng = random.Random(seed)
    cache = PagedKVCache(page_size=4, num_heads=2, head_dim=5)
    specs = []
    pid = 0
    for r in range(n_req):
        length = rng.randint(2, 11)
        blocks = (length + 3) // 4
        table = []
        flat_k, flat_v = [], []
        for b in range(blocks):
            ref = cache.allocate(f"s{r}", page_id=pid)
            pid += 1
            valid = min(4, length - b * 4)
            for s in range(valid):
                k = [[rng.uniform(-1, 1) for _ in range(5)] for _ in range(2)]
                v = [[rng.uniform(-1, 1) for _ in range(5)] for _ in range(2)]
                cache.write_slot(ref, s, k, v, session_id=f"s{r}")
                flat_k.append(k)
                flat_v.append(v)
            table.append(ref)
        q = [[rng.uniform(-1, 1) for _ in range(5)] for _ in range(2)]
        window = rng.choice([None, 0, 2, 3, 5])
        specs.append((f"s{r}", table, length, q, window, flat_k, flat_v))
    return cache, specs


class DecodeDeterminismTest(unittest.TestCase):
    def test_repeated_decode_bitwise_identical(self):
        cache, specs = _random_cache(11)
        op = PagedAttentionOperator(cache)
        for rid, table, length, q, window, _, _ in specs:
            p = length - 1
            runs = [
                op.decode(request_id=f"{rid}-{i}", session_id=rid, page_table=table,
                          kv_length=length, query=q, position=p, window=window).rows[0].output
                for i in range(8)
            ]
            for run in runs[1:]:
                self.assertEqual(run, runs[0])

    def test_batch_order_does_not_change_outputs(self):
        cache, specs = _random_cache(23, n_req=4)
        op = PagedAttentionOperator(cache)

        def one(spec):
            rid, table, length, q, window, _, _ = spec
            return op.decode(request_id=rid, session_id=rid, page_table=table,
                             kv_length=length, query=q, position=length - 1,
                             window=window, record=False).rows[0].output

        order_a = [one(s) for s in specs]
        order_b = [one(s) for s in reversed(specs)]
        self.assertEqual(order_a, list(reversed(order_b)))

    def test_decode_strict_error_bound(self):
        cache, specs = _random_cache(31, n_req=6)
        op = PagedAttentionOperator(cache)
        for rid, table, length, q, window, fk, fv in specs:
            p = length - 1
            res = op.decode(session_id=rid, page_table=table, kv_length=length,
                            query=q, position=p, window=window, record=False)
            expected, visible = brute_force_row(q, fk, fv, p, window=window)
            self.assertEqual(res.witnesses[0].visible_positions, visible)
            worst = max(
                abs(res.rows[0].output[h][d] - expected[h][d])
                for h in range(2) for d in range(5)
            )
            # 解码路径维持严格误差约束（相对 fsum 参考 < 1e-12）
            self.assertLess(worst, 1e-12)
            self.assertEqual(res.strategy, "exact")


class MixedBatchTest(unittest.TestCase):
    def test_batch_with_different_lengths_windows_and_phases(self):
        from operator_paged_attention import build_fixture_scenario, load_fixture
        scenario = build_fixture_scenario(load_fixture())
        op = scenario.operator
        payload = scenario.payload

        requests = []
        for spec in payload["requests"]:
            rid = spec["request_id"]
            pfx = int(spec["prefix_length"])
            w = int(spec["window"]) or None
            q = scenario.query_at(rid, pfx - 1)
            requests.append(dict(
                request_id=rid, session_id=rid,
                page_table=scenario.tables[rid], kv_length=pfx,
                query=q, position=pfx - 1, window=w, phase="decode",
            ))
        batch = op.submit_batch(requests)
        self.assertTrue(batch.all_succeeded, [o.error for o in batch.failures])
        # chat-a W=5 at p=5 -> 5 keys；chat-b window=0 不限制 -> 3 keys
        self.assertEqual(
            len(batch["chat-a"].result.witnesses[0].visible_positions), 5
        )
        self.assertEqual(
            len(batch["chat-b"].result.witnesses[0].visible_positions), 3
        )
        self.assertEqual([o.request_id for o in batch.outcomes], ["chat-a", "chat-b"])


if __name__ == "__main__":
    unittest.main()
