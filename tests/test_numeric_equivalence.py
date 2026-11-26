"""数值正确性：与独立暴力参考对比，prefill/decode 边界一致性与确定性。"""

import sys as _sys
from pathlib import Path as _Path
_T = str(_Path(__file__).resolve().parent)
if _T not in _sys.path:
    _sys.path.insert(0, _T)

import unittest

from _contract_util import brute_force_row, make_random_case, SRC  # noqa: F401

from operator_paged_attention import paged_attention


class RandomizedReferenceTest(unittest.TestCase):
    def test_exact_path_matches_brute_force_across_layouts(self):
        for seed in range(60):
            page_size = [1, 2, 3, 4, 8][seed % 5]
            length = 1 + (seed * 7) % 18
            window = [None, 0, 1, 2, 4, 99][seed % 6]
            conv = "inclusive" if seed % 2 == 0 else "exclusive"
            cache, table, flat_k, flat_v, queries = make_random_case(
                1000 + seed, page_size=page_size, length=length, window=window,
                convention=conv,
            )
            from operator_paged_attention import PagedAttentionOperator
            op = PagedAttentionOperator(cache)

            for i in range(length):
                res = op.prefill(
                    request_id=f"r{seed}", session_id="s", page_table=table,
                    kv_length=length, queries=[queries[i]], query_positions=[i],
                    window=window, window_convention=conv, strategy="exact",
                    record=False,
                )
                expected, visible = brute_force_row(
                    queries[i], flat_k, flat_v, i, window=window, convention=conv
                )
                self.assertEqual(res.witnesses[0].visible_positions, visible,
                                 msg=f"seed={seed} i={i} visibility")
                if expected is None:
                    self.assertTrue(res.rows[0].is_fully_masked)
                    continue
                worst = max(
                    abs(res.rows[0].output[h][d] - expected[h][d])
                    for h in range(2) for d in range(6)
                )
                self.assertLess(worst, 1e-12, msg=f"seed={seed} i={i} worst={worst}")

    def test_block_fast_within_loose_tolerance(self):
        for seed in range(20):
            cache, table, _, _, queries = make_random_case(
                2000 + seed, page_size=4, length=14, window=5,
            )
            from operator_paged_attention import PagedAttentionOperator
            op = PagedAttentionOperator(cache)
            exact = op.prefill(
                session_id="s", page_table=table, kv_length=14, queries=queries,
                window=5, strategy="exact", record=False,
            )
            fast = op.prefill_fast(
                session_id="s", page_table=table, kv_length=14, queries=queries,
                window=5, record=False,
            )
            # 遮罩语义必须完全一致
            self.assertEqual(
                [w.visible_positions for w in exact.witnesses],
                [w.visible_positions for w in fast.witnesses],
            )
            worst = 0.0
            for r1, r2 in zip(exact.rows, fast.rows):
                for h in range(2):
                    for d in range(6):
                        worst = max(worst, abs(r1.output[h][d] - r2.output[h][d]))
            self.assertLess(worst, 1e-8)
            self.assertTrue(fast.approximate)
            self.assertEqual(fast.strategy, "block_fast")


class PrefillDecodeBoundaryEquivalenceTest(unittest.TestCase):
    def test_window_five_six_tokens_boundary_both_conventions(self):
        cache, table, flat_k, flat_v, queries = make_random_case(
            42, page_size=4, length=6, head_dim=8,
        )
        from operator_paged_attention import PagedAttentionOperator
        op = PagedAttentionOperator(cache)

        # inclusive W=5: p=5 -> [1,5]（边界排除位置 0）
        pre = op.prefill(
            session_id="s", page_table=table, kv_length=6, queries=[queries[5]],
            query_positions=[5], window=5, window_convention="inclusive", record=False,
        )
        self.assertEqual(pre.witnesses[0].visible_positions, (1, 2, 3, 4, 5))

        # exclusive: 下界 p-W=0 -> [0,5]
        pre_ex = op.prefill(
            session_id="s", page_table=table, kv_length=6, queries=[queries[5]],
            query_positions=[5], window=5, window_convention="exclusive", record=False,
        )
        self.assertEqual(pre_ex.witnesses[0].visible_positions, (0, 1, 2, 3, 4, 5))

    def test_prefill_row_and_decode_step_bitwise_identical(self):
        for seed in range(25):
            cache, table, _, _, queries = make_random_case(
                300 + seed, page_size=3, length=9, window=4,
            )
            from operator_paged_attention import PagedAttentionOperator
            op = PagedAttentionOperator(cache)
            for p in range(9):
                pre = op.prefill(
                    request_id="pre", session_id="s", page_table=table, kv_length=9,
                    queries=[queries[p]], query_positions=[p], window=4, record=False,
                )
                dec = op.decode(
                    request_id="dec", session_id="s", page_table=table, kv_length=9,
                    query=queries[p], position=p, window=4,
                )
                # 同一份遮罩 + 同一条精确数值路径 => 必须逐位相同，
                # 这正是“窗口边界不再让两阶段得到不同 logits”的保证。
                self.assertEqual(pre.rows[0].output, dec.rows[0].output,
                                 msg=f"seed={seed} p={p}")
                self.assertEqual(
                    pre.witnesses[0].visible_positions,
                    dec.witnesses[0].visible_positions,
                )

    def test_functional_entry_matches_operator(self):
        cache, table, _, _, queries = make_random_case(7, length=5)
        refs = [{"page_id": r.page_id, "generation": r.generation} for r in table]
        result = paged_attention(
            queries=[queries[4]], page_table=refs, cache=cache,
            kv_length=5, position=4, window=2,
        )
        self.assertEqual(result.witnesses[0].visible_positions, (3, 4))


if __name__ == "__main__":
    unittest.main()
