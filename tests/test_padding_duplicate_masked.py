"""无效填充、重复页有效区间与完全遮罩行的处理契约。"""

import sys as _sys
from pathlib import Path as _Path
_T = str(_Path(__file__).resolve().parent)
if _T not in _sys.path:
    _sys.path.insert(0, _T)

import math
import unittest

from _contract_util import SRC  # noqa: F401

from operator_paged_attention import (
    FullyMaskedRowError,
    PagedAttentionOperator,
    PagedKVCache,
)


class PaddingDenominatorTest(unittest.TestCase):
    def _build(self, valid_tokens=2, kv_length=4):
        cache = PagedKVCache(page_size=4, num_heads=1, head_dim=3)
        op = PagedAttentionOperator(cache)
        ref = cache.allocate("s", page_id=0, valid_tokens=valid_tokens)
        real = []
        for slot in range(valid_tokens):
            k = [[0.3 * (slot + 1), -0.2, 0.1]]
            v = [[float(slot + 1), 2.0, -1.0]]
            cache.write_slot(ref, slot, k, v, session_id="s")
            real.append((k, v))
        # 在“无效填充槽”放置极端毒值
        for slot in range(valid_tokens, 4):
            cache.pages[0].keys[slot] = [[1e8, 1e8, 1e8]]
            cache.pages[0].values[slot] = [[1e8, 1e8, 1e8]]
        return cache, op, ref, real

    def test_invalid_padding_excluded_from_denominator(self):
        cache, op, ref, real = self._build(valid_tokens=2, kv_length=4)
        q = [[1.0, 0.0, 0.0]]
        res = op.decode(
            session_id="s", page_table=[ref], kv_length=4,
            query=q, position=3, causal=False, record=False,
        )
        w = res.witnesses[0]
        self.assertEqual(w.visible_positions, (0, 1))
        excluded = {e.position: e.reason for e in w.excluded}
        self.assertEqual(excluded.get(2), "invalid_padding")
        self.assertEqual(excluded.get(3), "invalid_padding")

        # 归一化分母只含两个有效 key
        scores = [
            (q[0][0] * real[j][0][0][0]) / math.sqrt(3) for j in range(2)
        ]
        m = max(scores)
        expected_norm = math.fsum(math.exp(s - m) for s in scores)
        self.assertAlmostEqual(w.normalizers[0], expected_norm, places=12)

        # 输出等于仅两个有效 KV 的加权结果，绝不可能被毒值放大
        weights = [math.exp(s - m) / expected_norm for s in scores]
        expected_out = [
            math.fsum(weights[j] * real[j][1][0][d] for j in range(2))
            for d in range(3)
        ]
        for d in range(3):
            self.assertAlmostEqual(res.rows[0].output[0][d], expected_out[d], places=12)
        self.assertTrue(all(abs(x) < 1e3 for x in res.rows[0].flat_output()))

    def test_prefill_fast_path_also_excludes_padding(self):
        cache, op, ref, _ = self._build(valid_tokens=2, kv_length=4)
        res = op.prefill_fast(
            session_id="s", page_table=[ref], kv_length=4,
            queries=[[[1.0, 0.0, 0.0]]], query_positions=[3],
            causal=False, record=False,
        )
        self.assertEqual(res.witnesses[0].visible_positions, (0, 1))
        self.assertTrue(all(abs(x) < 1e3 for x in res.rows[0].flat_output()))


class DuplicatePageTest(unittest.TestCase):
    def test_duplicate_page_entries_respect_declared_range(self):
        cache = PagedKVCache(page_size=4, num_heads=1, head_dim=2)
        op = PagedAttentionOperator(cache)
        ref = cache.allocate("s", page_id=3, valid_tokens=2)
        for slot in range(2):
            cache.write_slot(
                ref, slot,
                [[float(slot + 1), 0.5]],
                [[10.0 * (slot + 1), 1.0]],
                session_id="s",
            )
        for slot in range(2, 4):
            cache.pages[3].keys[slot] = [[1e9, 1e9]]
            cache.pages[3].values[slot] = [[1e9, 1e9]]

        # 同一物理页在页表中出现两次：逻辑块 0 与逻辑块 1
        table = [ref, ref]
        res = op.decode(
            session_id="s", page_table=table, kv_length=6,
            query=[[0.2, -0.3]], position=5, causal=True, record=False,
        )
        w = res.witnesses[0]
        # 块0: 位置0,1 有效；2,3 是页内填充。块1: 位置4,5 映射回 slot0,1 有效
        self.assertEqual(w.visible_positions, (0, 1, 4, 5))
        coords = [(v.position, v.slot) for v in w.visible]
        self.assertEqual(coords, [(0, 0), (1, 1), (4, 0), (5, 1)])
        # 四个有效出现都参与，分母有限
        self.assertEqual(len(w.visible), 4)
        self.assertGreater(w.normalizers[0], 0.0)
        self.assertTrue(all(abs(x) < 1e6 for x in res.rows[0].flat_output()))

    def test_optional_physical_dedup(self):
        cache = PagedKVCache(page_size=4, head_dim=2)
        op = PagedAttentionOperator(cache)
        ref = cache.allocate("s", page_id=3, valid_tokens=2)
        for slot in range(2):
            cache.write_slot(ref, slot, [[1.0, 0.0]], [[float(slot), 1.0]])
        res = op.decode(
            session_id="s", page_table=[ref, ref], kv_length=6,
            query=[[0.2, -0.3]], position=5, dedup_physical=True, record=False,
        )
        self.assertEqual(res.witnesses[0].visible_positions, (0, 1))
        reasons = {e.position: e.reason for e in res.witnesses[0].excluded}
        self.assertEqual(reasons[4], "duplicate_physical_page")
        self.assertEqual(reasons[5], "duplicate_physical_page")


class FullyMaskedRowTest(unittest.TestCase):
    def _empty_window_case(self, **kw):
        cache = PagedKVCache(page_size=4, head_dim=3)
        op = PagedAttentionOperator(cache)
        ref = cache.allocate("s", page_id=0, valid_tokens=3)
        for slot in range(3):
            cache.write_slot(ref, slot, [[0.1, 0.2, 0.3]], [[1.0, 2.0, 3.0]])
        # query 位于 p=3，但 KV 只有位置 0..2（当前 token 的 K/V 尚未写入）；
        # W=1 inclusive 下界=p=3，与 [0,3) 无交集 => 完全遮罩。
        return op.decode(
            session_id="s", page_table=[ref], kv_length=3,
            query=[[0.1, 0.2, 0.3]], position=3, window=1,
            record=False, **kw
        )

    def test_fully_masked_returns_handleable_status_by_default(self):
        result = self._empty_window_case()
        row = result.rows[0]
        self.assertTrue(row.is_fully_masked)
        self.assertIsNone(row.output)
        self.assertEqual(row.status, "fully_masked")
        self.assertTrue(result.witnesses[0].is_fully_masked)
        self.assertEqual(result.witnesses[0].visible_positions, ())
        # dense 张量仍保持形状且为零，不含 NaN
        self.assertEqual(result.dense_outputs[0], [[0.0, 0.0, 0.0]])
        self.assertNotIn("nan", str(result.dense_outputs).lower())
        # 排除原因可解释（全部落在窗口外）
        self.assertTrue(
            all(e.reason == "outside_window" for e in result.witnesses[0].excluded)
        )

    def test_fully_masked_zero_policy(self):
        result = self._empty_window_case(on_fully_masked="zero")
        self.assertTrue(result.rows[0].is_fully_masked)
        self.assertEqual(result.rows[0].output, [[0.0, 0.0, 0.0]])

    def test_fully_masked_raise_policy(self):
        with self.assertRaises(FullyMaskedRowError):
            self._empty_window_case(on_fully_masked="raise")

    def test_all_padding_page_is_fully_masked(self):
        cache = PagedKVCache(page_size=4, head_dim=2)
        op = PagedAttentionOperator(cache)
        ref = cache.allocate("s", page_id=0, valid_tokens=0)
        result = op.prefill(
            session_id="s", page_table=[ref], kv_length=2,
            queries=[[[0.1, 0.2]]], query_positions=[0],
            causal=False, record=False,
        )
        self.assertTrue(result.rows[0].is_fully_masked)
        reasons = {e.position: e.reason for e in result.witnesses[0].excluded}
        self.assertEqual(set(reasons.values()), {"invalid_padding"})


if __name__ == "__main__":
    unittest.main()
