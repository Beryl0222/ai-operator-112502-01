"""因果 + 滑动窗口 + 前缀共享同时生效的组合契约。"""

import sys as _sys
from pathlib import Path as _Path
_T = str(_Path(__file__).resolve().parent)
if _T not in _sys.path:
    _sys.path.insert(0, _T)

import unittest

from _contract_util import SRC, brute_force_row  # noqa: F401

from operator_paged_attention import PagedAttentionOperator, PagedKVCache


def _put(cache, ref, slot, seed, dim=4, heads=1):
    k = [[((seed * 7 + d * 3 + h) % 17 - 8) * 0.15 for d in range(dim)]
         for h in range(heads)]
    v = [[((seed * 11 + d * 5 + h) % 13 - 6) * 0.2 for d in range(dim)]
         for h in range(heads)]
    cache.write_slot(ref, slot, k, v)
    return k, v


class CombinedRulesTest(unittest.TestCase):
    def test_causal_window_and_shared_prefix_together(self):
        cache = PagedKVCache(page_size=4, num_heads=1, head_dim=4)
        op = PagedAttentionOperator(cache)

        shared = cache.allocate("prefix-svc", page_id=0, valid_tokens=4)
        cache.share_page(0)
        flat_k, flat_v = [], []
        for s in range(4):
            k, v = _put(cache, shared, s, s + 1)
            flat_k.append(k)
            flat_v.append(v)

        priv_a = cache.allocate("chat-a", page_id=1, valid_tokens=2)
        for s in range(2):
            k, v = _put(cache, priv_a, s, 20 + s)
            flat_k.append(k)
            flat_v.append(v)

        q = [[0.4, -0.8, 0.2, 0.6]]
        # p=5, W=4 inclusive -> lower=2；因果上界 5
        res = op.decode(
            session_id="chat-a", page_table=[shared, priv_a], kv_length=6,
            query=q, position=5, window=4, record=False,
        )
        self.assertEqual(res.witnesses[0].visible_positions, (2, 3, 4, 5))
        # 共享页只在其有效区间内出现：位置 2/3 -> page0 slot2/3
        self.assertEqual(
            [(v.page_id, v.slot) for v in res.witnesses[0].visible],
            [(0, 2), (0, 3), (1, 0), (1, 1)],
        )
        expected, visible = brute_force_row(q, flat_k, flat_v, 5, window=4)
        self.assertEqual(visible, (2, 3, 4, 5))
        for d in range(4):
            self.assertAlmostEqual(res.rows[0].output[0][d], expected[0][d], places=12)

        # 另一个会话 chat-b 复用同一共享页但有自己的私有延续
        priv_b = cache.allocate("chat-b", page_id=2, valid_tokens=2)
        bk, bv = [], []
        for s in range(2):
            k, v = _put(cache, priv_b, s, 40 + s)
            bk.append(k)
            bv.append(v)
        res_b = op.decode(
            session_id="chat-b", page_table=[shared, priv_b], kv_length=6,
            query=q, position=5, window=4, record=False,
        )
        self.assertEqual(res_b.witnesses[0].visible_positions, (2, 3, 4, 5))
        self.assertEqual(
            [v.page_id for v in res_b.witnesses[0].visible], [0, 0, 2, 2]
        )
        # 共享的两个前缀 KV 权重一致，私有部分不同 -> 总输出不同
        self.assertNotEqual(res.rows[0].output, res_b.rows[0].output)

    def test_duplicate_shared_page_under_window_and_causal(self):
        cache = PagedKVCache(page_size=4, num_heads=1, head_dim=3)
        op = PagedAttentionOperator(cache)
        ref = cache.allocate("prefix-svc", page_id=9, valid_tokens=2)
        cache.share_page(9)
        for s in range(2):
            _put(cache, ref, s, s + 5, dim=3)
        # 页表把同一共享页映射到逻辑块 0 和 1（前缀 fork/重复映射），
        # kv_length=6：有效逻辑位置为 0,1 与 4,5。
        q = [[0.3, -0.3, 0.0]]
        # p=5, W=3 -> lower=3；窗口先裁掉位置 0,1，块1 的 4,5 留下
        res = op.decode(
            session_id="chat-z", page_table=[ref, ref], kv_length=6,
            query=q, position=5, window=3, record=False,
        )
        self.assertEqual(res.witnesses[0].visible_positions, (4, 5))
        self.assertEqual(
            [(v.page_id, v.slot) for v in res.witnesses[0].visible],
            [(9, 0), (9, 1)],
        )
        reasons = {e.position: e.reason for e in res.witnesses[0].excluded}
        # 2,3 是无效填充；0,1 被窗口排除
        self.assertEqual(reasons[2], "invalid_padding")
        self.assertEqual(reasons[3], "invalid_padding")
        self.assertEqual(reasons[0], "outside_window")
        self.assertEqual(reasons[1], "outside_window")

    def test_prefill_and_decode_share_combined_boundary(self):
        cache = PagedKVCache(page_size=4, head_dim=4)
        op = PagedAttentionOperator(cache)
        r0 = cache.allocate("s", page_id=0, valid_tokens=4)
        r1 = cache.allocate("s", page_id=1, valid_tokens=2)
        for s in range(4):
            _put(cache, r0, s, s)
        for s in range(2):
            _put(cache, r1, s, s + 9)
        q = [[0.1, 0.2, -0.1, 0.0]]
        pre = op.prefill(
            session_id="s", page_table=[r0, r1], kv_length=6,
            queries=[q], query_positions=[5], window=4, record=False,
        )
        dec = op.decode(
            session_id="s", page_table=[r0, r1], kv_length=6,
            query=q, position=5, window=4,
        )
        self.assertEqual(pre.rows[0].output, dec.rows[0].output)
        self.assertEqual(
            pre.witnesses[0].visible_positions,
            dec.witnesses[0].visible_positions,
        )


if __name__ == "__main__":
    unittest.main()
