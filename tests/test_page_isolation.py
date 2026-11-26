"""跨会话隔离、属主校验与前缀共享测试。"""

import sys as _sys
from pathlib import Path as _Path
_T = str(_Path(__file__).resolve().parent)
if _T not in _sys.path:
    _sys.path.insert(0, _T)

import unittest

from _contract_util import SRC  # noqa: F401

from operator_paged_attention import (
    PageOwnershipError,
    PagedAttentionOperator,
    PagedKVCache,
)


def _kv(seed_val, heads=1, dim=4, scale=1.0):
    return (
        [[scale * (seed_val + 0.1 * d + 0.01 * h) for d in range(dim)] for h in range(heads)],
        [[scale * (seed_val + 0.2 * d - 0.01 * h) for d in range(dim)] for h in range(heads)],
    )


class CrossPageIsolationTest(unittest.TestCase):
    def setUp(self):
        self.cache = PagedKVCache(page_size=4, num_heads=1, head_dim=4)
        self.op = PagedAttentionOperator(self.cache)
        # 两个会话的物理页故意交错分配
        self.ref_a0 = self.cache.allocate("chat-a", page_id=0)
        self.ref_b0 = self.cache.allocate("chat-b", page_id=1)
        self.ref_a1 = self.cache.allocate("chat-a", page_id=2)
        for slot in range(4):
            k, v = _kv(slot)
            self.cache.write_slot(self.ref_a0, slot, k, v, session_id="chat-a")
        for slot in range(2):
            k, v = _kv(100 + slot, scale=5.0)
            self.cache.write_slot(self.ref_a1, slot, k, v, session_id="chat-a")
        for slot in range(4):
            # chat-b 的数据极其极端
            k, v = _kv(50 + slot, scale=1e6)
            self.cache.write_slot(self.ref_b0, slot, k, v, session_id="chat-b")

    def test_shuffled_cross_page_reads_stay_in_session(self):
        table = [self.ref_a0, self.ref_a1]
        q = [[0.3, -0.7, 0.2, 0.9]]
        res = self.op.decode(
            session_id="chat-a", page_table=table, kv_length=6,
            query=q, position=5, window=None, record=False,
        )
        visible = res.witnesses[0].visible
        # 位置 0..3 -> page0，4..5 -> page2；任何可见坐标都不属于 page1(chat-b)
        self.assertEqual([v.page_id for v in visible], [0, 0, 0, 0, 2, 2])
        self.assertTrue(all(v.session_id == "chat-a" for v in visible))
        # 数值有限且未被 chat-b 的 1e6 污染
        for head in res.rows[0].output:
            for x in head:
                self.assertTrue(abs(x) < 1e3)

    def test_mutating_other_session_page_does_not_change_output(self):
        table = [self.ref_a0, self.ref_a1]
        q = [[0.3, -0.7, 0.2, 0.9]]
        before = self.op.decode(
            session_id="chat-a", page_table=table, kv_length=6,
            query=q, position=5, record=False,
        ).rows[0].output

        # 覆写 chat-b 的整页为毒值
        poison = [[1e12] * 4]
        for slot in range(4):
            self.cache.pages[1].keys[slot] = poison
            self.cache.pages[1].values[slot] = poison

        after = self.op.decode(
            session_id="chat-a", page_table=table, kv_length=6,
            query=q, position=5, record=False,
        ).rows[0].output
        self.assertEqual(before, after)

    def test_ownership_violation_raises(self):
        # chat-a 的页表中混入 chat-b 的物理页且未共享 -> 显式越权错误
        with self.assertRaises(PageOwnershipError):
            self.op.prefill(
                session_id="chat-a",
                page_table=[self.ref_a0, self.ref_b0],
                kv_length=8, queries=[[[0.1] * 4]], query_positions=[7],
                record=False,
            )


class PrefixSharingTest(unittest.TestCase):
    def test_shared_prefix_page_read_by_two_sessions(self):
        cache = PagedKVCache(page_size=4, num_heads=1, head_dim=4)
        op = PagedAttentionOperator(cache)
        shared_ref = cache.allocate("prefix-svc", page_id=7, valid_tokens=4)
        for slot in range(4):
            k, v = _kv(20 + slot)
            cache.write_slot(shared_ref, slot, k, v, session_id="prefix-svc")
        cache.share_page(7, "chat-a", "chat-b")

        q = [[0.5, 0.5, -0.5, -0.5]]
        for sid in ("chat-a", "chat-b"):
            res = op.decode(
                session_id=sid, page_table=[shared_ref], kv_length=4,
                query=q, position=3, record=False,
            )
            self.assertEqual(res.witnesses[0].visible_positions, (0, 1, 2, 3))
            self.assertTrue(all(v.page_id == 7 for v in res.witnesses[0].visible))

    def test_unshared_prefix_denied(self):
        cache = PagedKVCache(page_size=4, head_dim=4)
        op = PagedAttentionOperator(cache)
        ref = cache.allocate("prefix-svc", page_id=7)
        with self.assertRaises(PageOwnershipError):
            op.decode(
                session_id="chat-x", page_table=[ref], kv_length=1,
                query=[[0.1] * 4], position=0, record=False,
            )

    def test_same_prefix_in_two_requests_matches_reference(self):
        # 两个请求共享同一前缀页，但各自追加不同的私有页
        cache = PagedKVCache(page_size=4, num_heads=1, head_dim=3)
        op = PagedAttentionOperator(cache)
        pre = cache.allocate("svc", page_id=0, valid_tokens=4)
        cache.share_page(0)
        for s in range(4):
            k, v = _kv(s, dim=3)
            cache.write_slot(pre, s, k, v)

        priv_a = cache.allocate("a", page_id=1, valid_tokens=1)
        priv_b = cache.allocate("b", page_id=2, valid_tokens=1)
        ka, va = _kv(9, dim=3)
        kb, vb = _kv(-9, dim=3)
        cache.write_slot(priv_a, 0, ka, va, session_id="a")
        cache.write_slot(priv_b, 0, kb, vb, session_id="b")

        qa = [[0.2, -0.4, 0.6]]
        ra = op.decode(session_id="a", page_table=[pre, priv_a], kv_length=5,
                       query=qa, position=4, record=False)
        rb = op.decode(session_id="b", page_table=[pre, priv_b], kv_length=5,
                       query=qa, position=4, record=False)
        # 前 4 个前缀 KV 相同，第 5 个私有 KV 不同 -> 输出必须不同，
        # 且 witness 明确显示各自只多看到自己的私有页。
        self.assertEqual([v.page_id for v in ra.witnesses[0].visible], [0, 0, 0, 0, 1])
        self.assertEqual([v.page_id for v in rb.witnesses[0].visible], [0, 0, 0, 0, 2])
        self.assertNotEqual(ra.rows[0].output, rb.rows[0].output)


if __name__ == "__main__":
    unittest.main()
