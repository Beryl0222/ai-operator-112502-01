"""API 适配面：BatchProcessor、对象元数据、位置约定、遮罩原因一致性。"""

import sys as _sys
from pathlib import Path as _Path
_T = str(_Path(__file__).resolve().parent)
if _T not in _sys.path:
    _sys.path.insert(0, _T)

import unittest
from types import SimpleNamespace

from _contract_util import SRC  # noqa: F401

from operator_paged_attention import (
    BatchProcessor,
    PagedAttentionOperator,
    PagedKVCache,
)


class BatchProcessorApiTest(unittest.TestCase):
    def test_queued_batch_process(self):
        cache = PagedKVCache(page_size=4, head_dim=2)
        ref = cache.allocate("s", page_id=0)
        cache.write_slot(ref, 0, [[0.1, 0.2]], [[0.3, 0.4]])
        bp = BatchProcessor(cache=cache)
        bp.add(request_id="a", session_id="s", page_table=[ref], kv_length=1,
               query=[0.1, 0.2], phase="decode")
        self.assertEqual(len(bp), 1)
        batch = bp.process()
        self.assertTrue(batch.all_succeeded)
        self.assertEqual(batch["a"].result.witnesses[0].visible_positions, (0,))
        # 别名指向同一实现函数
        self.assertIs(bp.run.__func__, BatchProcessor.process)
        self.assertIs(bp.submit.__func__, BatchProcessor.process)

    def test_coerce_plain_metadata_object(self):
        cache = PagedKVCache(page_size=4, head_dim=2)
        ref = cache.allocate("s", page_id=0)
        cache.write_slot(ref, 0, [[0.1, 0.2]], [[0.3, 0.4]])
        meta = SimpleNamespace(
            request_id="obj", session_id="s", page_table=[ref],
            context_length=1, query=[0.2, 0.1], query_position=0,
        )
        op = PagedAttentionOperator(cache)
        result = op.decode(meta)
        self.assertEqual(result.request_id, "obj")
        self.assertEqual(result.witnesses[0].query_position, 0)


class DecodePositionConventionTest(unittest.TestCase):
    def test_prefix_only_cache_default_position_after_prefix(self):
        cache = PagedKVCache(page_size=4, head_dim=2)
        op = PagedAttentionOperator(cache)
        ref = cache.allocate("s", page_id=0, valid_tokens=3)
        for s in range(3):
            cache.write_slot(ref, s, [[0.1 * s, 0.2]], [[0.3, 0.4 * s]])
        q = [0.5, -0.5]
        # KV 只有 0..2，新 token 位于 p=3；W=2 inclusive -> lower=2 -> {2}
        res = op.decode(
            session_id="s", page_table=[ref], kv_length=3, query=q,
            window=2, current_token_in_kv=False, record=False,
        )
        self.assertEqual(res.witnesses[0].query_position, 3)
        self.assertEqual(res.witnesses[0].visible_positions, (2,))

    def test_current_token_in_kv_default(self):
        cache = PagedKVCache(page_size=4, head_dim=2)
        op = PagedAttentionOperator(cache)
        ref = cache.allocate("s", page_id=0, valid_tokens=4)
        for s in range(4):
            cache.write_slot(ref, s, [[0.1 * s, 0.2]], [[0.3, 0.4 * s]])
        q = [0.5, -0.5]
        # kv_length=4 且当前 token 已写入 slot3 -> 默认 p=3，W=2 -> {2,3}
        res = op.decode(
            session_id="s", page_table=[ref], kv_length=4, query=q,
            window=2, record=False,
        )
        self.assertEqual(res.witnesses[0].query_position, 3)
        self.assertEqual(res.witnesses[0].visible_positions, (2, 3))

    def test_query_position_singular_alias(self):
        cache = PagedKVCache(page_size=4, head_dim=2)
        op = PagedAttentionOperator(cache)
        ref = cache.allocate("s", page_id=0, valid_tokens=2)
        for s in range(2):
            cache.write_slot(ref, s, [[0.1, 0.2]], [[0.3, 0.4]])
        res = op.attention(
            phase="decode", session_id="s", page_table=[ref], kv_length=2,
            query=[0.1, 0.1], query_position=1,
        )
        self.assertEqual(res.query_positions, (1,))


class FastExactMaskParityTest(unittest.TestCase):
    def test_excluded_reasons_identical_between_paths(self):
        cache = PagedKVCache(page_size=4, num_heads=1, head_dim=3)
        op = PagedAttentionOperator(cache)
        ref0 = cache.allocate("s", page_id=0, valid_tokens=4)
        ref1 = cache.allocate("s", page_id=1, valid_tokens=2)
        for pid, ref, n in [(0, ref0, 4), (1, ref1, 2)]:
            for s in range(n):
                cache.write_slot(ref, s, [[0.1 * (s + pid), 0.2, -0.1]],
                                 [[0.3, 0.4 * s, 0.0]])
        qs = [[[0.1 * i, -0.2, 0.3]] for i in range(6)]
        kw = dict(session_id="s", page_table=[ref0, ref1], kv_length=6,
                  queries=qs, window=3, record=False)
        exact = op.prefill(strategy="exact", **kw)
        fast = op.prefill_fast(**kw)
        for we, wf in zip(exact.witnesses, fast.witnesses):
            self.assertEqual(we.visible_positions, wf.visible_positions)
            self.assertEqual(
                [(e.position, e.reason) for e in we.excluded],
                [(e.position, e.reason) for e in wf.excluded],
            )
            self.assertEqual(we.window_lower, wf.window_lower)
            self.assertEqual(we.mask, wf.mask)


if __name__ == "__main__":
    unittest.main()
