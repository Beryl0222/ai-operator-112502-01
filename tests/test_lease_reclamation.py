"""租约结束/页回收：已完成 token 可查、未完成请求显式失败、不读新租约数据。"""

import sys as _sys
from pathlib import Path as _Path
_T = str(_Path(__file__).resolve().parent)
if _T not in _sys.path:
    _sys.path.insert(0, _T)

import unittest

from _contract_util import SRC  # noqa: F401

from operator_paged_attention import (
    LeaseExpiredError,
    PageReclaimedError,
    PagedAttentionOperator,
    PagedKVCache,
    RequestState,
)


def _fill(cache, ref, slots, dim=3, value_scale=1.0):
    for s in range(slots):
        k = [[value_scale * (s + 1) if d == 0 else (0.2 if d == 1 else -0.3)
              for d in range(dim)]]
        v = [[value_scale * (s + 2) if d == 0 else (0.1 if d == 1 else 0.0)
              for d in range(dim)]]
        cache.write_slot(ref, s, k, v)


class LeaseIdentityTest(unittest.TestCase):
    def test_generation_bump_makes_old_ref_fail(self):
        cache = PagedKVCache(page_size=4, head_dim=3)
        op = PagedAttentionOperator(cache)
        old = cache.allocate("s", page_id=0)
        _fill(cache, old, 2)

        # 旧引用在回收前可用
        ok = op.decode(page_table=[old], kv_length=2, query=[[0.1, 0.2, 0.3]],
                       position=1, session_id="s", record=False)
        self.assertTrue(ok.all_ok)

        cache.reclaim_page(0)
        with self.assertRaises(PageReclaimedError):
            op.decode(page_table=[old], kv_length=2, query=[[0.1, 0.2, 0.3]],
                      position=1, session_id="s", record=False)

    def test_reallocated_page_poison_not_silently_used(self):
        cache = PagedKVCache(page_size=4, head_dim=3)
        op = PagedAttentionOperator(cache)
        old = cache.allocate("chat-a", page_id=0)
        _fill(cache, old, 4, value_scale=1.0)
        snapshot = op.decode(
            page_table=[old], kv_length=4, query=[[0.5, -0.5, 0.5]],
            position=3, session_id="chat-a", record=False,
        ).rows[0].output

        cache.reclaim_page(0)
        new_lease = cache.begin_lease("chat-z", "lz")
        new = cache.allocate("chat-z", lease_id="lz", page_id=0, valid_tokens=4)
        for s in range(4):
            cache.write_slot(new, s, [[1e9, 1e9, 1e9]], [[-1e9, -1e9, -1e9]],
                            session_id="chat-z")

        # 旧租约引用必须因代际不匹配而失败，而不是算出毒数据结果
        with self.assertRaises(PageReclaimedError):
            op.decode(page_table=[old], kv_length=4, query=[[0.5, -0.5, 0.5]],
                      position=3, session_id="chat-a", record=False)

        # 新租约引用可以正常使用新数据
        new_res = op.decode(page_table=[new], kv_length=4,
                            query=[[0.5, -0.5, 0.5]], position=3,
                            session_id="chat-z", record=False)
        self.assertTrue(all(abs(x) > 1e6 for x in new_res.rows[0].flat_output()[:1]))
        # 旧快照没有被改变
        self.assertLess(max(abs(x) for row in snapshot for x in row), 10.0)

    def test_ended_lease_ref_fails(self):
        cache = PagedKVCache(page_size=4, head_dim=2)
        op = PagedAttentionOperator(cache)
        lease = cache.begin_lease("s", "L1")
        ref = cache.allocate("s", lease_id="L1", page_id=2)
        _fill(cache, ref, 1, dim=2)
        cache.end_lease("L1", reclaim=False)
        with self.assertRaises(LeaseExpiredError):
            op.decode(page_table=[ref], kv_length=1, query=[[0.1, 0.2]],
                      position=0, session_id="s", record=False)

    def test_block_fast_prefill_has_same_identity_check(self):
        cache = PagedKVCache(page_size=4, head_dim=2)
        op = PagedAttentionOperator(cache)
        ref = cache.allocate("s", page_id=1)
        _fill(cache, ref, 2, dim=2)
        cache.reclaim_page(1)
        # 近似预填充路径的缓存身份/回收行为必须与解码严格路径一致
        with self.assertRaises(PageReclaimedError):
            op.prefill_fast(page_table=[ref], kv_length=2,
                            queries=[[[0.1, 0.2]], [[0.3, 0.4]]],
                            session_id="s", record=False)


class JobLedgerLifecycleTest(unittest.TestCase):
    def _scenario(self):
        cache = PagedKVCache(page_size=4, num_heads=1, head_dim=3)
        op = PagedAttentionOperator(cache)
        lease = cache.begin_lease("chat-a", "la")
        r0 = cache.allocate("chat-a", lease_id="la", page_id=0)
        r1 = cache.allocate("chat-a", lease_id="la", page_id=1)
        _fill(cache, r0, 4)
        _fill(cache, r1, 2)
        return cache, op, [r0, r1]

    def test_completed_tokens_survive_reclamation_but_next_step_fails(self):
        cache, op, table = self._scenario()
        job = op.start_request(
            "chat-a", session_id="chat-a", page_table=table, kv_length=6,
            window=5, planned_outputs=8,
        )
        qs = [[[0.1 * i, 0.2, -0.1]] for i in range(6)]
        job.prefill(qs, kv_length=6)
        self.assertEqual(len(job.completed_positions), 6)

        k6 = [[0.7, 0.1, -0.2]]
        v6 = [[1.7, 0.2, 0.0]]
        job.decode([[0.3, 0.3, 0.3]], position=6, append_key=k6, append_value=v6)
        self.assertEqual(job.completed_positions, tuple(range(7)))
        token6_before = job.get_token(6)
        output6 = token6_before.output

        # 租约结束 + 回收，并把第二页交给毒会话
        cache.end_lease("la", reclaim=True)
        new_lease = cache.begin_lease("chat-z", "lz")
        new_ref = cache.allocate("chat-z", lease_id="lz", page_id=1, valid_tokens=4)
        for s in range(4):
            cache.write_slot(new_ref, s, [[1e8] * 3], [[1e8] * 3], session_id="chat-z")

        with self.assertRaises(PageReclaimedError):
            job.decode([[0.4, 0.1, 0.2]], position=7, kv_length=8)

        self.assertEqual(job.state, RequestState.FAILED)
        # 粘性：再次尝试仍然显式失败，不会因为新页表状态而静默
        with self.assertRaises(PageReclaimedError):
            job.decode([[0.4, 0.1, 0.2]], position=7, kv_length=8)

        # 已完成 token（含 witness）仍可查询，且输出是未被污染的快照
        for p in range(7):
            token = op.get_token("chat-a", p)
            self.assertEqual(token.position, p)
            self.assertIsNotNone(token.witness)
        token6_after = op.get_token("chat-a", 6)
        self.assertEqual(token6_after.output, output6)
        self.assertEqual(token6_after.witness.visible_positions, (2, 3, 4, 5, 6))

    def test_completed_request_still_queryable_after_lease_end(self):
        cache, op, table = self._scenario()
        job = op.start_request(
            "chat-a", session_id="chat-a", page_table=table, kv_length=6,
            planned_outputs=6,
        )
        qs = [[[0.1 * i, 0.2, -0.1]] for i in range(6)]
        job.prefill(qs, kv_length=6)
        self.assertEqual(job.state, RequestState.COMPLETED)

        cache.end_lease("la", reclaim=True)
        completed = op.get_completed("chat-a")
        self.assertEqual(len(completed), 6)
        # 账本快照可 JSON 化其 witness
        for t in completed:
            t.witness.to_dict()

    def test_batch_captures_lease_failure_per_request(self):
        cache = PagedKVCache(page_size=4, head_dim=2)
        op = PagedAttentionOperator(cache)
        good_ref = cache.allocate("g", page_id=0)
        bad_ref = cache.allocate("b", page_id=1)
        for ref in (good_ref, bad_ref):
            cache.write_slot(ref, 0, [[0.1, 0.2]], [[0.3, 0.4]])
        cache.reclaim_page(1)

        batch = op.submit_batch([
            dict(request_id="ok", session_id="g", page_table=[good_ref], kv_length=1,
                 query=[0.1, 0.2], phase="decode", record=False),
            dict(request_id="dead", session_id="b", page_table=[bad_ref], kv_length=1,
                 query=[0.1, 0.2], phase="decode", record=False),
        ])
        self.assertFalse(batch.all_succeeded)
        self.assertEqual(batch["ok"].status, "fulfilled")
        self.assertTrue(batch["ok"].ok)
        self.assertEqual(batch["dead"].status, "failed")
        self.assertIsInstance(batch["dead"].error, PageReclaimedError)

        from operator_paged_attention import BatchProcessingError
        with self.assertRaises(BatchProcessingError):
            op.submit_batch([
                dict(request_id="dead2", session_id="b", page_table=[bad_ref],
                     kv_length=1, query=[0.1, 0.2], phase="decode"),
            ], strict=True)


if __name__ == "__main__":
    unittest.main()
