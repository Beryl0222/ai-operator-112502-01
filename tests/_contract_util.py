"""测试公共工具：确保直接从 src/ 导入，并提供独立暴力参考实现。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import operator_paged_attention  # noqa: E402,F401


def brute_force_row(q, flat_k, flat_v, position, *, window=None, causal=True,
                    convention="inclusive", scale=None):
    """独立编写的扁平参考注意力（fsum 严格求和）。

    返回 ``(output_heads_or_None, visible_positions)``；无可见 KV 时
    output 为 ``None``。
    """
    heads = len(q)
    dim = len(q[0])
    if scale is None:
        scale = 1.0 / math.sqrt(dim)
    lower = None
    if window and window > 0:
        lower = position - window + 1 if convention == "inclusive" else position - window

    visible = []
    for j in range(len(flat_k)):
        if causal and j > position:
            continue
        if lower is not None and j < lower:
            continue
        visible.append(j)
    if not visible:
        return None, ()

    out = [[0.0] * dim for _ in range(heads)]
    for h in range(heads):
        scores = [
            math.fsum(q[h][d] * flat_k[j][h][d] for d in range(dim)) * scale
            for j in visible
        ]
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        normalizer = math.fsum(exps)
        weights = [e / normalizer for e in exps]
        for d in range(dim):
            out[h][d] = math.fsum(
                weights[i] * flat_v[visible[i]][h][d] for i in range(len(visible))
            )
    return out, tuple(visible)


def make_random_case(seed, *, page_size=4, num_heads=2, head_dim=6, length=10,
                     window=None, convention="inclusive", session_id="s"):
    """构造物理页号乱序、带填充毒值的随机分页场景。"""
    import random
    from operator_paged_attention import PagedKVCache

    rng = random.Random(seed)
    blocks = (length + page_size - 1) // page_size
    physical_ids = list(range(blocks + 3))
    rng.shuffle(physical_ids)

    cache = PagedKVCache(page_size, num_heads=num_heads, head_dim=head_dim)
    table = []
    flat_k, flat_v = [], []
    for b in range(blocks):
        pid = physical_ids[b]
        valid = min(page_size, length - b * page_size)
        ref = cache.allocate(session_id, page_id=pid, valid_tokens=valid)
        table.append(ref)
        for slot in range(valid):
            k = [[rng.uniform(-1, 1) for _ in range(head_dim)] for _ in range(num_heads)]
            v = [[rng.uniform(-1, 1) for _ in range(head_dim)] for _ in range(num_heads)]
            cache.write_slot(ref, slot, k, v, session_id=session_id)
            if b * page_size + slot < length:
                flat_k.append(k)
                flat_v.append(v)
        for slot in range(valid, page_size):
            cache.pages[pid].keys[slot] = [[1e7] * head_dim for _ in range(num_heads)]
            cache.pages[pid].values[slot] = [[1e7] * head_dim for _ in range(num_heads)]

    queries = [
        [[rng.uniform(-1, 1) for _ in range(head_dim)] for _ in range(num_heads)]
        for _ in range(length)
    ]
    return cache, table, flat_k, flat_v, queries
