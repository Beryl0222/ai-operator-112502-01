"""分页注意力计算引擎：身份校验、KV 解析、精确路径与分块快速路径。

两条计算路径（:func:`run_exact` 与 :func:`run_block_fast`）共用
:func:`prepare_kv` 的缓存身份/属主校验以及 :func:`masks.build_row_mask`
的遮罩结论，因此它们的缓存身份、遮罩语义与回收行为严格一致，差别只在
softmax 的数值累加方式。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from .cache import PageTable, PagedKVCache, PageRef
from .errors import FullyMaskedRowError, ShapeError
from .linalg import dot, stable_softmax, weighted_vector_sum, zeros_matrix
from .masks import build_row_mask
from .types import KVCoordinate, RowResult, VisibleKV, Witness


@dataclass
class PreparedKV:
    kv_length: int
    num_heads: int
    head_dim: int
    page_size: int
    table: PageTable
    coords: list[KVCoordinate]
    valid_flags: list[bool]
    keys: list[Any]   # [position] -> [head][dim] | None
    values: list[Any]


def normalize_queries(queries: Any, *, phase: str, num_heads: int | None, head_dim: int | None):
    """把多种便捷 Q 形状统一为 ``[q_len][heads][head_dim]``。

    返回 ``(rows, num_heads, head_dim)``。
    """
    if queries is None:
        raise ShapeError("queries must not be None")

    def depth(x) -> int:
        if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], (list, tuple)):
            return 1 + depth(x[0])
        return 0

    d = depth(queries)
    if d == 0:
        # 单个扁平向量 -> 单行单头（或显式声明的 H*D 扁平多头）
        vec = [float(x) for x in queries]
        if num_heads and head_dim and num_heads > 1 and len(vec) == num_heads * head_dim:
            rows = [[vec[h * head_dim:(h + 1) * head_dim] for h in range(num_heads)]]
            return rows, num_heads, head_dim
        dim = len(vec)
        if head_dim is not None and dim != head_dim and num_heads == 1:
            raise ShapeError(f"query length {dim} != head_dim {head_dim}")
        return [[vec]], num_heads or 1, head_dim or dim

    if d == 1:
        # [head][dim] 或单头预填充的 [q_len][dim]，由 phase 决定
        width = len(queries[0])
        if phase == "decode":
            heads = [[[float(x) for x in head] for head in queries]]
            h = len(queries)
            return heads, num_heads or h, head_dim or width
        # prefill：单头 [q][dim]
        rows = [[[float(x) for x in vec]] for vec in queries]
        return rows, num_heads or 1, head_dim or width

    if d == 2:
        # [q][head][dim]
        rows = [[[float(x) for x in head] for head in row] for row in queries]
        h = len(rows[0])
        dim = len(rows[0][0])
        return rows, num_heads or h, head_dim or dim

    raise ShapeError(f"queries have unsupported nesting depth {d + 1}")


def prepare_kv(
    cache: PagedKVCache,
    table: PageTable | Sequence,
    kv_length: int,
    *,
    session_id: str | None = None,
) -> PreparedKV:
    """校验页表覆盖的每个块的身份/属主，并解析全部 KV 位置。

    身份校验先于一切数据读取；代际/租约/属主不匹配直接抛显式异常。
    """
    if kv_length is None or kv_length < 0:
        raise ValueError("kv_length must be a non-negative integer")
    if not isinstance(table, PageTable):
        table = PageTable(cache.page_size, list(table))

    page_size = table.page_size

    # 提交上来的每个页表项都先校验身份/属主——即使它暂时落在 kv_length
    # 之外：租约一旦结束，整份未完成请求的页表都不再是可静默使用的身份。
    block_info: dict[int, tuple[Any, PageRef, int]] = {}
    for b, entry in enumerate(table.entries):
        page, checked = cache.validate(entry, session_id)
        limit = page.valid_tokens
        if checked.valid_tokens is not None:
            limit = min(limit, checked.valid_tokens)
        block_info[b] = (page, checked, max(0, limit))

    required_blocks = (kv_length + page_size - 1) // page_size if kv_length else 0
    if required_blocks > len(table.entries):
        from .errors import PageTableError
        raise PageTableError(
            f"kv_length={kv_length} requires {required_blocks} page-table blocks, "
            f"but table only has {len(table.entries)} entries"
        )

    coords: list[KVCoordinate] = []
    valid_flags: list[bool] = []
    keys: list[Any] = []
    values: list[Any] = []
    num_heads = cache.num_heads
    head_dim = cache.head_dim

    for j in range(kv_length):
        b = j // page_size
        slot = j % page_size
        page, checked, limit = block_info[b]
        coord = KVCoordinate(
            position=j,
            page_id=page.page_id,
            slot=slot,
            generation=page.generation,
            lease_id=page.lease_id,
            session_id=page.session_id,
        )
        coords.append(coord)
        if slot >= limit:
            # 页声明有效区间之外的填充槽：保留坐标用于解释，但标记无效。
            valid_flags.append(False)
            keys.append(None)
            values.append(None)
            continue
        k_slot = page.keys[slot] if isinstance(page.keys, list) else None
        v_slot = page.values[slot] if isinstance(page.values, list) else None
        if not isinstance(k_slot, list) or not isinstance(v_slot, list):
            from .errors import ContractError
            raise ContractError(
                f"page {page.page_id} slot {slot} is within declared valid_tokens="
                f"{page.valid_tokens} but has no stored K/V"
            )
        valid_flags.append(True)
        keys.append(k_slot)
        values.append(v_slot)
        if head_dim is None:
            head_dim = len(k_slot[0])
            num_heads = len(k_slot)

    return PreparedKV(
        kv_length=kv_length,
        num_heads=num_heads or 1,
        head_dim=head_dim or 0,
        page_size=page_size,
        table=table,
        coords=coords,
        valid_flags=valid_flags,
        keys=keys,
        values=values,
    )


def _extra_mask_allows(extra_mask, query_index: int):
    if extra_mask is None:
        return None
    row = extra_mask[query_index]

    def allows(j: int) -> bool:
        return bool(row[j])

    return allows


def _make_witness(
    *,
    request_id: str,
    query_index: int,
    position: int,
    phase: str,
    prepared: PreparedKV,
    row_mask,
    per_head_weights: list[list[float]],
    normalizers: list[float],
    causal: bool,
    window,
    convention: str,
    strategy: str,
) -> Witness:
    visible: list = []
    for idx, j in enumerate(row_mask.eligible):
        coord = prepared.coords[j]
        weights = tuple(per_head_weights[h][idx] for h in range(len(per_head_weights)))
        visible.append(
            VisibleKV(
                position=j,
                page_id=coord.page_id,
                slot=coord.slot,
                weights=weights,
                generation=coord.generation,
                session_id=coord.session_id,
            )
        )
    mask_tuple = tuple(j in row_mask.eligible for j in range(prepared.kv_length))
    return Witness(
        request_id=request_id,
        query_index=query_index,
        query_position=position,
        phase=phase,
        status="ok",
        kv_length=prepared.kv_length,
        causal=causal,
        window=window,
        window_convention=convention,
        window_lower=row_mask.window_lower,
        page_size=prepared.page_size,
        visible=tuple(visible),
        visible_positions=tuple(row_mask.eligible),
        excluded=tuple(row_mask.excluded),
        normalizers=tuple(normalizers),
        mask=mask_tuple,
        strategy=strategy,
    )


def _mask_for_row(prepared: PreparedKV, position: int, *, query_index: int,
                  causal: bool, window, convention: str, extra_mask, dedup_physical: bool):
    return build_row_mask(
        position,
        prepared.kv_length,
        valid_position=lambda j: prepared.valid_flags[j],
        coordinate_of=lambda j: (prepared.coords[j].page_id, prepared.coords[j].slot),
        causal=causal,
        window=window,
        convention=convention,
        allow_position=_extra_mask_allows(extra_mask, query_index),
        dedup_key=(lambda j: prepared.coords[j].physical_key()) if dedup_physical else None,
    )


def _fully_masked_result(request_id, query_index, position, phase, prepared, row_mask,
                         causal, window, convention, strategy, on_fully_masked):
    witness = Witness(
        request_id=request_id,
        query_index=query_index,
        query_position=position,
        phase=phase,
        status="fully_masked",
        kv_length=prepared.kv_length,
        causal=causal,
        window=window,
        window_convention=convention,
        window_lower=row_mask.window_lower,
        page_size=prepared.page_size,
        visible=(),
        visible_positions=(),
        excluded=tuple(row_mask.excluded),
        normalizers=(),
        mask=tuple(False for _ in range(prepared.kv_length)),
        strategy=strategy,
    )
    if on_fully_masked == "raise":
        raise FullyMaskedRowError(request_id, position)
    output = zeros_matrix(prepared.num_heads, prepared.head_dim) if on_fully_masked == "zero" else None
    return RowResult(
        query_index=query_index,
        query_position=position,
        status="fully_masked",
        output=output,
        witness=witness,
    )


def run_exact(
    *,
    request_id: str,
    phase: str,
    queries: list,
    positions: Sequence[int],
    prepared: PreparedKV,
    causal: bool,
    window,
    convention: str,
    scale: float,
    extra_mask=None,
    dedup_physical: bool = False,
    on_fully_masked: str = "status",
) -> list[RowResult]:
    """参考级精确路径：一次性 softmax + fsum 加权，顺序固定。"""
    results: list[RowResult] = []
    h_count = prepared.num_heads
    dim = prepared.head_dim

    for qi, position in enumerate(positions):
        row_mask = _mask_for_row(
            prepared, position, query_index=qi, causal=causal, window=window,
            convention=convention, extra_mask=extra_mask, dedup_physical=dedup_physical,
        )
        if row_mask.is_fully_masked:
            results.append(_fully_masked_result(
                request_id, qi, position, phase, prepared, row_mask,
                causal, window, convention, "exact", on_fully_masked,
            ))
            continue

        eligible = row_mask.eligible
        per_head_weights: list[list[float]] = []
        normalizers: list[float] = []
        output_heads: list[list[float]] = []

        for h in range(h_count):
            q = queries[qi][h] if len(queries[qi]) == h_count else queries[qi][0]
            scores = []
            kv_vectors = []
            for j in eligible:
                k_vec = prepared.keys[j][h] if len(prepared.keys[j]) == h_count else prepared.keys[j][0]
                scores.append(dot(q, k_vec) * scale)
                v_vec = prepared.values[j][h] if len(prepared.values[j]) == h_count else prepared.values[j][0]
                kv_vectors.append(v_vec)
            weights, normalizer, _ = stable_softmax(scores)
            per_head_weights.append(weights)
            normalizers.append(normalizer)
            output_heads.append(weighted_vector_sum(weights, kv_vectors, dim))

        witness = _make_witness(
            request_id=request_id, query_index=qi, position=position, phase=phase,
            prepared=prepared, row_mask=row_mask, per_head_weights=per_head_weights,
            normalizers=normalizers, causal=causal, window=window,
            convention=convention, strategy="exact",
        )
        results.append(RowResult(qi, position, "ok", output_heads, witness))
    return results


def run_block_fast(
    *,
    request_id: str,
    phase: str,
    queries: list,
    positions: Sequence[int],
    prepared: PreparedKV,
    causal: bool,
    window,
    convention: str,
    scale: float,
    block_size: int | None = None,
    extra_mask=None,
    dedup_physical: bool = False,
    on_fully_masked: str = "status",
) -> tuple[list[RowResult], dict[str, Any]]:
    """预填充按块快速路径：对**同一组可见位置**做分块在线 softmax。

    与精确路径的可见集合、身份校验、遮罩语义完全相同；近似仅来自分块
    重标定的浮点累加顺序。``block_size`` 默认等于页大小。
    """
    block_size = block_size or prepared.page_size or 16
    results: list[RowResult] = []
    h_count = prepared.num_heads
    dim = prepared.head_dim
    total_blocks = 0

    for qi, position in enumerate(positions):
        row_mask = _mask_for_row(
            prepared, position, query_index=qi, causal=causal, window=window,
            convention=convention, extra_mask=extra_mask, dedup_physical=dedup_physical,
        )
        if row_mask.is_fully_masked:
            results.append(_fully_masked_result(
                request_id, qi, position, phase, prepared, row_mask,
                causal, window, convention, "block_fast", on_fully_masked,
            ))
            continue

        eligible = list(row_mask.eligible)
        # 在线 softmax 累加器（每头）
        m_run = [float("-inf")] * h_count
        l_run = [0.0] * h_count
        acc = [zeros_matrix(1, dim)[0] for _ in range(h_count)]

        # 先保留每行每头的原始分数，用于结束后构造 witness 权重。
        final_scores: list[list[float]] = [[] for _ in range(h_count)]
        blocks = 0

        for start in range(0, len(eligible), block_size):
            chunk = eligible[start:start + block_size]
            blocks += 1
            for h in range(h_count):
                q = queries[qi][h] if len(queries[qi]) == h_count else queries[qi][0]
                chunk_scores = []
                chunk_vectors = []
                for j in chunk:
                    k_vec = prepared.keys[j][h] if len(prepared.keys[j]) == h_count else prepared.keys[j][0]
                    s = dot(q, k_vec) * scale
                    chunk_scores.append(s)
                    final_scores[h].append(s)
                    v_vec = prepared.values[j][h] if len(prepared.values[j]) == h_count else prepared.values[j][0]
                    chunk_vectors.append(v_vec)

                chunk_max = max(chunk_scores)
                m_new = max(m_run[h], chunk_max)
                if m_run[h] == float("-inf"):
                    old_factor = 0.0  # 首个块：旧累加器为零，不做无意义缩放
                else:
                    old_factor = math.exp(m_run[h] - m_new)
                l_run[h] *= old_factor
                for d in range(dim):
                    acc[h][d] *= old_factor

                # exp(s - m_new)，直接按新最大值累加，最后统一除以 l_run。
                for c, s in enumerate(chunk_scores):
                    w_new = math.exp(s - m_new)
                    l_run[h] += w_new
                    v_vec = chunk_vectors[c]
                    for d in range(dim):
                        acc[h][d] += w_new * v_vec[d]
                m_run[h] = m_new

        output_heads = []
        per_head_weights: list[list[float]] = []
        normalizers: list[float] = []
        for h in range(h_count):
            normalizer = l_run[h]
            output_heads.append([acc[h][d] / normalizer for d in range(dim)])
            weights = [math.exp(s - m_run[h]) / normalizer for s in final_scores[h]]
            per_head_weights.append(weights)
            normalizers.append(normalizer)

        witness = _make_witness(
            request_id=request_id, query_index=qi, position=position, phase=phase,
            prepared=prepared, row_mask=row_mask, per_head_weights=per_head_weights,
            normalizers=normalizers, causal=causal, window=window,
            convention=convention, strategy="block_fast",
        )
        results.append(RowResult(qi, position, "ok", output_heads, witness))
        total_blocks += blocks

    stats = {"block_size": block_size, "kv_blocks_processed": total_blocks}
    return results, stats
