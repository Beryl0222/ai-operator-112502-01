"""因果 / 滑动窗口 / 有效区间遮罩的唯一权威实现。

预填充快速路径、精确预填充与单步解码都必须通过 :func:`build_row_mask`
得到同一套可见位置，不允许各调用方自行拼接遮罩——这是消除“窗口边界在
prefill / decode 给出不同 logits”问题的关键约束。

坐标约定
--------
* KV 与 query 都使用**绝对序列位置** ``p`` / ``j``（从 0 开始）。
* 因果规则：``j <= p`` 时可见（包含 query 自身位置；若该位置的 K/V 尚未
  写入分页缓存，它自然落在 ``kv_length`` 之外，等价于只看之前的 token）。
* 滑动窗口（``window`` 为正整数时）默认采用与主流滑动窗口注意力一致的
  ``"inclusive"`` 边界：``j >= p - window + 1``，即每行最多 ``window`` 个
  key。``window=0`` 或 ``None`` 表示不限制窗口（与资料样例中 chat-b 的
  ``"window": 0`` 约定一致）。
* 还提供 ``convention="exclusive"``（下界 ``p - window``，每行至多
  ``window + 1`` 个 key）以适配其它边界定义；无论选择哪种，prefill 与
  decode 调用的都是同一个函数，边界不会再发生阶段间漂移。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

#: 每行恰好容纳 ``window`` 个 KV：区间 ``[p-window+1, p]``
WINDOW_INCLUSIVE = "inclusive"
#: 区间 ``[p-window, p]``，每行至多 ``window+1`` 个 KV
WINDOW_EXCLUSIVE = "exclusive"
WINDOW_CONVENTIONS = (WINDOW_INCLUSIVE, WINDOW_EXCLUSIVE)

# 排除原因码（同时作为 witness 中的稳定字符串）
REASON_INVALID_PADDING = "invalid_padding"
REASON_CAUSAL = "causal_masked"
REASON_WINDOW = "outside_window"
REASON_CUSTOM = "custom_masked"
REASON_DUPLICATE = "duplicate_physical_page"


def window_lower_bound(query_position: int, window: int | None, convention: str = WINDOW_INCLUSIVE) -> int | None:
    """返回窗口下界（含）；``None`` 表示不启用窗口。"""
    if window is None or window <= 0:
        return None
    if convention == WINDOW_INCLUSIVE:
        return query_position - int(window) + 1
    if convention == WINDOW_EXCLUSIVE:
        return query_position - int(window)
    raise ValueError(f"unknown window convention: {convention!r}, expected one of {WINDOW_CONVENTIONS}")


@dataclass(frozen=True)
class ExcludedPosition:
    """一个未参与计算的 KV 位置及其原因，用于解释输出。"""

    position: int
    reason: str
    page_id: int | None = None
    slot: int | None = None


@dataclass(frozen=True)
class RowMask:
    """单行 query 的完整遮罩结论。

    ``eligible`` 是按绝对位置升序排列的有效可见 KV 位置；
    ``excluded`` 记录上下文范围内被排除的位置与原因。
    """

    query_position: int
    kv_length: int
    eligible: tuple[int, ...]
    excluded: tuple[ExcludedPosition, ...]
    window_lower: int | None

    @property
    def is_fully_masked(self) -> bool:
        return len(self.eligible) == 0

    def reason_of(self, position: int) -> str | None:
        for item in self.excluded:
            if item.position == position:
                return item.reason
        return None


def build_row_mask(
    query_position: int,
    kv_length: int,
    *,
    valid_position: Callable[[int], bool] | None = None,
    coordinate_of: Callable[[int], tuple[int | None, int | None]] | None = None,
    causal: bool = True,
    window: int | None = None,
    convention: str = WINDOW_INCLUSIVE,
    allow_position: Callable[[int], bool] | None = None,
    dedup_key: Callable[[int], object] | None = None,
) -> RowMask:
    """构造一行 query 的权威遮罩。

    参数
    ----
    query_position:
        query 的绝对位置 ``p``。
    kv_length:
        上下文 KV 位置总数，候选位置为 ``0 <= j < kv_length``。
    valid_position:
        分页有效区间判定；返回 ``False`` 表示该位置是无效填充
        （页内 ``valid_tokens`` 之外），不进入归一化分母。
    coordinate_of:
        可选的 ``j -> (page_id, slot)``，仅用于让排除证据带上物理坐标。
    allow_position:
        可选的额外布尔遮罩；返回 ``False`` 的位置以 ``custom_masked`` 排除。
    dedup_key:
        可选的物理身份键；当两个逻辑位置指向同一物理 KV 坐标时，只保留
        第一次出现，后续以 ``duplicate_physical_page`` 排除。默认不启用，
        因为页表声明的是逻辑序列，重复映射本身代表不同逻辑位置。
    """
    lower = window_lower_bound(query_position, window, convention)
    eligible: list[int] = []
    excluded: list[ExcludedPosition] = []
    seen_physical: set[object] = set()

    for j in range(kv_length):
        page_id = slot = None
        if coordinate_of is not None:
            page_id, slot = coordinate_of(j)

        if valid_position is not None and not valid_position(j):
            excluded.append(ExcludedPosition(j, REASON_INVALID_PADDING, page_id, slot))
            continue
        if causal and j > query_position:
            excluded.append(ExcludedPosition(j, REASON_CAUSAL, page_id, slot))
            continue
        if lower is not None and j < lower:
            excluded.append(ExcludedPosition(j, REASON_WINDOW, page_id, slot))
            continue
        if allow_position is not None and not allow_position(j):
            excluded.append(ExcludedPosition(j, REASON_CUSTOM, page_id, slot))
            continue
        if dedup_key is not None:
            key = dedup_key(j)
            if key in seen_physical:
                excluded.append(ExcludedPosition(j, REASON_DUPLICATE, page_id, slot))
                continue
            seen_physical.add(key)
        eligible.append(j)

    return RowMask(
        query_position=query_position,
        kv_length=kv_length,
        eligible=tuple(eligible),
        excluded=tuple(excluded),
        window_lower=lower,
    )


def positions_for_prefill(
    prefix_length: int,
    query_count: int | None = None,
    *,
    start_position: int | None = None,
) -> tuple[int, ...]:
    """给出预填充 query 行的绝对位置。

    默认对前缀本身做预填充：``0 .. prefix_length-1``。
    若提供 ``start_position``，则生成 ``start_position .. +query_count-1``。
    """
    if start_position is None:
        start_position = 0
        if query_count is None:
            query_count = prefix_length
    if query_count is None:
        raise ValueError("query_count is required when start_position is given")
    return tuple(start_position + i for i in range(query_count))
