"""物理 KV 页、页表与缓存租约管理。

安全模型
--------
每个 :class:`Page` 都带有：

* ``page_id``：物理页号；
* ``session_id``：当前属主会话；
* ``lease_id``：当前缓存租约标识；
* ``generation``：每次分配/回收都会单调递增的代际号。

请求在提交页表时携带从 :meth:`PagedKVCache.allocate` 得到的
:class:`PageRef`（含 ``generation``/``lease_id``）。算子读取前执行
:meth:`PagedKVCache.validate`：

* 代际不一致 → :class:`~operator_paged_attention.errors.PageReclaimedError`；
* 租约已结束/更换 → :class:`~operator_paged_attention.errors.LeaseExpiredError`；
* 会话不匹配 → :class:`~operator_paged_attention.errors.PageOwnershipError`。

因此页面被回收并分配给新租约（甚至写入“毒”数据）后，旧请求只会显式失败，
绝不会静默把新会话的 K/V 当作旧前缀参与计算。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence
from collections.abc import Mapping
import itertools

from .errors import (
    LeaseExpiredError,
    PageIdentityMismatchError,
    PageNotFoundError,
    PageOwnershipError,
    PageReclaimedError,
    PageTableError,
)


@dataclass(frozen=True)
class PageRef:
    """页表项：逻辑块到物理页某一代身份的不可变引用。"""

    page_id: int
    generation: int | None = None
    lease_id: str | None = None
    #: 调用方声明的该映射有效长度上限；``None`` 时以物理页 ``valid_tokens`` 为准。
    valid_tokens: int | None = None

    def with_valid_tokens(self, valid_tokens: int | None) -> "PageRef":
        return PageRef(self.page_id, self.generation, self.lease_id, valid_tokens)


# 常见别名：部分调用方/资料把页表项称为 BlockRef。
BlockRef = PageRef


@dataclass
class Page:
    """一个物理 KV 页。

    ``keys`` / ``values`` 的存储布局为 ``[slot][head][head_dim]``，
    只有 ``slot < valid_tokens`` 的位置允许被读取。
    """

    page_id: int
    page_size: int
    session_id: str | None = None
    lease_id: str | None = None
    generation: int = 0
    valid_tokens: int = 0
    keys: list | None = None
    values: list | None = None
    reclaimed_count: int = 0
    #: 前缀共享：为 True 时任何持有有效代际身份的会话都可只读访问。
    is_shared: bool = False
    #: 额外被授权只读访问的会话（前缀共享的精细 ACL）。
    authorized_sessions: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        # 每个槽位为 None 或 [head][dim] 的浮点嵌套数组。
        if self.keys is None:
            self.keys = [None] * self.page_size
        if self.values is None:
            self.values = [None] * self.page_size


@dataclass
class Lease:
    lease_id: str
    session_id: str
    active: bool = True
    page_ids: list[int] = field(default_factory=list)
    ended: bool = False


class PagedKVCache:
    """带租约与代际身份的分页 KV 池（纯 Python 内存实现）。"""

    def __init__(self, page_size: int, *, num_heads: int = 1, head_dim: int | None = None):
        if not isinstance(page_size, int) or page_size <= 0:
            raise ValueError("page_size must be a positive integer")
        self.page_size = page_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.pages: dict[int, Page] = {}
        self.leases: dict[str, Lease] = {}
        self._lease_seq = itertools.count(1)
        self._page_seq = itertools.count(0)

    # ------------------------------------------------------------------ 租约
    def begin_lease(self, session_id: str, lease_id: str | None = None) -> Lease:
        if lease_id is None:
            lease_id = f"lease-{next(self._lease_seq)}"
        if lease_id in self.leases and self.leases[lease_id].active:
            raise LeaseExpiredError(f"lease {lease_id!r} already exists and is active")
        lease = Lease(lease_id=lease_id, session_id=session_id, active=True)
        self.leases[lease_id] = lease
        return lease

    def end_lease(self, lease_id: str, *, reclaim: bool = True) -> list[int]:
        """结束租约；默认同时回收其全部页。返回被回收的 page_id 列表。"""
        lease = self.leases.get(lease_id)
        if lease is None:
            raise LeaseExpiredError(f"unknown lease {lease_id!r}")
        lease.active = False
        lease.ended = True
        reclaimed: list[int] = []
        if reclaim:
            for page_id in list(lease.page_ids):
                self.reclaim_page(page_id, reason="lease-ended")
                reclaimed.append(page_id)
        return reclaimed

    # ------------------------------------------------------------------ 页分配
    def _new_page_id(self) -> int:
        while True:
            pid = next(self._page_seq)
            if pid not in self.pages:
                return pid

    def allocate(
        self,
        session_id: str,
        *,
        lease_id: str | None = None,
        page_id: int | None = None,
        valid_tokens: int = 0,
    ) -> PageRef:
        """分配/占用一个页并返回带当前代际身份的 :class:`PageRef`。

        若指定已存在的 ``page_id``（回收后重新分配的典型路径），generation
        会递增，从而使旧 :class:`PageRef` 立即失效。
        """
        lease: Lease | None = None
        if lease_id is not None:
            lease = self.leases.get(lease_id)
            if lease is None or not lease.active:
                raise LeaseExpiredError(f"lease {lease_id!r} is not active")
            if lease.session_id != session_id:
                raise PageOwnershipError(
                    f"lease {lease_id!r} belongs to {lease.session_id!r}, not {session_id!r}"
                )

        if page_id is None:
            page_id = self._new_page_id()

        existing = self.pages.get(page_id)
        if existing is not None and existing.session_id is not None:
            # 允许显式覆盖分配，但必须先回收，避免调用方绕过代际机制。
            raise PageOwnershipError(
                f"page {page_id} is still occupied by {existing.session_id!r}; "
                "reclaim it before reallocating"
            )

        if existing is None:
            page = Page(page_id=page_id, page_size=self.page_size)
            self.pages[page_id] = page
        else:
            page = existing
            # 回收壳可能保留旧数组；重新分配时整体换成空槽，杜绝残留读取。
            page.keys = [None] * self.page_size
            page.values = [None] * self.page_size

        page.session_id = session_id
        page.lease_id = lease_id
        page.generation += 1
        page.valid_tokens = valid_tokens
        if lease is not None and page_id not in lease.page_ids:
            lease.page_ids.append(page_id)
        return PageRef(page_id=page_id, generation=page.generation, lease_id=lease_id)

    def reclaim_page(self, page_id: int, *, reason: str = "reclaimed") -> Page:
        """回收物理页：递增代际、摘除属主并清空 K/V 数据。"""
        page = self.pages.get(page_id)
        if page is None:
            raise PageNotFoundError(f"page {page_id} does not exist")
        page.generation += 1
        page.reclaimed_count += 1
        page.session_id = None
        page.lease_id = None
        page.valid_tokens = 0
        # 用 None 覆盖，旧数据不再可达；即使有调用方持有旧 Page 对象引用，
        # 正式读取路径也必须先通过 generation 校验。
        page.keys = [None] * self.page_size
        page.values = [None] * self.page_size
        return page

    # ------------------------------------------------------------------ 写入
    def _normalize_kv(self, tensor: Sequence, where: str) -> list[list[float]]:
        """把 ``[H][D]`` 或单头 ``[D]`` 统一为 ``[H][D]``。"""
        if tensor is None:
            raise ValueError(f"{where} must not be None")
        first = tensor[0] if len(tensor) > 0 else None
        if isinstance(first, (list, tuple)):
            return [[float(x) for x in head] for head in tensor]
        return [[float(x) for x in tensor]]

    def write_slot(
        self,
        ref: PageRef | int,
        slot: int,
        key: Sequence[float] | Sequence[Sequence[float]],
        value: Sequence[float] | Sequence[Sequence[float]],
        *,
        session_id: str | None = None,
    ) -> PageRef:
        """向页内一个槽位写入 K/V，并把 ``valid_tokens`` 扩展到 ``slot+1``。"""
        page, checked_ref = self.validate(ref, session_id)
        if not isinstance(slot, int) or not (0 <= slot < self.page_size):
            raise ValueError(f"slot must be in [0, {self.page_size}), got {slot}")
        k = self._normalize_kv(key, "key")
        v = self._normalize_kv(value, "value")
        if self.head_dim is None:
            self.head_dim = len(k[0])
        if self.num_heads == 1 and len(k) > 1:
            self.num_heads = len(k)
        if len(k) != self.num_heads or len(v) != self.num_heads:
            raise ValueError(f"expected {self.num_heads} heads, got k={len(k)} v={len(v)}")
        if len(k[0]) != self.head_dim or len(v[0]) != self.head_dim:
            raise ValueError(f"expected head_dim={self.head_dim}")

        needs_alloc = (
            not isinstance(page.keys, list)
            or len(page.keys) != self.page_size
            or not isinstance(page.keys[0], list)
        )
        if needs_alloc:
            # 按已知 head 维度初始化整页；未写槽位保持零向量但永远不会被
            # 读取，因为槽位可见性由 valid_tokens / 遮罩决定。
            page.keys = [
                [[0.0] * self.head_dim for _ in range(self.num_heads)]
                for _ in range(self.page_size)
            ]
            page.values = [
                [[0.0] * self.head_dim for _ in range(self.num_heads)]
                for _ in range(self.page_size)
            ]
        page.keys[slot] = k
        page.values[slot] = v
        if slot + 1 > page.valid_tokens:
            page.valid_tokens = slot + 1
        return checked_ref

    # 便捷别名
    write = write_slot

    def append_token(
        self,
        table: "PageTable | Sequence[PageRef | int]",
        logical_position: int,
        key,
        value,
        *,
        session_id: str | None = None,
        lease_id: str | None = None,
    ) -> tuple[PageRef, int]:
        """把一个新 token 的 K/V 写到其逻辑位置对应的页内槽位。

        返回 ``(page_ref, slot)``。页表长度不足时不会自动分配新页——批处理器
        应先 :meth:`allocate` 并扩展页表；这样“页表提交什么就用什么”的边界
        更清晰。
        """
        table = table if isinstance(table, PageTable) else PageTable(self.page_size, list(table))
        entry = table.ref_for_position(logical_position)
        slot = logical_position % self.page_size
        ref = entry if isinstance(entry, PageRef) else PageRef(int(entry), lease_id=lease_id)
        self.write_slot(ref, slot, key, value, session_id=session_id)
        return ref, slot

    # ------------------------------------------------------------------ 校验/读取
    def validate(self, ref: PageRef | int, session_id: str | None = None) -> tuple[Page, PageRef]:
        """校验页身份与属主，返回 ``(page, normalized_ref)``。

        这是所有数值读取前的唯一门禁；校验失败抛显式异常，绝不返回
        新租约页的数据。
        """
        if isinstance(ref, int):
            ref = PageRef(page_id=ref)
        elif not isinstance(ref, PageRef):
            # 容忍其它带有 page_id/generation 属性的对象
            ref = PageRef(
                page_id=int(getattr(ref, "page_id")),
                generation=getattr(ref, "generation", None),
                lease_id=getattr(ref, "lease_id", None),
                valid_tokens=getattr(ref, "valid_tokens", None),
            )

        page = self.pages.get(ref.page_id)
        if page is None:
            raise PageNotFoundError(f"referenced page {ref.page_id} does not exist")

        if ref.generation is not None and page.generation != ref.generation:
            raise PageReclaimedError(
                f"page {ref.page_id} generation mismatch: request holds generation="
                f"{ref.generation}, cache now has {page.generation} (page was reclaimed/reallocated)"
            )
        if ref.lease_id is not None and page.lease_id != ref.lease_id:
            raise LeaseExpiredError(
                f"page {ref.page_id} lease mismatch: request holds lease={ref.lease_id!r}, "
                f"page current lease={page.lease_id!r}"
            )
        if ref.lease_id is not None:
            lease = self.leases.get(ref.lease_id)
            if lease is not None and not lease.active:
                raise LeaseExpiredError(f"lease {ref.lease_id!r} has ended")
        if (
            session_id is not None
            and page.session_id is not None
            and page.session_id != session_id
            and not page.is_shared
            and session_id not in page.authorized_sessions
        ):
            raise PageOwnershipError(
                f"session {session_id!r} attempted to read page {ref.page_id} "
                f"owned by {page.session_id!r}"
            )
        return page, ref

    def share_page(self, page_id: int, *sessions: str, shared: bool = True) -> PageRef:
        """把一个已分配页标记为前缀共享，或授权给指定会话只读访问。"""
        page = self.pages.get(page_id)
        if page is None:
            raise PageNotFoundError(f"page {page_id} does not exist")
        page.is_shared = shared
        page.authorized_sessions.update(sessions)
        return PageRef(page_id, page.generation, page.lease_id, None)

    def get_kv(self, ref: PageRef | int, slot: int, session_id: str | None = None):
        page, checked = self.validate(ref, session_id)
        limit = page.valid_tokens if checked.valid_tokens is None else min(page.valid_tokens, checked.valid_tokens)
        if slot >= limit:
            # 无效填充：返回 None，由上层遮罩排除，而不是读取残留数组。
            return None, None
        return page.keys[slot], page.values[slot]


class PageTable:
    """逻辑块索引 -> :class:`PageRef` 的有序页表。"""

    def __init__(self, page_size: int, entries: Sequence[PageRef | int | dict | Any] = ()):
        self.page_size = page_size
        self.entries: list[PageRef] = [self._coerce(e) for e in entries]

    @staticmethod
    def _coerce(entry) -> PageRef:
        if isinstance(entry, PageRef):
            return entry
        if isinstance(entry, int):
            return PageRef(entry)
        if isinstance(entry, dict):
            return PageRef(
                page_id=int(entry["page_id"] if "page_id" in entry else entry.get("id")),
                generation=entry.get("generation"),
                lease_id=entry.get("lease_id"),
                valid_tokens=entry.get("valid_tokens"),
            )
        page_id = int(getattr(entry, "page_id"))
        return PageRef(
            page_id=page_id,
            generation=getattr(entry, "generation", None),
            lease_id=getattr(entry, "lease_id", None),
            valid_tokens=getattr(entry, "valid_tokens", None),
        )

    def ref_for_position(self, position: int) -> PageRef:
        block = position // self.page_size
        if block < 0 or block >= len(self.entries):
            raise PageTableError(
                f"logical block {block} (position {position}) is not mapped; "
                f"page table has {len(self.entries)} entries"
            )
        return self.entries[block]

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def to_list(self) -> list[dict]:
        return [
            {"page_id": e.page_id, "generation": e.generation,
             "lease_id": e.lease_id, "valid_tokens": e.valid_tokens}
            for e in self.entries
        ]


def _extract_kv_arrays(record):
    """从多种页记录形态中取出 ``(keys, values)`` 原始嵌套数组。"""
    if isinstance(record, Page):
        return record.keys, record.values, record
    if isinstance(record, dict):
        keys = record.get("keys", record.get("key", record.get("k", record.get("key_cache"))))
        values = record.get("values", record.get("value", record.get("v", record.get("value_cache"))))
        return keys, values, None
    if isinstance(record, (tuple, list)) and len(record) == 2:
        return record[0], record[1], None
    keys = getattr(record, "keys", None) or getattr(record, "key_cache", None)
    values = getattr(record, "values", None) or getattr(record, "value_cache", None)
    return keys, values, None


def build_cache_from_pages(
    page_size: int,
    pages: Any,
    *,
    num_heads: int = 1,
    head_dim: int | None = None,
) -> PagedKVCache:
    """把调用方自管的页数据包装成带校验能力的 :class:`PagedKVCache`。

    ``pages`` 可以是：

    * 已经构造好的 :class:`PagedKVCache`（原样返回）；
    * ``{page_id: Page | dict | (keys, values)}`` 映射；
    * 以 page_id 为下标的列表。
    """
    if isinstance(pages, PagedKVCache):
        return pages

    cache = PagedKVCache(page_size, num_heads=num_heads, head_dim=head_dim)

    if isinstance(pages, Sequence) and not isinstance(pages, (str, bytes)) and pages and \
            isinstance(pages[0], (list, tuple)) and len(pages[0]) == page_size:
        # 单个 [page_size][...] 页数组的情况不应到这里（外层通常是页集合），
        # 保留给单页简写。
        pages = {0: pages}

    if isinstance(pages, Mapping):
        items = pages.items()
    elif isinstance(pages, Sequence):
        items = enumerate(pages)
    else:
        raise TypeError(f"unsupported pages container: {type(pages)!r}")

    for raw_id, record in items:
        if record is None:
            continue
        page_id = int(raw_id)
        keys, values, page_obj = _extract_kv_arrays(record)

        if page_obj is not None:
            cache.pages[page_id] = page_obj
            continue

        def _meta(*names, default=None):
            if isinstance(record, dict):
                for n in names:
                    if n in record:
                        return record[n]
            else:
                for n in names:
                    if hasattr(record, n):
                        return getattr(record, n)
            return default

        valid_tokens = _meta("valid_tokens", "valid_length", "length", default=None)
        generation = _meta("generation", "gen", default=0) or 0
        lease_id = _meta("lease_id", "lease", default=None)
        session_id = _meta("session_id", "session", "owner", default=None)

        # 推断有效长度与维度
        if valid_tokens is None:
            if isinstance(keys, list):
                valid_tokens = sum(1 for slot in keys[:page_size] if slot is not None)
            else:
                valid_tokens = 0

        page = Page(
            page_id=page_id,
            page_size=page_size,
            session_id=session_id,
            lease_id=lease_id,
            generation=int(generation),
            valid_tokens=int(valid_tokens),
        )
        # 规范化 K/V 到 [slot][head][dim]
        norm_keys: list = [None] * page_size
        norm_values: list = [None] * page_size
        inferred_dim = head_dim
        inferred_heads = num_heads
        for s in range(min(page_size, len(keys or []))):
            k_slot = keys[s]
            v_slot = values[s] if values is not None else None
            if k_slot is None:
                continue
            if k_slot and isinstance(k_slot[0], (list, tuple)):
                k_heads = [[float(x) for x in h] for h in k_slot]
                v_heads = [[float(x) for x in h] for h in v_slot]
            else:
                k_heads = [[float(x) for x in k_slot]]
                v_heads = [[float(x) for x in v_slot]]
            norm_keys[s] = k_heads
            norm_values[s] = v_heads
            inferred_heads = len(k_heads)
            inferred_dim = len(k_heads[0])
        page.keys = norm_keys
        page.values = norm_values
        cache.pages[page_id] = page
        if inferred_dim is not None:
            cache.head_dim = inferred_dim
        cache.num_heads = inferred_heads

    return cache
