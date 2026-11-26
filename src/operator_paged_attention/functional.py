"""函数式入口 ``paged_attention(...)``。

尽量宽容地接受常见调用形态（位置参数、``q``/``query`` 别名、独立的
``key_cache``/``value_cache`` 稠密页缓存等），但所有计算都收敛到
:class:`operator_paged_attention.operator.PagedAttentionOperator` 的同一套
身份校验与遮罩语义上。
"""

from __future__ import annotations

from typing import Any, Sequence

from .cache import PagedKVCache, build_cache_from_pages
from .operator import AttentionRequest, PagedAttentionOperator, _KNOWN_REQUEST_FIELDS


def _pages_from_dense_kv(key_cache, value_cache, page_size: int, valid_tokens=None,
                         page_metadata=None) -> dict:
    """把 ``[num_pages][page_size]...`` 的稠密 K/V 缓存转成 pages 记录。"""
    pages: dict[int, dict] = {}
    n = len(key_cache)
    for pid in range(n):
        vt = page_size
        if valid_tokens is not None:
            vt = valid_tokens[pid] if not isinstance(valid_tokens, dict) else valid_tokens.get(pid, page_size)
        meta = (page_metadata or {}).get(pid) if isinstance(page_metadata, dict) else None
        record = {
            "keys": key_cache[pid],
            "values": value_cache[pid],
            "valid_tokens": vt,
        }
        if meta:
            record.update(meta)
        pages[pid] = record
    return pages


def paged_attention(*args, **kwargs):
    """执行一次分页注意力。详细契约见 :class:`AttentionRequest`。

    支持的位置形态：

    * ``paged_attention(queries, page_table, pages, ...)``
    * ``paged_attention(q, key_cache, value_cache, page_table, ...)``
    """
    queries = kwargs.pop("queries", None) or kwargs.pop("q", None) or kwargs.pop("query", None)
    page_table = kwargs.pop("page_table", None) or kwargs.pop("block_table", None)
    pages = kwargs.pop("pages", None) or kwargs.pop("page_records", None)
    cache = kwargs.pop("cache", None)
    key_cache = (kwargs.pop("key_cache", None) or kwargs.pop("k_cache", None)
                 or kwargs.pop("keys_cache", None) or kwargs.pop("keys", None))
    value_cache = (kwargs.pop("value_cache", None) or kwargs.pop("v_cache", None)
                   or kwargs.pop("values_cache", None) or kwargs.pop("values", None))
    page_size = kwargs.pop("page_size", None)
    valid_tokens = kwargs.pop("valid_tokens", None) or kwargs.pop("page_valid_tokens", None)

    # 位置参数解析
    if args:
        a = list(args)
        if queries is None and a:
            queries = a.pop(0)
        # 形如 (q, key_cache, value_cache, page_table)
        if key_cache is None and value_cache is None and page_table is None and len(a) >= 3 \
                and isinstance(a[0], (list, tuple)) and isinstance(a[1], (list, tuple)) \
                and isinstance(a[2], (list, tuple)) and _looks_like_dense_cache(a[0]):
            key_cache = a.pop(0)
            value_cache = a.pop(0)
            page_table = a.pop(0)
        else:
            if page_table is None and a:
                page_table = a.pop(0)
            if pages is None and a:
                pages = a.pop(0)
        if a:
            raise TypeError(f"unexpected positional arguments: {len(a)}")

    if key_cache is not None or value_cache is not None:
        if page_size is None:
            page_size = len(key_cache[0]) if key_cache is not None else len(value_cache[0])
        dense_pages = _pages_from_dense_kv(
            key_cache, value_cache, page_size, valid_tokens=valid_tokens,
            page_metadata=kwargs.pop("page_metadata", None),
        )
        if pages is None:
            pages = dense_pages
        else:
            # 调用方同时给了元数据 pages：以 dense 数据补齐
            for pid, rec in dense_pages.items():
                if pid in pages:
                    continue
                pages[pid] = rec

    if cache is None and pages is not None and not isinstance(pages, PagedKVCache):
        if page_size is None:
            raise ValueError("page_size is required when constructing from raw pages")
        cache = build_cache_from_pages(page_size, pages)

    req_kwargs: dict[str, Any] = {
        "queries": queries,
        "page_table": page_table,
        "cache": cache if isinstance(cache, PagedKVCache) else None,
        "pages": None if isinstance(cache, PagedKVCache) else pages,
        "page_size": page_size,
    }
    # cache 可能就是 PagedKVCache
    if isinstance(cache, PagedKVCache):
        req_kwargs["cache"] = cache

    for key, value in kwargs.items():
        if key in _KNOWN_REQUEST_FIELDS:
            req_kwargs[key] = value
        # 未知键忽略会掩盖契约错误，因此直接报错更安全。
        else:
            raise TypeError(f"unknown paged_attention option: {key!r}")

    req = AttentionRequest(**req_kwargs)
    operator = PagedAttentionOperator(cache if isinstance(cache, PagedKVCache) else None)
    if not isinstance(cache, PagedKVCache) and pages is not None:
        req.pages = pages
        req.cache = None
    return operator.run(req)


def _looks_like_dense_cache(first_arg) -> bool:
    """粗略判断位置参数是否是 ``[num_pages][page_size]...`` 的 K/V 缓存。"""
    return bool(first_arg) and isinstance(first_arg[0], (list, tuple))


# 常见函数性别名，方便不同命名习惯的调用方发现。
paged_attention_forward = paged_attention
attention_forward = paged_attention
run_paged_attention = paged_attention
