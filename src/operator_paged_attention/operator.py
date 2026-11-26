"""可复用的分页注意力算子与请求生命周期管理。

典型用法
--------
>>> from operator_paged_attention import PagedAttentionOperator, PagedKVCache
>>> cache = PagedKVCache(page_size=4, num_heads=1, head_dim=8)
>>> op = PagedAttentionOperator(cache)
>>> lease = cache.begin_lease("s1", "l-1")
>>> ref = cache.allocate("s1", lease_id="l-1", page_id=0)
>>> # ...写入 K/V...
>>> result = op.prefill(page_table=[ref], kv_length=4, queries=q)
>>> result.witnesses[0].visible_positions
(0,)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .cache import PageTable, PagedKVCache, build_cache_from_pages
from .engine import normalize_queries, prepare_kv, run_block_fast, run_exact
from .errors import (
    BatchProcessingError,
    ContractError,
    RequestState as _RS,
)
from .masks import WINDOW_INCLUSIVE, WINDOW_CONVENTIONS
from .types import AttentionResult, BatchOutcome, BatchResult, ResultLedger


@dataclass
class AttentionRequest:
    """批处理器可提交的完整请求契约。

    元数据别名（``context_length`` / ``seq_len`` / ``sliding_window`` 等）
    都会被归一化，方便不同调用方的既有命名接入。
    """

    request_id: str = "request"
    page_table: Any = None
    block_table: Any = None
    queries: Any = None
    query: Any = None
    q: Any = None
    cache: PagedKVCache | None = None
    pages: Any = None
    page_size: int | None = None
    kv_length: int | None = None
    context_length: int | None = None
    seq_len: int | None = None
    sequence_length: int | None = None
    cached_length: int | None = None
    query_positions: Sequence[int] | None = None
    positions: Sequence[int] | None = None
    position: int | None = None
    query_position: int | None = None
    q_position: int | None = None
    start_position: int | None = None
    #: 解码默认位置约定：True 时当前 token 的 K/V 已在缓存中（p=kv_length-1）；
    #: False 时缓存只含历史 KV（新 token 位于 p=kv_length）。
    current_token_in_kv: bool = True
    phase: str | None = None
    causal: bool = True
    window: int | None = None
    sliding_window: int | None = None
    window_size: int | None = None
    window_convention: str = WINDOW_INCLUSIVE
    scale: float | None = None
    num_heads: int | None = None
    head_dim: int | None = None
    strategy: str | None = None
    approximate: bool | None = None
    fast: bool | None = None
    block_size: int | None = None
    extra_mask: Any = None
    attention_mask: Any = None
    mask: Any = None
    on_fully_masked: str = "status"
    session_id: str | None = None
    dedup_physical: bool = False
    planned_outputs: int | None = None
    record: bool = True

    # 兼容资料样例字段（prefix_pages 等），由 fixture 层进一步解析；
    # 这里仅吸收，避免 **kwargs 场景报错。
    prefix_pages: Any = None
    prefix_length: int | None = None
    decode_tokens: int | None = None

    def resolved_kv_length(self) -> int:
        for value in (self.kv_length, self.context_length, self.sequence_length,
                      self.seq_len, self.cached_length, self.prefix_length):
            if value is not None:
                return int(value)
        raise ContractError("kv_length/context_length is required")

    def resolved_window(self):
        for value in (self.window, self.sliding_window, self.window_size):
            if value is not None:
                return int(value) if value else 0
        return None

    def resolved_queries(self):
        if self.queries is not None:
            return self.queries
        for value in (self.query, self.q):
            if value is not None:
                return value
        raise ContractError("queries/query is required")

    def resolved_table(self):
        table = self.page_table if self.page_table is not None else self.block_table
        if table is None and self.prefix_pages is not None:
            table = self.prefix_pages
        if table is None:
            raise ContractError("page_table/block_table is required")
        return table

    def resolved_mask(self):
        return self.extra_mask if self.extra_mask is not None else (
            self.attention_mask if self.attention_mask is not None else self.mask
        )


_KNOWN_REQUEST_FIELDS = set(AttentionRequest.__dataclass_fields__)

# 外部序列元数据对象可能使用的属性名 -> 本算子字段
_ATTR_ALIASES = {
    "id": "request_id",
    "req_id": "request_id",
    "sequence_id": "request_id",
    "seq_id": "request_id",
    "block_table": "block_table",
    "page_table": "page_table",
    "context_len": "context_length",
    "kv_len": "kv_length",
    "cached_len": "cached_length",
    "seq_length": "sequence_length",
    "sliding_window_size": "sliding_window",
    "query_len": None,
}


def coerce_request(item: AttentionRequest | dict | Any) -> AttentionRequest:
    if isinstance(item, AttentionRequest):
        return item
    if isinstance(item, dict):
        known = {k: v for k, v in item.items() if k in _KNOWN_REQUEST_FIELDS}
        return AttentionRequest(**known)
    # 容忍调用方自己的 SequenceMetadata/Request 对象：抽取已知属性。
    data: dict[str, Any] = {}
    for name in _KNOWN_REQUEST_FIELDS:
        if hasattr(item, name):
            data[name] = getattr(item, name)
    for attr, target in _ATTR_ALIASES.items():
        if target and attr != target and hasattr(item, attr) and target not in data:
            data[target] = getattr(item, attr)
    if data:
        return AttentionRequest(**data)
    raise TypeError(f"cannot build AttentionRequest from {type(item)!r}")


class PagedAttentionOperator:
    """分页注意力算子：统一入口、账本与批处理。

    参数
    ----
    cache:
        可选的共享 :class:`PagedKVCache`。批处理器通常持有一个长生命周期缓存。
    ledger:
        可选的结果账本；默认新建一个，用于“已完成 token 仍可查询”。
    default_prefill_strategy:
        ``"exact"``（默认，参考级精度）或 ``"block_fast"``（按块近似快速路径）。
    """

    def __init__(
        self,
        cache: "PagedKVCache | int | None" = None,
        *,
        ledger: ResultLedger | None = None,
        default_prefill_strategy: str = "exact",
        page_size: int = 16,
        num_heads: int = 1,
        head_dim: int | None = None,
        window_convention: str = WINDOW_INCLUSIVE,
    ):
        # 宽容 ``PagedAttentionOperator(page_size_int)`` 的位置参数习惯。
        if isinstance(cache, int):
            page_size = cache
            cache = None
        self.cache = cache or PagedKVCache(page_size, num_heads=num_heads, head_dim=head_dim)
        self.ledger = ledger or ResultLedger()
        self.default_prefill_strategy = default_prefill_strategy
        self.window_convention = window_convention

    # -------------------------------------------------------------- 主执行体
    def run(self, req: AttentionRequest) -> AttentionResult:
        """执行一次 attention 调用（prefill 或 decode）。"""
        phase = (req.phase or "").lower() or None
        queries_raw = req.resolved_queries()
        table_raw = req.resolved_table()
        kv_length = req.resolved_kv_length()
        window = req.resolved_window()
        convention = req.window_convention or self.window_convention
        if convention not in WINDOW_CONVENTIONS:
            raise ContractError(f"window_convention must be one of {WINDOW_CONVENTIONS}")

        cache = req.cache or self.cache
        if req.pages is not None and not isinstance(req.pages, PagedKVCache):
            cache = build_cache_from_pages(
                req.page_size or cache.page_size,
                req.pages,
                num_heads=req.num_heads or cache.num_heads,
                head_dim=req.head_dim or cache.head_dim,
            )
        elif req.pages is not None:
            cache = req.pages

        page_size = req.page_size or cache.page_size
        table = table_raw if isinstance(table_raw, PageTable) else PageTable(page_size, list(table_raw))

        # 阶段推断（影响 Q 形状归一化与默认位置）
        if phase is None:
            if req.position is not None or (req.query is not None or req.q is not None) \
                    and req.queries is None:
                phase = "decode"
            else:
                phase = "prefill"

        queries, q_heads, q_dim = normalize_queries(
            queries_raw, phase=phase,
            num_heads=req.num_heads or cache.num_heads,
            head_dim=req.head_dim or cache.head_dim,
        )

        prepared = prepare_kv(cache, table, kv_length, session_id=req.session_id)

        # 维度一致性：以实际 KV 为准，Q 必须匹配
        head_count = prepared.num_heads
        head_dim = prepared.head_dim or q_dim
        if prepared.head_dim == 0:
            # KV 中可能还没有任何有效槽（例如完全遮罩场景），用 Q 维度兜底
            prepared.head_dim = q_dim
            prepared.num_heads = q_heads
            head_count, head_dim = q_heads, q_dim
        if q_dim != head_dim:
            from .errors import ShapeError
            raise ShapeError(f"query head_dim={q_dim} does not match KV head_dim={head_dim}")
        if q_heads not in (head_count, 1):
            from .errors import ShapeError
            raise ShapeError(f"query heads={q_heads} does not match KV heads={head_count}")

        positions = self._resolve_positions(req, phase, len(queries), kv_length)

        scale = req.scale
        if scale is None:
            scale = 1.0 / (head_dim ** 0.5)

        strategy = self._select_strategy(req, phase)
        extra_mask = req.resolved_mask()

        common = dict(
            request_id=req.request_id,
            phase=phase,
            queries=queries,
            positions=positions,
            prepared=prepared,
            causal=req.causal,
            window=window,
            convention=convention,
            scale=scale,
            extra_mask=extra_mask,
            dedup_physical=req.dedup_physical,
            on_fully_masked=req.on_fully_masked,
        )

        stats: dict[str, Any] = {}
        if strategy == "block_fast":
            rows, stats = run_block_fast(block_size=req.block_size, **common)
            approximate = True
        else:
            rows = run_exact(**common)
            approximate = False
            strategy = "exact"

        outputs = [row.output if row.output is not None else None for row in rows]
        result = AttentionResult(
            request_id=req.request_id,
            phase=phase,
            outputs=outputs,
            rows=rows,
            witnesses=[r.witness for r in rows],
            query_positions=tuple(positions),
            strategy=strategy,
            num_heads=head_count,
            head_dim=head_dim,
            approximate=approximate,
            stats=stats,
        )

        if req.record and self.ledger is not None:
            for row in rows:
                if row.ok:
                    self.ledger.record_row(req.request_id, row)
        return result

    @staticmethod
    def _resolve_positions(req: AttentionRequest, phase: str, q_count: int, kv_length: int):
        explicit = req.query_positions if req.query_positions is not None else req.positions
        if explicit is not None:
            positions = tuple(int(p) for p in explicit)
            if len(positions) != q_count:
                raise ContractError(
                    f"got {len(positions)} query_positions for {q_count} query rows"
                )
            return positions
        if req.position is not None:
            if q_count != 1:
                raise ContractError("single position requires exactly one query row")
            return (int(req.position),)
        if req.query_position is not None or req.q_position is not None:
            if q_count != 1:
                raise ContractError("single query_position requires exactly one query row")
            return (int(req.query_position if req.query_position is not None else req.q_position),)
        if req.start_position is not None:
            start = int(req.start_position)
            return tuple(start + i for i in range(q_count))
        if phase == "decode":
            # 两种常见约定都支持：KV 已含当前 token -> p=kv_length-1；
            # KV 仅含历史 -> 新 token 位置 p=kv_length。
            if req.current_token_in_kv:
                return (max(0, kv_length - 1),)
            return (kv_length,)
        # prefill 默认 query 占据上下文末尾的 q_count 个绝对位置；
        # 当 kv_length == q_count 时即为 0..q_count-1（前缀预填充）。
        if q_count > kv_length:
            raise ContractError(
                f"prefill has {q_count} query rows but kv_length={kv_length}; "
                "provide query_positions explicitly"
            )
        start = kv_length - q_count
        return tuple(start + i for i in range(q_count))

    def _select_strategy(self, req: AttentionRequest, phase: str) -> str:
        wanted = req.strategy
        if wanted is not None:
            wanted = wanted.lower()
            if wanted in ("exact", "reference", "strict"):
                return "exact"
            if wanted in ("block", "block_fast", "fast", "approximate", "flash"):
                return "block_fast"
            if wanted == "auto":
                return "block_fast" if phase == "prefill" else "exact"
            raise ContractError(f"unknown strategy {req.strategy!r}")
        if req.approximate is True or req.fast is True:
            return "block_fast"
        if req.approximate is False or req.fast is False:
            return "exact"
        # 指定分块大小本身即表达“走按块路径”的意图（仅预填充允许）。
        if phase == "prefill" and req.block_size is not None:
            return "block_fast"
        # 解码永远走严格精确路径，即使全局默认是 block_fast。
        if phase == "decode":
            return "exact"
        return self.default_prefill_strategy

    # -------------------------------------------------------------- 便捷方法
    def prefill(self, request: AttentionRequest | dict | None = None, **kwargs) -> AttentionResult:
        req = coerce_request(request) if request is not None else AttentionRequest()
        for k, v in kwargs.items():
            setattr(req, k, v) if k in _KNOWN_REQUEST_FIELDS else _raise_unknown(k)
        req.phase = "prefill"
        return self.run(req)

    def prefill_fast(self, request: AttentionRequest | dict | None = None, **kwargs) -> AttentionResult:
        kwargs.setdefault("strategy", "block_fast")
        return self.prefill(request, **kwargs)

    def decode(self, request: AttentionRequest | dict | None = None, **kwargs) -> AttentionResult:
        req = coerce_request(request) if request is not None else AttentionRequest()
        for k, v in kwargs.items():
            setattr(req, k, v) if k in _KNOWN_REQUEST_FIELDS else _raise_unknown(k)
        req.phase = "decode"
        req.strategy = "exact"  # 解码阶段强制严格路径
        return self.run(req)

    def __call__(self, *args, **kwargs) -> AttentionResult:
        return self.attention(*args, **kwargs)

    def attention(self, request: AttentionRequest | dict | None = None, **kwargs) -> AttentionResult:
        req = coerce_request(request) if request is not None else AttentionRequest()
        for k, v in kwargs.items():
            if k in _KNOWN_REQUEST_FIELDS:
                setattr(req, k, v)
            else:
                _raise_unknown(k)
        return self.run(req)

    #: 常见 nn.Module 风格别名
    def forward(self, *args, **kwargs) -> AttentionResult:
        return self.attention(*args, **kwargs)

    #: 其它常见方法命名
    def compute(self, *args, **kwargs) -> AttentionResult:
        return self.attention(*args, **kwargs)

    def execute(self, *args, **kwargs) -> AttentionResult:
        return self.attention(*args, **kwargs)

    # -------------------------------------------------------------- 批处理
    def submit_batch(
        self,
        requests: Sequence[AttentionRequest | dict],
        *,
        strict: bool = False,
        record: bool | None = None,
    ) -> BatchResult:
        """提交一批带页表与元数据的请求，逐项捕获显式失败。

        缓存身份/租约/属主错误会变成 ``BatchOutcome(status="failed")``，
        不会让整批静默通过；``strict=True`` 时改为抛出
        :class:`BatchProcessingError`。
        """
        outcomes: list[BatchOutcome] = []
        for item in requests:
            req = coerce_request(item)
            if record is not None:
                req.record = record
            try:
                result = self.run(req)
                outcomes.append(BatchOutcome(req.request_id, "fulfilled", result=result))
            except Exception as exc:  # noqa: BLE001 - 批边界需要把失败显式交付
                from .errors import PagedAttentionError
                if not isinstance(exc, PagedAttentionError):
                    raise
                outcomes.append(BatchOutcome(req.request_id, "failed", error=exc))
        batch = BatchResult(outcomes)
        if strict and batch.failures:
            raise BatchProcessingError(
                f"{len(batch.failures)} of {len(outcomes)} requests failed",
                failures=batch.failures,
            )
        return batch

    # -------------------------------------------------------------- 请求作业
    def start_request(
        self,
        request_id: str,
        *,
        session_id: str | None = None,
        page_table: Any = None,
        kv_length: int = 0,
        window: int | None = None,
        causal: bool = True,
        planned_outputs: int | None = None,
        total_tokens: int | None = None,
        window_convention: str | None = None,
        dedup_physical: bool = False,
        on_fully_masked: str = "status",
        block_size: int | None = None,
    ) -> "RequestJob":
        return RequestJob(
            operator=self,
            request_id=request_id,
            session_id=session_id,
            page_table=page_table,
            kv_length=kv_length,
            window=window,
            causal=causal,
            planned_outputs=planned_outputs if planned_outputs is not None else total_tokens,
            window_convention=window_convention or self.window_convention,
            dedup_physical=dedup_physical,
            on_fully_masked=on_fully_masked,
            block_size=block_size,
        )

    # -------------------------------------------------------------- 账本代理
    def get_token(self, request_id: str, position: int):
        return self.ledger.get_token(request_id, position)

    def find_token(self, request_id: str, position: int):
        return self.ledger.find_token(request_id, position)

    def get_completed(self, request_id: str):
        return self.ledger.get_completed(request_id)

    def completed_positions(self, request_id: str):
        return self.ledger.completed_positions(request_id)


def _raise_unknown(name: str):
    raise TypeError(f"unknown attention option: {name!r}")


class RequestJob:
    """有状态请求句柄：跟踪已完成 token 数与 active/completed/failed 生命周期。

    租约失效后，下一次 :meth:`decode`/:meth:`prefill` 会把作业置为
    ``failed`` 并重新抛出显式异常；此前已完成的 token 仍可通过
    :attr:`ledger` 或算子查询。
    """

    def __init__(
        self,
        *,
        operator: PagedAttentionOperator,
        request_id: str,
        session_id: str | None,
        page_table: Any,
        kv_length: int,
        window: int | None,
        causal: bool,
        planned_outputs: int | None,
        window_convention: str,
        dedup_physical: bool,
        on_fully_masked: str,
        block_size: int | None,
    ):
        self.operator = operator
        self.request_id = request_id
        self.session_id = session_id
        self.page_table = page_table
        self.kv_length = int(kv_length)
        self.window = window
        self.causal = causal
        self.planned_outputs = planned_outputs
        self.window_convention = window_convention
        self.dedup_physical = dedup_physical
        self.on_fully_masked = on_fully_masked
        self.block_size = block_size
        self.state = _RS.ACTIVE
        self.last_error: BaseException | None = None

    @property
    def completed_positions(self) -> tuple[int, ...]:
        return self.operator.ledger.completed_positions(self.request_id)

    @property
    def completed_count(self) -> int:
        return len(self.completed_positions)

    def _base_kwargs(self) -> dict:
        return dict(
            request_id=self.request_id,
            session_id=self.session_id,
            page_table=self.page_table,
            kv_length=self.kv_length,
            window=self.window,
            causal=self.causal,
            window_convention=self.window_convention,
            dedup_physical=self.dedup_physical,
            on_fully_masked=self.on_fully_masked,
            block_size=self.block_size,
        )

    def _refresh_completion(self):
        if self.planned_outputs is not None and self.completed_count >= self.planned_outputs:
            self.state = _RS.COMPLETED

    def prefill(self, queries, *, kv_length: int | None = None,
                strategy: str | None = None, **overrides) -> AttentionResult:
        if self.state == _RS.FAILED:
            assert self.last_error is not None
            raise self.last_error
        if self.state == _RS.COMPLETED:
            raise ContractError(f"request {self.request_id!r} is already completed")
        if kv_length is not None:
            self.kv_length = int(kv_length)
        kwargs = self._base_kwargs()
        kwargs.update(queries=queries, phase="prefill", strategy=strategy, **overrides)
        try:
            result = self.operator.run(AttentionRequest(**kwargs))
        except Exception as exc:
            self.state = _RS.FAILED
            self.last_error = exc
            raise
        self._refresh_completion()
        return result

    def decode(
        self,
        query,
        *,
        position: int | None = None,
        append_key=None,
        append_value=None,
        kv_length: int | None = None,
        **overrides,
    ) -> AttentionResult:
        if self.state == _RS.FAILED:
            # 粘性失败：不允许在租约失效后“换个页表再试”而静默蒙混。
            assert self.last_error is not None
            raise self.last_error
        if self.state == _RS.COMPLETED:
            raise ContractError(f"request {self.request_id!r} is already completed")

        if kv_length is not None:
            self.kv_length = int(kv_length)

        target_position = position if position is not None else self.kv_length
        if append_key is not None or append_value is not None:
            if append_key is None or append_value is None:
                raise ContractError("append_key and append_value must be provided together")
            self.operator.cache.append_token(
                self.page_table, target_position, append_key, append_value,
                session_id=self.session_id,
            )
            self.kv_length = target_position + 1
            # 写入后默认 query 位于新 token 自身位置
            if position is None:
                position = target_position

        kwargs = self._base_kwargs()
        kwargs.update(query=query, phase="decode", position=position, **overrides)
        try:
            result = self.operator.run(AttentionRequest(**kwargs))
        except Exception as exc:
            self.state = _RS.FAILED
            self.last_error = exc
            raise
        self._refresh_completion()
        return result

    def mark_completed(self) -> None:
        self.state = _RS.COMPLETED

    def get_completed(self):
        return self.operator.get_completed(self.request_id)

    def get_token(self, position: int):
        return self.operator.get_token(self.request_id, position)


class BatchProcessor:
    """批处理器友好的排队/提交封装。

    用法
    ----
    >>> bp = BatchProcessor(cache=cache, strict=False)
    >>> bp.add(request_dict_a).add(request_dict_b)
    >>> batch = bp.process()
    >>> batch.all_succeeded
    True
    """

    def __init__(
        self,
        operator: PagedAttentionOperator | None = None,
        *,
        cache: PagedKVCache | None = None,
        strict: bool = False,
        record: bool = True,
    ):
        self.operator = operator or PagedAttentionOperator(cache)
        self.strict = strict
        self.record = record
        self.queued: list[AttentionRequest] = []

    def add(self, request: AttentionRequest | dict | Any | None = None, **kwargs) -> "BatchProcessor":
        req = coerce_request(request) if request is not None else AttentionRequest()
        for k, v in kwargs.items():
            if k in _KNOWN_REQUEST_FIELDS:
                setattr(req, k, v)
            else:
                _raise_unknown(k)
        self.queued.append(req)
        return self

    def __len__(self) -> int:
        return len(self.queued)

    def process(
        self,
        requests: Sequence[AttentionRequest | dict] | None = None,
        *,
        strict: bool | None = None,
    ) -> BatchResult:
        items = list(requests) if requests is not None else self.queued
        use_strict = self.strict if strict is None else strict
        return self.operator.submit_batch(items, strict=use_strict, record=self.record)

    # 语义别名
    run = process
    submit = process

    def get_token(self, request_id: str, position: int):
        return self.operator.get_token(request_id, position)

    def get_completed(self, request_id: str):
        return self.operator.get_completed(request_id)


# 类名别名，便于 ``from operator_paged_attention.operator import PagedAttention``
PagedAttention = PagedAttentionOperator
PagedAttentionOp = PagedAttentionOperator
