"""输出对象、可见性证据（witness）与已完成 token 账本。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .errors import BatchStatus, RowStatus


@dataclass(frozen=True)
class KVCoordinate:
    """一个逻辑 KV 位置解析到的物理坐标。"""

    position: int
    page_id: int
    slot: int
    generation: int | None = None
    lease_id: str | None = None
    session_id: str | None = None

    def physical_key(self) -> tuple[int, int, int | None]:
        """用于重复物理页判定的身份键（含代际，避免 ABA 误判）。"""
        return self.page_id, self.slot, self.generation


@dataclass(frozen=True)
class VisibleKV:
    """实际参与了某行 softmax 的一个 KV，以及它在每个头上的权重。"""

    position: int
    page_id: int
    slot: int
    weights: tuple[float, ...]  # 按 head 排列；单头时长度为 1
    generation: int | None = None
    session_id: str | None = None


@dataclass(frozen=True)
class Witness:
    """解释“某个输出只看到了哪些有效 KV”的完整证据。

    * ``visible``：通过了全部遮罩且实际参与归一化的 KV（含权重）；
    * ``visible_positions``：仅逻辑位置的便捷元组；
    * ``excluded``：上下文范围内被排除的位置与稳定原因码；
    * ``normalizers``：每个 head 的 softmax 分母（只含有效可见 KV）；
    * ``mask``：长度 ``kv_length`` 的布尔可见序列，便于直接断言。
    """

    request_id: str
    query_index: int
    query_position: int
    phase: str
    status: str
    kv_length: int
    causal: bool
    window: int | None
    window_convention: str
    window_lower: int | None
    page_size: int
    visible: tuple[VisibleKV, ...]
    visible_positions: tuple[int, ...]
    excluded: tuple["ExcludedPositionLike", ...]  # noqa: F821
    normalizers: tuple[float, ...]
    mask: tuple[bool, ...]
    strategy: str = "exact"

    @property
    def is_fully_masked(self) -> bool:
        return self.status == RowStatus.FULLY_MASKED.value

    def explain(self) -> str:
        """生成人类可读的可见性解释。"""
        lines = [
            f"request={self.request_id!r} phase={self.phase} "
            f"query_index={self.query_index} position={self.query_position} "
            f"status={self.status} strategy={self.strategy}",
            f"rules: causal={self.causal} window={self.window} "
            f"convention={self.window_convention} lower={self.window_lower} "
            f"kv_length={self.kv_length}",
            f"visible positions ({len(self.visible_positions)}): {list(self.visible_positions)}",
        ]
        for item in self.visible:
            lines.append(
                f"  + pos={item.position:<4} page={item.page_id} slot={item.slot} "
                f"weights={['%.6g' % w for w in item.weights]}"
            )
        if self.excluded:
            lines.append("excluded:")
            for ex in self.excluded:
                lines.append(
                    f"  - pos={ex.position:<4} reason={ex.reason} "
                    f"page={ex.page_id} slot={ex.slot}"
                )
        lines.append(f"normalizers={list(self.normalizers)}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "query_index": self.query_index,
            "query_position": self.query_position,
            "phase": self.phase,
            "status": self.status,
            "kv_length": self.kv_length,
            "causal": self.causal,
            "window": self.window,
            "window_convention": self.window_convention,
            "window_lower": self.window_lower,
            "visible_positions": list(self.visible_positions),
            "visible": [
                {
                    "position": v.position,
                    "page_id": v.page_id,
                    "slot": v.slot,
                    "generation": v.generation,
                    "session_id": v.session_id,
                    "weights": list(v.weights),
                }
                for v in self.visible
            ],
            "excluded": [
                {"position": e.position, "reason": e.reason,
                 "page_id": e.page_id, "slot": e.slot}
                for e in self.excluded
            ],
            "normalizers": list(self.normalizers),
            "mask": list(self.mask),
            "strategy": self.strategy,
        }


# 避免导入 masks.ExcludedPosition 造成循环，直接用其结构化协议。
ExcludedPositionLike = Any


@dataclass
class RowResult:
    """单行输出。

    正常行：``output`` 为 ``[head][dim]``（单头时也保持二维）。
    完全遮罩行：状态为 ``fully_masked``，``output is None``（除非调用方
    选择零向量策略），由调用方决定降级/报错/跳过。
    """

    query_index: int
    query_position: int
    status: str
    output: list | None
    witness: Witness

    @property
    def ok(self) -> bool:
        return self.status == RowStatus.OK.value

    @property
    def is_fully_masked(self) -> bool:
        return self.status == RowStatus.FULLY_MASKED.value

    def flat_output(self) -> list[float] | None:
        if self.output is None:
            return None
        if len(self.output) == 1:
            return list(self.output[0])
        return [x for head in self.output for x in head]


@dataclass
class AttentionResult:
    """一次 prefill（多行）或 decode（单行）调用的完整结果。"""

    request_id: str
    phase: str
    outputs: list  # [query][head][dim]；完全遮罩行在 dense_outputs 中为零向量
    rows: list[RowResult]
    witnesses: list[Witness]
    query_positions: tuple[int, ...]
    strategy: str
    dense_outputs: list | None = None
    num_heads: int = 1
    head_dim: int = 0
    approximate: bool = False
    stats: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dense_outputs is None:
            self.dense_outputs = [
                row.output if row.output is not None
                else [[0.0] * self.head_dim for _ in range(self.num_heads)]
                for row in self.rows
            ]

    @property
    def output(self):
        """便捷属性：decode 时返回唯一行的 ``[head][dim]``，否则返回 outputs。"""
        if self.phase == "decode" and len(self.rows) == 1:
            return self.rows[0].output
        return self.outputs

    # 让结果对象也可以直接当作稠密输出序列使用：
    # result[0] -> 第一行 [head][dim]（完全遮罩行为零向量）。
    def __getitem__(self, index):
        return self.dense_outputs[index]

    def __iter__(self):
        return iter(self.dense_outputs)

    def __len__(self) -> int:
        return len(self.dense_outputs)

    @property
    def visible_positions(self) -> list[tuple[int, ...]]:
        return [w.visible_positions for w in self.witnesses]

    #: 可见位置的常见命名别名
    @property
    def visible_indices(self) -> list[tuple[int, ...]]:
        return self.visible_positions

    @property
    def visibility_mask(self) -> list[tuple[bool, ...]]:
        return [w.mask for w in self.witnesses]

    # 可见性证据的常见命名别名
    @property
    def visibility(self) -> list[Witness]:
        return self.witnesses

    @property
    def provenance(self) -> list[Witness]:
        return self.witnesses

    @property
    def traces(self) -> list[Witness]:
        return self.witnesses

    def explain(self, row: int = 0) -> str:
        """返回第 ``row`` 行输出的人类可读可见性解释。"""
        return self.witnesses[row].explain()

    def explain_all(self) -> list[str]:
        return [w.explain() for w in self.witnesses]

    @property
    def all_ok(self) -> bool:
        return all(r.ok for r in self.rows)

    def flat_outputs(self) -> list[list[float]]:
        """每行一个扁平向量（多头拼接）；完全遮罩行为 ``None``。"""
        return [row.flat_output() for row in self.rows]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "phase": self.phase,
            "strategy": self.strategy,
            "approximate": self.approximate,
            "query_positions": list(self.query_positions),
            "outputs": self.outputs,
            "dense_outputs": self.dense_outputs,
            "row_status": [r.status for r in self.rows],
            "visible_positions": [list(v) for v in self.visible_positions],
            "witnesses": [w.to_dict() for w in self.witnesses],
            "stats": self.stats,
        }


@dataclass(frozen=True)
class CompletedToken:
    """账本中一个已完成 token 的不可变记录。

    输出在入库时被转成 tuple 快照，之后缓存回收/覆写都不可能改变它。
    """

    request_id: str
    position: int
    sequence_index: int
    phase: str
    output: tuple                      # tuple[head][dim-floats]
    flat_output: tuple[float, ...]
    witness: Witness
    completed_index: int


class ResultLedger:
    """按 ``(request_id, position)`` 保存已完成 token，租约失效后仍可查询。"""

    def __init__(self) -> None:
        self._tokens: dict[tuple[str, int], CompletedToken] = {}
        self._by_request: dict[str, list[int]] = {}
        self._completion_counter = 0

    @staticmethod
    def _freeze_output(output) -> tuple:
        return tuple(tuple(float(x) for x in head) for head in output)

    def record_row(self, request_id: str, row: RowResult, *, sequence_index: int | None = None) -> CompletedToken:
        if row.output is None:
            raise ValueError("fully masked row cannot be recorded as a completed numeric token")
        frozen = self._freeze_output(row.output)
        flat = tuple(x for head in frozen for x in head)
        if sequence_index is None:
            sequence_index = len(self._by_request.get(request_id, []))
        token = CompletedToken(
            request_id=request_id,
            position=row.query_position,
            sequence_index=sequence_index,
            phase=row.witness.phase,
            output=frozen,
            flat_output=flat,
            witness=row.witness,
            completed_index=self._completion_counter,
        )
        self._completion_counter += 1
        self._tokens[(request_id, row.query_position)] = token
        positions = self._by_request.setdefault(request_id, [])
        if row.query_position not in positions:
            positions.append(row.query_position)
            positions.sort()
        return token

    def get_token(self, request_id: str, position: int) -> CompletedToken:
        return self._tokens[(request_id, int(position))]

    def find_token(self, request_id: str, position: int) -> CompletedToken | None:
        return self._tokens.get((request_id, int(position)))

    def completed_positions(self, request_id: str) -> tuple[int, ...]:
        return tuple(self._by_request.get(request_id, ()))

    def get_completed(self, request_id: str) -> list[CompletedToken]:
        return [self._tokens[(request_id, p)] for p in self.completed_positions(request_id)]

    def has(self, request_id: str, position: int) -> bool:
        return (request_id, int(position)) in self._tokens

    def requests(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_request))


@dataclass
class BatchOutcome:
    request_id: str
    status: str
    result: AttentionResult | None = None
    error: BaseException | None = None

    @property
    def ok(self) -> bool:
        return self.status == BatchStatus.FULFILLED.value and self.error is None


@dataclass
class BatchResult:
    """批处理器提交一批请求后的逐项结果（保持提交顺序）。"""

    outcomes: list[BatchOutcome]

    def __post_init__(self) -> None:
        self.by_id = {o.request_id: o for o in self.outcomes}

    @property
    def results(self) -> dict[str, AttentionResult]:
        return {o.request_id: o.result for o in self.outcomes if o.result is not None}

    @property
    def failures(self) -> list[BatchOutcome]:
        return [o for o in self.outcomes if not o.ok]

    @property
    def all_succeeded(self) -> bool:
        return all(o.ok for o in self.outcomes)

    # 语义化别名
    @property
    def ok(self) -> bool:
        return self.all_succeeded

    def get(self, request_id: str) -> BatchOutcome | None:
        return self.by_id.get(request_id)

    def __getitem__(self, request_id: str) -> BatchOutcome:
        return self.by_id[request_id]

    def __iter__(self):
        return iter(self.outcomes)

    def __len__(self) -> int:
        return len(self.outcomes)
