"""异常类型与调用方可处理的状态枚举。

所有算子抛出的异常都继承 :class:`PagedAttentionError`，批处理器可以用它
统一捕获；租约/回收相关异常单独成树，便于把“必须显式失败的未完成请求”
与普通参数契约错误区分开。
"""

from __future__ import annotations

from enum import Enum


class PagedAttentionError(Exception):
    """分页注意力算子所有显式错误的根类型。"""


class ContractError(PagedAttentionError, ValueError):
    """调用方提交的请求、页表或张量形状不满足契约。"""


class PageTableError(ContractError):
    """页表缺失逻辑块、块大小不一致或映射非法。"""


class ShapeError(ContractError):
    """Q/K/V 的维数、头数或嵌套结构与声明不一致。"""


class PageAccessError(PagedAttentionError):
    """读取物理页时的访问安全错误根类型。"""


class PageOwnershipError(PageAccessError, PermissionError):
    """请求会话与页的属主会话不一致（潜在的跨会话读取）。"""


class LeaseError(PageAccessError):
    """缓存租约/页代际身份校验失败的根类型。

    这类错误意味着提交的页表项所代表的缓存身份已经失效，算子不会读取
    当前物理页中可能属于新租约的数据。
    """


class LeaseExpiredError(LeaseError):
    """租约已经结束（或句柄中的 lease token 与页当前租约不符）。"""


class PageReclaimedError(LeaseError):
    """物理页已被回收并可能被新租约占用（generation 不匹配）。"""


class PageIdentityMismatchError(PageReclaimedError):
    """页身份不匹配的别名，语义等同 :class:`PageReclaimedError`。"""


class PageNotFoundError(LeaseError):
    """引用的物理页在缓存池中不存在。"""


class FullyMaskedRowError(PagedAttentionError):
    """当调用方选择 ``on_fully_masked="raise"`` 且某一行没有任何可见 KV 时抛出。"""

    def __init__(self, request_id: str | None = None, query_position: int | None = None):
        self.request_id = request_id
        self.query_position = query_position
        where = []
        if request_id is not None:
            where.append(f"request={request_id!r}")
        if query_position is not None:
            where.append(f"position={query_position}")
        detail = " (" + ", ".join(where) + ")" if where else ""
        super().__init__(f"attention row is fully masked; no valid KV participates{detail}")


class BatchProcessingError(PagedAttentionError):
    """严格批处理模式下存在失败请求时抛出，``failures`` 保留逐项结果。"""

    def __init__(self, message: str, failures=None):
        super().__init__(message)
        self.failures = list(failures or [])


class RowStatus(str, Enum):
    """单个 query 行的计算状态。"""

    OK = "ok"
    FULLY_MASKED = "fully_masked"

    def __str__(self) -> str:  # pragma: no cover - 简单透传
        return self.value


class RequestState(str, Enum):
    """请求生命周期状态。"""

    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"

    def __str__(self) -> str:  # pragma: no cover
        return self.value


class BatchStatus(str, Enum):
    FULFILLED = "fulfilled"
    FAILED = "failed"
