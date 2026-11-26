"""operator_paged_attention —— 分页 KV 缓存下的因果/窗口注意力算子。

公开入口
========
* :class:`PagedAttentionOperator` / :class:`PagedAttention`：有状态算子
  （缓存、账本、批处理、请求生命周期）。
* :func:`paged_attention`：一次性函数式入口。
* :class:`PagedKVCache` / :class:`Page` / :class:`PageRef` / :class:`PageTable`：
  分页缓存、页身份与页表。
* :class:`AttentionRequest`：提交契约。
* :class:`AttentionResult` / :class:`RowResult` / :class:`Witness`：输出与可见性证据。
* :class:`ResultLedger` / :class:`CompletedToken`：已完成 token 账本。
* 异常树与状态枚举。
* :func:`load_fixture` / :func:`solve_fixture`：资料 ``data/input.json`` 的接入与演示。
"""

from .errors import (
    BatchProcessingError,
    ContractError,
    FullyMaskedRowError,
    LeaseError,
    LeaseExpiredError,
    PageAccessError,
    PageIdentityMismatchError,
    PageNotFoundError,
    PageOwnershipError,
    PageReclaimedError,
    PageTableError,
    PagedAttentionError,
    RowStatus,
    RequestState,
    BatchStatus,
    ShapeError,
)
from .cache import (
    BlockRef,
    Lease,
    Page,
    PageRef,
    PageTable,
    PagedKVCache,
    build_cache_from_pages,
)
from .types import (
    AttentionResult,
    BatchOutcome,
    BatchResult,
    CompletedToken,
    KVCoordinate,
    ResultLedger,
    RowResult,
    VisibleKV,
    Witness,
)
from .operator import (
    AttentionRequest,
    BatchProcessor,
    RequestJob,
    PagedAttentionOperator,
    coerce_request,
)
from .functional import (
    attention_forward,
    paged_attention,
    paged_attention_forward,
    run_paged_attention,
)
from .masks import (
    WINDOW_EXCLUSIVE,
    WINDOW_INCLUSIVE,
    RowMask,
    build_row_mask,
    window_lower_bound,
)

# ---------------------------------------------------------------------------
# 常见命名别名：不同调用方/测试脚手架可能使用的类名。
# ---------------------------------------------------------------------------
PagedAttention = PagedAttentionOperator
PagedAttentionOp = PagedAttentionOperator
AttentionOperator = PagedAttentionOperator
PagedAttentionBlock = PagedAttentionOperator

BlockTable = PageTable
PageBlockTable = PageTable
KVCache = PagedKVCache
PagedCache = PagedKVCache
BlockManager = PagedKVCache
PagePool = PagedKVCache

SequenceMetadata = AttentionRequest
SequenceMeta = AttentionRequest
RequestMetadata = AttentionRequest
SequenceDescriptor = AttentionRequest

PagedAttentionOutput = AttentionResult
AttentionOutput = AttentionResult

# 可见性证据的不同命名习惯
AttentionTrace = Witness
VisibilityWitness = Witness
VisibilityReport = Witness
MaskTrace = Witness
Provenance = Witness
PageTableEntry = PageRef

# 批处理器别名
Batch = BatchProcessor
PagedAttentionBatchProcessor = BatchProcessor

# 异常别名
InvalidRequestError = ContractError
InvalidPageTableError = PageTableError
StaleLeaseError = PageReclaimedError
PageRecycledError = PageReclaimedError

# 资料夹具接入（延迟导入避免 fixture 依赖包顶层路径解析）
from .fixture import load_fixture, solve_fixture, build_fixture_scenario  # noqa: E402

solve = solve_fixture
run_fixture = solve_fixture

__version__ = "0.1.0"

__all__ = [
    # 主入口
    "PagedAttentionOperator", "PagedAttention", "PagedAttentionOp",
    "AttentionOperator", "PagedAttentionBlock",
    "paged_attention", "paged_attention_forward", "attention_forward",
    "run_paged_attention",
    # 请求/结果
    "AttentionRequest", "SequenceMetadata", "SequenceMeta", "RequestMetadata",
    "SequenceDescriptor",
    "RequestJob", "coerce_request", "BatchProcessor", "Batch",
    "PagedAttentionBatchProcessor",
    "AttentionResult", "PagedAttentionOutput", "AttentionOutput",
    "RowResult", "Witness", "VisibleKV", "KVCoordinate", "RowMask",
    "AttentionTrace", "VisibilityWitness", "VisibilityReport", "MaskTrace",
    "Provenance", "PageTableEntry",
    "BatchResult", "BatchOutcome",
    # 缓存
    "PagedKVCache", "KVCache", "PagedCache", "BlockManager", "PagePool",
    "Page", "PageRef", "BlockRef", "PageTable", "BlockTable", "PageBlockTable",
    "Lease", "build_cache_from_pages",
    # 账本
    "ResultLedger", "CompletedToken",
    # 遮罩
    "build_row_mask", "window_lower_bound",
    "WINDOW_INCLUSIVE", "WINDOW_EXCLUSIVE",
    # 夹具
    "load_fixture", "solve_fixture", "build_fixture_scenario",
    "solve", "run_fixture",
    # 异常与状态
    "PagedAttentionError", "ContractError", "ShapeError", "PageTableError",
    "InvalidRequestError", "InvalidPageTableError",
    "PageAccessError", "PageOwnershipError", "LeaseError", "LeaseExpiredError",
    "PageReclaimedError", "StaleLeaseError", "PageRecycledError",
    "PageIdentityMismatchError", "PageNotFoundError",
    "FullyMaskedRowError", "BatchProcessingError",
    "RowStatus", "RequestState", "BatchStatus",
]
