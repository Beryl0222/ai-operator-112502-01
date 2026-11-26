# 分页 KV 缓存下的因果注意力块算子

一个**零第三方依赖、仅需 Python 标准库**的可复用分页注意力算子。批处理器提交
带页表与序列元数据的请求，即可得到预填充（prefill，多行）与单步解码
（decode，单行）的注意力输出；每个输出都附带 witness，解释它实际看到了哪些
有效 KV（逻辑位置 → 物理页/槽位、注意力权重、归一化分母与被排除原因）。

资料文件 `data/input.json` 描述了两个边界请求（不同前缀长度、窗口、decode
token 数与页 `valid_tokens`），但不含数值张量；`solve`/CLI 会用固定种子合成
确定性 Q/K/V，并在无效填充槽写入极端“毒值”，任何越界读取都会立刻造成数值爆炸。

## 环境与验证

```bash
python -m pip install -e .
python -m pip install -r requirements-test.txt   # 仅锁定的打包工具，测试本身不需要
python -m unittest discover -s tests -v
python -m build --wheel --no-isolation
python -m operator_paged_attention --data data/input.json   # 输出 JSON 验收报告
```

运行时只依赖标准库（Python ≥ 3.11）。测试共 56 个用例，另有用 300 组随机
乱序物理页布局对照独立暴力参考的数值校验脚本（见下文“正确性保证”）。

## 30 秒上手

```python
from operator_paged_attention import PagedAttentionOperator, PagedKVCache

cache = PagedKVCache(page_size=4, num_heads=1, head_dim=4)
op = PagedAttentionOperator(cache)

lease = cache.begin_lease("chat-a", "lease-1")
ref0 = cache.allocate("chat-a", lease_id="lease-1", page_id=0)  # 带 generation 身份
ref1 = cache.allocate("chat-a", lease_id="lease-1", page_id=1)
# cache.write_slot(ref0, slot, key_head_major, value_head_major, session_id="chat-a")
# ...

result = op.decode(
    request_id="chat-a", session_id="chat-a",
    page_table=[ref0, ref1], kv_length=5,
    query=[[0.3, -0.2, 0.1, 0.4]], position=4,
    window=3, causal=True,
)
print(result.output)                       # [head][dim]
print(result.witnesses[0].explain())       # 可见位置/物理坐标/权重/排除原因
```

函数式一次性入口（支持裸 pages 字典或稠密 `key_cache/value_cache`）：

```python
from operator_paged_attention import paged_attention
result = paged_attention(
    q=[0.2, -0.1], block_table=[0, 1], pages={...},
    page_size=4, kv_length=5, position=4, window=2,
)
```

## 核心契约

### 坐标与遮罩（prefill/decode 唯一权威实现）

* KV 与 query 都使用**绝对序列位置**；逻辑位置 `j` 经
  `page_table[j // page_size] + j % page_size` 解析到物理页槽位，跨页由页表
  决定，物理页号可以任意乱序。
* 因果：`j <= p`（包含自身；若当前 token 的 K/V 尚未写入，该位置自然落在
  `kv_length` 之外，等价于只看历史）。
* 滑动窗口：`window=0/None` 表示不限制（与资料中 chat-b 一致）；默认
  `window_convention="inclusive"` 时下界 `p - window + 1`，每行至多 `window`
  个 key。也提供 `"exclusive"`（下界 `p - window`）。**两种阶段调用同一个
  `masks.build_row_mask`，窗口边界不会再在 prefill/decode 间漂移。**
* 有效区间：只有 `slot < page.valid_tokens`（或页表项声明的更紧
  `valid_tokens`）的位置是有效 KV；区间外的页内填充标记为
  `invalid_padding`，不进入 softmax 分母。
* 完全遮罩行：返回 `RowStatus.FULLY_MASKED` 状态，`RowResult.output is None`，
  `dense_outputs` 中为保持形状的零向量（无 NaN）。可用
  `on_fully_masked="zero"|"status"|"raise"` 选择策略。

### 可见性证据（Witness）

每个 `Witness` 含：`visible_positions`、每项的物理
`page_id/slot/generation/session_id` 与各头权重、每头 `normalizers`（softmax
分母）、长度 `kv_length` 的布尔 `mask`、以及带稳定原因码的 `excluded`
（`causal_masked` / `outside_window` / `invalid_padding` / `custom_masked` /
`duplicate_physical_page`）。`witness.explain()` 给人类可读解释，
`witness.to_dict()` 可直接 JSON 序列化。

### 缓存身份、租约与回收

* 每个 `PageRef(page_id, generation, lease_id)` 是分配时刻的身份快照；
  `Page` 在每次分配/回收时递增 `generation`。
* 任何计算前 `PagedKVCache.validate` 校验：generation 不符 →
  `PageReclaimedError`；租约结束/更换 → `LeaseExpiredError`；会话不符且页未
  共享 → `PageOwnershipError`。校验先于数据读取，因此**页面被新租约占用并
  写入毒数据后，旧请求只会显式失败，绝不会读到新会话内容**。
* 前缀共享：`cache.share_page(page_id, *sessions)` 或 `Page(is_shared=True)`
  允许不同会话凭有效代际身份只读同一物理页。
* `ResultLedger` 按 `(request_id, position)` 保存已完成 token 的**不可变快照**
  （输出转 tuple、witness 冻结）。租约结束、页回收后，
  `op.get_token(rid, p)` / `op.get_completed(rid)` 仍可查询；未完成的下一次
  计算抛显式异常，`RequestJob` 进入 `failed` 且错误是粘性的。

```python
job = op.start_request("chat-a", session_id="chat-a", page_table=table,
                       kv_length=6, window=5, planned_outputs=8)
job.prefill(queries)          # 6 个已完成 token 入账
job.decode(q6, append_key=k6, append_value=v6)   # 第 7 个
cache.end_lease("lease-1", reclaim=True)
try:
    job.decode(q7)            # -> PageReclaimedError，job.state == failed
except PageReclaimedError:
    pass
op.get_token("chat-a", 6)     # 此前完成的 token 与 witness 仍可查询
```

### 批处理

`op.submit_batch([AttentionRequest | dict, ...])` 保持提交顺序返回
`BatchResult`：成功项有 `result`，租约/契约错误被捕获为逐项
`BatchOutcome(status="failed", error=...)`，不会让整批静默通过；
`strict=True` 时改抛 `BatchProcessingError`（含全部 failures）。
另有 `BatchProcessor` 排队封装。不同请求可以有各自的 `kv_length`、`window`、
位置与阶段。

### Prefill 快速路径 vs Decode 严格路径

* decode 永远走 `strategy="exact"`：一次性 softmax + `math.fsum` 严格求和，
  固定按位置升序计算，重复调用逐位相同，批顺序不影响结果。
* prefill 默认也走 exact（最大数值可复现）；需要按块快速路径时用
  `op.prefill_fast(...)` 或 `strategy="block_fast"` / `approximate=True`：
  对**同一组权威可见位置**做分块在线 softmax，`block_size` 默认等于页大小。
  近似仅来自分块重标定的浮点累加顺序（夹具上最大偏差约 `5e-17`，随机测试
  中 < 1e-8）。
* 两条路径共用 `prepare_kv`（身份/属主校验）与 `build_row_mask`（遮罩），
  因此缓存身份、遮罩语义、回收行为完全一致；测试断言它们的可见位置、排除
  原因、窗口下界与 stale-ref 异常都相同。

## 公开 API 地图

| 角色 | 名称（`operator_paged_attention` 顶层导出） |
| --- | --- |
| 算子 | `PagedAttentionOperator`（别名 `PagedAttention`/`PagedAttentionOp`/`AttentionOperator`），方法 `.run/.attention/.forward/.prefill/.prefill_fast/.decode/.submit_batch/.start_request` |
| 函数式 | `paged_attention`（别名 `paged_attention_forward`/`run_paged_attention`/`attention_forward`） |
| 请求 | `AttentionRequest`（别名 `SequenceMetadata`/`SequenceMeta`）、`coerce_request`、`RequestJob`、`BatchProcessor` |
| 页/缓存 | `PagedKVCache`（别名 `KVCache`/`BlockManager`）、`Page`、`PageRef`/`BlockRef`、`PageTable`/`BlockTable`、`Lease` |
| 结果/证据 | `AttentionResult`、`RowResult`、`Witness`/`AttentionTrace`/`VisibilityReport`、`VisibleKV`、`KVCoordinate`、`ResultLedger`、`CompletedToken`、`BatchResult`/`BatchOutcome` |
| 遮罩 | `build_row_mask`、`window_lower_bound`、`WINDOW_INCLUSIVE`/`WINDOW_EXCLUSIVE` |
| 状态/异常 | `RowStatus`、`RequestState`、`PagedAttentionError`、`ContractError`、`ShapeError`、`PageTableError`、`PageOwnershipError`、`LeaseExpiredError`、`PageReclaimedError`（别名 `StaleLeaseError`）、`PageNotFoundError`、`FullyMaskedRowError`、`BatchProcessingError` |
| 夹具 | `load_fixture`、`build_fixture_scenario`、`solve_fixture`/`solve` |

兼容导入：仓库根的 `paged_attention.py` 支持 `import paged_attention`；
`operator_paged_attention.api` 与 `operator_paged_attention.paged_attention`
镜像顶层入口。

## 资料夹具的预期可见集合（page_size=4）

* chat-a，prefix=6，W=5 inclusive：prefill p=0..5 分别看到
  `{0}`、`{0,1}`、`{0,1,2}`、`{0..3}`、`{0..4}`、`{1..5}`；decode 到
  p=6/7 时为 `{2..6}`、`{3..7}`。
* chat-b，prefix=3，window=0（不限）：prefill 为
  `{0}`、`{0,1}`、`{0,1,2}`，decode p=3 为 `{0,1,2,3}`。

## 正确性保证

* `tests/` 中 56 个 `unittest` 用例覆盖：夹具可见集合、跨页物理坐标、跨会话
  隔离与毒数据不变性、前缀共享、重复页有效区间与可选物理去重、填充不进分母、
  完全遮罩三种策略、租约/代际/属主失败、账本快照、作业生命周期、混合元数据
  批处理、逐位确定性、prefill/decode 逐位一致、快速/精确遮罩 parity、CLI/JSON。
* 开发期另用 300 组随机用例（随机 page_size∈{1..8}、长度、窗口约定、物理页
  乱序、填充毒值）对照独立 `math.fsum` 暴力参考：exact 最大误差 < 1e-12，
  block_fast < 1e-9，同位置 prefill/decode 一致。

## 目录结构

```
src/operator_paged_attention/
  errors.py       异常树与 RowStatus/RequestState
  linalg.py       确定性纯 Python 数值内核（fsum、稳定 softmax）
  masks.py        因果/窗口/有效区间/去重的唯一遮罩权威
  cache.py        Page/PageRef/PageTable/PagedKVCache/Lease 与代际身份
  types.py        Witness/RowResult/AttentionResult/账本/批结果
  engine.py       KV 解析校验 + exact 与 block_fast 两条计算路径
  operator.py     AttentionRequest/PagedAttentionOperator/RequestJob/BatchProcessor
  functional.py   paged_attention(...) 函数式入口与稠密缓存适配
  fixture.py      data/input.json 加载、合成数据、solve_fixture 验收报告
  __main__.py     python -m operator_paged_attention CLI
tests/            56 个契约/数值/生命周期测试
```
