"""资料 ``data/input.json`` 的加载、确定性合成 K/V 与可执行验收演示。

资料只给了页元数据（属主、``valid_tokens``），没有数值张量，因此这里用
固定种子的 FNV + LCG 生成可复现的合成 Q/K/V，并在无效填充槽写入极端
“毒”值：如果算子错误地读取了有效区间之外的页内填充，输出会立刻爆炸，
从而把跨页/填充 bug 暴露成数值失败。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .cache import PageRef, PagedKVCache
from .operator import PagedAttentionOperator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE_PATH = REPO_ROOT / "data" / "input.json"

_MASK64 = (1 << 64) - 1


def _resolve_fixture_path(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path)
    import os
    candidates = []
    env_path = os.environ.get("OPA_FIXTURE")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(DEFAULT_FIXTURE_PATH)
    candidates.append(Path.cwd() / "data" / "input.json")
    cur = Path.cwd()
    for _ in range(5):
        candidates.append(cur / "data" / "input.json")
        cur = cur.parent
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return DEFAULT_FIXTURE_PATH


def load_fixture(path: str | Path | None = None) -> dict[str, Any]:
    target = _resolve_fixture_path(path)
    return json.loads(target.read_text(encoding="utf-8"))


def fnv_seed(*parts: Any) -> int:
    """把任意字符串/整数部件混合成 64 位确定性种子（不使用随机化的 hash()）。"""
    h = 14695981039346656037
    text = "/".join(str(p) for p in parts)
    for b in text.encode("utf-8"):
        h ^= b
        h = (h * 1099511628211) & _MASK64
    return h


def synth_vector(seed: int, count: int, lo: float = -0.35, hi: float = 0.35) -> list[float]:
    """LCG 生成固定区间的确定性浮点向量。"""
    state = seed & _MASK64
    span = hi - lo
    out: list[float] = []
    for _ in range(count):
        state = (state * 6364136223846793005 + 1442695040888963407) & _MASK64
        out.append(lo + (state / _MASK64) * span)
    return out


def synth_heads(seed: int, num_heads: int, head_dim: int) -> list[list[float]]:
    flat = synth_vector(seed, num_heads * head_dim)
    return [flat[h * head_dim:(h + 1) * head_dim] for h in range(num_heads)]


@dataclass
class FixtureScenario:
    payload: dict
    page_size: int
    num_heads: int
    head_dim: int
    cache: PagedKVCache
    operator: PagedAttentionOperator
    tables: dict[str, list[PageRef]] = field(default_factory=dict)
    sessions: dict[str, str] = field(default_factory=dict)
    planned_outputs: dict[str, int] = field(default_factory=dict)

    def query_at(self, request_id: str, position: int):
        return synth_heads(fnv_seed("Q", "v1", request_id, position), self.num_heads, self.head_dim)

    def token_kv_at(self, request_id: str, position: int):
        k = synth_heads(fnv_seed("K", "v1", request_id, "gen", position), self.num_heads, self.head_dim)
        v = synth_heads(fnv_seed("V", "v1", request_id, "gen", position), self.num_heads, self.head_dim)
        return k, v


def build_fixture_scenario(
    payload: dict | str | Path | None = None,
    *,
    head_dim: int = 8,
    num_heads: int = 2,
    poison_padding: bool = True,
) -> FixtureScenario:
    """根据资料构造带真实（合成）K/V 的缓存、页表与算子。"""
    if payload is None or isinstance(payload, (str, Path)):
        payload = load_fixture(payload)
    page_size = int(payload["page_size"])
    cache = PagedKVCache(page_size, num_heads=num_heads, head_dim=head_dim)
    operator = PagedAttentionOperator(cache)

    scenario = FixtureScenario(
        payload=payload,
        page_size=page_size,
        num_heads=num_heads,
        head_dim=head_dim,
        cache=cache,
        operator=operator,
    )

    for spec in payload["requests"]:
        rid = spec["request_id"]
        session = rid
        scenario.sessions[rid] = session
        prefix_length = int(spec["prefix_length"])
        decode_tokens = int(spec.get("decode_tokens", 0))
        scenario.planned_outputs[rid] = prefix_length + decode_tokens

        lease = cache.begin_lease(session, lease_id=f"lease:{rid}")
        table: list[PageRef] = []
        for raw_pid in spec["prefix_pages"]:
            pid = int(raw_pid)
            meta = payload.get("pages", {}).get(str(raw_pid)) or payload.get("pages", {}).get(pid)
            owner = str(meta["session"]) if meta and "session" in meta else session
            valid_tokens = int(meta["valid_tokens"]) if meta and "valid_tokens" in meta else page_size

            if pid in cache.pages and isinstance(cache.pages[pid].keys[0], list):
                # 前缀共享：物理页已由别的请求构建，直接拿当前代际身份。
                page = cache.pages[pid]
                ref = PageRef(pid, page.generation, page.lease_id)
                if owner != session:
                    cache.share_page(pid, session)
            else:
                ref = cache.allocate(
                    owner,
                    lease_id=lease.lease_id if owner == session else None,
                    page_id=pid,
                    valid_tokens=valid_tokens,
                )
                page = cache.pages[pid]
                for slot in range(valid_tokens):
                    k = synth_heads(fnv_seed("K", "v1", pid, slot), num_heads, head_dim)
                    v = synth_heads(fnv_seed("V", "v1", pid, slot), num_heads, head_dim)
                    cache.write_slot(ref, slot, k, v)
                if poison_padding:
                    # 有效区间外的填充槽写入极端值；正确算子绝不能读到它们。
                    for slot in range(valid_tokens, page_size):
                        page.keys[slot] = [[1.0e6] * head_dim for _ in range(num_heads)]
                        page.values[slot] = [[9.0e9] * head_dim for _ in range(num_heads)]
                if owner != session:
                    cache.share_page(pid, session)
            table.append(ref)

        scenario.tables[rid] = table

        # 健全性：prefix_length 必须与各页 valid_tokens 之和一致（资料契约）。
        total_valid = sum(
            min(cache.pages[r.page_id].valid_tokens,
                r.valid_tokens if r.valid_tokens is not None else page_size)
            for r in table
        )
        if total_valid < prefix_length:
            from .errors import ContractError
            raise ContractError(
                f"{rid}: prefix_length={prefix_length} exceeds declared valid ranges "
                f"sum={total_valid}"
            )

    return scenario


def _max_abs_diff(a, b) -> float:
    worst = 0.0
    for ra, rb in zip(a, b):
        for ha, hb in zip(ra, rb):
            for x, y in zip(ha, hb):
                worst = max(worst, abs(x - y))
    return worst


def _result_summary(result, *, include_outputs: bool = True) -> dict:
    return {
        "phase": result.phase,
        "strategy": result.strategy,
        "approximate": result.approximate,
        "query_positions": list(result.query_positions),
        "row_status": [r.status for r in result.rows],
        "visible_positions": [list(v) for v in result.visible_positions],
        "outputs": result.outputs if include_outputs else None,
    }


def solve_fixture(
    payload: dict | str | Path | None = None,
    *,
    path: str | Path | None = None,
    include_reclamation_demo: bool = True,
) -> dict[str, Any]:
    """跑通资料中的两个请求：prefill + 逐 token decode，并给可见性证据。

    返回 JSON 可序列化的报告字典；同时内嵌租约回收后“已完成 token 可查、
    未完成请求显式失败”的演示证据。
    """
    if payload is None:
        payload = load_fixture(path)
    scenario = build_fixture_scenario(payload)
    op = scenario.operator
    ps = scenario.page_size

    report: dict[str, Any] = {
        "page_size": ps,
        "num_heads": scenario.num_heads,
        "head_dim": scenario.head_dim,
        "requests": {},
    }

    for spec in payload["requests"]:
        rid = spec["request_id"]
        window = int(spec["window"]) if spec.get("window") is not None else None
        if window == 0:
            window = None
        causal = bool(spec.get("causal", True))
        prefix_length = int(spec["prefix_length"])
        decode_tokens = int(spec.get("decode_tokens", 0))
        table = list(scenario.tables[rid])

        job = op.start_request(
            rid,
            session_id=rid,
            page_table=table,
            kv_length=prefix_length,
            window=window,
            causal=causal,
            planned_outputs=scenario.planned_outputs[rid],
        )

        q_prefill = [scenario.query_at(rid, p) for p in range(prefix_length)]
        prefill = job.prefill(q_prefill, kv_length=prefix_length)

        # 同样输入再走一遍按块快速路径，记录与精确路径的最大偏差。
        prefill_fast = op.prefill_fast(
            request_id=rid + "#fast-check",
            session_id=rid,
            page_table=table,
            kv_length=prefix_length,
            queries=q_prefill,
            window=window,
            causal=causal,
        )
        fast_diff = _max_abs_diff(prefill.dense_outputs, prefill_fast.dense_outputs)

        decode_summaries = []
        determinism_ok = True
        for t in range(decode_tokens):
            position = prefix_length + t
            q = scenario.query_at(rid, position)
            k, v = scenario.token_kv_at(rid, position)
            out = job.decode(q, position=position, append_key=k, append_value=v)
            # 只读重放一次，验证解码确定性（逐位相同）。
            replay = op.decode(
                request_id=rid + "#replay",
                session_id=rid,
                page_table=table,
                kv_length=position + 1,
                query=q,
                position=position,
                window=window,
                causal=causal,
            )
            if out.rows[0].output != replay.rows[0].output:
                determinism_ok = False
            decode_summaries.append(_result_summary(out))

        report["requests"][rid] = {
            "window": window,
            "causal": causal,
            "prefix_length": prefix_length,
            "decode_tokens": decode_tokens,
            "state_after_run": str(job.state.value if hasattr(job.state, "value") else job.state),
            "prefill": _result_summary(prefill),
            "prefill_fast_max_abs_diff": fast_diff,
            "decode": decode_summaries,
            "decode_deterministic_bitwise": determinism_ok,
            "last_prefill_explanation": prefill.witnesses[-1].explain(),
            "last_decode_explanation": job.get_token(prefix_length + decode_tokens - 1).witness.explain(),
        }

    if include_reclamation_demo:
        report["reclamation_demo"] = _reclamation_demo(payload)

    return report


def _reclamation_demo(payload: dict) -> dict[str, Any]:
    """构造“差最后一个 token 未完成时租约结束/页被新租约占用”的证据。"""
    scenario = build_fixture_scenario(payload)
    op = scenario.operator

    # 选第一个请求做回收演示（chat-a：6 prefix + 2 decode，差最后一步）。
    spec = payload["requests"][0]
    rid = spec["request_id"]
    window = int(spec["window"]) or None
    prefix_length = int(spec["prefix_length"])
    table = list(scenario.tables[rid])

    job = op.start_request(
        rid, session_id=rid, page_table=table, kv_length=prefix_length,
        window=window, causal=bool(spec.get("causal", True)),
        planned_outputs=scenario.planned_outputs[rid],
    )
    q_prefill = [scenario.query_at(rid, p) for p in range(prefix_length)]
    job.prefill(q_prefill, kv_length=prefix_length)

    pos6 = prefix_length
    q6 = scenario.query_at(rid, pos6)
    k6, v6 = scenario.token_kv_at(rid, pos6)
    job.decode(q6, position=pos6, append_key=k6, append_value=v6)
    positions_before = list(job.completed_positions)

    # 结束租约并回收全部页，然后把关键页分配给新会话并写入毒数据。
    reclaimed = scenario.cache.end_lease(f"lease:{rid}", reclaim=True)
    target_pid = table[1].page_id  # 位置 7 落在第二个页块
    new_lease = scenario.cache.begin_lease("chat-z", lease_id="lease:chat-z")
    new_ref = scenario.cache.allocate(
        "chat-z", lease_id="lease:chat-z", page_id=target_pid, valid_tokens=scenario.page_size
    )
    for slot in range(scenario.page_size):
        scenario.cache.write_slot(
            new_ref, slot,
            [[1.0e8] * scenario.head_dim for _ in range(scenario.num_heads)],
            [[-1.0e8] * scenario.head_dim for _ in range(scenario.num_heads)],
            session_id="chat-z",
        )

    pos7 = prefix_length + 1
    q7 = scenario.query_at(rid, pos7)
    failure_type = None
    failure_message = None
    try:
        job.decode(q7, position=pos7, kv_length=pos7 + 1)
        failed_explicitly = False
    except Exception as exc:  # 预期：PageReclaimedError
        failed_explicitly = True
        failure_type = type(exc).__name__
        failure_message = str(exc)

    still_queryable = {
        str(p): op.find_token(rid, p) is not None for p in positions_before
    }
    ledger_output_unchanged = op.get_token(rid, pos6).position == pos6

    return {
        "request_id": rid,
        "completed_before_reclamation": positions_before,
        "reclaimed_page_ids": reclaimed,
        "page_reallocated_to": "chat-z",
        "incomplete_request_failed_explicitly": failed_explicitly,
        "failure_type": failure_type,
        "failure_message": failure_message,
        "completed_tokens_still_queryable": still_queryable,
        "ledger_snapshot_intact": ledger_output_unchanged,
        "job_state": str(job.state.value if hasattr(job.state, "value") else job.state),
    }
