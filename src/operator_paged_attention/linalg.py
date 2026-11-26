"""确定性的纯 Python 数值内核。

只使用标准库：

* 点积/加权求和使用 :func:`math.fsum` 固定求和顺序之外的精度损失；
* softmax 使用最大值平移，避免溢出；
* 所有遍历都按位置升序进行，不依赖 ``set``/``dict`` 的不确定顺序，
  因此同一份输入重复调用得到逐位相同的 ``float`` 结果。
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

Number = float
Vector = list[float]
Matrix = list[list[float]]
Tensor3 = list[Matrix]


def to_float(value) -> float:
    f = float(value)
    if math.isnan(f) or math.isinf(f):
        raise ValueError("non-finite numeric value is not supported")
    return f


def zeros_vector(dim: int) -> Vector:
    return [0.0] * dim


def zeros_matrix(rows: int, cols: int) -> Matrix:
    return [[0.0] * cols for _ in range(rows)]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    # fsum 对误差有严格界；生成器按坐标升序喂入。
    return math.fsum(x * y for x, y in zip(a, b))


def stable_softmax(scores: Sequence[float]) -> tuple[list[float], float, float]:
    """返回 ``(weights, normalizer, max_score)``。

    ``normalizer`` 是平移后的指数和 ``sum(exp(s - max))``，即 softmax 的
    归一化分母；调用方可以据此证明只有真正有效的 KV 进入了分母。
    空序列返回 ``([], 0.0, -inf)``，由上层解释为“完全遮罩行”。
    """
    if not scores:
        return [], 0.0, float("-inf")
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    normalizer = math.fsum(exps)
    if normalizer == 0.0:  # 理论上不会发生（至少 exp(0)=1），防御性处理
        return [0.0] * len(scores), 0.0, m
    weights = [e / normalizer for e in exps]
    return weights, normalizer, m


def weighted_vector_sum(weights: Sequence[float], vectors: Sequence[Sequence[float]], dim: int) -> Vector:
    """``out[d] = sum_i weights[i] * vectors[i][d]``，逐坐标 fsum。"""
    out: Vector = []
    for d in range(dim):
        out.append(math.fsum(weights[i] * vectors[i][d] for i in range(len(vectors))))
    return out


def vector_add_scaled(acc: Vector, factor: float, vec: Sequence[float], scale: float = 1.0) -> None:
    """``acc += factor * scale * vec``（原地，固定坐标顺序）。"""
    mul = factor * scale
    for d in range(len(acc)):
        acc[d] += mul * vec[d]


def infer_head_dim(kv_pages_or_vectors) -> int | None:  # pragma: no cover - 辅助
    for vector in kv_pages_or_vectors:
        if vector is not None:
            return len(vector)
    return None
