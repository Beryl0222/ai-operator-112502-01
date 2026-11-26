"""兼容性别名：允许从仓库根目录直接 ``import paged_attention``。

正式分发包名/导入名为 :mod:`operator_paged_attention`；本文件只把 ``src/``
加入路径并全量转导出，方便不同测试脚手架的导入习惯。
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from operator_paged_attention import *  # noqa: E402,F401,F403
import operator_paged_attention as _opa  # noqa: E402

globals().update({k: getattr(_opa, k) for k in getattr(_opa, "__all__", [])})
__version__ = _opa.__version__
