"""``python -m operator_paged_attention``：在资料夹具上运行并输出 JSON 报告。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .fixture import load_fixture, solve_fixture


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paged causal/sliding-window attention operator")
    parser.add_argument("--data", type=str, default=None, help="path to data/input.json")
    parser.add_argument("--indent", type=int, default=2)
    parser.add_argument(
        "--no-reclamation-demo", action="store_true",
        help="skip the lease/reclamation failure demonstration",
    )
    args = parser.parse_args(argv)

    payload = load_fixture(Path(args.data)) if args.data else load_fixture()
    report = solve_fixture(payload, include_reclamation_demo=not args.no_reclamation_demo)
    json.dump(report, sys.stdout, ensure_ascii=False, indent=args.indent)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
