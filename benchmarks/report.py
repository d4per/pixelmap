#!/usr/bin/env python3
"""Summarise run.py's CSV as one markdown table per (dataset, thread count).

    python report.py                      # benchmarks/results/results.csv
    python report.py path/to/results.csv --json summary.json

Each cell is the mean over the dataset's pairs, except `EPE med`, which is the median
of the per-pair median EPE. Pairs on which a method failed are counted in `failed`
and left out of its means. Lower is better for EPE, Fl and seconds; higher is better
for coverage and PCK.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import methods

HERE = Path(__file__).resolve().parent

COLUMNS = [
    # (csv field, header, format, aggregate)
    ("coverage", "coverage", "{:.1%}", "mean"),
    ("epe", "EPE", "{:.2f}", "mean"),
    ("epe_med", "EPE med", "{:.2f}", "median"),
    ("epe_shared", "EPE shared", "{:.2f}", "mean"),
    ("fl", "Fl", "{:.1%}", "mean"),
    ("pck1", "PCK@1", "{:.1%}", "mean"),
    ("pck3", "PCK@3", "{:.1%}", "mean"),
    ("pck10", "PCK@10", "{:.1%}", "mean"),
    ("pck3_noc", "PCK@3 noc", "{:.1%}", "mean"),
    ("seconds", "s/pair", "{:.2f}", "mean"),
]


def _num(s: str) -> float:
    try:
        v = float(s)
    except (TypeError, ValueError):
        return math.nan
    return v


def summarise(rows: list[dict]) -> dict:
    groups: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[(r["dataset"], r["threads"])][r["method"]].append(r)

    order = {m: i for i, m in enumerate(methods.ALL)}
    out = {}
    for (dataset, threads), by_method in sorted(groups.items()):
        table = {}
        for method in sorted(by_method, key=lambda m: order.get(m, 999)):
            rs = by_method[method]
            ok = [r for r in rs if not r.get("error")]
            entry = {"pairs": len(rs), "failed": len(rs) - len(ok)}
            for field, _, _, agg in COLUMNS:
                vals = [v for v in (_num(r.get(field)) for r in ok) if not math.isnan(v)]
                if not vals:
                    entry[field] = None
                elif agg == "median":
                    entry[field] = statistics.median(vals)
                else:
                    entry[field] = statistics.fmean(vals)
            table[method] = entry
        out[f"{dataset} @ {threads} threads"] = table
    return out


def markdown(summary: dict) -> str:
    parts = []
    for title, table in summary.items():
        cols = [c for c in COLUMNS if any(e[c[0]] is not None for e in table.values())]
        head = ["method", "pairs"] + [c[1] for c in cols]
        lines = [f"### {title}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for method, e in table.items():
            pairs = str(e["pairs"]) + (f" ({e['failed']} failed)" if e["failed"] else "")
            cells = [method, pairs] + [
                "–" if e[f] is None else fmt.format(e[f]) for f, _, fmt, _ in cols
            ]
            lines.append("| " + " | ".join(cells) + " |")
        parts.append("\n".join(lines))
    return "\n\n".join(parts) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path, nargs="?", default=HERE / "results" / "results.csv")
    p.add_argument("--json", type=Path, help="also write the summary as JSON")
    a = p.parse_args()
    with open(a.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    summary = summarise(rows)
    print(markdown(summary), end="")
    if a.json:
        a.json.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
