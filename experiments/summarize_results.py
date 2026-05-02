"""
Scan results/ directory and produce a unified comparison table.

Generates:
  - results/SUMMARY.md   (human-readable markdown table)
  - results/SUMMARY.tex  (LaTeX table for the paper)

Usage:
    python -m experiments.summarize_results
    python -m experiments.summarize_results --filter mll  # only MLL runs
"""

import os
import sys
import json
import argparse
import glob
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import RESULTS_DIR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--filter", default="", help="Substring filter on run_name")
    return p.parse_args()


def scan_results(filter_str=""):
    rows = []
    for d in sorted(os.listdir(RESULTS_DIR)):
        if filter_str and filter_str not in d:
            continue
        path = os.path.join(RESULTS_DIR, d)
        if not os.path.isdir(path):
            continue

        # Try summary.json first (main method), then test_results.json (baseline)
        summary_path = os.path.join(path, "summary.json")
        results_path = os.path.join(path, "test_results.json")

        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            if "game_test" in summary:
                rows.append({
                    "run": d,
                    "method": "Bidding Game (Ours)",
                    **summary["game_test"],
                })
                # Also add per-expert results
                for k in ["expert_a_test", "expert_b_test", "expert_c_test"]:
                    if k in summary:
                        rows.append({
                            "run": f"{d}/{k}",
                            "method": k.replace("_test", "").replace("_", " ").title(),
                            **summary[k],
                        })
                if "p1_test" in summary:
                    rows.append({
                        "run": f"{d}/p1",
                        "method": "P1 Baseline",
                        **summary["p1_test"],
                    })
        elif os.path.exists(results_path):
            with open(results_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            method_name = d.split("_")[-1]
            if method_name in {"42", "0", "123"}:  # seed suffix
                method_name = d.split("_")[-2]
            rows.append({
                "run": d,
                "method": method_name,
                **metrics,
            })
    return rows


def make_markdown_table(rows):
    if not rows:
        return "No results found.\n"
    cols = ["run", "method", "acc", "macro_f1", "weighted_f1", "head_f1", "tail_f1"]
    lines = ["| " + " | ".join(c.upper() for c in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "-")
            if isinstance(v, float):
                v = f"{v:.4f}"
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def make_latex_table(rows):
    """Generate LaTeX table for a paper."""
    if not rows:
        return "% No results\n"
    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Comparison of methods on long-tail classification.}\n"
        "\\label{tab:main}\n"
        "\\begin{tabular}{lccccc}\n"
        "\\toprule\n"
        "Method & Acc & Macro F1 & Wtd F1 & Head F1 & Tail F1 \\\\\n"
        "\\midrule\n"
    )
    body = ""
    for r in rows:
        method = r.get("method", "?")
        acc = r.get("acc", 0)
        mf1 = r.get("macro_f1", 0)
        wf1 = r.get("weighted_f1", 0)
        hf1 = r.get("head_f1", 0)
        tf1 = r.get("tail_f1", 0)
        body += f"{method} & {acc:.3f} & {mf1:.3f} & {wf1:.3f} & {hf1:.3f} & {tf1:.3f} \\\\\n"
    footer = "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return header + body + footer


def main():
    args = parse_args()
    rows = scan_results(args.filter)
    print(f"Found {len(rows)} result entries")

    md = make_markdown_table(rows)
    md_path = os.path.join(RESULTS_DIR, "SUMMARY.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Results Summary{' (filter: ' + args.filter + ')' if args.filter else ''}\n\n")
        f.write(md)
    print(f"Markdown saved: {md_path}")

    tex = make_latex_table(rows)
    tex_path = os.path.join(RESULTS_DIR, "SUMMARY.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"LaTeX saved: {tex_path}")

    # Print to console
    print("\n" + md)


if __name__ == "__main__":
    main()
