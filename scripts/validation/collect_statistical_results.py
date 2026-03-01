#!/usr/bin/env python3
"""收集统计验证（5-fold CV、LOSO）结果，生成可报告格式"""

import pickle
import json
from pathlib import Path

RESULTS_ROOT = Path("/root/autodl-tmp/TwoTST/results")
OUTPUT_MD = RESULTS_ROOT / "statistical_validation_summary.md"


def load_results(name):
    """加载 results.pkl"""
    pkl_path = RESULTS_ROOT / name / "results.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def main():
    lines = [
        "# 统计验证结果汇总",
        "",
        "## 1. 5-fold CV (Subject-level, mean ± std, 95% CI)",
        "",
    ]

    for name in ["cv5_baseline_concat", "cv5_baseline_cross_attention"]:
        r = load_results(name)
        if r is None:
            lines.append(f"### {name}: 未完成或未找到")
            lines.append("")
            continue
        mm = r.get("mean_metrics", {})
        sm = r.get("std_metrics", {})
        ci = r.get("ci_95_metrics", {})
        n_folds = r.get("reproducibility", {}).get("n_folds", "?")
        lines.append(f"### {name} (n_folds={n_folds})")
        lines.append("")
        lines.append("| 指标 | Mean ± Std | 95% CI |")
        lines.append("|------|------------|--------|")
        for k in ["accuracy", "auc", "f1", "precision", "recall"]:
            if k in mm:
                lo, hi = ci.get(k, (0, 0))
                lines.append(f"| {k} | {mm[k]:.4f} ± {sm[k]:.4f} | [{lo:.4f}, {hi:.4f}] |")
        lines.append("")
        lines.append("")

    lines.append("## 2. LOSO (Leave-One-Site-Out, 跨站点泛化)")
    lines.append("")

    r = load_results("loso_baseline_concat")
    if r is None:
        lines.append("### loso_baseline_concat: 未完成或未找到")
    else:
        mm = r.get("mean_metrics", {})
        sm = r.get("std_metrics", {})
        ci = r.get("ci_95_metrics", {})
        n_folds = r.get("reproducibility", {}).get("n_folds", "?")
        lines.append(f"### loso_baseline_concat (n_sites={n_folds})")
        lines.append("")
        lines.append("| 指标 | Mean ± Std | 95% CI |")
        lines.append("|------|------------|--------|")
        for k in ["accuracy", "auc", "f1", "precision", "recall"]:
            if k in mm:
                lo, hi = ci.get(k, (0, 0))
                lines.append(f"| {k} | {mm[k]:.4f} ± {sm[k]:.4f} | [{lo:.4f}, {hi:.4f}] |")
        lines.append("")
        lines.append("*LOSO 每次留出一个站点作测试集，评估模型在未见扫描仪/协议上的泛化能力。*")
        lines.append("")

    content = "\n".join(lines)
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(content)
    print(content)
    print(f"\n已保存到 {OUTPUT_MD}")


if __name__ == "__main__":
    main()
