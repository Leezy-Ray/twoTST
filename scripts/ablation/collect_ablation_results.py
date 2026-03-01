#!/usr/bin/env python3
"""收集消融实验 (baseline + freeze) 的 results.json 并生成汇总表"""

import json
from pathlib import Path

FINETUNE_DIR = Path("/root/autodl-tmp/TwoTST/checkpoints/finetune")
OUTPUT_JSON = Path("/root/autodl-tmp/TwoTST/results/ablation_summary.json")
OUTPUT_CSV = Path("/root/autodl-tmp/TwoTST/results/ablation_summary.csv")

# 48 个消融实验的预期名称（不含早期运行的 base 配置）
EXPECTED_BASELINE = [
    "baseline_concat", "baseline_gated", "baseline_cross_attention",
    "baseline_bilinear", "baseline_attention_pooling",
]
EXPECTED_PROJECTION = [
    "projection_tst1_only", "projection_tst2_only",
    "projection_fusion_concat", "projection_fusion_gated",
    "projection_fusion_cross_attention", "projection_fusion_bilinear",
    "projection_fusion_attention_pooling",
]
FREEZE_SUFFIXES = ["unfrozen", "freeze_both", "freeze_tst1", "freeze_tst2"]


def build_expected_names():
    names = []
    for base in EXPECTED_BASELINE:
        for s in FREEZE_SUFFIXES:
            names.append(f"{base}_{s}")
    for base in EXPECTED_PROJECTION:
        for s in FREEZE_SUFFIXES:
            names.append(f"{base}_{s}")
    return names


def main():
    expected = set(build_expected_names())
    rows = []
    missing = []

    for name in sorted(expected):
        res_path = FINETUNE_DIR / name / "results.json"
        if not res_path.exists():
            missing.append(name)
            continue
        with open(res_path) as f:
            data = json.load(f)
        auc = data.get("auc", 0)
        acc = data.get("accuracy", 0)
        f1 = data.get("f1", 0)
        best_val = data.get("best_val_auc", 0)
        rows.append({
            "name": name,
            "auc": round(auc, 4),
            "accuracy": round(acc, 4),
            "f1": round(f1, 4),
            "best_val_auc": round(best_val, 4),
        })

    summary = {
        "total_expected": 48,
        "collected": len(rows),
        "missing": missing,
        "experiments": rows,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # CSV 便于制表
    with open(OUTPUT_CSV, "w", encoding="utf-8") as f:
        f.write("name,auc,accuracy,f1,best_val_auc\n")
        for r in rows:
            f.write(f"{r['name']},{r['auc']},{r['accuracy']},{r['f1']},{r['best_val_auc']}\n")

    print(f"已收集 {len(rows)}/48 个实验")
    if missing:
        print(f"缺失: {missing}")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"CSV:  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
