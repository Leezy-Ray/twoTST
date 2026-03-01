#!/usr/bin/env python3
"""
构建 CC200 ROI 与脑区学名（AAL）的对应表
可选：提供 Craddock 200 的 NIfTI，通过质心与 AAL 重叠得到学名；
否则尝试使用 nilearn 的 Craddock 2012 中 200-ROI 体积（若存在）生成。
输出：cc200_roi_labels.csv（roi_index, region_name_en, [lobe]）
"""

import os
import sys
import csv
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 默认输出路径
DEFAULT_LABELS_DIR = PROJECT_ROOT / "data" / "labels"
DEFAULT_CSV = DEFAULT_LABELS_DIR / "cc200_roi_labels.csv"


def build_from_craddock_nii(craddock_nii_path, aal_atlas, output_csv, n_rois=200):
    """
    从 Craddock 200 的 3D NIfTI（体素值为 1..200）提取每个 ROI 的质心，
    再在 AAL 图谱上查该 MNI 坐标对应的脑区名称。
    """
    import numpy as np
    import nibabel as nib
    from nilearn.image import resample_to_img

    img = nib.load(craddock_nii_path)
    data = img.get_fdata()
    # 假设 3D，或 4D 取第一帧
    if data.ndim == 4:
        data = data[..., 0]
    assert data.ndim == 3

    aal_img = nib.load(aal_atlas.maps)
    aal_data = aal_img.get_fdata()
    if aal_data.ndim == 4:
        aal_data = aal_data[..., 0]

    # 将 Craddock 重采样到 AAL 空间以便坐标一致（若需）
    from nilearn.image import new_img_like
    try:
        from nilearn.image import resample_to_img
        craddock_resampled = resample_to_img(img, aal_img, interpolation='nearest')
        data = craddock_resampled.get_fdata()
        if data.ndim == 4:
            data = data[..., 0]
    except Exception:
        pass  # 若形状一致则直接用

    # 每个 ROI 的质心（MNI 坐标）
    rows = []
    for roi_id in range(1, n_rois + 1):
        mask = (data == roi_id)
        if not np.any(mask):
            rows.append((roi_id - 1, f"CC200_roi_{roi_id - 1}", ""))
            continue
        xs, ys, zs = np.where(mask)
        cx, cy, cz = np.mean(xs), np.mean(ys), np.mean(zs)
        # 体素坐标转 MNI（若 AAL 与 Craddock 同空间则可直接用体素索引查 AAL）
        ix, iy, iz = int(round(cx)), int(round(cy)), int(round(cz))
        if (0 <= ix < aal_data.shape[0] and 0 <= iy < aal_data.shape[1] and 0 <= iz < aal_data.shape[2]):
            aal_val = int(aal_data[ix, iy, iz])
            if aal_val > 0 and hasattr(aal_atlas, 'labels') and aal_val <= len(aal_atlas.labels):
                name = aal_atlas.labels[aal_val - 1]  # AAL 常为 1-based
            else:
                name = f"CC200_roi_{roi_id - 1}"
        else:
            name = f"CC200_roi_{roi_id - 1}"
        rows.append((roi_id - 1, name, ""))
    return rows


def build_from_nilearn(output_csv, n_rois=200):
    """
    使用 nilearn 的 AAL 与 Craddock 2012（若有 200-ROI 体积）生成标签。
    Craddock 2012 在 nilearn 中为 4D，43 个体积，未必有 200；若无则只输出 AAL 参考。
    """
    try:
        from nilearn import datasets
        from nilearn.image import index_img
        import numpy as np
    except ImportError:
        return None

    # AAL：116 个脑区，有 labels
    aal = datasets.fetch_atlas_aal()
    aal_labels = getattr(aal, 'labels', None) or []
    if isinstance(aal_labels, np.ndarray):
        aal_labels = aal_labels.tolist()

    # 尝试 Craddock 2012
    try:
        craddock = datasets.fetch_atlas_craddock_2012()
        # 4D: (47, 56, 46, 43) -> 取一帧看唯一值数量
        img = craddock['scorr_mean']
        if isinstance(img, str):
            import nibabel as nib
            img = nib.load(img)
        data = img.get_fdata() if hasattr(img, 'get_fdata') else np.asarray(img.dataobj)
        if data.ndim == 4:
            n_vols = data.shape[-1]
            best_vol = None
            best_n = 0
            for v in range(n_vols):
                uniq = np.unique(data[..., v])
                uniq = uniq[uniq > 0]
                if len(uniq) <= n_rois and len(uniq) > best_n:
                    best_n = len(uniq)
                    best_vol = v
            if best_vol is not None and best_n >= 100:
                # 用该体积生成标签，需 AAL 重叠则要 resample；这里简化：只生成索引名
                rows = [(i, f"CC200_roi_{i}", "") for i in range(n_rois)]
                return rows
    except Exception:
        pass

    # 无 Craddock 200 时：输出占位 + 用 AAL 名称循环填充（便于与医学对照时手动校对）
    rows = []
    for i in range(n_rois):
        if aal_labels and i < len(aal_labels):
            name = aal_labels[i]
        elif aal_labels:
            name = aal_labels[i % len(aal_labels)] + f"_{i // len(aal_labels)}"
        else:
            name = f"CC200_roi_{i}"
        rows.append((i, name, ""))
    return rows


def load_default_placeholder(n_rois=200):
    """返回占位表：roi_index, region_name_en（无 nilearn 时用）。"""
    return [(i, f"CC200_roi_{i}", "") for i in range(n_rois)]


def main():
    parser = argparse.ArgumentParser(description='Build CC200 ROI index -> brain region name (AAL)')
    parser.add_argument('--craddock_nii', type=str, default=None,
                        help='Path to Craddock 200 3D NIfTI (voxel values 1..200)')
    parser.add_argument('--output', type=str, default=str(DEFAULT_CSV),
                        help='Output CSV path')
    parser.add_argument('--n_rois', type=int, default=200)
    args = parser.parse_args()

    rows = None
    if args.craddock_nii and os.path.exists(args.craddock_nii):
        try:
            from nilearn import datasets
            aal = datasets.fetch_atlas_aal()
            rows = build_from_craddock_nii(args.craddock_nii, aal, args.output, args.n_rois)
        except Exception as e:
            print(f"Build from NIfTI failed: {e}")
    if rows is None:
        rows = build_from_nilearn(args.output, args.n_rois) or load_default_placeholder(args.n_rois)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['roi_index', 'region_name_en', 'lobe'])
        w.writerows(rows)
    print(f"Saved {len(rows)} rows to {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
