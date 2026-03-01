"""
数据划分模块
实现受试者级划分与站点分层K折，避免滑窗导致的信息泄漏
"""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def get_subject_level_fold_splits(
    labels,
    subject_indices,
    site_ids=None,
    n_splits=5,
    val_ratio=0.15,
    seed=42,
):
    """
    生成受试者级、站点分层（可选）的K折划分。
    同一受试者的所有样本（含滑窗）只会出现在 train/val 或 test 之一中。

    Args:
        labels: ndarray (n_samples,), 样本标签
        subject_indices: ndarray (n_samples,), 每个样本对应的受试者索引
        site_ids: ndarray (n_samples,), 每个样本对应的站点ID，用于站点分层（可选）
        n_splits: 折数
        val_ratio: 从训练集中划分验证集的比例
        seed: 随机种子

    Returns:
        splits: list of dict, 每折包含:
            - train_idx: 训练样本索引
            - val_idx: 验证样本索引
            - test_idx: 测试样本索引
            - train_subjects: 训练集受试者索引
            - test_subjects: 测试集受试者索引
    """
    unique_subjects = np.unique(subject_indices)
    n_subjects = len(unique_subjects)

    # 每个受试者取一个样本得到 subject -> label 映射
    subject_to_label = {}
    subject_to_site = {}
    for s in unique_subjects:
        mask = subject_indices == s
        subject_to_label[s] = int(labels[mask][0])
        if site_ids is not None:
            subject_to_site[s] = site_ids[mask][0]

    subject_labels = np.array([subject_to_label[s] for s in unique_subjects])
    subject_sites = np.array([subject_to_site.get(s, "unknown") for s in unique_subjects]) if site_ids is not None else None

    # 使用 StratifiedGroupKFold: groups=subject_indices 等价于按受试者划分
    # 需传入样本级 groups，同一 subject 的样本有相同 group id
    groups = subject_indices

    try:
        kfold = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_iter = kfold.split(np.arange(len(labels)), labels, groups)
    except TypeError:
        # sklearn < 0.24 无 StratifiedGroupKFold，退化为按受试者手动划分
        fold_iter = _fallback_subject_stratified_kfold(
            unique_subjects, subject_labels, n_splits, seed
        )
        # fold_iter yields (train_subject_indices, test_subject_indices)
        # 需要转换为样本索引
        fold_iter = _convert_subject_splits_to_sample_splits(
            fold_iter, subject_indices, unique_subjects
        )

    splits = []
    for fold_idx, (train_val_sample_idx, test_sample_idx) in enumerate(fold_iter):
        train_val_subjects = np.unique(subject_indices[train_val_sample_idx])
        test_subjects = np.unique(subject_indices[test_sample_idx])

        # 从 train_val 中再划分 train / val，仍按受试者级
        train_val_labels_subj = np.array([subject_to_label[s] for s in train_val_subjects])
        n_val_subjects = max(1, int(len(train_val_subjects) * val_ratio))
        n_val_subjects = min(n_val_subjects, len(train_val_subjects) - 1)

        if n_val_subjects < 1:
            train_subjects = train_val_subjects
            val_subjects = np.array([], dtype=np.int64)
        else:
            try:
                train_subj_idx, val_subj_idx = train_test_split(
                    np.arange(len(train_val_subjects)),
                    test_size=val_ratio,
                    random_state=seed,
                    stratify=train_val_labels_subj,
                )
                train_subjects = train_val_subjects[train_subj_idx]
                val_subjects = train_val_subjects[val_subj_idx]
            except ValueError:
                train_subjects = train_val_subjects
                val_subjects = np.array([], dtype=np.int64)

        train_idx = np.where(np.isin(subject_indices, train_subjects))[0]
        val_idx = np.where(np.isin(subject_indices, val_subjects))[0]

        splits.append({
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_sample_idx,
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "test_subjects": test_subjects,
        })

    return splits


def _fallback_subject_stratified_kfold(unique_subjects, subject_labels, n_splits, seed):
    """当无 StratifiedGroupKFold 时，按受试者做分层K折"""
    from sklearn.model_selection import StratifiedKFold

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(unique_subjects, subject_labels):
        yield unique_subjects[train_idx], unique_subjects[test_idx]


def _convert_subject_splits_to_sample_splits(fold_iter, subject_indices, unique_subjects):
    """将受试者级划分转换为样本级划分"""
    for train_subjects, test_subjects in fold_iter:
        train_val_sample_idx = np.where(np.isin(subject_indices, train_subjects))[0]
        test_sample_idx = np.where(np.isin(subject_indices, test_subjects))[0]
        yield train_val_sample_idx, test_sample_idx


def get_loso_fold_splits(
    labels,
    subject_indices,
    site_ids,
    val_ratio=0.15,
    seed=42,
):
    """
    Leave-One-Site-Out (LOSO) 划分：每次留一个站点作为测试集，其余站点作为训练集。
    用于评估跨站点泛化能力。

    Args:
        labels: (n_samples,)
        subject_indices: (n_samples,)
        site_ids: (n_samples,), 站点ID
        val_ratio: 从训练集中划分验证集的比例（按受试者）
        seed: 随机种子

    Returns:
        splits: list of dict, 每折对应一个被留出的站点
    """
    if site_ids is None:
        raise ValueError("LOSO requires site_ids")

    unique_sites = np.unique(site_ids)
    subject_to_label = {s: int(labels[subject_indices == s][0]) for s in np.unique(subject_indices)}
    subject_to_site = {s: site_ids[subject_indices == s][0] for s in np.unique(subject_indices)}

    splits = []
    for test_site in unique_sites:
        train_val_subjects = np.array([s for s in np.unique(subject_indices) if subject_to_site[s] != test_site])
        test_subjects = np.array([s for s in np.unique(subject_indices) if subject_to_site[s] == test_site])

        if len(test_subjects) == 0:
            continue

        # 从 train_val 中划分 val
        train_val_labels = np.array([subject_to_label[s] for s in train_val_subjects])
        n_val = max(1, int(len(train_val_subjects) * val_ratio))
        n_val = min(n_val, len(train_val_subjects) - 1)
        if n_val < 1:
            train_subjects = train_val_subjects
            val_subjects = np.array([], dtype=np.int64)
        else:
            try:
                train_idx, val_idx = train_test_split(
                    np.arange(len(train_val_subjects)),
                    test_size=val_ratio,
                    random_state=seed,
                    stratify=train_val_labels,
                )
                train_subjects = train_val_subjects[train_idx]
                val_subjects = train_val_subjects[val_idx]
            except ValueError:
                train_subjects = train_val_subjects
                val_subjects = np.array([], dtype=np.int64)

        train_idx = np.where(np.isin(subject_indices, train_subjects))[0]
        val_idx = np.where(np.isin(subject_indices, val_subjects))[0]
        test_idx = np.where(np.isin(subject_indices, test_subjects))[0]

        splits.append({
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "test_site": str(test_site),
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "test_subjects": test_subjects,
        })

    return splits


def get_subject_level_train_val_test_split(
    labels,
    subject_indices,
    site_ids=None,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    seed=42,
):
    """
    受试者级的单次 train/val/test 划分（用于预训练等）。

    Args:
        labels: (n_samples,)
        subject_indices: (n_samples,)
        site_ids: (n_samples,), 可选
        train_ratio, val_ratio, test_ratio: 比例，需和为 1
        seed: 随机种子

    Returns:
        train_idx, val_idx, test_idx: 样本索引
    """
    unique_subjects = np.unique(subject_indices)
    subject_to_label = {s: int(labels[subject_indices == s][0]) for s in unique_subjects}
    subject_labels = np.array([subject_to_label[s] for s in unique_subjects])

    # 先分 train vs (val+test)
    train_subj, temp_subj = train_test_split(
        unique_subjects,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=subject_labels,
    )
    temp_labels = np.array([subject_to_label[s] for s in temp_subj])
    val_subj, test_subj = train_test_split(
        temp_subj,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=seed,
        stratify=temp_labels,
    )

    train_idx = np.where(np.isin(subject_indices, train_subj))[0]
    val_idx = np.where(np.isin(subject_indices, val_subj))[0]
    test_idx = np.where(np.isin(subject_indices, test_subj))[0]

    return train_idx, val_idx, test_idx
