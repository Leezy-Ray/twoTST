"""
Mean Relative Change (指标 C)：逐参数相对变化
用于衡量微调前后各模块参数变化程度
"""

import torch


def mean_relative_change(init_state: dict, final_state: dict, eps: float = 1e-8) -> float:
    """
    计算两个 state_dict 之间的 Mean Relative Change（指标 C）
    Mean Relative Change = (1/n) * sum_i( |θ_final,i - θ_init,i| / (|θ_init,i| + ε) )

    init_state, final_state: 同一模块的 state_dict，key 需一致
    """
    all_rels = []
    for k in init_state:
        if k not in final_state:
            continue
        init_p = init_state[k].float().flatten()
        final_p = final_state[k].float().flatten()
        denom = init_p.abs() + eps
        rel = ((final_p - init_p).abs() / denom)
        all_rels.append(rel)
    if not all_rels:
        return float("nan")
    return torch.cat(all_rels).mean().item()


def extract_module_state(model_state: dict, prefix: str) -> dict:
    """从完整 model.state_dict() 中提取以 prefix 开头的子模块 state"""
    out = {}
    for k, v in model_state.items():
        if k.startswith(prefix) and k[len(prefix):]:
            subk = k[len(prefix):].lstrip(".")
            out[subk] = v
    return out


def compute_module_changes(cont_ckpt: dict, model_state: dict, eps: float = 1e-8) -> dict:
    """
    计算 tst1, tst2, proj_head1, proj_head2 四个模块的 Mean Relative Change

    cont_ckpt: contrastive checkpoint dict，含 tst1_state_dict, tst2_state_dict, proj_head1_state_dict, proj_head2_state_dict
    model_state: 微调后的 model.state_dict()
    """
    modules = ["tst1", "tst2", "proj_head1", "proj_head2"]
    changes = {}
    for name in modules:
        ckpt_key = f"{name}_state_dict"
        prefix = name + "."
        if ckpt_key not in cont_ckpt:
            changes[name] = float("nan")
            continue
        init_sd = cont_ckpt[ckpt_key]
        final_sd = extract_module_state(model_state, prefix)
        changes[name] = mean_relative_change(init_sd, final_sd, eps=eps)
    return changes
