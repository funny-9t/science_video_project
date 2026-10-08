from typing import Dict, List, Tuple

import torch


def collate_pair(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    def stack(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        keys = samples[0].keys()
        result = {}
        for k in keys:
            tensors = [x[k] for x in samples]
            if not isinstance(tensors[0], torch.Tensor):
                result[k] = tensors
                continue
            if k in {"frame_features", "cover_temporal_feat"}:
                # 变长序列: 找到最大长度，pad
                max_len = max(t.size(0) for t in tensors)
                feature_dim = next((t.size(-1) for t in tensors if t.ndim == 2 and t.size(0) > 0), 768)
                if max_len == 0:
                    result[k] = torch.zeros(len(samples), 0, feature_dim)
                else:
                    padded = torch.zeros(len(samples), max_len, feature_dim)
                    for i, t in enumerate(tensors):
                        if t.size(0) > 0:
                            if t.size(-1) != feature_dim:
                                raise ValueError(
                                    f"Inconsistent frame feature dims: {t.size(-1)} vs {feature_dim}"
                                )
                            padded[i, : t.size(0), :] = t
                    result[k] = padded
            else:
                result[k] = torch.stack(tensors, dim=0)
        return result

    pos_batch = stack([x[0] for x in batch])
    neg_batch = stack([x[1] for x in batch])
    return pos_batch, neg_batch
