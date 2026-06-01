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
            if k == "frame_features":
                # 变长序列: 找到最大长度，pad
                max_len = max(t.size(0) for t in tensors)
                if max_len == 0:
                    result[k] = torch.zeros(len(samples), 0, 512)
                else:
                    padded = torch.zeros(len(samples), max_len, 512)
                    for i, t in enumerate(tensors):
                        if t.size(0) > 0:
                            padded[i, : t.size(0), :] = t
                    result[k] = padded
            else:
                result[k] = torch.stack(tensors, dim=0)
        return result

    pos_batch = stack([x[0] for x in batch])
    neg_batch = stack([x[1] for x in batch])
    return pos_batch, neg_batch
