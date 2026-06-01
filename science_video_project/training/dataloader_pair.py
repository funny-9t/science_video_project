from typing import Dict, List, Tuple

import torch


def collate_pair(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    def stack(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {k: torch.stack([x[k] for x in samples], dim=0) for k in samples[0].keys()}

    pos_batch = stack([x[0] for x in batch])
    neg_batch = stack([x[1] for x in batch])
    return pos_batch, neg_batch
