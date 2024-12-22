import typing as tp

from torch.nn import functional as F
from torch import nn

class LossModule(nn.Module):
    def __init__(self, name: str, weight: float = 1.0):
        super().__init__()

        self.name = name
        self.weight = weight

    def forward(self, info, *args, **kwargs):
        raise NotImplementedError
    
class ValueLoss(LossModule):
    def __init__(self, key: str, name, weight: float = 1.0):
        super().__init__(name=name, weight=weight)

        self.key = key
    
    def forward(self, info):
        return self.weight * info[self.key]

class L1Loss(LossModule):
    def __init__(self, key_a: str, key_b: str, weight: float = 1.0, mask_key: str = None, name: str = 'l1_loss'):
        super().__init__(name=name, weight=weight)

        self.key_a = key_a
        self.key_b = key_b

        self.mask_key = mask_key
    
    def forward(self, info):
        mse_loss = F.l1_loss(info[self.key_a], info[self.key_b], reduction='none')    

        if self.mask_key is not None and self.mask_key in info:
            mse_loss = mse_loss[info[self.mask_key]]

        mse_loss = mse_loss.mean()

        return self.weight * mse_loss
    
class MSELoss(LossModule):
    def __init__(self, key_a: str, key_b: str, key_c: float, weight: float = 1.0, mask_key: str = None, name: str = 'mse_loss'):
        super().__init__(name=name, weight=weight)

        self.key_a = key_a
        self.key_b = key_b
        self.key_c = key_c

        self.mask_key = mask_key
    
    def forward(self, info):
        print(info[self.key_a].shape, info[self.key_b].shape, info[self.mask_key].shape, (info[self.mask_key]==False).sum())
        mse_loss = F.mse_loss(info[self.key_a], info[self.key_b], reduction='none') 

        if self.mask_key is not None and self.mask_key in info and info[self.mask_key] is not None:
            mask = info[self.mask_key]

            if mask.ndim == 2 and mse_loss.ndim == 3:
                mask = mask.unsqueeze(1)

            if mask.shape[1] != mse_loss.shape[1]:
                mask = mask.repeat(1, mse_loss.shape[1], 1)

            mse_loss = mse_loss[mask]
        
        if self.key_c is not None and self.key_c in info:
            sample_weight = info[self.key_c]

            # Ensure sample_weight is broadcastable to the loss shape
            if sample_weight.ndim == 1 and mse_loss.ndim == 3:
                sample_weight = sample_weight.unsqueeze(1).unsqueeze(2) # [2,3] -> [[[2], [3]]]

            # Apply sample-wise weight to the loss
            mse_loss = mse_loss * sample_weight

        mse_loss = mse_loss.mean()

        return self.weight * mse_loss
    
class AuralossLoss(LossModule):
    def __init__(self, auraloss_module, input_key: str, target_key: str, name: str, weight: float = 1):
        super().__init__(name, weight)

        self.auraloss_module = auraloss_module

        self.input_key = input_key
        self.target_key = target_key

    def forward(self, info):
        loss = self.auraloss_module(info[self.input_key], info[self.target_key])

        return self.weight * loss
    
class MultiLoss(nn.Module):
    def __init__(self, losses: tp.List[LossModule]):
        super().__init__()

        self.losses = nn.ModuleList(losses)

    def forward(self, info):
        total_loss = 0

        losses = {}

        for loss_module in self.losses:
            module_loss = loss_module(info)
            total_loss += module_loss
            losses[loss_module.name] = module_loss

        return total_loss, losses