import torch

from model.loss import Criterion

@torch.no_grad()
def loss_policy(criterion: Criterion, 
                prediction_dict: dict[str, torch.Tensor], 
                target_dict: dict[str, torch.Tensor]):
    """Goes for the criterion for the loss"""
    loss = criterion.loss_policy(prediction_dict, target_dict)
    return loss['policy']
    
@torch.no_grad()
def loss_value(criterion: Criterion, 
               prediction_dict: dict[str, torch.Tensor], 
               target_dict: dict[str, torch.Tensor]):
    """Goes for the criterion for the loss"""
    loss = criterion.loss_value(prediction_dict, target_dict)
    return loss['value']
