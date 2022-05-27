from typing import Dict

import torch

from model.loss import Criterion

@torch.no_grad()
def loss_policy(criterion: Criterion, 
                prediction_dict: Dict[str, torch.Tensor], 
                target_dict: Dict[str, torch.Tensor]):
    """Goes for the criterion for the loss"""
    loss = criterion.loss_policy(prediction_dict, target_dict)
    return loss['policy']
    
@torch.no_grad()
def loss_value(criterion: Criterion, 
               prediction_dict: Dict[str, torch.Tensor], 
               target_dict: Dict[str, torch.Tensor]):
    """Goes for the criterion for the loss"""
    loss = criterion.loss_value(prediction_dict, target_dict)
    return loss['value']
