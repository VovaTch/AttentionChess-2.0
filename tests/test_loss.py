import unittest
import copy

import torch
import chess

from model import AttentionChess2
from model.loss import Criterion
from data_loader.data_loaders import collate_fn, LichessLoader

class test_loss(unittest.TestCase):
    
    def setUp(self) -> None:
        self.device = 'cuda:0'
        self.model = AttentionChess2(device='cuda:0', aux_outputs=True)
        self.model = self.model.to(self.device)
        self.board = chess.Board()
        self.model.eval()
        self.criteria = Criterion(['loss_policy', 'loss_value'])
        self.lichess_loader = LichessLoader(batch_size=5, collate_fn=collate_fn, base_multiplier=0.95)
        
        return super().setUp()
    
    def test_loss_policy(self):
        
        # Create targets
        for target_dict in self.lichess_loader:
            for target_key in target_dict:
                target_dict[target_key] = target_dict[target_key].to(self.device)
            break
        
        # Create output
        inputs = [copy.deepcopy(self.board) for _ in range(5)]
        output_dict = self.model(inputs)
        
        # Compute loss
        loss_dict = self.criteria(output_dict, target_dict)
        entries_existing_policy = \
            'policy' in loss_dict and\
            'policy_aux_0' in loss_dict and\
            'policy_aux_1' in loss_dict and\
            'policy_aux_2' in loss_dict and\
            'policy_aux_3' in loss_dict and\
            'policy_aux_4' in loss_dict
        entries_existing_value = \
            'value' in loss_dict and\
            'value_aux_0' in loss_dict and\
            'value_aux_1' in loss_dict and\
            'value_aux_2' in loss_dict and\
            'value_aux_3' in loss_dict and\
            'value_aux_4' in loss_dict
            
        self.assertTrue(entries_existing_policy)
        self.assertTrue(entries_existing_value)
        
        
if __name__ == '__main__':
    unittest.main()