import unittest

import chess
import copy

from model import AttentionChess2
from utils import num_param


class test_model(unittest.TestCase):
    
    def setUp(self):
        self.device = 'cuda:0'
        self.model = AttentionChess2(device='cuda:0', aux_outputs=True)
        self.model = self.model.to(self.device)
        self.board = chess.Board()
        self.model.eval()
        num_param_tot = num_param(self.model)
        print(f'Number of parameters is {num_param_tot:,}.')
        return super().setUp()
    
    def test_forward_2_boards(self):
        
        # Run the boards through the net
        board_list = [copy.deepcopy(self.board), copy.deepcopy(self.board)]
        output_dict = self.model(board_list)
        
        # Check the sizes
        policy_size = list(output_dict['policy'].size())
        value_size = list(output_dict['value'].size())
        
        self.assertEqual(policy_size, [2, 4864])
        self.assertEqual(value_size, [2])
        
        # Check the sizes for the aux modules
        for idx in range(5):
            policy_size = list(output_dict[f'policy_aux_{idx}'].size())
            value_size = list(output_dict[f'value_aux_{idx}'].size())
            
            self.assertEqual(policy_size, [2, 4864])
            self.assertEqual(value_size, [2])
            
    def test_post_processing(self):
            
        # Run the boards through the net
        board_list = [copy.deepcopy(self.board), copy.deepcopy(self.board)]
        output_dict = self.model(board_list)
        policy_list, value_list = self.model.post_process(board_list, output_dict, print_output=False)
        
        # Check sizes
        self.assertEqual(len(policy_list), 2)
        self.assertEqual(len(value_list), 2)
        
        # Check sizes of inner dict of the policy
        self.assertEqual(len(policy_list[0]), 20)
            
if __name__ == '__main__':
    unittest.main()