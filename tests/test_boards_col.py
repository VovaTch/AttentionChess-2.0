import unittest
import pickle

import torch


class test_boards_col(unittest.TestCase):
    
    def setUp(self):
        
        # load 2 pickle files
        with open('data/boards_data/lichess_boards_001.pkl', 'rb') as f:
            self.pkl_data_1 = pickle.load(f)
        with open('data/boards_data/lichess_boards_001.pkl', 'rb') as f:
            self.pkl_data_2 = pickle.load(f)
            
        return super().setUp()
    
    def test_size_1(self):
        
        boards_tensor = self.pkl_data_1[0]
        policy_tensor = self.pkl_data_1[1]
        value_tensor = self.pkl_data_1[2]
        
        self.assertEqual(list(boards_tensor.size()), [50000, 16, 8, 8])
        self.assertEqual(list(policy_tensor.size()), [50000, 4864])
        self.assertEqual(list(value_tensor.size()), [50000])

    
    def test_size_2(self):
        
        boards_tensor = self.pkl_data_2[0]
        policy_tensor = self.pkl_data_2[1]
        value_tensor = self.pkl_data_2[2]
        
        self.assertEqual(list(boards_tensor.size()), [50000, 16, 8, 8])
        self.assertEqual(list(policy_tensor.size()), [50000, 4864])
        self.assertEqual(list(value_tensor.size()), [50000])
        
    def test_overlap(self):
        
        policy_tensor = self.pkl_data_2[1]
        policy_tensor_sum = torch.sum(policy_tensor)
        
        self.assertNotEqual(4864, policy_tensor_sum.item())
        
        
if __name__ == '__main__':
    unittest.main()