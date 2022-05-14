import unittest

from data_loader.data_loaders import LichessLoader, collate_fn


class test_loaders(unittest.TestCase):
    
    def setUp(self) -> None:
        self.lichess_loader = LichessLoader(batch_size=7, collate_fn=collate_fn, base_multiplier=0.95)
        return super().setUp()
    
    def test_lichess_loader(self):
        for data_dict in self.lichess_loader:
            break
        
        board_size = list(data_dict['board'].size())
        policy_size = list(data_dict['policy'].size())
        value_size = list(data_dict['value'].size())
        win_size = list(data_dict['win'].size())
        
        # Assert sizes
        self.assertEqual(board_size, [7, 16, 8, 8])
        self.assertEqual(policy_size, [7, 4864])
        self.assertEqual(value_size, [7])
        self.assertEqual(win_size, [7])
        

if __name__ == '__main__':
    unittest.main()
        