import unittest
import copy

from model.model import AttentionChess2
from data_loader.data_loaders import LichessLoader, BoardsLoader, FullSelfPlayLoader, collate_fn
from data_loader.mcts import MCTS


class test_loaders(unittest.TestCase):
    
    def setUp(self) -> None:
        self.lichess_loader = LichessLoader(batch_size=7, collate_fn=collate_fn, base_multiplier=0.95)
        self.boards_loader = BoardsLoader(batch_size=7, collate_fn=collate_fn, base_multiplier=0.95)
        self.model = AttentionChess2()
        self.mcts = MCTS(model_good=copy.deepcopy(self.model), model_evil=copy.deepcopy(self.model), num_sims=10)
        self.fsp_loader = FullSelfPlayLoader(batch_size=7, mcts=self.mcts, collate_fn=collate_fn, num_of_sims=10, min_counts=5, move_limit=10, num_workers=0)
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
        
    def test_boards_loader(self):
        for data_dict in self.boards_loader:
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
        
    def test_fsp_loader(self):
        for data_dict in self.fsp_loader:
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
        self.assertGreaterEqual(len(self.fsp_loader), 10)

if __name__ == '__main__':
    unittest.main()
        