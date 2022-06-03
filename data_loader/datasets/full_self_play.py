import copy
import time

import chess
import torch
from torch.utils.data import Dataset
from colorama import Fore

from utils.util import is_game_end
from data_loader.mcts import MCTS

class FullSelfPlayDataset(Dataset):
    
    def __init__(self, 
                 mcts: MCTS,
                 num_of_sims: int=100, 
                 min_counts: int=10, 
                 simultaneous_mcts: int=32, 
                 move_limit: int=300, 
                 buffer_size: int=1e5,
                 ignore_loss_lim: float=1.0):
        super().__init__()
        
        # Initiate engines, will later assert that they aren't empty.
        self.white_engine = None
        self.black_engine = None
        
        # Initiate variables from outside
        self.num_of_sims: int = num_of_sims
        self.min_counts: int = min_counts
        self.simultaneous_mcts: int = simultaneous_mcts
        self.move_limit: int = move_limit
        self.buffer_size: int = buffer_size
        self.ignore_loss_lim: float = ignore_loss_lim
        
        # Initialize MCTS:
        self.mcts: MCTS = mcts
        
        # Initiate buffer
        self.buffer = {
            'board': torch.zeros((0, 16, 8, 8)),
            'policy': torch.zeros((0, 4864)),
            'value': torch.zeros((0))
        }
        self.follow_idx = 0
        while self.buffer['board'].size()[0] < self.buffer_size:
            self.load_game()
            print_size = self.buffer['board'].size()[0]
            print(f'Current number of positions: {print_size}')
        
    def __getitem__(self, sample_idx):
        
        # Assert engines are inputed
        assert self.mcts is not None, 'Must load an MCTS object into the dataloader'

        sampled_board = self.buffer['board'][sample_idx, :, :, :]
        sampled_policy = self.buffer['policy'][sample_idx, :]
        sampled_value = self.buffer['value'][sample_idx]
        sampled_win = True if sampled_value >= -self.ignore_loss_lim else False

        self.follow_idx += 1
        if self.follow_idx == self.buffer['board'].size()[0]:
            self.follow_idx = 0
            self.load_game()

        return {'board': sampled_board, 
                'policy': sampled_policy, 
                'value': sampled_value, 
                'win': sampled_win}
    
    def load_game(self):
        """
        Batch game running method
        """
        
        # Setting up simultaneous boards
        boards = [chess.Board() for _ in range(self.simultaneous_mcts)]
        
        for move_idx in range(self.move_limit):
            
            # Perform MCTS for each node per search
            sample_nodes_ending_tuples = [(is_game_end(board)) for board in boards]
            self._count_results(sample_nodes_ending_tuples)
            boards_active = [boards[idx] for idx in range(len(boards)) if not sample_nodes_ending_tuples[idx][0]]
            if len(boards_active) == 0: # Get out of the loop if all games have ended.
                break
                
            # if white and black engines are available, overwrite the current engines in the mcts.
            if self.white_engine is not None and self.black_engine is not None:
                color_white_flag = boards_active[0].turn
                self.mcts.model_good = self.white_engine if color_white_flag else self.black_engine
                self.mcts.model_evil = self.black_engine if color_white_flag else self.white_engine
                
            sample_nodes = self.mcts.run_multi(boards_active)
            
            # Collect all individual data from the nodes
            for sample_node in sample_nodes:
                
                
                # TODO: Activate CUDA
                board_collection, policy_collection, value_collection = self.mcts.collect_nodes_for_training(sample_node, 
                                                                                                min_counts=self.min_counts)
                
                # Collect data for variables
                self.buffer['board'] = torch.cat((self.buffer['board'], board_collection.to('cpu')), dim=0)
                self.buffer['policy'] = torch.cat((self.buffer['policy'], policy_collection.to('cpu')), dim=0)
                if self.buffer['value'].nelement == 0:
                    self.buffer['value'] = value_collection
                else:
                    self.buffer['value'] = torch.cat((self.buffer['value'], value_collection.to('cpu')), dim=0)
                    
                # Prune if the buffer size exceeds the limit
                if self.buffer['board'].size()[0] > self.buffer_size:
                    self.buffer['board'] = self.buffer['board'][-self.buffer_size:, :, :, :]
                    self.buffer['policy'] = self.buffer['policy'][-self.buffer_size:, :]
                    self.buffer['value'] = self.buffer['value'][-self.buffer_size:]
                
            # Push and print the moves
            samples = [sample_node.select_action(temperature=0.25) for sample_node in sample_nodes]
            sample_string = ', '.join(samples)
            print(f'[FullSelfPlay]: Pushed moves: ' + Fore.YELLOW + sample_string + Fore.RESET + f'\tMove: {(move_idx + 2) // 2}')
            for idx, board in enumerate(boards_active):
                board.push_san(samples[idx])
            boards = boards_active
            
            
    @staticmethod
    def _count_results(result_list):
        """
        Print result counts if a game has ended.
        """
        
        white_wins = 0
        black_wins = 0
        draws = 0
        
        for ind_result_tuple in result_list:
            
            if ind_result_tuple[0]:
                
                if ind_result_tuple[1] == 1:
                    white_wins += 1
                elif ind_result_tuple[1] == -1:
                    black_wins += 1
                elif ind_result_tuple[1] == 0:
                    draws += 1
                    
        if white_wins > 0:
            print('[FullSelfPlay] ' + Fore.RED + f'{white_wins} white wins.' + Fore.RESET)
        if black_wins > 0:
            print('[FullSelfPlay] ' + Fore.RED + f'{black_wins} black wins.' + Fore.RESET)
        if draws > 0:
            print('[FullSelfPlay] ' + Fore.RED + f'{draws} draws.' + Fore.RESET)
            

    def __len__(self):
        return self.buffer_size