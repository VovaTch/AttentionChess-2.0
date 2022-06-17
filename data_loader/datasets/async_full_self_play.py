import asyncio

import chess
import torch
from torch.utils.data import Dataset
from colorama import Fore

from utils.util import is_game_end
from data_loader.async_mcts import MCTS

class FullSelfPlayAsyncDataset(Dataset):
    
    def __init__(self, 
                 mcts: MCTS,
                 num_of_sims: int=100, 
                 min_counts: int=10, 
                 simultaneous_mcts: int=32,
                 simultaneous_mcts_pre_generated: int=64, 
                 move_limit: int=300, 
                 buffer_size: int=1e5,
                 ignore_loss_lim: float=1.0):
        super().__init__()
        
        # Initialize event loop
        self.loop = asyncio.get_event_loop()
        
        # Initiate engines, will later assert that they aren't empty.
        self.white_engine = None
        self.black_engine = None
        
        # Initiate variables from outside
        self.num_of_sims: int = num_of_sims
        self.min_counts: int = min_counts
        self.simultaneous_mcts: int = simultaneous_mcts_pre_generated # Generate moves from a larger batch to fill the memory buffer.
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
        
        # Trying here the asynchronous mcts
        background_tasks = set()
        
        while self.buffer['board'].size()[0] < self.buffer_size:
            task = asyncio.create_task(self.load_game())
            background_tasks.add(task)
            print_size = self.buffer['board'].size()[0]
            print(f'Current number of positions: {print_size}')
            task.add_done_callback(background_tasks.discard)
            
        self.simultaneous_mcts = simultaneous_mcts
        
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
    
    async def load_game(self):
        """
        Async loading of a game; maybe it will be faster than the vanilla run_multiple?
        """
        
        board = chess.Board()
        
        for move_idx in range(self.move_limit):
            
            # Perform MCTS for the selected node
            game_end, _ = is_game_end(board)
            if game_end:
                break
        
            # if white and black engines are available, overwrite the current engines in the mcts.
            if self.white_engine is not None and self.black_engine is not None:
                color_white_flag = board.turn
                self.mcts.model_good = self.white_engine if color_white_flag else self.black_engine
                self.mcts.model_evil = self.black_engine if color_white_flag else self.white_engine
                
            root_node = await self.mcts.run(board)
            
            # Collect all the data from the training
            board_collection, policy_collection, value_collection = self.mcts.collect_nodes_for_training(root_node, 
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
            
            sample = root_node.select_action(temperature=0.25)
            print(f'[AsyncFSP]: Pushed sample for move {(move_idx + 2) // 2}: {sample}')
            board.push_san(sample)
            
            
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