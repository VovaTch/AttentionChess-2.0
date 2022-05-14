from email.mime import base
import os
import requests
import copy
import logging

import torch
from torch.utils.data import Dataset
import chess
import chess.pgn

from utils import board_to_representation, move_to_word


class LichessDataset(Dataset):
    """
    A simple dataset that takes a PGN file and create training data from it. TODO: Implement a FFCV dataset out of that.
    """
    logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)
    
    def __init__(self,
                 dataset_path: str,
                 base_multiplier: float=1.0,
                 dataset_download_url: str='https://database.lichess.org/standard/lichess_db_standard_rated_2016-09.pgn.bz2'):
        
        super(LichessDataset).__init__()
        
        # Download the dataset if it doesn't exist
        if not os.path.isfile(dataset_path):
            r = requests.get(dataset_download_url, allow_redirects=True)
            open(dataset_path, 'wb').write(r.content)
        self.pgn = open(dataset_path, encoding='utf-8')
            
        self.follow_idx = 0
        self.game_length = 0
        self.board_batch = None
        self.policy_batch = None
        self.value_batch = None
        self.win_flag_batch = None
        self.base_multiplier = base_multiplier
        
    def load_game(self):
        
        # Load a game that ends in a victory/draw/defeat
        while True:
            game = chess.pgn.read_game(self.pgn)
            move_counter = 0
            for _ in enumerate(game.mainline_moves()):
                move_counter += 1
            
            if 'Termination' in game.headers and game.headers['Termination'] == 'Normal' and move_counter > 0:
                break
            
        # Initiate the board and enter the result
        board = game.board()
        result = game.headers['Result']
        if result == '1-0':
            base_eval = 5
        elif result == '0-1':
            base_eval = -5
        else:
            base_eval = 0
        turn_constant = 1
            
        # Reset the batch variables
        self.board_batch = torch.zeros((0, 16, 8, 8))
        self.policy_batch = torch.zeros((0, 4864))
        self.value_batch = torch.zeros(0)
        self.win_flag_batch = []
        
        # Loop over the game moves
        for idx, move in enumerate(game.mainline_moves()):
            
            # Create board representation
            board_tensor = board_to_representation(board)
            
            # Handle the policy
            policy_move = torch.zeros((1, 4864))
            if turn_constant * base_eval > 0:
                move_idx = move_to_word(move)
                policy_move[0, move_idx] = 1
                self.win_flag_batch.append(True)
            else:
                self.win_flag_batch.append(False)
                
            # Handle the value
            value = torch.tanh(torch.tensor(base_eval) * self.base_multiplier ** (move_counter - idx)) * turn_constant
            
            # Forward the board and collect data
            self.board_batch = torch.cat((self.board_batch, board_tensor.unsqueeze(0)), dim=0)
            self.policy_batch = torch.cat((self.policy_batch, policy_move), dim=0)
            if self.value_batch.nelement() == 0:
                self.value_batch = value
                self.value_batch = self.value_batch.unsqueeze(0)
            else:
                self.value_batch = torch.cat((self.value_batch, value.unsqueeze(0)), dim=0)
            board.push(move)
            turn_constant *= -1
            
        # Add the last position of the board
        board_tensor = board_to_representation(board)
        policy_move = torch.zeros((1, 4864))
        value = torch.tanh(torch.tensor(base_eval * turn_constant))
        self.win_flag_batch.append(False)
            
        # Forward the last board
        self.board_batch = torch.cat((self.board_batch, board_tensor.unsqueeze(0)), dim=0)
        self.policy_batch = torch.cat((self.policy_batch, policy_move), dim=0)
        self.value_batch = torch.cat((self.value_batch, value.unsqueeze(0)), dim=0)
        
    def __getitem__(self, _):
        """
        Outputs the wanted item. If the game moves end, load another game.
        """
        
        if self.follow_idx == 0:
            while self.game_length == 0:
                self.load_game()
                self.game_length = self.board_batch.size()[0]

        sampled_board = self.board_batch[self.follow_idx, :, :, :].clone()
        sampled_quality_batch = self.policy_batch[self.follow_idx, :].clone()
        sampled_board_value_batch = self.value_batch[self.follow_idx].clone()
        sampled_win_flag = copy.deepcopy(self.win_flag_batch[self.follow_idx])

        self.follow_idx += 1
        if self.follow_idx == self.game_length:
            self.follow_idx = 0
            self.game_length = 0
            
        return {'board': sampled_board, 
                'policy': sampled_quality_batch, 
                'value': sampled_board_value_batch, 
                'win': sampled_win_flag}
        
    def __len__(self):
        return int(1e6)

        
                
        