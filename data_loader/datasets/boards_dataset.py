from email.mime import base
import os
from itertools import cycle
import pickle

import torch
from torch.utils.data import Dataset
import chess


class BoardsDataset(Dataset):
    """
    The dataset class that takes data from the lichess_boards pickle files and relays them to AttentionChess. 
    This form of supervised training should be more balanced than the lichess dataset.
    """
    def __init__(self,
                 dataset_path: str = 'data/boards_data/',
                 base_multiplier: float = 1.0):
        
        super().__init__()
        
        # List all files in a directory
        self.file_list = os.listdir(dataset_path)
        self.file_path_list = [dataset_path + file for file in self.file_list]
        self.cycle_iter = cycle(self.file_path_list)
        self.current_file = next(self.cycle_iter)
        self.pkl_data = self._load_pkl(self.current_file)
        self.file_len = self.pkl_data[0].size()[0]
        
        # Initialize constants
        self.position_counter = 0
        self.base_multiplier = base_multiplier
    
    @staticmethod
    def _load_pkl(file_path):
        """
        A simple method to load a pkl file
        """
        with open(file_path, 'rb') as f:
            pkl_data = pickle.load(f)
        return pkl_data
    
    def __getitem__(self, index):
        """The get item method gives a data sample for training. If the file reaches the end, load the next file"""
        
        # Load the sample
        self.position_counter += 1
        board_sample = self.pkl_data[0][index, :, :, :]
        policy_sample = self.pkl_data[1][index, :]
        value_sample = self.pkl_data[2][index]
        win_sample = False if torch.sum(policy_sample).item() == 0 else True
        
        # If we reach the file size limit, load the next file
        if self.position_counter == self.file_len:
            self.position_counter = 0
            self.current_file = next(self.cycle_iter)
            self.pkl_data = self._load_pkl(self.current_file)
            self.file_len = self.pkl_data[0].size()[0]
            print(f'Reset position counter, the current file is {self.current_file}')
        
        # Return the sample
        return {'board': board_sample,
                'policy': policy_sample,
                'value': value_sample,
                'win': win_sample}
        
    def __len__(self):
        return self.file_len
        
        
    
    