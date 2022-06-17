import torch
from base import BaseDataLoader
from model.model import AttentionChess2

from .datasets import LichessDataset,\
                      BoardsDataset,\
                      FullSelfPlayDataset,\
                      SwarmSelfPlayDataset,\
                      FullSelfPlayAsyncDataset
from data_loader.mcts import MCTS
from data_loader.async_mcts import MCTS as MCTS_async


class LichessLoader(BaseDataLoader):
    """
    The dataloader of the Lichess dataset
    """
    def __init__(self, batch_size, collate_fn, dataset_path='data/lichess_data.pgn',
                 shuffle=True, validation_split=0.0, num_workers=0, training=True, base_multiplier=1.0):

        self.dataset_path = dataset_path
        self.dataset = LichessDataset(dataset_path, base_multiplier=base_multiplier)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)
        
class BoardsLoader(BaseDataLoader):
    """
    The dataloader of the Boards dataset
    """
    def __init__(self, batch_size, collate_fn, dataset_path='data/boards_data/',
                shuffle=True, validation_split=0.0, num_workers=0, training=True, base_multiplier=1.0):

        self.dataset_path = dataset_path
        self.dataset = BoardsDataset(dataset_path=dataset_path, base_multiplier=base_multiplier)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)

class FullSelfPlayLoader(BaseDataLoader):
    """
    Data loader for self playing games with moves from database.
    """
    def __init__(self, batch_size, mcts, collate_fn, simultaneous_mcts=16, simultaneous_mcts_pre_generated=64,
                 shuffle=True, validation_split=0.0, num_workers=0, training=True, 
                 num_of_sims=100, min_counts=10, move_limit=300, buffer_size=1e5, ignore_loss_sim=1.0):

        self.dataset: FullSelfPlayDataset = FullSelfPlayDataset(num_of_sims=num_of_sims, 
                                                                mcts=mcts,
                                                                simultaneous_mcts_pre_generated=simultaneous_mcts_pre_generated,
                                                                min_counts=min_counts, 
                                                                simultaneous_mcts=simultaneous_mcts, 
                                                                move_limit=move_limit, 
                                                                buffer_size=buffer_size, 
                                                                ignore_loss_lim=ignore_loss_sim)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)
        
    def set_mcts(self, mcts: MCTS):
        
        self.dataset.mcts = mcts
        
    def set_white_engine(self, engine):
        
        self.dataset.white_engine = engine
        
    def set_black_engine(self, engine):
        
        self.dataset.black_engine = engine
        
class FullSelfPlayAsyncLoader(BaseDataLoader):
    """
    Data loader for self playing games with moves from database.
    """
    def __init__(self, batch_size, mcts, collate_fn, simultaneous_mcts=16, simultaneous_mcts_pre_generated=64,
                 shuffle=True, validation_split=0.0, num_workers=0, training=True, 
                 num_of_sims=100, min_counts=10, move_limit=300, buffer_size=1e5, ignore_loss_sim=1.0):

        self.dataset: FullSelfPlayAsyncDataset = FullSelfPlayAsyncDataset(num_of_sims=num_of_sims, 
                                                                mcts=mcts,
                                                                simultaneous_mcts_pre_generated=simultaneous_mcts_pre_generated,
                                                                min_counts=min_counts, 
                                                                simultaneous_mcts=simultaneous_mcts, 
                                                                move_limit=move_limit, 
                                                                buffer_size=buffer_size, 
                                                                ignore_loss_lim=ignore_loss_sim)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)
        
    def set_mcts(self, mcts: MCTS_async):
        
        self.dataset.mcts = mcts
        
    def set_white_engine(self, engine):
        
        self.dataset.white_engine = engine
        
    def set_black_engine(self, engine):
        
        self.dataset.black_engine = engine
        
class SwarmSelfPlayLoader(BaseDataLoader):
    """
    Data loader for low quality self playing games for quick game generation, hopefully faster convergence.
    """
    def __init__(self, batch_size, engine, collate_fn, simultaneous_games=128,
                 shuffle=True, validation_split=0.0, num_workers=0, training=True, 
                 move_limit=100, buffer_size=1e5, base_multiplier=1.0, win_only=False):
        
        self.dataset: SwarmSelfPlayDataset = SwarmSelfPlayDataset(engine=engine,
                                                                  simultaneous_games=simultaneous_games,
                                                                  move_limit=move_limit,
                                                                  buffer_size=buffer_size,
                                                                  base_multiplier=base_multiplier,
                                                                  win_only=win_only)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)
        
    def set_engine(self, engine: AttentionChess2):
        
        self.dataset.engine = engine


def collate_fn(batch):
    """
    Collate function for all the dictionary-style outputs
    """
    output_dict = {}
    
    for dict in batch:
        for key, value in dict.items():
            
            # Initialize if key is not in the output dict
            if key not in output_dict:
                output_dict[key] = []
                
            output_dict[key].append(torch.tensor(value))
            
    # Create a torch tensor for each key
    for key in output_dict:
        output_dict[key] = torch.stack(output_dict[key])
        
    return output_dict