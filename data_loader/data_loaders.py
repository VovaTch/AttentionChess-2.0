import torch
from base import BaseDataLoader

from .datasets import LichessDataset


class LichessLoader(BaseDataLoader):
    """
    The dataloader of the Lichess dataset
    """
    def __init__(self, batch_size, collate_fn, data_dir='data/lichess_data.pgn',
                 shuffle=True, validation_split=0.0, num_workers=0, training=True, base_multiplier=1.0):

        self.dataset_path = data_dir
        self.dataset = LichessDataset(data_dir, base_multiplier=base_multiplier)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)


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