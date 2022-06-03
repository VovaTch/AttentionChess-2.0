from typing import Dict, List
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np

from base import BaseModel
from model.switch_transformer import SwitchTransformerLayer, SwitchTransformer, SwitchFeedForward
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.transformers.mha import MultiHeadAttention
from utils import move_to_word, board_to_representation


class AttentionChess2(BaseModel):
    """
    The second version of AttentionChess, now going for a more feature-heavy input and an encoder-only architecture, 
    turning it into a 4810 classes classification problem for the policy. The value should still remain the same, 
    an unormalized scalar beween -inf to inf, with later turning it into one from -1 (black victory) to 1 (white victory).
    This uses the expert-type sparse encoder. TODO: Download this one or implement myself.
    """
    
    def __init__(self, 
                 hidden_dim: int = 256, 
                 num_encoders: int = 6,
                 num_experts: int = 16,
                 expert_dim: int = 2048,
                 dropout: float = 0.01,
                 aux_outputs: bool = False,
                 device: str = 'cpu'):
        super().__init__()
        
        # Insert the constants
        assert hidden_dim % 16 == 0, 'The hidden size must be divisable by 16'
        self.hidden_dim = hidden_dim
        self.num_encoders = num_encoders
        self.num_experts = num_experts
        self.num_heads = hidden_dim // 16
        self.aux_outputs = aux_outputs
        self.expert_dim = expert_dim
        self.device = device
        
        # Initialize the layers
        self.individual_expert = FeedForward(hidden_dim, expert_dim)
        self.mha = MultiHeadAttention(heads=self.num_heads, d_model=self.hidden_dim, dropout_prob=dropout)
        self.sff = SwitchFeedForward(capacity_factor=1.2, drop_tokens=True, is_scale_prob=False, 
                                     n_experts=self.num_experts, expert=self.individual_expert, d_model=hidden_dim)
        self.switch_transformer_layer = SwitchTransformerLayer(d_model=hidden_dim, attn=self.mha, dropout_prob=dropout, feed_forward=self.sff)
        self.switch_transformer = SwitchTransformer(layer=self.switch_transformer_layer, n_layers=num_encoders)
        self.policy_cls_head = EndHead(hidden_dim=hidden_dim, num_tokens=64, dim_out=4864)
        self.value_head = EndHead(hidden_dim=hidden_dim, num_tokens=64, dim_out=1)
        
    def forward(self, boards: List[chess.Board]) -> Dict[str, torch.Tensor]:
        """
        Gets a list of boards in the form of chess.Board, and outputs a dictionary with entries "policy" and "value", with the values being tensors.
        The policy is sized BS x 4864
        The value is sized BS.
        """
        boards_tensor = torch.stack([board_to_representation(board) for board in boards]).to(self.device)
        output_dict = self.forward_raw(boards_tensor)
        return output_dict
        
    def forward_raw(self, boards: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        The raw forward method that gets a tensor sized BS x 64 x 8 x 8. It outputs a dictionary with policy and value. 
        If the aux outputs flag is enabled, it outputs them as well.
        """
        # Prepare the input
        boards_flattened = boards.flatten(2, 3)
        transformer_input = boards_flattened.permute((2, 0, 1)) # To get 64 x BS x Hidden_dim
        transformer_input = transformer_input.repeat((1, 1, self.num_heads))
        
        # Transformer
        mask = torch.zeros((64, 64, boards_flattened.size()[0])).to(transformer_input.device) + 1
        transformer_output, memory, _, _, _, _ = self.switch_transformer(transformer_input, mask) # TODO: Program Aux outputs in
        transformer_output = transformer_output.permute((1, 0, 2)) # To get BS x 64 x Hidden_dim
        
        # Ending heads
        policy_output = self.policy_cls_head(transformer_output)
        value_output = self.value_head(transformer_output).squeeze(-1)
        
        output_dict = {'policy': policy_output, 'value': value_output}
        if self.aux_outputs:
            for idx, aux_out in enumerate(memory):
                if idx != self.num_encoders - 1:
                    
                    # Ending heads and update the output dict
                    aux_out = aux_out.permute((1, 0, 2))
                    policy_output = self.policy_cls_head(aux_out)
                    value_output = self.value_head(aux_out).squeeze(-1)
                    aux_dict = {f'policy_aux_{idx}': policy_output, f'value_aux_{idx}': value_output}
                    output_dict.update(aux_dict)
        
        return output_dict
    
    @staticmethod
    def post_process(boards: List[chess.Board], 
                     output_dict: Dict[str, torch.Tensor], 
                     print_output: bool = False):
        """
        A method for processing the raw outputs to output and potentially show them in a presentable fashion. Takes into account legal moves.
        """
        # Initialize the lists
        value_list = []
        policy_list = []
        
        # Run over each board in the batch
        for num_idx, board in enumerate(boards):
            
            # Filter out legal moves and extract the policy based on that
            legal_moves = {move_to_word(legal_move): board.san(legal_move) for legal_move in board.legal_moves}
            policy_filtered = output_dict['policy'][num_idx, np.ix_(list(legal_moves.keys()))]
            
            # policy_filtered_all = torch.index_select(output_dict['policy'], 1, torch.tensor(list(legal_moves.keys())))
            # policy_filtered = policy_filtered_all[num_idx, :].unsqueeze(0)
            
            policy_filtered = F.softmax(policy_filtered, dim=1)
            ind_policy_dict = {legal_move_san: policy_prob.item() for legal_move_san, policy_prob in zip(legal_moves.values(), policy_filtered[0])}
            policy_list.append(ind_policy_dict)
            
            # Extract the value
            turn_value = 1 if board.turn else -1
            value_list.append(torch.tanh(output_dict['value'][num_idx]).item() * turn_value)
            
        # In case we want to print the value
        if print_output:
            for (board, policy, value) in zip(boards, policy_list, value_list):
                print(f'Board value: {value:.3f}')
                
                # Print the policy probabilities
                policy_present = {key: round(value, 3) for key, value in policy.items()}
                print(f'Board policy: {policy_present}')
                    
        return policy_list, value_list
        

        
class EndHead(nn.Module):
    """
    This is the ending head for the transformer. It gets BS x N_tokens x H_dim input and converts it into BS x Out output via 2 linear layers.
    """
    def __init__(self, 
                 hidden_dim: int, 
                 num_tokens: int, 
                 dim_out: int):
        
        # Initialize the scalars
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.dim_out = dim_out
        super().__init__()
        
        # Initialize the layers
        self.linear_1 = nn.Linear(hidden_dim, 1)
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(num_tokens, dim_out)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: an input tensor dimensions BS x N_tokens x H_dim.
        output: an output tensor dimensions BS x Out
        """
        x = self.linear_1(x)
        x = self.activation(x)
        x = x.squeeze(-1)
        x = self.linear_2(x)
        return x

        
        
        