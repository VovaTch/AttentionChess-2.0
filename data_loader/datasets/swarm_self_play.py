import math

import chess
import torch
from torch.utils.data import Dataset
from colorama import Fore
import numpy as np

from model.model import AttentionChess2
from utils import is_game_end, move_to_word, board_to_representation

class SwarmSelfPlayDataset(Dataset):
    """
    Generates low-quality games in high quantity for training like the board dataset.
    """
    def __init__(self,
                 engine: AttentionChess2,
                 simultaneous_games: int=128,
                 move_limit: int=600,
                 buffer_size: int=1e5,
                 base_multiplier: float=1.0,
                 win_only: bool=False):
        super().__init__()
        
        self.simultanous_games: int = simultaneous_games
        self.move_limit: int = move_limit
        self.buffer_size: int = buffer_size
        self.engine: AttentionChess2 = engine
        self.base_multiplier: float = base_multiplier
        self.buffer = {}
        self.follow_idx = 0 # Games counted, when it reaches the buffer size, it resets and another game batch is loaded
        self.win_only: bool = win_only
        
        # Load games until the buffer is full
        while len(self.buffer) < self.buffer_size:
            self.load_game()
            
        print(f'Loaded the buffer with generated games.')
        
    def load_game(self):
        """
        Load a bunch of random games, save positions 
        """
        
        win_count_white = 0
        win_count_black = 0
        draw_count = 0
        
        running_boards = [chess.Board() for _ in range(self.simultanous_games)]
        running_games = [[[board.fen(), 9999, 0]] for board in running_boards] # Format: fen, move, value
        finished_games = []

        num_finished = 0
        
        while len(running_boards) != 0:
            
            # run the board list through the neural network
            output_dict = self.engine(running_boards)
            policy_list, _ = self.engine.post_process(running_boards, output_dict)
            
            finished_idx = []
            
            # Iterate over every game to sample from the probability distribution
            for idx, (running_board, running_game, policy) in enumerate(zip(running_boards, running_games, policy_list)):
                
                # Generate a move and a word
                prob_vector = np.array(list(policy.values()))
                moves_vector = list(policy.keys())
                sampled_move = np.random.choice(moves_vector, p=prob_vector / np.sum(prob_vector))
                move_uci = running_board.parse_san(sampled_move).uci()
                word = move_to_word(chess.Move.from_uci(move_uci))
                running_game[-1][1] = word
                
                # Push the moves into the running boards and running games, check if the game has ended
                running_board.push_san(sampled_move)
                game_ended, result = is_game_end(running_board)
                if game_ended:
                    
                    # input the correct values into the running games
                    value = 5 * abs(result)
                    if not self.win_only or result != 0:
                        for position in reversed(running_game):
                            position[2] = math.tanh(value)
                            value *= - self.base_multiplier
            
                        # Move the finished game into the finished games list
                        finished_games.append(running_game)
                    num_finished += 1
                    finished_idx.append(idx)
                    
                    if result == 1:
                        win_count_white += 1
                    elif result == -1:
                        win_count_black += 1
                    else:
                        draw_count += 1
                        
                    print(f'Processed ' + Fore.MAGENTA + f'{num_finished}' + Fore.RESET + ' games,' + Fore.CYAN + 
                          f'white wins {win_count_white} games, ' 
                          f'black wins {win_count_black} games, '
                          f'and {draw_count} games are drawn.' + Fore.RESET, end='\r')
                else:
                    
                    # append the new board into the running games
                    running_game.append([running_board.fen(), 9999, 0])
                    
                # If the limit has been reached without a conclusion, just remove the games.
                if len(running_game) > self.move_limit:
                    finished_idx.append(idx)
                    
            # Removed finished games
            idx_fix = 0
            for fin_idx in finished_idx:
                del running_boards[fin_idx - idx_fix], running_games[fin_idx - idx_fix]
                idx_fix += 1
                    
        # with the finished games, handle stack update
        buffer_update = {}
        for finished_game in finished_games:
            for position in finished_game:
                
                # If the entry is new
                if position[0] not in buffer_update:
                    
                    buffer_update[position[0]] = []
                    buffer_update[position[0]].append(board_to_representation(chess.Board(fen=position[0])))
                    buffer_update[position[0]].append(torch.zeros((4864)))
                    buffer_update[position[0]][1][position[1]] = 1 if position[2] > 0 else 0
                    buffer_update[position[0]].append(position[2])
                    buffer_update[position[0]].append(1)
                    
                else:
                    
                    if position[2] > 0:
                        buffer_update[position[0]][1][position[1]] += 1
                    buffer_update[position[0]][3] += 1
                    buffer_update[position[0]][2] += (position[2] - buffer_update[position[0]][2]) / buffer_update[position[0]][3]
                    
        # Update the buffer and limit its size
        self.buffer.update(buffer_update)
        while len(self.buffer) > self.buffer_size:
            keys = list(self.buffer.keys())
            self.buffer.pop(keys[0])
                    
    def __getitem__(self, idx):
        
        extracted_position = list(self.buffer.values())[idx]
        sampled_board = extracted_position[0]
        sampled_policy = extracted_position[1]
        sampled_value = extracted_position[2]
        sampled_win = True if torch.sum(sampled_policy).item() != 0 else False
        
        self.follow_idx += 1
        if self.follow_idx == self.buffer_size:
            self.follow_idx = 0
            self.load_game()
            
        return {'board': sampled_board, 
                'policy': sampled_policy, 
                'value': sampled_value, 
                'win': sampled_win}
        
    def __len__(self):
        return self.buffer_size
                    