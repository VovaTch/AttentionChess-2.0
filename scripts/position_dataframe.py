import argparse
import pickle
import os

import chess
import chess.pgn
import torch

from utils import move_to_word, board_to_representation


def load_game(pgn_handle, args):
    
    base_multiplier = args.base_multiplier
    
    # Load a game that ends in a victory/draw/defeat
    while True:
        game = chess.pgn.read_game(pgn_handle)
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
    board_batch = torch.zeros((0, 16, 8, 8))
    policy_batch = torch.zeros((0, 4864))
    value_batch = torch.zeros(0)
    win_flag_batch = []
    board_fen_list = []
    
    # Loop over the game moves
    for idx, move in enumerate(game.mainline_moves()):
        
        # Create board representation
        board_tensor = board_to_representation(board)
        board_fen_list.append(board.fen())
        
        # Handle the policy
        policy_move = torch.zeros((1, 4864))
        if turn_constant * base_eval > 0:
            move_idx = move_to_word(move)
            policy_move[0, move_idx] = 1
            win_flag_batch.append(True)
        else:
            win_flag_batch.append(False)
            
        # Handle the value
        value = torch.tanh(torch.tensor(base_eval) * base_multiplier ** (move_counter - idx)) * turn_constant
        
        # Forward the board and collect data
        board_batch = torch.cat((board_batch, board_tensor.unsqueeze(0)), dim=0)
        policy_batch = torch.cat((policy_batch, policy_move), dim=0)
        if value_batch.nelement() == 0:
            value_batch = value
            value_batch = value_batch.unsqueeze(0)
        else:
            value_batch = torch.cat((value_batch, value.unsqueeze(0)), dim=0)
        board.push(move)
        turn_constant *= -1
        
    # Add the last position of the board
    board_tensor = board_to_representation(board)
    policy_move = torch.zeros((1, 4864))
    value = torch.tanh(torch.tensor(base_eval * turn_constant))
    win_flag_batch.append(False)
    board_fen_list.append(board.fen())
        
    # Forward the last board
    board_batch = torch.cat((board_batch, board_tensor.unsqueeze(0)), dim=0)
    policy_batch = torch.cat((policy_batch, policy_move), dim=0)
    value_batch = torch.cat((value_batch, value.unsqueeze(0)), dim=0)
    
    return board_fen_list, board_batch, policy_batch, value_batch

def main(args):
    
    pgn_database = open(args.path, encoding="utf-8")
    game_count = 0
        
    position_count = 0
    file_count = 0
    
    # Organize all data into a massive dict. Save later as a json/csv
    while position_count <= args.position_limit:
        pgn_dict = {}
        file_count += 1
        while len(pgn_dict) < args.chunk_size:
    
            board_fen_list, board_batch, policy_batch, value_batch = load_game(pgn_database, args)
            game_count += 1
            # Check if the game list has ended
            if board_fen_list is None:
                break
    
            # Run over every position
            for (board_fen_ind, board_ind, policy_ind, value_ind) in\
                    zip(board_fen_list, board_batch, policy_batch, value_batch):
                
                if board_fen_ind not in pgn_dict:
                    
                    # Create an entry for the pgn dictionary
                    pgn_dict[board_fen_ind] = []
                    pgn_dict[board_fen_ind].append(board_ind)
                    pgn_dict[board_fen_ind].append(policy_ind)
                    pgn_dict[board_fen_ind].append(value_ind)
                    pgn_dict[board_fen_ind].append(1)
                    
                    # Check if the length exceeds the limit
                    if len(pgn_dict) >= args.chunk_size:
                        break

                    
                else:
                    
                    # Update the dictionary entry
                    pgn_dict[board_fen_ind][1] += policy_ind
                    pgn_dict[board_fen_ind][3] += 1
                    pgn_dict[board_fen_ind][2] += (value_ind - pgn_dict[board_fen_ind][2]) / pgn_dict[board_fen_ind][3]
                    
            print(f'Game processed: {game_count}', end='\r')
                    
        # Collect all data and save it into a file
        position_count += len(pgn_dict)
        board_col = []
        policy_col = []
        value_col = []
        for value in pgn_dict.values():
            board_col.append(value[0])
            policy_col.append(value[1])
            value_col.append(value[2])
        board_col = torch.stack(board_col)
        policy_col = torch.stack(policy_col)
        value_col = torch.stack(value_col)
        
        if not os.path.exists('data/boards_data'):
            os.makedirs('data/boards_data')
        
        pkl_file_name = f'data/boards_data/lichess_boards_{file_count:03}.pkl'
        with open(pkl_file_name, 'wb') as file:
            pickle.dump([board_col, policy_col, value_col], file)
        print(f'Saved file {pkl_file_name}')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A script to save a database as csv file with non-repeating positions.')
    parser.add_argument('-p', '--path', type=str, default = 'data/lichess_data.pgn', 
                        help='The path of the game dataset.')
    parser.add_argument('-pl', '--position_limit', type=int, default=1e6, 
                        help='Number of games processed. Infinite - all the games in the file.')
    parser.add_argument('-c', '--chunk_size', type=int, default=5e4,
                        help='Number of positions per each data file.')
    parser.add_argument('-m', '--base_multiplier', type=float, default=0.95,
                        help='Multiplier for decaying reward for long games')
    args = parser.parse_args()
    main(args) 