import time
import pickle

import chess
import torch
import numpy as np
         

def board_to_representation(board: chess.Board) -> torch.Tensor: 
    """
    Converts a py-chess board to a tensor representation of size 8 X 8 X 15. Encapsulates the pieces, the turn, the coordinates, 
    castling, en passant, and area that pieces can move for richer features and faster training.
    """
    
    # Python chess uses flattened representation of the board
    pos_total = np.zeros((64, 16)) # 12 pieces (white, black) (p, N, B, R, Q, K), 2 coordinates (col, row), 1 turn
    
    for pos in range(64):
        
        # Extract the piece
        piece = board.piece_type_at(pos)
    
        # Place piece on board
        color = int(bool(board.occupied_co[chess.BLACK] & chess.BB_SQUARES[pos]))
        color_add = 6 if color else 0
        col = int(pos % 8)
        row = int(pos / 8)
        
        if piece:
            
            # Mark horizontals, verticals, and diagonals if bishop, rook, or queen
            if piece in [4, 5, 10, 11]:
                pos_total = _mark_hor_ver(pos_total, pos, piece - 1 + color_add)
            if piece in [3, 5, 9 ,11]:
                pos_total = _mark_diag(pos_total, pos, piece - 1 + color_add)
            
            # Show influence on board
            for attacked_square in list(board.attacks(pos)):
                pos_total[attacked_square, piece - 1 + color_add] = pos_total[attacked_square, piece - 1 + color_add] + 0.25\
                    if pos_total[attacked_square, piece - 1 + color_add] >= 0.25\
                        else 0.25
                if color:
                    pos_total[attacked_square, 15] -= 1
                else:
                    pos_total[attacked_square, 15] += 1
            
            # En passant    
            if piece == 1:
                if board.ep_square:
                    ep_square = np.floor(board.ep_square)
                    pos_total[int(ep_square), piece - 1 + color_add] = 0.33
            
            # Piece placement
            pos_total[row * 8 + col, piece - 1 + color_add] = 1
            
            # Castling
            if piece == 6:
                if piece == 6 and not color and board.has_castling_rights(chess.WHITE):
                    pos_total[row * 8 + col, 5] = 2
                if piece == 6 and color and board.has_castling_rights(chess.BLACK):
                    pos_total[row * 8 + col, 11] = 2
            
        # Encapsulate coordinates        
        pos_total[row * 8 + col, 12], pos_total[row * 8 + col, 13] = col, row
        
    # Encapsulate turn
    pos_total[:, 14] = 0 if board.turn else 1
    representation_board = pos_total.reshape((8, 8, 16))
    representation_board = np.transpose(representation_board, (2, 0, 1))
    representation_board = torch.from_numpy(representation_board).float()
    
    return representation_board
            
            
def move_to_word(move: chess.Move) -> int:
    """
    Converts a move object to an int that represents a move index. There are overall 4810 possible indices, 
    the output of the policy model is the classification score between them. Possibly filtering out the illegal moves.
    """
    
    # coordinates
    from_square = move.from_square
    to_square = move.to_square
    
    # handle promotions
    if move.promotion is not None:
        promotion_fix = 8 if from_square >= 48 else -8
        direction = from_square + promotion_fix - to_square
        promotion_symbol = move.promotion
        if promotion_symbol is chess.QUEEN:
            to_square = 65 + direction
        elif promotion_symbol is chess.ROOK:
            to_square = 68 + direction
        elif promotion_symbol is chess.BISHOP:
            to_square = 71 + direction
        else:
            to_square = 74 + direction
            
    word = from_square + 64 * to_square
    return word
    
    
def word_to_move(word) -> chess.Move:
    """
    Converts an int index of a move into a move in uci. Possible 4810 moves, accounting also for promotions.
    Outputs a move object.
    """

    coor_col = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    promotion_char = ['n', 'b', 'r', 'q']

    # Decompose to individual move coordinates
    coordinates_from_to = (int(word % 64), 
                           torch.div(word, 64, rounding_mode='floor').int())
    coordinates_from = (int(coordinates_from_to[0] % 8), coordinates_from_to[0] // 8)  # 0 is a,b,c... 1 is numbers

    coor_char_from = coor_col[coordinates_from[0]] + str(int(coordinates_from[1] + 1))

    # If not promoting
    if coordinates_from_to[1] < 64:
        coordinates_to = (int(coordinates_from_to[1] % 8), 
                          torch.div(coordinates_from_to[1], 8, rounding_mode='floor').int())
        coor_char_to = coor_col[coordinates_to[0]] + str(int(coordinates_to[1] + 1))

    # If promoting
    else:
        if 64 <= coordinates_from_to[1] < 67:
            coor_shift = 65 - coordinates_from_to[1]
            coor_up_down = 0 if coordinates_from[1] == 1 else 7
            coor_char_to = coor_col[coordinates_from[0] + coor_shift] + str(coor_up_down + 1) + promotion_char[3]
        elif 67 <= coordinates_from_to[1] < 70:
            coor_shift = 68 - coordinates_from_to[1]
            coor_up_down = 0 if coordinates_from[1] == 1 else 7
            coor_char_to = coor_col[coordinates_from[0] + coor_shift] + str(coor_up_down + 1) + promotion_char[2]
        elif 70 <= coordinates_from_to[1] < 73:
            coor_shift = 71 - coordinates_from_to[1]
            coor_up_down = 0 if coordinates_from[1] == 1 else 7
            coor_char_to = coor_col[coordinates_from[0] + coor_shift] + str(coor_up_down + 1) + promotion_char[1]
        else:
            coor_shift = 74 - coordinates_from_to[1]
            coor_up_down = 0 if coordinates_from[1] == 1 else 7
            coor_char_to = coor_col[coordinates_from[0] + coor_shift] + str(coor_up_down + 1) + promotion_char[0]

    move = chess.Move.from_uci(coor_char_from + coor_char_to)

    return move
    

def board_to_embedding_coord(board: chess.Board):
    """
    Used in AttaChess 1, converts a board to embedding coordinates, used here for play_game_gui.py
    """

    # Python chess uses flattened representation of the board
    x = torch.zeros(64, dtype=torch.float)
    for pos in range(64):
        piece = board.piece_type_at(pos)
        if piece:
            color = int(bool(board.occupied_co[chess.BLACK] & chess.BB_SQUARES[pos]))
            col = int(pos % 8)
            row = int(pos / 8)
            x[row * 8 + col] = -piece if color else piece
    x = x.reshape(8, 8)
    x += 6

    if board.ep_square:
        coordinates = (np.floor(board.ep_square / 8), board.ep_square % 8)
        if x[int(coordinates[0]), int(coordinates[1])] == 5:
            x[int(coordinates[0]), int(coordinates[1])] = 16
        else:
            x[int(coordinates[0]), int(coordinates[1])] = 17

    # Check for castling rights
    if board.has_castling_rights(chess.WHITE):
        x[0, 4] = 14
    if board.has_castling_rights(chess.BLACK):
        x[7, 4] = 15

    x += (not board.turn) * 18
    x = x.int()
    return x

def _fen_to_bitmap_rep(pos_total: np.ndarray, fen: str):
    """
    A function that converts a fen representation into pytorch position board without influence areas.
    """
    
    CONVERSION_DICT = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                       'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
    
    # Create a visual board representation
    fen_split = fen.split()
    fen_board = fen_split[0]
    for num_empty in range(8):
        replacer = '-' * (num_empty + 1)
        fen_board = fen_board.replace(str(num_empty + 1), replacer)
    fen_board_split = fen_board.split('/')
    for idx_row, row in enumerate(reversed(fen_board_split)):
        for idx_col, piece in enumerate(row):
            if piece != '-':
                pos_total[idx_col + 8 * idx_row, CONVERSION_DICT[piece]] = 1
                
    # Encapsulate turn
    pos_total[:, 14] = 0 if fen_split[1] == 'w' else 1
    
    return pos_total
    

def _mark_hor_ver(pos_total, pos, piece_idx):
    """
    Marks horizontal and vertical lines, used for rook and queen attack lines.
    """
    
    # Vertical up
    mark_idx = pos
    while mark_idx < 64:
        pos_total[mark_idx, piece_idx] = 0.1 if pos_total[mark_idx, piece_idx] == 0 else pos_total[mark_idx, piece_idx]
        mark_idx += 8
        
    # Vertical down
    mark_idx = pos
    while mark_idx >= 0:
        pos_total[mark_idx, piece_idx] = 0.1 if pos_total[mark_idx, piece_idx] == 0 else pos_total[mark_idx, piece_idx]
        mark_idx -= 8
        
    # Horizontal right
    mark_idx = pos
    while mark_idx % 8 != 0:
        pos_total[mark_idx, piece_idx] = 0.1 if pos_total[mark_idx, piece_idx] == 0 else pos_total[mark_idx, piece_idx]
        mark_idx += 1
        
    # Horizontal left
    mark_idx = pos
    while mark_idx % 8 != 0:
        mark_idx -= 1
        pos_total[mark_idx, piece_idx] = 0.1 if pos_total[mark_idx, piece_idx] == 0 else pos_total[mark_idx, piece_idx]
        
    return pos_total

def _mark_diag(pos_total, pos, piece_idx):
    """
    Marks diagonals, used for bishop and queen attack lines.
    """

    
    # Diagonal down right
    col = int(pos % 8)
    row = int(pos / 8)
    while row >=0 and col < 8:
        pos_total[col + row * 8, piece_idx] = 0.1 if pos_total[col + row * 8, piece_idx] == 0 else pos_total[col + row * 8, piece_idx]
        row -= 1
        col += 1
        
    # Diagonal up right
            
    col = int(pos % 8)
    row = int(pos / 8)
    while row < 8 and col < 8:
        pos_total[col + row * 8, piece_idx] = 0.1 if pos_total[col + row * 8, piece_idx] == 0 else pos_total[col + row * 8, piece_idx]
        row += 1
        col += 1
        
    # Diagonal down left
            
    col = int(pos % 8)
    row = int(pos / 8)
    while row >= 0 and col >= 0:
        pos_total[col + row * 8, piece_idx] = 0.1 if pos_total[col + row * 8, piece_idx] == 0 else pos_total[col + row * 8, piece_idx]
        row -= 1
        col -= 1
              
    # Diagonal down left
            
    col = int(pos % 8)
    row = int(pos / 8)
    while row < 8 and col >= 0:
        pos_total[col + row * 8, piece_idx] = 0.1 if pos_total[col + row * 8, piece_idx] == 0 else pos_total[col + row * 8, piece_idx]
        row += 1
        col -= 1
        
    return pos_total