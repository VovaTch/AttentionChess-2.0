import chess
import torch
import numpy as np


def board_to_representation(board: chess.Board) -> torch.Tensor: 
    """
    Converts a py-chess board to a tensor representation of size 8 X 8 X 15. Encapsulates the pieces, the turn, the coordinates, 
    castling, en passant, and area that pieces can move for richer features and faster training.
    """
    
    # Python chess uses flattened representation of the board
    pos_total = torch.zeros((64, 16)) # 12 pieces (white, black) (p, N, B, R, Q, K), 2 coordinates (col, row), 1 turn
    
    for pos in range(64):
        
        # Extract the piece
        piece = board.piece_type_at(pos)
      
        # Place piece on board
        color = int(bool(board.occupied_co[chess.BLACK] & chess.BB_SQUARES[pos]))
        color_add = 6 if color else 0
        col = int(pos % 8)
        row = int(pos / 8)
        
        if piece:

            pos_total[row * 8 + col, piece - 1 + color_add] = 1
            
            # Show influence on board
            for attacked_square in list(board.attacks(pos)):
                pos_total[attacked_square, piece - 1 + color_add] += 0.25
                if color:
                    pos_total[attacked_square, 15] -= 1
                else:
                    pos_total[attacked_square, 15] += 1
            
            # En passant    
            if board.ep_square:
                ep_square = np.floor(board.ep_square)
                pos_total[ep_square, piece - 1 + color_add] = 0.33
            
            # Castling
            if piece == 6 and not color and board.has_castling_rights(chess.WHITE):
                pos_total[row * 8 + col, 5] = 2
            if piece == 6 and color and board.has_castling_rights(chess.BLACK):
                pos_total[row * 8 + col, 11] = 2
        
        # Encapsulate coordinates        
        pos_total[row * 8 + col, 12], pos_total[row * 8 + col, 13] = col, row
        
    # Encapsulate turn
    pos_total[:, 14] = 0 if board.turn else 1
    representation_board = pos_total.reshape((8, 8, 16))
    representation_board = representation_board.permute((2, 0, 1))
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
    
    