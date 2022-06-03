import torch
import pickle
import numpy as np

"""This script creates a piece influence map, such that we can create it via a look-up table to make it faster"""


def main():
    
    inf_white_pawn = np.zeros((64, 64))
    inf_black_pawn = np.zeros((64, 64))
    inf_knight = np.zeros((64, 64))
    inf_bishop = np.zeros((64, 64))
    inf_rook = np.zeros((64, 64))
    inf_queen = np.zeros((64, 64))
    inf_king = np.zeros((64, 64))
    
    for pos in range(64):
        
        # columns and rows
        col = int(pos % 8)
        row = int(pos / 8)
        
        # Handle pawns
        if col > 0:
            if row < 7:
                inf_white_pawn[pos, pos + 7] = 1
            if row > 0:
                inf_black_pawn[pos, pos - 9] = 1
        if col < 7:
            if row < 7:
                inf_white_pawn[pos, pos + 9] = 1
            if row > 0:
                inf_black_pawn[pos, pos - 7] = 1
                
        # Handle knights
        for col_n, row_n in zip([-2, -1, 1, 2, 2, 1, -1, -2], [1, 2, 2, 1, -1, -2, -2, -1]):
            if 8 > col + col_n >= 0 and 8 > row + row_n >= 0:
                inf_knight[pos, col_n + col + 8 * (row_n + row)] = 1
                
        # Handle bishops
        inf_bishop[pos, :] = _mark_diag(inf_bishop[pos, :], pos)
        
        # Handle rooks
        inf_rook[pos, :] = _mark_hor_ver(inf_rook[pos, :], pos)
        
        # Handle queens
        inf_queen[pos, :] = _mark_diag(inf_queen[pos, :], pos)
        inf_queen[pos, :] = _mark_hor_ver(inf_queen[pos, :], pos)
        
        # Handle kings
        for col_n, row_n in zip([-1, -1, 0, 1, 1, 1, 0, -1], [0, 1, 1, 1, 0, -1, -1, -1]):
            if 8 > col + col_n >= 0 and 8 > row + row_n >= 0:
                inf_king[pos, col_n + col + 8 * (row_n + row)] = 1
                
    # Saving the lookup table
    inf_stack = np.stack([inf_white_pawn, inf_black_pawn, inf_knight, inf_bishop, inf_rook, inf_queen, inf_king])
        
    with open('data/inf_lookup.pkl', 'wb') as f:
        pickle.dump(inf_stack, f)
        
    print(f'Finished creating piece influence lookup table.')

def _mark_hor_ver(pos_total: np.ndarray, pos: int):
    """
    Marks horizontal and vertical lines, used for rook and queen attack lines.
    """
    
    col = int(pos % 8)
    row = int(pos / 8)
    pos_total_square = pos_total.reshape((8, 8))
    pos_total_square[:, col] = 1
    pos_total_square[row, :] = 1
    pos_total_square[row, col] = 0
        
    return pos_total

def _mark_diag(pos_total, pos):
    """
    Marks diagonals, used for bishop and queen attack lines.
    """

    
    # Diagonal down right
    col = int(pos % 8)
    row = int(pos / 8)
    while row >=0 and col < 8:
        pos_total[col + row * 8] = 1
        row -= 1
        col += 1
        
    # Diagonal up right
            
    col = int(pos % 8)
    row = int(pos / 8)
    while row < 8 and col < 8:
        pos_total[col + row * 8] = 1
        row += 1
        col += 1
        
    # Diagonal down left
            
    col = int(pos % 8)
    row = int(pos / 8)
    while row >= 0 and col >= 0:
        pos_total[col + row * 8] = 1
        row -= 1
        col -= 1
              
    # Diagonal down left
            
    col = int(pos % 8)
    row = int(pos / 8)
    while row < 8 and col >= 0:
        pos_total[col + row * 8] = 1
        row += 1
        col -= 1
        
    pos_total[pos] = 0
        
    return pos_total



if __name__ == '__main__':
    main()