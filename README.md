# AttentionChess-2.0

<img src="https://github.com/VovaTch/AttentionChess-2.0/blob/main/Attchess.png" alt="drawing" width="500"/>


A personal project for creating a transformer encoder-decoder-based chess engine. This is the sequel version, bigger and better! This project is based on PyTorch and Python Chess. Here I use a sparse encoder-only architecture with ~32 million parameters to output a policy (as a 4864-dimension classification vector) and value, ranging from 1 to -1. In the post-processed outputs, 1 is white player winning and -1 is black player winning. Currently it's 

[Weights available in Google Drive](https://drive.google.com/file/d/1QOoo4FKA2kCCpRhDRqsSsNX-YGkb2In3/view?usp=sharing)

## Current playing instructions:

* Clone the repository `git clone https://github.com/VovaTch/AttentionChess-2.0.git`.
* Download weights from the above link and put them in the repository main folder.
* Run `python play_game_gui.py` for a simple chess board gui.
* Move pieces by selecting them and pressing the destination square. A highlight for the piece and the legal moves will appear.
* Press twice on the same square to cancel the move.
* When promoting, a display of promotion options will appear, press on the wanted piece.
* Move the king 2 squares to the right or left to castle.

### GUI Playing Flags:

* `-l` for the number of leaves for the MCTS.
* `-t` temperature of randomness for the bot to select a move.
  - 0.0 -> Always select the best move.
  - 1.0 -> Select from a distribution of the normalized visit count from the MCTS.
  - Infinity -> Select a move from a uniform random distribution.
* `--no_mcts` disable MCTS for move selection and rely only on policy network outputs. Fast, but plays poorer than MCTS version with ~50 leaves or more.

### Hotkeys:

* Press **space** for the bot to perform a move.
* **ctrl+z** to undo.
* **ctrl+f** to flip board.
* **ctrl+r** to reset game.
* **ctrl+q** to quit the game.

## Current WIP: 
* Full self play, both from random positions and the beginning, with a memory buffer.
* Trying a version with more parameters (~60 Mil, more than that my computer runs out of GPU memory).
