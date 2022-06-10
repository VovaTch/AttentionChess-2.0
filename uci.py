import sys
import traceback
import argparse
import copy

import torch
import chess
import chess.pgn
import yaml

import model.model as module_arch
from model.model import AttentionChess2
import data_loader.mcts as module_mcts
from parse_config import ConfigParser
from utils import prepare_device

'''
Based on a0lite's implementation of the UCI protocol. 
AttentionChess-2.0 uses a modified MCTS with a Neural Network to output policy and value of boards.
Creating an EXE must account for the network, the weights, and the MCTS.
'''

CACHE_SIZE = 200000
MINTIME = 0.1
TIMEDIV = 25.0
NODES = 800
C = 3.0


logfile = open("ac2.log", "w")
LOG = True

def log(msg):
    if LOG:
        logfile.write(str(msg))
        logfile.write("\n")
        logfile.flush()

def send(str):
    log(">{}".format(str))
    sys.stdout.write(str)
    sys.stdout.write("\n")
    sys.stdout.flush()

def process_position(tokens):
    board = chess.Board()

    offset = 0

    if tokens[1] ==  'startpos':
        offset = 2
    elif tokens[1] == 'fen':
        fen = " ".join(tokens[2:8])
        board = chess.Board(fen=fen)
        offset = 8

    if offset >= len(tokens):
        return board

    if tokens[offset] == 'moves':
        for i in range(offset+1, len(tokens)):
            board.push_uci(tokens[i])

    # deal with cutechess bug where a drawn positions is passed in
    if board.can_claim_draw():
        board.clear_stack()
    return board



def load_network(config):
    '''
    Loads AttentionChess-2.0 and MCTS instances
    '''
    
    log("Loading network")
    
    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    
    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        
    # Resume from checkpoint
    if config.resume is not None:
        checkpoint = torch.load(config.resume, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])

    log('Network loaded')
    
    log('Loading MCTS instance')
    
    # setup data_loader instances and mcts
    mcts = config.init_obj('mcts', module_mcts, 
                           model_good=copy.deepcopy(model), model_evil=copy.deepcopy(model), 
                           num_sims=config['data_loader']['args']['num_of_sims'])
    
    log('Loaded MCTS instance')
    
    return model, mcts


def main(config):

    send("AttentionChess-2.0")
    board = chess.Board()
    nn = None

    while True:
        line = sys.stdin.readline()
        line = line.rstrip()
        log("<{}".format(line))
        tokens = line.split()
        if len(tokens) == 0:
            continue

        if tokens[0] == "uci":
            send('id name AttentionChess-2.0')
            send('id author Vladimir Tchuiev')
            send('uciok')
        elif tokens[0] == "quit":
            exit(0)
        elif tokens[0] == "isready":
            if nn == None:
                nn, mcts = load_network(config)
            send("readyok")
        elif tokens[0] == "ucinewgame":
            board = chess.Board()

        elif tokens[0] == 'position':
            board = process_position(tokens)

        elif tokens[0] == 'go':
            my_nodes = NODES
            my_time = None
            if (len(tokens) == 3) and (tokens[1] == 'nodes'):
                my_nodes = int(tokens[2])
            if (len(tokens) == 3) and (tokens[1] == 'movetime'):
                my_time = int(tokens[2])/1000.0
                if my_time < MINTIME:
                    my_time = MINTIME
            if (len(tokens) == 9) and (tokens[1] == 'wtime'):
                wtime = int(tokens[2])
                btime = int(tokens[4])
                winc = int(tokens[6])
                binc = int(tokens[8])
                if (wtime > 5*winc):
                    wtime += 5*winc
                else:
                    wtime += winc
                if (btime > 5*binc):
                    btime += 5*binc
                else:
                    btime += binc
                if board.turn:
                    my_time = wtime/(TIMEDIV*1000.0)
                else:
                    my_time = btime/(TIMEDIV*1000.0)
                if my_time < MINTIME:
                    my_time = MINTIME   
            if nn == None:
                nn = load_network()


            if my_time != None:
                mcts.num_sims = 1000000
                root = mcts.run(time_limit=my_time, board=board)
                score = root.value_avg() * 100
                best_san = root.select_action()
                best = board.parse_san(best_san).uci()
            else:
                mcts.num_sims = my_nodes
                root = mcts.run(board=board)
                score = root.value_avg() * 100
                best_san = root.select_action()
                best = board.parse_san(best_san).uci()
            send("bestmove {}".format(best))

if __name__ == '__main__':
    
    torch.multiprocessing.set_start_method('spawn')  # Necessary for this to work; maybe it will run out of memory like that
    
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default='config/config_uci.yaml', type=str,
                    help='config file path (default: config/config.yaml)')
    args.add_argument('-r', '--resume', default='model_post_fsp.pth', type=str,
                    help='path to latest checkpoint (default: model_post_fsp.pth)')
    args.add_argument('-d', '--device', default=None, type=str,
                    help='indices of GPUs to enable (default: all)')
    config = ConfigParser.from_args(args)
    main(config)


# try:
#     main()
# except:
#     exc_type, exc_value, exc_tb = sys.exc_info()
#     log(traceback.format_exception(exc_type, exc_value, exc_tb))

# logfile.close()