import torch
import torch.nn.functional as F
import chess
import copy
import numpy as np
import math

from model.model import AttentionChess2
from utils import move_to_word, word_to_move



class Node:
    
    def __init__(self, board: chess.Board, prior_prob: float, device='cpu', use_dir=False) -> None:
        """
        Initiates the node for the MCTS
        """
        
        self.device: str = device
        self.prior_prob: float = prior_prob
        self.turn: bool = board.turn
        self.half_move = None  # Used to compute the cost function
        
        self.children: dict[str, Node] = {}
        self.visit_count: int = 0
        self.value_candidates: dict[str, float] = {}
        self.value_sum: float = 0.0
        self.board: chess.Board = board
        self.use_dir: bool = use_dir
        
    def expanded(self):
        """
        Return a boolian to represent if it's an expanded node or not
        """
        return len(self.children) > 0
    
    def visit_count_children(self) -> dict[str: int]:
        """
        Create a dictionary of visit counts per child move. The format is 'san_move': visit count.
        """
        visit_count_dict = {move: child.visit_count for move, child in self.children.items()}
        return visit_count_dict
    
    def select_action(self, temperature: float=0.0, print_action_count: bool=False):
        """
        Select action according to the visit count distribution and the temperature.
        """
        visit_counts = torch.tensor(list(self.visit_count_children().values())).float()
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[torch.argmax(visit_counts)]
        elif temperature == float("inf"):
            
            logits = torch.tensor([0.0 for _ in actions]).to(self.device)
            cat_dist = torch.distributions.Categorical(logits=logits)
            action = actions[cat_dist.sample()]
                    
        else:
            
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / torch.sum(visit_count_distribution)
            cat_dist = torch.distributions.Categorical(probs=visit_count_distribution)
            action = actions[cat_dist.sample()]
            
        if print_action_count:
            action_value = {act: round(float(np.tanh(move.value_avg())), 5) for act, move in zip(actions, self.children.values())}
            print(f'Action values: {action_value}')
            action_dict = {act: int(vis_c) for act, vis_c in zip(actions, visit_counts)}
            print(f'Action list: {action_dict}')

        return action
    
    def select_child(self):
        """
        Select the child with the highest UCB score
        """
        ucb_score_dict = ucb_scores(self, self.children, dir_noise=self.use_dir)
        max_score_move = max(ucb_score_dict, key=ucb_score_dict.get)
        
        best_child = self.children[max_score_move]
        
        return max_score_move, best_child
    
    def value_avg(self):
        """
        Return the average of the children's value
        """

        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

    
    def value_max(self):
        """
        Return the maximum value of the children
        """
        
        non_none_candidate = {move: value for move, value in self.value_candidates.items() if self.value_candidates[move] is not None}
        
        if len(non_none_candidate) == 0:
            return None

        if self.turn:
            return non_none_candidate[max(non_none_candidate, key=non_none_candidate.get)]
        else:
            return non_none_candidate[min(non_none_candidate, key=non_none_candidate.get)]
        
    def expand(self, policy_vec: dict[str, float]):
        """
        Expand the node to include all possible moves.
        """
        
        for move, prob in policy_vec.items():
            new_board = copy.deepcopy(self.board)
            
            if type(move) is str:
                new_board.push_san(move)
            else:
                new_board.push(move)
                
            self.children[move] = Node(prior_prob=prob, board=new_board, device=self.device, use_dir=self.use_dir)
            self.value_candidates[move] = None
            
    def find_greedy_value(self, value_multiplier=1.0):
        
        # Flag for if no visits for the children
        child_no_visit_flag = True
        for child in self.children.values():
            if child.visit_count != 0:
                child_no_visit_flag = False
                break
        
        # Return the current value if leaf node
        if child_no_visit_flag:
            return self.value_avg()
        
        # Otherwise continue to the best child recursively.
        else:
            temp = self.use_dir
            self.use_dir = False
            _, best_child = self.select_child()
            self.use_dir = temp
            greedy_value = best_child.find_greedy_value() * value_multiplier
            return greedy_value
            
    def __repr__(self):
        """
        Display important information quickly
        """
        prior = "{0:.3f}".format(self.prior_prob)
        copy_board = copy.deepcopy(self.board)
        try:
            copy_board.pop()
            parent_move = copy_board.san(self.board.peek())
        except:
            parent_move = 'Start'
        return f'{parent_move} p: {float(prior)}, c: {int(self.visit_count)}, v: {float(self.value_avg()):.3f}'
            
    # def __repr__(self):
    #     """
    #     Debug: display pretty info
    #     """
        
    #     prior = "{0:.3f}".format(self.prior_prob)
    #     return "{} Prior: {} Count: {} Value: {}".format(self.board.move_stack, float(prior), int(self.visit_count), np.tanh(float(self.value_avg())))
        
        
      
def ucb_scores(parent: Node, children: dict[str, Node], dir_noise: bool=False):
    """
    The score for an action that would transition between the parent and child.
    """
    c_puct = 1.5
    dir_alpha = 0.35
    x_dir = 0.6
    
    prior_scores = {move: child.prior_prob * math.sqrt(parent.visit_count) / (child.visit_count + 1) for move, child in children.items()}
    
    # Insert Dirichlet noise like it was done in the AlphaZero paper.
    if dir_noise:
        dir_noise_obj = torch.distributions.Dirichlet(torch.tensor([dir_alpha for _ in children.values()]))
        dir_noise_sample = dir_noise_obj.sample()
        
        for idx, (key, prior_pro_value) in enumerate(prior_scores.items()):
            prior_scores[key] = (x_dir * prior_pro_value + (1 - x_dir) * dir_noise_sample[idx])\
                * math.sqrt(parent.visit_count) / (children[key].visit_count + 1)
    
    value_scores = {}
    for move, child in children.items():
        if child.visit_count > 0 and child.value_avg() is not None:
            value_scores[move] = - torch.tensor(child.value_avg())
    
        else:
            value_scores[move] = 0
            
        # Mate magnet edition
        if move[-1] == '#':
            value_scores[move] = np.inf
    
    collector_scores = {move: value_scores[move] + c_puct * prior_scores[move] for move, _ in children.items()}

    return collector_scores
        
        
class MCTS:
    """
    A class to run MCTS search; compared to simple AlphaZero, need to configure this to run simultaneous 
    board evaluations to levarage CUDA capabilities.
    """
    
    def __init__(self, model_good: AttentionChess2, model_evil: AttentionChess2, num_sims, device='cpu', use_dir=False):
        self.model_good = model_good
        self.model_evil = model_evil
        self.num_sims = num_sims
        self.model_good_flag = True
        self.device = device
        self.use_dir = use_dir
        
        self.model_good.to(self.device)
        self.model_evil.to(self.device)
        
        self.board_list = []
        self.move_count_vec = torch.zeros((0, 256)).to(self.device)
        self.board_value_vec = torch.zeros(0).to(self.device)
        
    @torch.no_grad()
    def run_engine(self, boards: list[chess.Board]):
        out_dict = self.model_good(boards) if self.model_good_flag else self.model_evil(boards)
        policy_list, value_list = self.model_good.post_process(boards, out_dict)
        return policy_list, value_list
        
    def get_endgame_value(self, board: chess.Board):
        """
        Get final value if the game ended, else get None
        """
        game_end_flag, result = _is_game_end(board)
        if game_end_flag:
            return result * 5.0
        else:
            return None
        
    def run(self, board: chess.Board, verbose=False):
        
        self.model_good_flag = True
        root = Node(board, 0.0, device=self.device, use_dir=self.use_dir)
        
        # Expand the root node
        policy_list, _ = self.run_engine([board])
        root.expand(policy_list[0])
        
        for _ in range(self.num_sims):
            node = root
            search_path = [node]
            
            # Select move to make
            while node.expanded():
                move, node = node.select_child()
                search_path.append(node)
                
            parent = search_path[-2]
            board = parent.board
            next_board = copy.deepcopy(board)
            next_board.push_san(move)
            
            value = self.get_endgame_value(next_board)
            
            if value is None:
                # Expand if game not ended
                policy_list, value_list = self.run_engine([next_board])
                node.expand(policy_list[0])
                value = value_list[0]
            else:
                value = math.tanh(value)
                node.half_move = 1 # Used to compute the value function
                
            self.backpropagate(search_path, value)
                
            if verbose:
                for node in search_path:
                    print(node)
        print(root)
        return root
    
    
    def collect_nodes_for_training(self, node: Node, min_counts = 5):
        """
        Consider all nodes that have X or more visits for future training of self play.
        """
        
        board_collection: list[chess.Board] = [node.board]
        policy_collection: torch.Tensor = self._create_policy_vector(node)
        value_collection: torch.Tensor = torch.tensor([node.value_avg()]).to(self.device)
        
        for child in node.children.values():
            
            # If the visit criteria is met
            if child.visit_count >= min_counts:
                
                board_add, policy_add, value_add = self.collect_nodes_for_training(child, min_counts=min_counts)
                
                # Recursivelly add the nodes with the correct count number
                board_collection.extend(board_add)
                policy_collection = torch.cat((policy_collection, policy_add), dim=0)
                value_collection = torch.cat((value_collection, value_add), dim=0)
                
            # Endgame position should be considered as an endgame position by the network.
            game_end, value_end = _is_game_end(child.board)
            if game_end:
                board_collection.append(child.board)
                policy_collection = torch.cat((policy_collection, torch.zeros(1, 4864).to(self.device)), dim=0)
                value_collection = torch.cat((value_collection, torch.tensor(value_end).to(self.device)), dim=0)
        
        return board_collection, policy_collection, value_collection    
    
    
    def run_multi(self, boards: list[chess.Board], verbose=False, print_enchors=True):
        
        self.model_good_flag = True
        roots = [Node(board, 0.0, self.device, use_dir=self.use_dir) for board in boards]
        root_boards = [node.board for node in roots]
        
        # Expand every root node
        policy_list, _  = self.run_engine(root_boards)
        for idx, root in enumerate(roots):
            root.expand(policy_list[idx])
            
        # Create win/loss/draw counters for printing
        white_win_count = 0
        black_win_count = 0
        draw_count = 0
            
        # Run sim for every board
        for _ in range(self.num_sims):
            
            node_edge_list = [None for _ in roots] # Need to do this, otherwise the roots will be overridden by the leaf nodes
            search_path_list = [[node] for node in roots]
            
            # Select a move to make per each node
            value_list = [None for _ in roots]
            board_slice_list = []
            
            # Expand tree nodes and input values until every value in the list is filled
            for idx, node in enumerate(roots):
                while node.expanded():
                    move, node = node.select_child()
                    node_edge_list[idx] = node
                    search_path_list[idx].append(node)

                # necessary to not break the loop when the game ended in one of the branches
                if len(search_path_list[idx]) >= 2:
                    
                    parent = search_path_list[idx][-2]
                    board = parent.board 
                    
                    next_board = copy.deepcopy(board)
                    next_board.push_san(move)
                    
                else:
                    
                    next_board = search_path_list[idx][-1].board
                
                value = self.get_endgame_value(next_board)
                value_list[idx] = value
                
                if value == 5:
                    white_win_count += 1
                elif value == -5:
                    black_win_count += 1
                elif value == 0:
                    draw_count += 1
                
                if value is None:
                    board_slice_list.append(next_board)
                else:
                    value = math.tanh(value)
                    node.half_move = 1 # Used to compute the value function in old version
                 
                    
            # Forward all boards through the net
            if len(board_slice_list) > 0:
                self.model_good_flag = not self.model_good_flag
                policy_list, value_list_out = self.run_engine(board_slice_list)
            
            # Expand every node that didn't reach the end
            node_selection_idx = 0
            for idx, node in enumerate(node_edge_list):
                if value_list[idx] is None:
                    node.expand(policy_list[node_selection_idx])
                    value_list[idx] = torch.atanh(value_list_out[node_selection_idx])
                    node_selection_idx += 1
                    
            for idx, search_path in enumerate(search_path_list):
                self.backpropagate(search_path, value_list[idx])
                
                if verbose:
                    for node in search_path:
                        print(node)
                        
        if print_enchors:
            print(f'Out of {self.num_sims} simulations, {len(roots)} roots, {white_win_count} white wins, {black_win_count} black wins, {draw_count} draws.')
                        
        return roots
                
            

    def backpropagate(self, search_path: list[Node], value: float, value_multiplier: float=0.95):
        
        turn_sign = 1 if search_path[-1].board.turn else -1
        
        for node_idx, node in reversed(list(enumerate(search_path))):
            node.value_sum += value * turn_sign
            
            if node_idx != 0:
                prior_board = copy.deepcopy(node.board)
                prior_board.pop()
                last_move = prior_board.san(node.board.peek())
                
                if search_path[node_idx - 1].value_candidates[last_move] is None:
                    search_path[node_idx - 1].value_candidates[last_move] = math.atanh(value * value_multiplier * -1)
                
                elif not node.board.turn and search_path[node_idx - 1].value_candidates[last_move] > value * value_multiplier:   
                    search_path[node_idx - 1].value_candidates[last_move] = math.atanh(value * value_multiplier * -1)
                    
                elif node.board.turn and search_path[node_idx - 1].value_candidates[last_move] < value * value_multiplier:    
                    search_path[node_idx - 1].value_candidates[last_move] = math.atanh(value * value_multiplier * -1)
            
            node.visit_count += 1
            value = math.tanh(math.atanh(value) * -value_multiplier)
            
            
    def backpropagate_new(self, search_path: list[Node], value: float, value_multiplier: float=1.0):
        """
        Backpropagation according to the paper that claims that the most greedy leaf should determine the value
        """
        
        for node_idx, node in reversed(list(enumerate(search_path))):
            
            node.visit_count += 1
            if node_idx == len(search_path) - 1:
                node.value_sum = node.visit_count * value
            else:
                node.value_sum = node.visit_count * node.find_greedy_value(value_multiplier=value_multiplier)
                
    def _create_policy_vector(self, node: Node):
        """
        Create a policy vector from a node. Converts the visit count dictionary into a probability vector sized 4864.
        """
        policy_vector = torch.zeros((1, 4864)).to(self.device)
        visit_count_dict = node.visit_count()
        
        for move_key, count_value in visit_count_dict.items():
            move = node.board.parse_san(move_key)
            word = move_to_word(move)
            policy_vector[word] = count_value
            
        policy_vector = F.normalize(policy_vector, p=1)
        return policy_vector
            

def _is_game_end(board: chess.Board):
    """Checks if the game ends."""
    if board.is_checkmate():
        result_const = -1 if board.turn else 1
        return True, result_const
    elif board.is_stalemate() or board.is_repetition() or \
            board.is_seventyfive_moves() or board.is_insufficient_material():
        return True, 0
    return False, 0
