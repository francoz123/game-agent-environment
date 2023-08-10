import random
import time

from tictactoe_game_environment import TicTacToeGameEnvironment as gm

def selection_policy(node, target_player):
    successors = node.get_successors()
    best_successor = None
    for successor in successors:
        if successor.is_leaf_node():
            return successor
        if best_successor is None or is_best(successor):
            best_successor = successor
    return best_successor

def random_playout(initial_node):
    current_state = initial_node.get_state()
    while not gm.is_terminal(current_state):
        actions = gm.get_legal_actions(current_state.get_state())
        action = random.choice(actions)
        current_state = gm.transition_result(current_state, action)
    return gm.get_winner(current_state)

def mcts(root_node, target_player, max_time=1):
    start_time = time.time()

    # Performing simulations until the time is up
    while (time.time() - start_time) < max_time:
        # SELECTION PHASE
        current_node = root_node
        while not gm.is_terminal(current_node.get_state()) and not current_node.is_leaf_node():
            current_node = selection_policy(current_node, target_player)
        
        # EXPANSION PHASE
        selected_node = current_node
        legal_moves = gm.get_legal_actions(selected_node.get_state())
        for a in legal_moves:
            if not selected_node.was_action_expanded(a):
                successor_state = gm.transition_result(selected_node.get_state(), a)
                selected_node.add_successor(successor_state, a)
        
        # SIMULATION PHASE
        winner = random_playout(selected_node)
        if winner is None:
            # we consider a tie as a win
            # if we don't do that, we might find an optimal opponent
            # always 1 step ahead of us and the only best option
            # for us is to achieve a tie
            winner = target_player

        # BACKPROPAGATION PHASE
        selected_node.backpropagate(winner)
    
    # Time is up, choosing the child of the root node with highest wins
    best_node = None
    max_wins = 0
    for successor in root_node.get_successors():
        if best_node is None or successor.wins(target_player)/successor.n() > max_wins:
            best_node = successor
            max_wins = successor.wins(target_player) / successor.n()
    return best_node.get_action()

def is_best(initail_node, target_player):
    successors = initail_node.get_successors()
    best_node = None
    for successor in successors:
        if successor.wins(target_player) > initail_node.wins(target_player):
            best_node = successor
    return best_node