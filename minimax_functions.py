from tictactoe_game_environment import TicTacToeGameEnvironment as gm
from une_ai.models import GraphNode

def minimax(node, player, depth):
    print('{0}'.format(node))
    best_move = None
    gs = node.get_state()
    player_turn = gs['player-turn']
    is_maximizing = player_turn == player

    if is_maximizing:
        best_payoff = float('-inf')
    else:
        best_payoff = float('+inf')
    
    if gm.is_terminal(gs) or depth <= 0:
        best_payoff = gm.payoff(gs, player)
        return best_payoff, best_move
    
    actions = gm.get_legal_actions(gs)

    for action in actions:
        future_state = gm.transition_result(gs, action)
        (payoff, _) = minimax(GraphNode(future_state, node, action, 1), player, depth-1)

        if (is_maximizing and payoff > best_payoff) \
            or not is_maximizing and payoff < best_payoff:
            best_payoff = payoff
            best_move = action
    return (best_payoff, best_move)

def minimax_alpha_beta(node, player, alpha, beta, depth):
    print('{0}'.format(node))
    best_move = None
    gs = node.get_state()
    player_turn = gs['player-turn']
    is_maximizing = player_turn == player

    if is_maximizing:
        best_payoff = float('-inf')
    else:
        best_payoff = float('+inf')
    
    if gm.is_terminal(gs) or depth <= 0:
        best_payoff = gm.payoff(gs, player)
        return best_payoff, best_move
    
    actions = gm.get_legal_actions(gs)

    for action in actions:
        future_state = gm.transition_result(gs, action)
        (payoff, _) = minimax(GraphNode(future_state, node, action, 1), player, depth-1)

        if (is_maximizing and payoff > best_payoff) \
            or not is_maximizing and payoff < best_payoff:
            best_payoff = payoff
            best_move = action
    return (best_payoff, best_move)

def optimised_minimax(node, player, tt, depth):
    return 0, None