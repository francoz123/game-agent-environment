from une_ai.tictactoe import TicTacToeGame
from une_ai.tictactoe import TicTacToePlayer

from tictactoe_game_environment import TicTacToeGameEnvironment
from tictactoe_ttable import TicTacToeTTable

from agent_programs import agent_program_random
from agent_programs import agent_program_minimax, agent_program_minimax_alpha_beta, agent_program_optimised_minimax
from agent_programs import agent_program_mcts

if __name__ == '__main__':
    # Creating the two players
    # To change their behaviour, change the second parameter
    # of the constructor with the desired agent program function
    player_X = TicTacToePlayer('X', agent_program_random)
    player_O = TicTacToePlayer('O', agent_program_random)

    # DO NOT EDIT THE FOLLOWING INSTRUCTIONS!
    environment = TicTacToeGameEnvironment()
    environment.add_player(player_X)
    environment.add_player(player_O)

    game = TicTacToeGame(player_X, player_O, environment)