import numpy as np
from scipy.signal import convolve2d
from une_ai.models import GameEnvironment, GridMap, Agent

class IllegalMove(Exception):
    pass

class TicTacToeGameEnvironment(GameEnvironment):

    def __init__(self, board_size=3):
        super().__init__("Tic Tac Toe")

        self.board_size = board_size
        self.game_state = {
            'game-board': GridMap(board_size, board_size, None),
            'player-turn':'X'
        }

    # TODO
    # implement the abstract method add_player
    # the GameEnvironment superclass uses a dictionary self._players 
    # to store the players of the game.
    # For this game, we must limit the players to 2 players and
    # The first added player will be X and the second O
    def add_player(self, player):
        assert isinstance(player, Agent), "The parameter player must be an instance of a subclass of the class Agent"
        assert len(self._players) < 2, "It is not possible to add more than 2 players for this game."

        if len(self._players) == 0:
            self._players['X'] = player
            return 'X'
        if len(self._players) == 1:
            self._players['O'] = player
            return 'O'

    # TODO
    # implement the abstract method get_game_state
    # the method must return the current state of the game
    # as a dictionary with the following keys:
    # 'game-board' -> a copy of the game board (as 3x3 GridMap)
    # 'player-turn' -> 'X' or 'O' depending on the current player turn
    # You may first create properties in the constructor function __init__
    # to store the game board and the current turn
    def get_game_state(self):
        return self.game_state.copy()
    
    # TODO
    # implement the abstract method get_percepts
    # this method returns a dictionary with keys the sensors of the agent
    # and values the percepts gathered for that sensor at time t
    # the sensors are:
    # 'game-board-sensor' -> the 'game-board' value from the current game state
    # 'turn-taking-indicator' -> the 'player-turn' value from the current game state
    def get_percepts(self):
        return {
            'game-board-sensor':self.get_game_state()['game-board'],
            'turn-taking-indicator':self.get_game_state()['player-turn']
        }

    # TODO
    # implement the abstract method get_legal_actions
    # This method is a static method (i.e. we do not have access to self
    # and it can only be accessed via the class TicTacToeGameEnvironment)
    # It takes a game_state as input and it returns the list of
    # legal actions in that game state
    # An action is legal in a given game state if the game board cell 
    # for that action is free from marks
    def get_legal_actions(game_state):
        actions = []
        board = game_state['game-board']
        for x in range(board.get_width()):
            for y in range(board.get_height()):
                if board.get_item_value(x, y) == None:
                    actions.append('mark-{0}-{1}'.format(x,y))
        return actions
    
    # TODO
    # implement the abstract method transition_result
    # This method is a static method (i.e. we do not have access to self
    # and it can only be accessed via the class TicTacToeGameEnvironment)
    # It takes a game_state and an action to perform as input and it returns
    # the new game state.
    def transition_result(game_state, action):
        if not action in TicTacToeGameEnvironment.get_legal_actions(game_state):
            raise(IllegalMove('Illegal move attempted'))
        tokens = action.split('-')
        x, y = (int(tokens[1]), int(tokens[2]))
        board = game_state['game-board'].copy()
        player = TicTacToeGameEnvironment.turn(game_state)
        board.set_item_value(x, y, player)

        return {
            'game-board': board,
            'player-turn': 'X' if player == 'O' else 'O'
        }
    
    # TODO
    # implement the abstract method state_transition
    # this method takes as input the agent's actuators
    # and it changes the game environment state based
    # on the values of the agent's actuators
    # This agent has only one actuator, 'marker'
    # the value of this actuator is a tuple with the x and y
    # coordinates where the agent will place its marker on the game board
    # We can implement this method by re-using the static method
    # transition_result we just implemented
    def state_transition(self, agent_actuators):
        assert agent_actuators['marker'] is not None, "During a turn, the player must have set the 'marker' actuator value to a coordinate (x, y) of the game board where to place the marker."
        
        x, y = agent_actuators['marker']
        gs = self.get_game_state()
        action =  'mark-{0}-{1}'.format(x, y)
        new_gs = TicTacToeGameEnvironment.transition_result(gs, action)
        self.game_state['game-board'] = new_gs['game-board'].copy()
        self.game_state['player-turn'] = new_gs['player-turn']
    
    # This method is a static method (i.e. we do not have access to self
    # and it can only be accessed via the class TicTacToeGameEnvironment)
    # It returns the turn of the player given a game state.
    def turn(game_state):
        assert 'player-turn' in game_state.keys(), "Invalid game state. A game state must have the key 'player-turn'"

        return game_state['player-turn']
    
    # This method is a static method (i.e. we do not have access to self
    # and it can only be accessed via the class TicTacToeGameEnvironment)
    # It takes a game_state as input and it returns the winner ('X' or 'O') if there is any
    # or None if there is no winner (a tie or a non-terminal state)
    # This method is already provided to you. You should look at its implementation
    # and try to understand how it is finding a winner with the convolution operation
    def get_winner(game_state):
        game_board = game_state['game-board']

        horizontal_kernel = np.array([[ 1, 1, 1]])
        vertical_kernel = np.transpose(horizontal_kernel)
        diag_kernel = np.eye(3, dtype=np.uint8)
        flipped_diag_kernel = np.fliplr(diag_kernel)
        detection_kernels = [horizontal_kernel, vertical_kernel, diag_kernel, flipped_diag_kernel]

        for marker in ['X', 'O']:
            player_markers = game_board.get_map() == marker
            for kernel in detection_kernels:
                convolved_values = convolve2d(player_markers, kernel, mode="valid")
                if (convolved_values == 3).any():
                    return marker
        
        return None

    # TODO
    # implement the abstract method is_terminal
    # This method is a static method (i.e. we do not have access to self
    # and it can only be accessed via the class TicTacToeGameEnvironment)
    # It takes a game_state as input and it returns True if the game state
    # is terminal and False otherwise.
    # In this game, a state is terminal if there are no more legal actions
    # or if there is a winner.
    def is_terminal(game_state):
        return len(TicTacToeGameEnvironment.get_legal_actions(game_state)) == 0\
            or TicTacToeGameEnvironment.get_winner(game_state) is not None

    # TODO
    # implement the abstract method payoff
    # This method is a static method (i.e. we do not have access to self
    # and it can only be accessed via the class TicTacToeGameEnvironment)
    # It takes a game_state and the player name ('X' or 'O') as input and it returns
    # the payoff value for that player in the given game state
    # In this scenario, we are only considering terminal states with a winner
    # if there is not a winner yet (or there is a tie) we return 0
    # In other games the payoff function may be more complex
    def payoff(game_state, player_name):
        if TicTacToeGameEnvironment.get_winner(game_state) == player_name:
            return 1
        if TicTacToeGameEnvironment.get_winner(game_state) is not None:
            return -1
        return 0
     