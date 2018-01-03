"""
Game 2048 3x3

Following the implementation of Georg Wiese: 
https://github.com/georgwiese/2048-rl

1) Editing the game setting to a 3x3 format with the purpose
to reduce the number of states. Game states are represented as shape (3, 3) numpy arrays 
whose entries are 0 for empty fields and ln2(value) for any tiles.

2) Encapsulate experience
    
3) Play the game
    
4) Strategies: random, static, highest reward, greedy, epsilon greedy

Algorithms and strategies to play 2048 and collect experience.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import Numby library for array calculations
import numby as np 


# Define Actions
ACTION_NAMES = ["left", "up", "right", "down"]
ACTION_LEFT = 0
ACTION_UP = 1
ACTION_RIGHT = 2
ACTION_DOWN = 3


class Game(object):

  def __init__(self, state=None, initial_score=0): 
    """ first step of game
    arguments:
        state: (3,3) numby array to initialize the state with. If None,
        the state will be initialized with with two random tiles (as done in the original game).
        initial_score: Score to initialize the Game with.
    """
    self._score = initial_score

    if state is None:  # if start of the game -> state =None 
      self._state = np.zeros((3, 3), dtype=np.int) 
      # edit (3,3), start of the game -> empty arrays (zeros)
      self.add_random_tile() # add two random tiles at the beginning
      self.add_random_tile()
    else:
      self._state = state # if not begin of the game...
         

  def copy(self):
    """Return a copy of self."""
    return Game(np.copy(self._state), self._score)

## define game over, available actions, do actions, add tiles 

  def game_over(self):
    """Whether the game is over."""

    for action in range(4): # four possible actions
      if self.is_action_available(action):
        return False
    return True

  def available_actions(self):
    """Computes the set of actions that are available."""
    return [action for action in range(4) if self.is_action_available(action)]

  def is_action_available(self, action):
    """Determines whether action is available.
    That is, executing it would change the state.
    """

    temp_state = np.rot90(self._state, action) # rotate array by 90 degrees
    return self._is_action_available_left(temp_state)

  def _is_action_available_left(self, state):
    """Determines whether action 'Left' is available."""

    # True if any field is 0 (empty) on the left of a tile or two tiles can
    # be merged.
    for row in range(3):
      has_empty = False
      for col in range(3):
        has_empty |= state[row, col] == 0
        if state[row, col] != 0 and has_empty:
          return True # left is possible 
        if (state[row, col] != 0 and col > 0 and
            state[row, col] == state[row, col - 1]):
          return True # left is possible

    return False # else, left is impossible


  def do_action(self, action):
    """Execute action, add a new tile, update the score & return the reward."""

    temp_state = np.rot90(self._state, action)
    reward = self._do_action_left(temp_state)
    self._state = np.rot90(temp_state, -action)
    self._score += reward # add value to variable (e.g. if merged)

    self.add_random_tile()

    return reward

  def _do_action_left(self, state):
    """Exectures action 'Left'."""

    reward = 0

    for row in range(3): 
      # Always the rightmost tile in the current row that was already moved
      merge_candidate = -1 # tile one to the left of rightmost tile
      merged = np.zeros((3,), dtype=np.bool) # bool: binary array (e.g. true/flase
      

      for col in range(3): 
        if state[row, col] == 0:
          continue

        if (merge_candidate != -1 and
            not merged[merge_candidate] and
            state[row, merge_candidate] == state[row, col]):
          # Merge tile with merge_candidate
          state[row, col] = 0
          merged[merge_candidate] = True
          state[row, merge_candidate] += 1 # add value to variable
          reward += 2 ** state[row, merge_candidate]

        else:
          # Move tile to the left
          merge_candidate += 1
          if col != merge_candidate:
            state[row, merge_candidate] = state[row, col]
            state[row, col] = 0

    return reward

  def add_random_tile(self):
    """Adds a random tile to the grid. Assumes that it has empty fields."""

    x_pos, y_pos = np.where(self._state == 0)
    assert len(x_pos) != 0
    empty_index = np.random.choice(len(x_pos))
    value = np.random.choice([1, 2], p=[0.9, 0.1]) #add new tile 90% chance add 1, 10% chance add 2

    self._state[x_pos[empty_index], y_pos[empty_index]] = value

  def print_state(self):
    """Prints the current state."""

    def tile_string(value):
      """Concert value to string."""
      if value > 0:
        return '% 5d' % (2 ** value,)
      return "     "

    print ("-" * 25)
    for row in range(4): 
      print ("|" + "|".join([tile_string(v) for v in self._state[row, :]]) + "|")
      print ("-" * 25)

  def state(self):
    """Return current state."""
    return self._state

  def score(self):
    """Return current score."""
    return self._score

############################################3

class Experience(object):
  """Struct to encapsulate the experience of a single turn."""

  def __init__(self, state, action, reward, next_state, game_over,
               not_available, next_state_available_actions):
    """Initialize Experience

    Args:
      state: Shape (3, 3) numpy array, the state before the action was executed
      action: Number in range(4), action that was taken
      reward: Number, experienced reward
      next_state: Shape (3, 3) numpy array, the state after the action was
          executed
      game_over: boolean, whether next_state is a terminal state
      not_available: boolean, whether action was not available from state
      next_state_available_actions: Available actions from the next state
    """
    self.state = state
    self.action = action
    self.reward = reward
    self.next_state = next_state
    self.game_over = game_over
    self.not_available = not_available
    self.next_state_available_actions = next_state_available_actions

  def __str__(self):
    return str((self.state, self.action, self.reward, self.next_state,
                self.game_over, self.next_state_available_actions))

  def __repr__(self):
    return self.__str__()


############################################
    
def play(strategy, verbose=False, allow_unavailable_action=True):
  """Plays a single game, using a provided strategy.

  Args:
    strategy: A function that takes as argument a state and a list of available
        actions and returns an action from the list.
    allow_unavailable_action: Boolean, whether strategy is passed all actions
        or just the available ones.
    verbose: If true, prints game states, actions and scores.

  Returns:
    score, experiences where score is the final score and experiences is the
        list Experience instances that represent the collected experience.
  """

  game = Game()

  state = game.state().copy()
  game_over = game.game_over()
  experiences = []

  while not game_over:
    if verbose:
      print("Score:", game.score())
      game.print_state()

    old_state = state
    next_action = strategy(
        old_state, range(4) if allow_unavailable_action
                            else game.available_actions())

    if game.is_action_available(next_action):

      reward = game.do_action(next_action)
      state = game.state().copy()
      game_over = game.game_over()

      if verbose:
        print("Action:", ACTION_NAMES[next_action])
        print("Reward:", reward)

      experiences.append(Experience(old_state, next_action, reward, state,
                                    game_over, False, game.available_actions()))

    else:
      experiences.append(Experience(state, next_action, 0, state, False, True,
                                    game.available_actions()))

  if verbose:
    print("Score:", game.score())
    game.print_state()
    print("Game over.")

  return game.score(), experiences

################# Strategies ###########################

def random_strategy(_, actions):
  """Strategy that always chooses actions at random."""
  return np.random.choice(actions)


def static_preference_strategy(_, actions):
  """Always prefer left over up over right over top."""

  return min(actions)


def highest_reward_strategy(state, actions):
  """Strategy that always chooses the action of highest immediate reward.

  If there are any ties, the strategy prefers left over up over right over down.
  """

  sorted_actions = np.sort(actions)[::-1]
  rewards = map(lambda action: Game(np.copy(state)).do_action(action),
                sorted_actions)
  action_index = np.argsort(rewards, kind="mergesort")[-1]
  return sorted_actions[action_index]

def make_greedy_strategy(get_q_values, verbose=False):
  """Makes greedy_strategy."""

  def greedy_strategy(state, actions):
    """Strategy that always picks the action of maximum Q(state, action)."""
    q_values = get_q_values(state)
    if verbose:
      print("State:")
      print(state)
      print("Q-Values:")
      for action, q_value, action_name in zip(range(4), q_values, ACTION_NAMES):
        not_available_string = "" if action in actions else "(not available)"
        print("%s:\t%.2f %s" % (action_name, q_value, not_available_string))
    sorted_actions = np.argsort(q_values)
    action = [a for a in sorted_actions if a in actions][-1]
    if verbose:
      print("-->", ACTION_NAMES[action])
    return action

  return greedy_strategy


def make_epsilon_greedy_strategy(get_q_values, epsilon):
  """Makes epsilon_greedy_strategy."""

  greedy_strategy = make_greedy_strategy(get_q_values)

  def epsilon_greedy_strategy(state, actions):
    """Picks random action with prob. epsilon, otherwise greedy_strategy."""
    do_random_action = np.random.choice([True, False], p=[epsilon, 1 - epsilon])
    if do_random_action:
      return random_strategy(state, actions)
    return greedy_strategy(state, actions)

  return epsilon_greedy_strategy
