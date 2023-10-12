from lle import Action
from mdp import MDP, S, A


def max_value_action(mdp: MDP[A, S], state: S, max_depth: int) -> (float, A):
    best_action = Action.STAY
    best_value = float('-inf')
    for action in mdp.available_actions(state):
        value = value_action(mdp, mdp.transition(state, action), max_depth - 1)[0]
        if value > best_value:
            best_value = value
            best_action = action
    return best_value, best_action

def min_value_action(mdp: MDP[A, S], state: S, max_depth: int) -> (float, A):
    best_action = Action.STAY
    best_value = float('inf')
    for action in mdp.available_actions(state):
        value = value_action(mdp, mdp.transition(state, action), max_depth - 1)[0]
        if value < best_value:
            best_value = value
            best_action = action
    return best_value, best_action

def value_action(mdp: MDP[A, S], state: S, max_depth: int) -> (float, A):
    if mdp.is_final(state) or max_depth == 0:
        return state.value, Action.STAY
    if state.current_agent == 0:
        return max_value_action(mdp, state, max_depth)
    else:
        return min_value_action(mdp, state, max_depth)

def minimax(mdp: MDP[A, S], state: S, max_depth: int) -> A:
    """Returns the best action for the current agent to take in the given state, according to the minimax algorithm."""
    if state.current_agent != 0: raise ValueError("The current agent must be 0.")
    return value_action(mdp, state, max_depth)[1]

def alpha_beta(mdp: MDP[A, S], state: S, max_depth: int) -> A:
    ...


def expectimax(mdp: MDP[A, S], state: S, max_depth: int) -> A:
    ...
