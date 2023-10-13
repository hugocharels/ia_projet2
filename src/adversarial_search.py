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
        new_state = mdp.transition(state, action)
        value = value_action(mdp, new_state, max_depth - 1 if new_state.current_agent == 0 else max_depth)[0]
        if value < best_value:
            best_value = value
            best_action = action
    return best_value, best_action

def value_action(mdp: MDP[A, S], state: S, max_depth: int) -> (float, A):
    if mdp.is_final(state) or max_depth == 0:
        return state.value, None
    if state.current_agent == 0:
        return max_value_action(mdp, state, max_depth)
    else:
        return min_value_action(mdp, state, max_depth)

def minimax(mdp: MDP[A, S], state: S, max_depth: int) -> A:
    """Returns the best action for the current agent to take in the given state, according to the minimax algorithm."""
    if state.current_agent != 0:  raise ValueError("The current agent must be 0.")
    if max_depth < 1: raise ValueError("The maximum depth must be at least 1.")
    return value_action(mdp, state, max_depth)[1]


def alpha_beta_max_value_action(mdp: MDP[A, S], state: S, max_depth: int, alpha: float, beta: float) -> (float, A):
    best_action = Action.STAY
    best_value = float('-inf')
    for action in mdp.available_actions(state):
        value = alpha_beta_value_action(mdp, mdp.transition(state, action), max_depth - 1, alpha, beta)[0]
        if value > best_value:
            best_value = value
            best_action = action
        if value >= beta:
            return value, action
        alpha = value if value > alpha else alpha
    return best_value, best_action

def alpha_beta_min_value_action(mdp: MDP[A, S], state: S, max_depth: int, alpha: float, beta: float) -> (float, A):
    best_action = Action.STAY
    best_value = float('inf')
    for action in mdp.available_actions(state):
        new_state = mdp.transition(state, action)
        value = alpha_beta_value_action(mdp, new_state, max_depth - 1 if new_state.current_agent == 0 else max_depth, alpha, beta)[0]
        if value < best_value:
            best_value = value
            best_action = action
        if value <= alpha:
            return value, action
        beta = value if value < beta else beta
    return best_value, best_action

def alpha_beta_value_action(mdp: MDP[A, S], state: S, max_depth: int, alpha: float, beta: float) -> (float, A):
    if mdp.is_final(state) or max_depth == 0:
        return state.value, Action.STAY
    if state.current_agent == 0:
        return alpha_beta_max_value_action(mdp, state, max_depth, alpha, beta)
    else:
        return alpha_beta_min_value_action(mdp, state, max_depth, alpha, beta)


def alpha_beta(mdp: MDP[A, S], state: S, max_depth: int) -> A:
    """Returns the best action for the current agent to take in the given state, according to the alpha-beta algorithm."""
    if state.current_agent != 0: raise ValueError("The current agent must be 0.")
    return alpha_beta_value_action(mdp, state, max_depth, float('-inf'), float('inf'))[1]


def expectimax(mdp: MDP[A, S], state: S, max_depth: int) -> A:
    ...
