from lle import Action
from mdp import MDP, S, A
from queue import Queue as LifoQueue
from typing import Optional, Generic, TypeVar
from dataclasses import dataclass
from abc import ABC, abstractmethod
from random import choice


def checker(func):
    def wrapper(mdp: MDP[A, S], state: S, max_depth: int):
        if state.current_agent != 0:  raise ValueError("The current agent must be 0.")
        if max_depth < 1: raise ValueError("The maximum depth must be at least 1.")
        return func(mdp, state, max_depth)
    return wrapper

@checker
def minimax(mdp: MDP[A, S], state: S, max_depth: int) -> A:
    return MinimaxSearch(mdp).search(state, max_depth)[1]

@checker
def alpha_beta(mdp: MDP[A, S], state: S, max_depth: int) -> A:
    return AlphaBetaSearch(mdp).search(state, max_depth)[1]

@checker
def expectimax(mdp: MDP[A, S], state: S, max_depth: int) -> A:
    return ExpectimaxSearch(mdp).search(state, max_depth)[1]

class AdversarialSearch(ABC, Generic[A, S]):
    def __init__(self, mdp: MDP[A, S]):
        self.mdp = mdp

    @abstractmethod
    def compare(self, maximize: bool, *args) -> (float, A, bool):
        ...

    def _get_successors(self, state: S, maximize: bool) -> [S]:
        for action in self.mdp.available_actions(state):
            yield self.mdp.transition(state, action), action

    def search(self, state: S, max_depth: int, alpha: float=float('-inf'), beta: float=float('inf')) -> (float, A):
        if max_depth == 0 or self.mdp.is_final(state):
            return state.value, None
        maximize = True if state.current_agent == 0 else False
        best_value = float('-inf') if maximize else float('inf')
        best_action = None
        for new_state, action in self._get_successors(state, maximize):
            value = self.search(new_state, max_depth - 1 if maximize or new_state.current_agent == 0 else max_depth, alpha, beta)[0]
            best_value, best_action, stop = self.compare(maximize, best_value, value, best_action, action, alpha, beta)
            if stop: break
            if maximize: alpha = value if value > alpha else alpha
            else: beta = value if value < beta else beta
        return best_value, best_action

class MinimaxSearch(AdversarialSearch):
    def compare(self, maximize: bool, best_value: float, value: float, best_action: A, action: A, *_) -> (float, A, bool):
        return (value, action, False) if (maximize and value > best_value) or (not maximize and value < best_value) else (best_value, best_action, False)

class AlphaBetaSearch(AdversarialSearch):
    def compare(self, maximize: bool, best_value: float, value: float, best_action: A, action: A, alpha: float, beta: float, *_) -> (float, A, bool):
        if maximize:
            if value > best_value:
                return value, action, value >= beta
        else:
            if value < best_value:
                return value, action, value <= alpha
        return best_value, best_action, False

class ExpectimaxSearch(AlphaBetaSearch):
    def _get_successors(self, state: S, maximize: bool) -> [S]:
        if maximize:
            for action in self.mdp.available_actions(state):
                yield self.mdp.transition(state, action), action
        else:
            action = choice(self.mdp.available_actions(state))
            yield self.mdp.transition(state, action), action
