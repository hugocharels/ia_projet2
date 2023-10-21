from lle import Action
from mdp import MDP, S, A
from world_mdp import MY_AGENT, override
from queue import Queue as LifoQueue
from typing import Optional, Generic, TypeVar
from dataclasses import dataclass
from abc import ABC, abstractmethod
from random import choice


def checker(func):
    """ Decorator that checks the arguments of the decorated function """
    def wrapper(mdp: MDP[A, S], state: S, max_depth: int):
        if state.current_agent != MY_AGENT:  raise ValueError("The current agent must be 0.")
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

    def _compare(self, maximize: bool, best_value: float, value: float, best_action: A, action: A, *_) -> (float, A, bool):
        return (value, action, False) if (maximize and value > best_value) or (not maximize and value < best_value) else (best_value, best_action, False)

    def _get_successors(self, state: S, maximize: bool) -> [S]:
        for action in self.mdp.available_actions(state):
            yield self.mdp.transition(state, action), action

    @abstractmethod
    def search(self, state: S, max_depth: int, *_) -> (float, A):
        ...


class MinimaxSearch(AdversarialSearch):

    @override(AdversarialSearch)
    def search(self, state: S, max_depth: int) -> (float, A):
        if max_depth == 0 or self.mdp.is_final(state):
            return state.value, None
        maximize = True if state.current_agent == MY_AGENT else False
        best_value = float('-inf') if maximize else float('inf')
        best_action = None
        for new_state, action in self._get_successors(state, maximize):
            value = self.search(new_state, max_depth - 1 if maximize or new_state.current_agent == MY_AGENT else max_depth)[0]
            best_value, best_action, stop = self._compare(maximize, best_value, value, best_action, action)
        return best_value, best_action


class AlphaBetaSearch(AdversarialSearch):

    @override(AdversarialSearch)
    def _compare(self, maximize: bool, best_value: float, value: float, best_action: A, action: A, alpha: float, beta: float) -> (float, A, bool):
        new_best_value, new_best_action, _ = super()._compare(maximize, best_value, value, best_action, action)
        return new_best_value, new_best_action, True if (maximize and new_best_value >= beta) or (not maximize and new_best_value <= alpha) else False

    @override(AdversarialSearch)
    def search(self, state: S, max_depth: int, alpha: float=float('-inf'), beta: float=float('inf')) -> (float, A):
        if max_depth == 0 or self.mdp.is_final(state):
            return state.value, None
        maximize = True if state.current_agent == MY_AGENT else False
        best_value = float('-inf') if maximize else float('inf')
        best_action = None
        for new_state, action in self._get_successors(state, maximize):
            value = self.search(new_state, max_depth - 1 if maximize or new_state.current_agent == MY_AGENT else max_depth, alpha, beta)[0]
            best_value, best_action, stop = self._compare(maximize, best_value, value, best_action, action, alpha, beta)
            if stop: break
            if maximize: alpha = max(alpha, value)
            else: beta = min(beta, value)
        return best_value, best_action


class ExpectimaxSearch(MinimaxSearch):

    @override(AdversarialSearch)
    def search(self, state: S, max_depth: int) -> (float, A):
        if max_depth == 0 or self.mdp.is_final(state):
            return state.value, None
        maximize = True if state.current_agent == MY_AGENT else False
        if not maximize:
            successors = list(self._get_successors(state, maximize))
            # Calculate the expected value for chance nodes
            expected_value = sum(self.search(new_state, max_depth - 1)[0] for new_state, _ in successors) / len(successors)
            return expected_value, None
        best_value = float('-inf') if maximize else float('inf')
        best_action = None
        for new_state, action in self._get_successors(state, maximize):
            value = self.search(new_state, max_depth - 1)[0]
            best_value, best_action, stop = self._compare(maximize, best_value, value, best_action, action)
        return best_value, best_action
