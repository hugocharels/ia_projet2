from lle import Action
from mdp import MDP, S, A
from queue import Queue as LifoQueue
from typing import Optional, Generic, TypeVar
from dataclasses import dataclass
from abc import ABC, abstractmethod


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
    #return value_action(mdp, state, max_depth)[1]
    return AdversarialSearch(mdp, max_depth, "AdversarialSearchStrategy.MINIMAX").search(state)


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





class AdversarialSearch:

    def __init__(self, mdp: MDP[A, S], max_depth: int, strategy):#: AdversarialSearchStrategy):
        self.mdp = mdp
        self.max_depth = max_depth
        self.strategy = strategy

    def search(self, state: S) -> A:
        if state.current_agent != 0: raise ValueError("The current agent must be 0.")
        stack = Stack()
        stack.push(Node(state, None, 0))
        best_action = None
        best_value = 0
        while not stack.is_empty():
            node = stack.pop()
            if self.mdp.is_final(node.state) or node.depth == self.max_depth: continue 
            maximize = True if node.state.current_agent == 0 else False
            best_value = float('-inf') if maximize else float('inf')
            
            for action in self.mdp.available_actions(node.state):
                new_state = self.mdp.transition(node.state, action)
                stack.push(Node(new_state, action, node.depth + 1 if not maximize and new_state.current_agent == 0 else node.depth))
                if maximize:
                    if new_state.value > best_value:
                        best_value = new_state.value
                        best_action = action
                else:
                    if new_state.value < best_value:
                        best_value = new_state.value
                        best_action = action

            if maximize and best_value > node.state.value: best_action = node.action
            elif not maximize and best_value < node.state.value: best_action = node.action

        return best_action











#################### frontier.py ####################

T = TypeVar("T")

class Frontier(ABC, Generic[T]):

    @abstractmethod
    def push(node: T) -> None:
        """ Add a node to the frontier. """

    @abstractmethod
    def pop() -> T:
        """ Remove and return the next node from the frontier. """

    @abstractmethod
    def is_empty() -> bool:
        """ Return True if the frontier is empty. """


class Stack(Frontier[T]):

    def __init__(self):
        self.queue = LifoQueue()

    #@override(Frontier)
    def push(self, node: T) -> None:
        self.queue.put(node)

    #@override(Frontier)
    def pop(self) -> T:
        return self.queue.get()

    #@override(Frontier)
    def is_empty(self) -> bool:
        return self.queue.empty()


@dataclass
class Node:

    state: S
    action: Optional[A]
    depth: int

    def __repr__(self):
        return f"<Node(state={self.state},parent={self.parent},action={self.action},cost={self.cost},depth={self.depth})>"