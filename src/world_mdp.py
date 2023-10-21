from dataclasses import dataclass
import lle
from lle import World, Action, WorldState
from mdp import MDP, State


MY_AGENT = 0


def override(abstract_class):
    """ Override decorator """
    def overrider(method):
        assert(method.__name__ in dir(abstract_class))
        return method
    return overrider


@dataclass
class MyWorldState(State):
    def __init__(self, value: int, current_agent: int, world_state: WorldState):
        self.value = value
        self.current_agent = current_agent
        self.world_state = world_state

    def __repr__(self):
        return f"<MyWorldState(value={self.value},current_agent={self.current_agent},world_state={self.world_state})>"


class WorldMDP(MDP[Action, MyWorldState]):
    def __init__(self, world: World):
        self.world = world

    def reset(self):
        self.n_expanded_states = 0
        self.world.reset()
        return MyWorldState(0, 0, self.world.get_state())

    def available_actions(self, state: MyWorldState) -> list[Action]:
        return self.world.available_actions()[state.current_agent]

    def is_final(self, state: MyWorldState) -> bool:
        self.world.set_state(state.world_state)
        return self.world.done

    def _compute_value(self, state: MyWorldState, step_reward: float) -> float:
        return (state.value + step_reward if not self.world.agents[state.current_agent].is_dead else lle.REWARD_AGENT_DIED) if state.current_agent == MY_AGENT else state.value

    def transition(self, state: MyWorldState, action: Action) -> MyWorldState:
        self.n_expanded_states += 1
        self.world.set_state(state.world_state)
        actions = [Action.STAY] * self.world.n_agents
        actions[state.current_agent] = action
        step_reward = self.world.step(actions)
        return MyWorldState(self._compute_value(state, step_reward), (state.current_agent + 1) % self.world.n_agents, self.world.get_state())

    def __repr__(self):
        return f"<WorldMDP(world={self.world.world_string})>"

class BetterValueFunction(WorldMDP):

    def _gems_remaining(self, state: MyWorldState) -> int:
        return self.world.n_gems - sum(state.world_state.gems_collected)

    def _compute_value(self, state: MyWorldState, step_reward: float) -> float:
        if self.world.agents[state.current_agent].is_dead: return lle.REWARD_AGENT_DIED
        if step_reward == 1: return state.value + self.world.n_gems - self._gems_remaining(state)
        return state.value + step_reward
