from dataclasses import dataclass
import lle
from lle import World, Action, WorldState
from mdp import MDP, State


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

    def transition(self, state: MyWorldState, action: Action) -> MyWorldState:
        self.n_expanded_states += 1
        self.world.set_state(state.world_state)
        actions = [Action.STAY] * self.world.n_agents
        actions[state.current_agent] = action
        value = self.world.step(actions)
        new_value = state.value
        if state.current_agent == 0:
            new_value = value + state.value if not self.world.agents[state.current_agent].is_dead else lle.REWARD_AGENT_DIED
        return MyWorldState(new_value, (state.current_agent + 1) % self.world.n_agents, self.world.get_state())

    def __repr__(self):
        return f"<WorldMDP(world={self.world.world_string})>"

class BetterValueFunction(WorldMDP):
    def transition(self, state: MyWorldState, action: Action) -> MyWorldState:
        self.n_expanded_states += 1
        self.world.set_state(state.world_state)
        actions = [Action.STAY] * self.world.n_agents
        actions[state.current_agent] = action
        value = self.world.step(actions)
        new_value = state.value
        if state.current_agent == 0:
            new_value = value + state.value if not self.world.agents[state.current_agent].is_dead else lle.REWARD_AGENT_DIED
        if state.current_agent != 0:
            new_value = value + state.value if not self.world.agents[state.current_agent].is_dead else -lle.REWARD_AGENT_DIED
        return MyWorldState(new_value, (state.current_agent + 1) % self.world.n_agents, self.world.get_state())
