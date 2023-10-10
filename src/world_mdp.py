from dataclasses import dataclass
import lle
from lle import World, Action, WorldState
from mdp import MDP, State


@dataclass
class MyWorldState(State):
    world_state = WorldState

    def __init__(self, value: int, current_agent: int, world_state: WorldState):
        self.value = value
        self.current_agent = current_agent
        self.world_state = world_state


class WorldMDP(MDP[Action, MyWorldState]):
    def __init__(self, world: World):
        self.world = world
        self.last_values = [0] * self.world.n_agents

    def reset(self):
        self.n_expanded_states = 0
        self.world.reset()
        return MyWorldState(0, 0, self.world.get_state())

    def available_actions(self, state: MyWorldState) -> list[Action]:
        return self.world.available_actions()[state.current_agent]

    def is_final(self, state: MyWorldState) -> bool:
        self.world.set_state(state.world_state)
        return all(state.world_state.gems_collected) and (self.world.agents[0].has_arrived or self.world.agents[0].is_dead)

    def update_last_values(self, state: MyWorldState):
        self.last_values[state.current_agent] = state.value

    def last_value(self, agent_id: int) -> int:
        return self.last_values[agent_id]

    def transition(self, state: MyWorldState, action: Action) -> MyWorldState:
        self.world.set_state(state.world_state)
        actions = [Action.STAY] * self.world.n_agents
        actions[state.current_agent] = action
        value = self.world.step(actions)
        new_value = state.value + self.last_value(state.current_agent) if not self.world.agents[state.current_agent].is_dead else lle.REWARD_AGENT_DIED
        self.update_last_values(state)
        return MyWorldState(new_value, (state.current_agent + 1) % self.world.n_agents, self.world.get_state())

class BetterValueFunction(WorldMDP):
    def transition(self, state: MyWorldState, action: Action) -> MyWorldState:
        # Change the value of the state here.
        ...
