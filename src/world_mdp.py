from dataclasses import dataclass
import lle
from lle import World, Action, WorldState
from mdp import MDP, State


@dataclass
class MyState(State):
    world_state = WorldState

    def __init__(self, value: int, current_agent: int, world_state: WorldState):
        self.value = value
        self.current_agent = current_agent
        self.world_state = world_state

class WorldMDP(MDP[Action, MyState]):
    def __init__(self, world: World):
        self.world = world

    def reset(self):
        self.n_expanded_states = 0
        self.world.reset()
        return MyState(0, 0, self.world.get_state())

    def available_actions(self, state: MyState) -> list[Action]:
        return self.world.available_actions()[state.current_agent]

    def is_final(self, state: MyState) -> bool:
        self.world.set_state(state.world_state)
        return all(state.world_state.gems_collected)# and self.

    def transition(self, state: MyState, action: Action) -> MyState:
        self.world.set_state(state.world_state)
        actions = [Action.STAY] * self.world.n_agents
        actions[state.current_agent] = action
        value = self.world.step(actions)
        new_value = state.value + value if state.value >= 0 else lle.REWARD_AGENT_DIED
        return MyState(new_value, (state.current_agent + 1) % self.world.n_agents, self.world.get_state())


class BetterValueFunction(WorldMDP):
    def transition(self, state: MyState, action: Action) -> MyState:
        # Change the value of the state here.
        ...
