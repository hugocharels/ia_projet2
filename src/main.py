#!/usr/bin/env python3
from lle import World, Action
from world_mdp import WorldMDP
from lle import REWARD_AGENT_DIED


def main():
    world = WorldMDP(
        World(
            """
    S0 . G
    S1 X X
"""
        )
    )
    s = world.reset()
    actions = [Action.EAST, Action.EAST, Action.EAST, Action.STAY, Action.SOUTH]
    for action in actions:
        print(s)
        s = world.transition(s, action)
    print(s)


if __name__ == "__main__":
	main()
