#!/usr/bin/env python3
from lle import World, Action
from world_mdp import WorldMDP
from lle import REWARD_AGENT_DIED
from adversarial_search import minimax
import cv2
import sys
sys.path.append("..")
from tests.graph_mdp import GraphMDP


def main():
    mdp = GraphMDP.parse("tests/graphs/vary-depth.graph")
    s0 = mdp.reset()
    s = mdp.transition(s0, "Right")
    


def main2():
    world = WorldMDP(
        World(
            """
        .  . . . G G S0
        .  . . @ @ @ G
        S2 . . X X X G
        .  . . . G G S1
"""
        )
    )
    
    action = minimax(world, world.reset(), 3)
    print(action)
    
    action = minimax(world, world.reset(), 7)
    print(action)


if __name__ == "__main__":
	main()
