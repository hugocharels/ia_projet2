#!/usr/bin/env python3
from lle import World, Action
from world_mdp import WorldMDP
from lle import REWARD_AGENT_DIED
from adversarial_search import minimax
import cv2



def main():
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
    
    cv2.imwrite("world.png", world.world.get_image())
    
    action = minimax(world, world.reset(), 3)
    print(action)
    
    # action = minimax(world, world.reset(), 7)
    # print(action)


if __name__ == "__main__":
	main()
