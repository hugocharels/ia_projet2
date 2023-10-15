#!/usr/bin/env python3
from lle import World, Action
from world_mdp import WorldMDP, BetterValueFunction
from lle import REWARD_AGENT_DIED
from adversarial_search import minimax, alpha_beta, expectimax
import cv2



def main():
    

    w = World(
        """
        .  . . . G G S0
        .  . . @ @ @ G
        S2 . . X X X G
        .  . . . G G S1
    """
    )

    depths = [*range(1, 10)]
    for depth in depths:
        print(f"----------------------------------\nfor depth={depth}")
        for WMDP in (WorldMDP, BetterValueFunction):
            for algo in (alpha_beta,):
                world = WMDP(w)
                print(f"<WorldMDP={WMDP.__name__},algo={algo.__name__},depth={depth} : {algo(world, world.reset(), depth)}>")
                print(f"Number of expanded states: {world.n_expanded_states}")

if __name__ == "__main__":
	main()
