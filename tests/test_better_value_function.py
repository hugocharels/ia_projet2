from lle import World, Action
from world_mdp import WorldMDP, BetterValueFunction
from adversarial_search import minimax, alpha_beta, expectimax
import csv

filename = "comparing.csv"
WORLDS = [
World("""
S0 . G G
G  @ @ @
.  . X X
S1 . . .
"""
)
,World("""
.  . . . G G S0
.  . . @ @ @ G
S2 . . X X X G
.  . . . G G S1
"""
),
]



DEPTHS = [*range(4, 12, 2)]

WMDPS = (WorldMDP, BetterValueFunction)

# ALGOS = ((minimax, "minimax"), (alpha_beta, "alpha_beta"))
ALGOS = ((alpha_beta, "alpha_beta"),)


def test_better_value():
    for i in range(len(WORLDS)):
        print("--------------------------------")
        for depth in DEPTHS:
            comparing = []
            actions = []
            for WMDP in WMDPS:
                for algo, name in ALGOS:
                    world = WMDP(WORLDS[i])
                    action = algo(world, world.reset(), depth)
                    n_states = world.n_expanded_states
                    # results.append([i, depth, WMDP.__name__, name, action, n_states])
                    comparing.append(n_states)
                    actions.append(action)
            print (f"world {i}, depth={depth}")
            print (f"Expanded nodes : world={comparing[0]}, better_value={comparing[1]}")
            print(actions)
            assert comparing[0] >= comparing[1]
