#!/usr/bin/env python3
from lle import World, Action
from world_mdp import WorldMDP, BetterValueFunction
from adversarial_search import minimax, alpha_beta, expectimax
import csv
import cv2


WORLDS = [
World("""
S1 G S0 G . X
. . . . . X
"""
),
World("""
. . . G .
X . X @ @
. G @ G G
. S0 . G G
. S1 . @ @
"""),

]

#WORLDS[0].reset()
#WORLDS[0].step([Action.WEST, Action.EAST])


DEPTHS = [*range(1, 14, 2)]

WMDPS = (WorldMDP,BetterValueFunction)

ALGOS = ((minimax, "minimax"), (alpha_beta, "alpha_beta"))


def main():
    results = []
    for i in range(1, len(WORLDS)):
        cv2.imwrite(f"world_{i+1}.png", WORLDS[i].get_image())
        for depth in DEPTHS:
            print(depth)
            for WMDP in WMDPS:
                for algo, name in ALGOS:
                    if BetterValueFunction == WMDP and algo == minimax:continue

                    world = WMDP(WORLDS[i])
                    s0 = world.reset()

                    action = algo(world, s0, depth)
                    n_states = world.n_expanded_states
                    results.append([i, depth, WMDP.__name__, name, action, n_states])

    # Écrivez les résultats dans un fichier CSV
    with open('results_newworld1.csv', 'w', newline='') as csvfile:
        fieldnames = ['World', 'Depth', 'WMDP', 'Algorithm', 'Expanded States']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow({'World': str(result[0]), 'Depth': result[1], 'WMDP': result[2], 'Algorithm': result[3], 'Expanded States': result[5]})

    print("Les résultats ont été enregistrés dans le fichier results.csv")


if __name__ == "__main__":
	main()
