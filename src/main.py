#!/usr/bin/env python3
from lle import World, Action
from world_mdp import WorldMDP, BetterValueFunction
from adversarial_search import minimax, alpha_beta, expectimax
import csv
import cv2


WORLDS = [
World("""
. . . . .
. G . G .
. . . . .
. X . X .
. S0 . S1 .
"""
),
World("""
S1 G S0 G . X
. . . . . X
"""
),
]

#WORLDS[0].reset()
#WORLDS[0].step([Action.WEST, Action.EAST])


DEPTHS = [*range(1, 11)]

WMDPS = (WorldMDP,BetterValueFunction)

ALGOS = ((alpha_beta, "alpha_beta"),)


def main():
    results = []
    for i in range(len(WORLDS)-1):
        cv2.imwrite("world_{}.png".format(i), WORLDS[i].get_image())
        for depth in DEPTHS:
            print(depth)
            for WMDP in WMDPS:
                for algo, name in ALGOS:
                    world = WMDP(WORLDS[i])
                    s0 = world.reset()

                    action = algo(world, s0, depth)
                    n_states = world.n_expanded_states
                    results.append([i, depth, WMDP.__name__, name, action, n_states])

    # Écrivez les résultats dans un fichier CSV
    with open('results_newworld1.csv', 'w', newline='') as csvfile:
        fieldnames = ['World', 'Depth', 'WMDP', 'Algorithm', 'Algorithm Name', 'Expanded States']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow({'World': str(result[0]), 'Depth': result[1], 'WMDP': result[2], 'Algorithm': result[3], 'Algorithm Name': result[4], 'Expanded States': result[5]})

    print("Les résultats ont été enregistrés dans le fichier results.csv")


if __name__ == "__main__":
	main()
