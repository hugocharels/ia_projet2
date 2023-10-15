#!/usr/bin/env python3
from lle import World, Action
from world_mdp import WorldMDP, BetterValueFunction
from adversarial_search import minimax, alpha_beta, expectimax
import csv


WORLDS = [
World("""
.  . . . G G S0
.  . . @ @ @ G
S2 . . X X X G
.  . . . G G S1
"""
),

]


DEPTHS = [*range(1, 11)]

WMDPS = (WorldMDP, BetterValueFunction)

ALGOS = ((alpha_beta, "alpha_beta"), (expectimax, "expectimax"))


def main():
    results = []
    for i in range(len(WORLDS)):
        for depth in DEPTHS:
            for WMDP in WMDPS:
                for algo, name in ALGOS:
                    world = WMDP(WORLDS[i])
                    action = algo(world, world.reset(), depth)
                    n_states = world.n_expanded_states
                    results.append([i, depth, WMDP.__name__, name, action, n_states])

    # Écrivez les résultats dans un fichier CSV
    with open('results.csv', 'w', newline='') as csvfile:
        fieldnames = ['World', 'Depth', 'WMDP', 'Algorithm', 'Algorithm Name', 'Expanded States']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow({'World': str(result[0]), 'Depth': result[1], 'WMDP': result[2], 'Algorithm': result[3], 'Algorithm Name': result[4], 'Expanded States': result[5]})

    print("Les résultats ont été enregistrés dans le fichier results.csv")


if __name__ == "__main__":
	main()
