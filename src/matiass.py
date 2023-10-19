from lle import World, Action
from world_mdp import WorldMDP, BetterValueFunction
from adversarial_search import minimax, alpha_beta, expectimax
import csv


WORLDS = [
World("""
      . . . S2 S0 S1 .
      G . . . . . G
      L0E . . . . . .
      . . @ G . . .
      G . . . . . G
      . . . . @ G .
      . . X . . @ G
      G X X . . . .
      """
),

World("""
      . . . S2 S0 S1 .
      G . . . . . G
      @ . . @ @ @ @
      . . G G . . .
      G . . . . @ G
      . . . . @ G .
      . . . . . @ G
      G X X X . . .
      """
),

World("""
      . S0 S1 S2 . . .
      . . . . . . G
      L1E . . . @ . .
      . G . . G . G
      . . @ . . . L2W
      . . . . G . @
      . . @ . . . .
      . G . X X X G
      """
)

]

depth = 5

WMDPS = (WorldMDP, BetterValueFunction)

ALGOS = ((alpha_beta, "Alpha-Beta"),)


def main():
    results = []
    for i in range(len(WORLDS)):

        for WMDP in WMDPS:
            for algo, name in ALGOS:
                world = WMDP(WORLDS[i])
                action = algo(world, world.reset(), depth)
                n_states = world.n_expanded_states
                results.append([i, depth, WMDP.__name__, name, action, n_states])

    # Écrivez les résultats dans un fichier CSV
    with open('results.csv', 'w', newline='') as csvfile:
        fieldnames = ['World', 'Depth', 'WMDP', 'Algorithm', 'Action', 'Expanded States']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow({'World': str(result[0]), 'Depth': result[1], 'WMDP': result[2], 'Algorithm': result[3], 'Action': result[4], 'Expanded States': result[5]})

    print("Les résultats ont été enregistrés dans le fichier results.csv")


if __name__ == "__main__":
    main()

