#!/usr/bin/env python3
import matplotlib.pyplot as plt
from lle import World, Action
from world_mdp import WorldMDP, BetterValueFunction
from adversarial_search import minimax, alpha_beta, expectimax
import csv
import cv2

VERBOSE = True
# VERBOSE = False

GENERATE_DATA = True
#GENERATE_DATA = False

# DATA_FILENAME = 'resultForRapport.csv'
# DATA_FILENAME = '../resultForRapport.csv'
DATA_FILENAME = 'temp.csv'
# DATA_FILENAME = 'temp2.csv'

DEPTH_MAX = 10
# DEPTH_MAX = 6

GENERATE_GRAPHICS = True
#GENERATE_GRAPHICS = False

GENERATE_IMAGES = True
#GENERATE_IMAGES = False
IMAGE_FILENAME = 'monde{i}.png'

W1_FILENAME = 'graph1'
W2_FILENAME = 'graph2'
W3_FILENAME = 'graph3'


WORLD1 = World("""
S1 G S0 G . X
. . . . . X
"""
)

WORLD2 = World("""
. . . G .
X . X @ @
. G @ G G
. S0 . G G
. S1 . @ @
""")

WORLD3 = World("""
. . @ G G @ . .
@ @ @ G G @ @ @
. G . . . . G .
G G . S0 S1 . G .
@ @ . . . . @ @
. . X G G X . .
""")



WORLDS, WMDPS = [WORLD1, WORLD2, WORLD3], (WorldMDP, BetterValueFunction)

ALGOS = ((minimax, "minimax"), (alpha_beta, "alpha_beta"))
# ALGOS = ((minimax, "minimax"), (alpha_beta, "alpha_beta"), (expectimax, "expectimax"))

def generateImages():
    for i in range(len(WORLDS)):
        if VERBOSE:
            print('generating image for world', i)
        img = WORLDS[i].get_image()
        cv2.imwrite(IMAGE_FILENAME.format(i=i+1), img)
        print(f"Les images ont été enregistrées dans le fichier {IMAGE_FILENAME}")

def generateData():
    d_max = DEPTH_MAX
    DEPTHS = [*range(1, d_max + 1)]
    with open(DATA_FILENAME, 'w', newline='') as csvfile:
        fieldnames = ['World', 'Depth', 'WMDP', 'Algorithm', 'Algorithm Name', 'Expanded States']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for i in range(len(WORLDS)):
            for depth in DEPTHS:
                scores = []
                for algo, name in ALGOS:
                    for WMDP in WMDPS:
                        if name == 'minimax' and WMDP == BetterValueFunction:
                            continue
                        if VERBOSE:
                            print('calculating world {}, depth {}, WMDP {}, algo {}'.format(i, depth, WMDP.__name__, name))
                        world = WMDP(WORLDS[i])
                        action = algo(world, world.reset(), depth)
                        n_states = world.n_expanded_states
                        writer.writerow({'World': str(i), 'Depth': depth, 'WMDP': WMDP.__name__, 'Algorithm': name, 'Expanded States': n_states})
                        scores.append((name, n_states))
    print (f"Les résultats ont été enregistrés dans le fichier {DATA_FILENAME}")

def extractData(world, name, algo):
    filename = DATA_FILENAME
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.split(',')
            if line[0] == world and line[2] == name and line[3] == algo:
                data.append(int(line[5]))
    return data

def generateTripleBarChart(filename, barename1, barename2, barename3, color1, color2, color3, data1, data2, data3):
    if VERBOSE:
        print('Generating graphics for', filename)
    
    labels = [str(i) for i in range(1, DEPTH_MAX + 1)]

    plt.plot(labels, data1, label=barename1)
    plt.plot(labels, data2, label=barename2)
    plt.plot(labels, data3, label=barename3)

    plt.yscale('log')
    plt.ylabel('Nombre de nœuds étendus (échelle logarithmique)')
    plt.xlabel('Profondeur')
    plt.title('Nombre de nœuds étendus par profondeur')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def generateGraphics():
    dataset = [ ['Minimax', 'blue', extractData('0', 'WorldMDP', 'minimax')],
                ['AlphaBeta', 'red', extractData('0', 'WorldMDP', 'alpha_beta')],
                ['BetterValue AlphaBeta', 'green', extractData('0', 'BetterValueFunction', 'alpha_beta')],
                ['MiniMax', 'blue', extractData('1', 'WorldMDP', 'minimax')],
                ['AlphaBeta', 'red', extractData('1', 'WorldMDP', 'alpha_beta')],
                ['BetterValue AlphaBeta', 'green', extractData('1', 'BetterValueFunction', 'alpha_beta')],
                ['MiniMax', 'blue', extractData('2', 'WorldMDP', 'minimax')],
                ['AlphaBeta', 'red', extractData('2', 'WorldMDP', 'alpha_beta')],
                ['BetterValue AlphaBeta', 'green', extractData('2', 'BetterValueFunction', 'alpha_beta')] ]

    if GENERATE_GRAPHICS:
        generateTripleBarChart(W1_FILENAME, dataset[0][0], dataset[1][0], dataset[2][0], dataset[0][1], dataset[1][1], dataset[2][1], dataset[0][2], dataset[1][2], dataset[2][2])
        generateTripleBarChart(W2_FILENAME, dataset[3][0], dataset[4][0], dataset[5][0], dataset[3][1], dataset[4][1], dataset[5][1], dataset[3][2], dataset[4][2], dataset[5][2])
        generateTripleBarChart(W3_FILENAME, dataset[6][0], dataset[7][0], dataset[8][0], dataset[6][1], dataset[7][1], dataset[8][1], dataset[6][2], dataset[7][2], dataset[8][2])
        
        print (f"Les graphiques ont été enregistrés dans les fichiers {W1_FILENAME}, {W2_FILENAME} et {W3_FILENAME}")
    else:
        i=0
        print('data for latex')
        for data in dataset:
            print('world',i//len(WORLDS),data[0],data[2])
            i+=1

if __name__ == '__main__':
    if GENERATE_IMAGES:
        generateImages()
    if GENERATE_DATA:
        generateData()
    generateGraphics()

