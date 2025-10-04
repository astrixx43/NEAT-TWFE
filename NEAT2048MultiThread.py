import random
import threading

from TWFENoUI import TWFENoUI
from viz import drawGenomeGraph
from viz import drawProgressGraph
from viz import updateProgressGraph
from viz import drawAverageFitnessGraph
from viz import updateAverageFitnessGraph
import viz
from DeadThread import DeadThread
import math
import time

ButtonNames = [
    "Up",
    "Down",
    "Left",
    "Right"
]

BoardSize = 4
InputSize = BoardSize**2

Inputs = InputSize + 1      # +1 for the bias
Outputs = len(ButtonNames)

Population = 1000
DeltaDisjoint = 2.0
DeltaWeights = 0.4

DeltaThreshold = 2.7
MinDeltaThreshold = 0.1
MaxDeltaThreshold = 7.0
DeltaThresholdAdjustmentStep = 0.05
TargetSpeciesCount = 100
# DeltaThreshold = 2.7

StaleSpecies = 15

MutateConnectionsChance = 0.25
PerturbChance = .9




CrossoverChance = 0.75
LinkMutationChance = 6.7
NodeMutationChance = 1.50
BiasMutationChance = 0.40
StepSize = 0.1
DisableMutationChance = 0.2
EnableMutationChance = 0.3

ConfidenceThreshold = 0.5
MaxTimeBetweenEachMove = 5

MaxNodes = 1000000

Scores = []

Generations = []
DrawProgressGraph = True
TopScoresOfGeneration = []
BestTopScores = []
DrawAverageFitness = True
AverageFitness = []
FitnessOfGeneration = []
DrawGenomeNeuralNetwork = False
DrawLargestTileGraph = True
LargestTilesOverGenerations = []
LargestTileOfGeneration = []

Boards = []
THREADS = []
THREADMAX = 100

NumParentThreads = 20
NumChildThreads = 10


def initializeBoards():
    for i in range(NumParentThreads):
        Boards.append([])
        THREADS.append(DeadThread())
        THREADS.append([])
        Scores.append([])
        for j in range(NumChildThreads):
            Boards[i].append(TWFENoUI())
            Scores[i].append([0, 0, 0])
            THREADS[(i * 2) + 1].append(DeadThread())


def getScore(indexOfBoard: tuple[int, int]):
    scoreArray = Scores[indexOfBoard[0]][indexOfBoard[1]]
    board = Boards[indexOfBoard[0]][indexOfBoard[1]]

    scoreArray[0] = board.score()
    scoreArray[1] = board.getMergeCount()
    scoreArray[2] = board.getNumMoves()


def getTiles(x: int, y: int, indexOfBoard: tuple[int, int]) -> float:
    num = Boards[indexOfBoard[0]][indexOfBoard[1]].pos_to_value((x, y))

    # Normalize inputs
    if num == 0:
        return -1

    return math.log2(num) / 16

    # return int(num)


def getInputs(indexOfBoard: tuple[int, int]):
    inps = []
    for dy in range(BoardSize):
        for dx in range(BoardSize):
            inps.append(getTiles(dx, dy, indexOfBoard))

    inps.append(1.0)

    return inps


def sigmoid(x) -> float:
    return 1 / (1 + math.exp(-x))
    # return 2 / (1 + math.exp(-3 * x)) -1
    # return 1 / (1 + math.exp(-5.9 * x))


def newInnovation():
    pool["innovation"] += 1
    return pool["innovation"]


def newPool():
    pool = {}
    pool['species'] = list()
    pool["generation"] = 0
    pool["innovation"] = Outputs
    pool['currentFrame'] = 0
    pool["maxFitness"] = 0
    pool['topScore'] = 0

    return pool


def newSpecies():
    species = {}
    species["topFitness"] = 0
    species["staleness"] = 0
    species['genomes'] = []
    species["averageFitness"] = 0

    return species


def newGenome():
    genome = {}
    genome["genes"] = []
    genome["fitness"] = 0
    genome["adjustedFitness"] = 0
    genome['network'] = {}
    genome['maxneuron'] = 0
    genome["globalRank"] = 0
    genome["mutationRates"] = {}
    genome["mutationRates"]["connections"] = MutateConnectionsChance
    genome["mutationRates"]["link"] = LinkMutationChance
    genome["mutationRates"]["bias"] = BiasMutationChance
    genome["mutationRates"]["node"] = NodeMutationChance
    genome["mutationRates"]["enable"] = EnableMutationChance
    genome["mutationRates"]["disable"] = DisableMutationChance
    genome["mutationRates"]['step'] = StepSize

    return genome


def copyGenome(genome):
    genome2 = newGenome()

    for g in genome["genes"]:
        genome2['genes'].append(copyGene(g))

    # copyNetwork(genome, genome2)
    genome2['fitness'] = 0

    genome2['maxneuron'] = genome['maxneuron']
    genome2["mutationRates"]["connections"] = genome["mutationRates"][
        "connections"]
    genome2["mutationRates"]["link"] = genome["mutationRates"]["link"]
    genome2["mutationRates"]["bias"] = genome["mutationRates"]["bias"]
    genome2["mutationRates"]["node"] = genome["mutationRates"]["node"]
    genome2["mutationRates"]["enable"] = genome["mutationRates"]["enable"]
    genome2["mutationRates"]["disable"] = genome["mutationRates"]["disable"]

    return genome2


def basicGenome():
    genome = newGenome()
    innovation = 1
    genome['maxneuron'] = Inputs

    for o in range(Outputs):
        newLink = newGene()
        newLink['into'] = random.randint(0, Inputs - 1)
        newLink['out'] = MaxNodes + o
        newLink["weight"] = random.uniform(-2, 2)
        newLink["innovation"] = newInnovation()
        genome['genes'].append(newLink)

    mutate(genome)

    return genome


def newGene():
    gene = {}
    gene["into"] = 0
    gene["out"] = 0
    gene["weight"] = random.uniform(-2, 2)
    gene['enabled'] = True
    gene["innovation"] = 0

    return gene


def copyGene(gene):
    gene2 = newGene()
    gene2["into"] = gene["into"]
    gene2["out"] = gene["out"]
    gene2["weight"] = gene["weight"]
    gene2['enabled'] = gene['enabled']
    gene2["innovation"] = gene["innovation"]

    return gene2


def newNeuron():
    neuron = {}
    neuron['incoming'] = []
    neuron["value"] = 0.0
    return neuron


def generateNetwork(genome):
    network = {}
    network['neurons'] = {}

    for i in range(Inputs):
        network['neurons'][i] = newNeuron()

    for j in range(Outputs):
        network['neurons'][MaxNodes + j] = newNeuron()

    genome['genes'] = sorted(genome['genes'], key=lambda d: d['out'])

    for k in range(len(genome["genes"])):
        gene = genome['genes'][k]

        if gene['enabled']:
            if gene['out'] not in network['neurons']:
                network['neurons'][gene['out']] = newNeuron()

            neuron = network['neurons'][gene['out']]
            neuron['incoming'].append(gene)

            if gene['into'] not in network['neurons']:
                network['neurons'][gene['into']] = newNeuron()

    genome['network'] = network


def evaluateNetwork(network, inputs):
    # inputs.append(1)

    # if len(inputs) != Inputs:
    #     print("Incorrect number of neural network inputs.")
    #     return {}

    for i in range(Inputs):
        network['neurons'][i]["value"] = inputs[i]

    for _, neuron in network['neurons'].items():
        val = 0
        for j in range(len(neuron['incoming'])):
            incoming = neuron['incoming'][j]
            other = network['neurons'][incoming['into']]
            val += (incoming["weight"] * other["value"])

        if len(neuron['incoming']) > 0:
            neuron["value"] = sigmoid(val)

    outputs = {"Up": False, "Down": False, "Left": False, "Right": False}
    # most = [(0, -1)]
    # totalSum = 0
    # for o in range(Outputs):
    #     neuron_val = network['neurons'][MaxNodes + o]["value"]
    #     totalSum += neuron_val
    #     if 0.05 < neuron_val - most[0][0]:
    #         most = [(neuron_val, o)]
    #     elif -0.05 < (neuron_val - most[0][0]) < 0.05:
    #         most.append((neuron_val, o))

    # probs = []
    # for o in range(Outputs):
    #     val = ((network['neurons'][MaxNodes + o]["value"] / totalSum), o)
    #     for i in range(math.floor((val[0] / totalSum) * 100)):
    #         probs.append(val)
    #
    # choice = random.choice(probs)
    # outputs[ButtonNames[choice[1]]] = True

    # move_confidence = most[0][0]
    # most_confident_move = most[0][1]
    #
    move_confidence = 0
    move_index = 0
    for o in range(Outputs):
        neuron_val = network['neurons'][MaxNodes + o]["value"]

        if move_confidence < neuron_val:
            move_confidence = neuron_val
            move_index = o

    if move_confidence < ConfidenceThreshold:
        return outputs
    #
    # if len(most) > 2:
    #     choice = random.choice(most)
    #     move_confidence = choice[0]
    #     most_confident_move = choice[1]
    #
    outputs[ButtonNames[move_index]] = True

    return outputs


# look at function below
def crossover(g1, g2):
    if g2['fitness'] > g1['fitness']:
        temp = g1
        g1 = g2
        g2 = temp

    child = newGenome()

    innovations2 = {}
    for i in range(len(g2['genes'])):
        gene = g2['genes'][i]
        innovations2[gene["innovation"]] = gene

    for i in range(len(g1['genes'])):
        gene1 = g1['genes'][i]

        if (gene1["innovation"] in innovations2 and random.randint(1, 2) == 1
                and innovations2[gene1["innovation"]]['enabled']):
            child['genes'].append(copyGene(innovations2[gene1["innovation"]]))
        else:
            child['genes'].append(copyGene(gene1))

    child['maxneuron'] = max(g1['maxneuron'], g2['maxneuron'])

    for mutation, rate in g1['mutationRates'].items():
        child['mutationRates'][mutation] = rate

    return child


# look at function below
def randomNeuron(genes, nonInput):
    neurons = {}

    if not nonInput:
        for i in range(Inputs):
            neurons[i] = True

    for o in range(Outputs):
        neurons[MaxNodes + o] = True

    for k in range(len(genes)):
        if (not nonInput) or (genes[k]['into'] > Inputs):
            neurons[genes[k]['into']] = True

        if (not nonInput) or (genes[k]['out'] > Inputs):
            neurons[genes[k]['out']] = True

    count = 0

    for _ in neurons:
        count += 1

    n = random.randint(1, count)

    for i in neurons:
        n -= 1
        if n == 0:
            return i

    return 0


def containsLink(genes, link) -> bool:
    for i in range(len(genes)):
        gene = genes[i]
        if gene['into'] == link['into'] and gene['out'] == link['out']:
            return True
    return False


def pointMutate(genome):
    step = genome["mutationRates"]['step']
    for gene in genome['genes']:
        if random.random() < PerturbChance:
            gene["weight"] += (random.random() * step)
        else:
            gene["weight"] = random.random() * 4 - 2


def linkMutate(genome, forceBias):
    neuron1 = randomNeuron(genome['genes'], False)
    neuron2 = randomNeuron(genome['genes'], True)

    while neuron1 < Inputs and neuron2 < Inputs:
        # Both input nodes
        return None

    # neuron1 = randomNeuron(genome['genes'], False)
    # neuron2 = randomNeuron(genome['genes'], True)

    newLink = newGene()

    if neuron2 < Inputs:
        temp = neuron2
        neuron2 = neuron1
        neuron1 = temp

    # if neuron1 > neuron2:
    #     # Swap output and input
    #     temp = neuron1
    #     neuron1 = neuron2
    #     neuron2 = temp

    newLink['into'] = neuron1
    newLink['out'] = neuron2

    if forceBias:
        newLink['into'] = Inputs

    # print("\n")
    # print("Link Mutate")

    if containsLink(genome['genes'], newLink):
        # print("Skipping")
        return

    # print("Not Skipping")

    newLink["innovation"] = newInnovation()
    newLink["weight"] = ((random.random() * 4) - 2)

    genome['genes'].append(newLink)
    return None


def nodeMutate(genome):
    if len(genome['genes']) == 0:
        return

    genome['maxneuron'] += 1

    gene = genome['genes'][random.randint(0, len(genome['genes']) - 1)]

    if not gene['enabled']:
        return

    gene['enabled'] = False

    gene1 = copyGene(gene)
    gene1['out'] = genome['maxneuron']
    gene1["weight"] = 1.0
    gene1["innovation"] = newInnovation()
    gene1['enabled'] = True
    genome['genes'].append(gene1)

    gene2 = copyGene(gene)
    gene2['into'] = genome['maxneuron']
    gene2["innovation"] = newInnovation()
    gene2['enabled'] = True
    genome['genes'].append(gene2)


def enableDisableMutate(ge, enable: bool):
    candidate = []

    for gene in ge['genes']:
        if gene['enabled'] == (not enable):
            candidate.append(gene)

    if len(candidate) == 0:
        return

    gene = candidate[random.randint(0, len(candidate) - 1)]
    gene['enabled'] = not gene['enabled']


def mutate(genome):

    for mutation, rate in genome["mutationRates"].items():
        # genome["mutationRates"][mutation] *= adjustment
        if random.randint(1, 2) == 1:
            genome["mutationRates"][mutation] = .95 * rate
        else:
            genome["mutationRates"][mutation] = 1.05263 * rate

    if random.random() < genome["mutationRates"]['connections']:
        pointMutate(genome)

    p = genome["mutationRates"]["link"]
    while p > 0:
        if random.random() < p:
            linkMutate(genome, False)
        p -= 1

    p = genome["mutationRates"]["bias"]
    while p > 0:
        if random.random() < p:
            linkMutate(genome, True)
        p -= 1

    p = genome["mutationRates"]["node"]
    while p > 0:
        if random.random() < p:
            nodeMutate(genome)
        p -= 1

    p = genome["mutationRates"]["enable"]
    while p > 0:
        if random.random() < p:
            enableDisableMutate(genome, True)
        p -= 1

    p = genome["mutationRates"]["disable"]

    while p > 0:
        if random.random() < p:
            enableDisableMutate(genome, False)
        p -= 1


def disjoint(genes1: list, genes2: list):
    i1 = dict()

    for g in genes1:
        i1[g["innovation"]] = True

    i2 = dict()
    for g in genes2:
        i2[g["innovation"]] = True

    disjoint_count = 0
    for g in genes1:
        if g["innovation"] not in i2.keys():
            disjoint_count += 1
    for g in genes2:
        if g["innovation"] not in i1.keys():
            disjoint_count += 1

    n = max(len(genes1), len(genes2))
    return disjoint_count / n
    #
    # i1 = {}
    # for i in range(len(genes1)):
    #     gene = genes1[i]
    #     i1[gene["innovation"]] = True
    #
    # i2 = {}
    # for i in range(len(genes2)):
    #     gene = genes2[i]
    #     i2[gene["innovation"]] = True
    #
    # disjointGenes = 0
    #
    # for i in range(len(genes1)):
    #     gene = genes1[i]
    #     if gene["innovation"] > len(i2):
    #         disjointGenes += 1
    #
    # for i in range(len(genes2)):
    #     gene = genes2[i]
    #     if gene["innovation"] >= len(i1):
    #         disjointGenes += 1
    #
    # n = max(len(genes1), len(genes2))
    #
    #
    # return disjointGenes / n

    #
    # innov1 = {g["innovation"] for g in genes1}
    # innov2 = {g["innovation"] for g in genes2}
    #
    # disjoint_genes = len(innov1 - innov2) + len(innov2 - innov1)
    # return disjoint_genes / max(len(genes1), len(genes2))


def weight(genes1: list, genes2: list):
    i2 = {}

    for i in range(len(genes2)):
        gene = genes2[i]
        i2[gene["innovation"]] = gene

    val = 0
    coincident = 0
    for i in range(len(genes1)):
        gene = genes1[i]
        if gene["innovation"] in i2:
            gene2 = i2[gene["innovation"]]
            val += abs(gene["weight"] - gene2["weight"])
            coincident += 1

    # This shouldn't be here
    return float('inf') if coincident == 0 else val / coincident
    return val / coincident


def sameSpecies(genome1, genome2):
    dd = DeltaDisjoint * disjoint(genome1['genes'], genome2['genes'])
    dw = DeltaWeights * weight(genome1['genes'], genome2['genes'])

    return (dd + dw) < DeltaThreshold


def adjustDeltaThreshold():
    global DeltaThreshold
    global pool
    if len(pool['species']) > TargetSpeciesCount:
        DeltaThreshold += DeltaThresholdAdjustmentStep
        DeltaThreshold = min(MaxDeltaThreshold, DeltaThreshold)
    elif len(pool['species']) < TargetSpeciesCount:
        DeltaThreshold -= DeltaThresholdAdjustmentStep
        DeltaThreshold = max(MinDeltaThreshold, DeltaThreshold)


def rankGlobally():
    globe = []
    for s in range(len(pool['species'])):
        species = pool['species'][s]
        for g in range(len(species['genomes'])):
            globe.append(species['genomes'][g])

    globe = sorted(globe, key=lambda d: d['fitness'])

    for g in range(len(globe)):
        globe[g]['globalRank'] = g


def calculateAverageFitness(species):
    tots = 0

    for g in range(len(species['genomes'])):
        genome = species['genomes'][g]
        # tots += genome['globalRank']
        tots += genome['fitness']

    species['averageFitness'] = tots / len(species['genomes'])


def totalAverageFitness():
    total = 0
    for s in range(len(pool['species'])):
        species = pool['species'][s]
        total += species['averageFitness']

    return total


def cullSpecies(cutToOne):
    for s in range(len(pool['species'])):
        species = pool['species'][s]

        species['genomes'] = sorted(species['genomes'],
                                    key=lambda d: d['fitness'], reverse=True)

        # temp = []
        # for i in range(len(species['genomes'])):
        #     temp.append((species['genomes'][i]['fitness'], i))
        #
        # temp.sort()
        # boob = []
        # for i in temp:
        #     boob.append(species['genomes'][i[1]])
        #
        # species['genomes'] = boob


        remaining = math.ceil(len(species['genomes']) / 2)

        if cutToOne:
            remaining = math.ceil(len(species['genomes']) * .3)

        while len(species['genomes']) > remaining:
            species['genomes'].pop()


def breedChild(species):
    if random.random() < CrossoverChance:
        g1 = species['genomes'][random.randint(0, len(species['genomes']) - 1)]
        g2 = species['genomes'][random.randint(0, len(species['genomes']) - 1)]
        child = crossover(g1, g2)
    else:
        g = species['genomes'][random.randint(0, len(species['genomes']) - 1)]
        child = copyGenome(g)

    mutate(child)

    return child


def removeStaleSpecies():
    survived = []

    for s in range(len(pool['species'])):
        species = pool['species'][s]

        species['genomes'] = sorted(species['genomes'],
                                    key=lambda d: d['fitness'], reverse=True)

        if species['genomes'][0]['fitness'] > species['topFitness']:
            species['topFitness'] = species['genomes'][0]['fitness']
            species['staleness'] = 0
        else:
            species['staleness'] += 1

        if (species['staleness'] < StaleSpecies or species['topFitness'] >=
                pool['maxFitness']):
            survived.append(species)

    pool['species'] = survived


def removeWeakSpecies():
    survived = []

    total_sum_fitness = totalAverageFitness()

    for s in range(len(pool['species'])):
        species = pool['species'][s]
        # breed = (species['averageFitness'] /() total_sum_fitness)
        breed = math.floor(
            (species['averageFitness'] / total_sum_fitness) * Population)
        if breed >= 1:
            survived.append(species)

    pool['species'] = survived

    # pool['species'] = sorted(pool['species'],
    # key=lambda d:  d['averageFitness'], reverse=True)

    # pool['species'] = pool['species'][ 0 : math.ceil(
    # (len(pool['species'])-1)/2)]

    # pool['species'] = pool['species'][ 0 : random.randint(
    # math.ceil((len(pool['species'])-1)/2), (len(pool['species'])-1) )]


def startExtinctionEvent():
    pool['species'] = sorted(pool['species'], key=lambda d: d['averageFitness'],
                             reverse=True)
    pool['species'] = pool['species'][
        0: random.randint(math.ceil((len(pool['species']) - 1) / 10),
                          math.ceil(7 * (len(pool['species']) - 1) / 10))]


def addToSpecies(child):
    foundSpecies = False

    for s in range(len(pool['species'])):
        species = pool['species'][s]
        for g in species['genomes']:
            if (not foundSpecies and sameSpecies(child, g)):
                species['genomes'].append(child)
                foundSpecies = True

    if not foundSpecies:
        childSpecies = newSpecies()
        childSpecies['genomes'].append(child)
        pool['species'].append(childSpecies)


def newGeneration():
    global TopScoresOfGeneration
    global FitnessOfGeneration
    global LargestTileOfGeneration

    Generations.append(pool['generation'] + 1)

    if DrawProgressGraph:
        BestTopScores.append(max(TopScoresOfGeneration))
        updateProgressGraph(Generations, BestTopScores, Population)

    AverageFitness.append(sum(FitnessOfGeneration)/len(FitnessOfGeneration))

    if DrawAverageFitness:
        updateAverageFitnessGraph(Generations, AverageFitness, Population)

    LargestTilesOverGenerations.append(max(LargestTileOfGeneration))

    if DrawLargestTileGraph:
        viz.updateLargestTileGraph(Generations,
                                   LargestTilesOverGenerations, Population)
    if DrawGenomeNeuralNetwork:
        bestFitness = {'fitness': float('-inf')}
        s = 0
        g = 0
        for currentSpecies in range(len(pool['species'])):
            for currentGenome in range(len(pool['species'][currentSpecies]['genomes'])):
                if bestFitness['fitness'] < pool['species'][currentSpecies]['genomes'][currentGenome]['fitness']:
                    bestFitness = pool['species'][currentSpecies]['genomes'][currentGenome]
                    s = currentSpecies
                    g = currentGenome

        plt = drawGenomeGraph(
            bestFitness,
            s, g, Generations[-1])
        plt.close()

    TopScoresOfGeneration = []
    FitnessOfGeneration = []
    LargestTileOfGeneration = []

    cullSpecies(False)  # Cull the bottom half of the species
    rankGlobally()
    removeStaleSpecies()
    rankGlobally()

    for s in range(len(pool['species'])):
        species = pool['species'][s]
        calculateAverageFitness(species)

    removeWeakSpecies()

    # if random.randint(0, 10000) <= 1:
    #     startExtinctionEvent()

    total_sum_fitness = totalAverageFitness()
    children = []

    for s in range(len(pool['species'])):
        species = pool['species'][s]

        breed = math.floor(
            (species['averageFitness'] / total_sum_fitness) * Population)
        for i in range(breed):
            children.append(breedChild(species))

    cullSpecies(True)  # Cull all but the top member of the each species

    while (len(children) + len(pool['species'])) < Population:
        species = pool['species'][random.randint(0, len(pool['species']) - 1)]

        children.append(breedChild(species))

    for child in children:
        addToSpecies(child)

    pool['generation'] += 1
    adjustDeltaThreshold()
    global MAX_TIME
    # if pool['generation'] < 20:
    #     MAX_TIME *= .9 # sets lower bound for MAX_Time

    # global DeltaThreshold
    # if pool['generation'] == 20:
    #     DeltaThreshold *= 1.4
    # elif pool['generation'] == 100:
    #     DeltaThreshold *= 1.8

    # text = "backup." + str(pool['generation']) + "."


def initializePool():
    global pool
    pool = newPool()
    for i in range(Population):
        basic = basicGenome()
        addToSpecies(basic)
        # new_species = newSpecies()
        # new_species['genomes'].append(basic)
        # pool['species'].append(new_species)


def initializeRun(currentSpecies, currentGenome):
    # Savestate.load(Filename);
    # <-- Probobly dont need this. Might be a rest method

    species = pool['species'][currentSpecies]
    genome = species['genomes'][currentGenome]

    generateNetwork(genome)


def evaluateGenome(currentSpecies, currentGenome,
                   indexOfBoard: tuple[int, int]):
    global pool

    initializeRun(currentSpecies, currentGenome)

    Boards[indexOfBoard[0]][indexOfBoard[1]].reset()

    species = pool['species'][currentSpecies]
    genome = species['genomes'][currentGenome]

    start = time.process_time()
    end = time.process_time()

    board = Boards[indexOfBoard[0]][indexOfBoard[1]]

    while (not board.is_game_over() and end - start <
           MaxTimeBetweenEachMove):

        global inputs

        inputs = getInputs(indexOfBoard)
        old_numMoves = board.getNumMoves()

        controller = evaluateNetwork(genome['network'], inputs)
        board.input_movements(controller)

        if old_numMoves >= board.getNumMoves():
            end = time.process_time()

    board.end()

    fitness = calculateFitness(indexOfBoard)

    if fitness <= 4:
        fitness = -1

    genome['fitness'] = fitness

    # pool['maxFitness'] = max(pool['maxFitness'], fitness)
    #
    # pool['topScore'] = max(new_score, pool['topScore'])

    new_score = Scores[indexOfBoard[0]][indexOfBoard[1]][0]

    string_to_print = ("\nCurrent Generation: " +
                       str(pool['generation']) +
                       "\nCurrent Species: " +
                       str(currentSpecies + 1) +
                       ", Out of " + str(len(pool['species'])) +
                       "\nCurrent Genome: " + str(currentGenome + 1) +
                       ", Out of: " + str(len(species['genomes'])) +
                       "\nNeuron Size: " +
                       str(len(species['genomes'][currentGenome]['genes'])) +
                       "\nCurrent Score: " + str(new_score) +
                       "\nCurrent Fitness: " + str(genome['fitness']) + "\n")

    print(string_to_print)


def nextGenome(currentSpecies: int, currentGenome: int) -> tuple[int, int]:
    currentGenome += 1

    if currentGenome >= len(
            pool['species'][currentSpecies]['genomes']):

        currentGenome = -1

        if currentSpecies >= len(pool['species']):
            # newGeneration()
            currentSpecies = 0

    return (currentSpecies, currentGenome)


def fitnessAlreadyMeasured(currentSpecies, currentGenome):
    species = pool['species'][currentSpecies]
    genome = species['genomes'][currentGenome]

    return genome['fitness'] != 0


def playTop(currentSpecies, currentGenome):
    pass
    # maxfitness = 0
    # maxs = 0
    # maxg = 0
    #
    # for s, species in pool['species']:
    #     for g, genome in species['genomes']:
    #         if genome['fitness'] > maxfitness:
    #             maxfitness = genome['fitness']
    #             maxs = s
    #             maxg = g
    # pool[currentSpecies] = maxs
    # pool[currentGenome] = maxg
    # pool['maxFitness'] = maxfitness
    # pool['currentFrame'] += 1

# def writeFile(filename):
#     file = open(filename, "w")
#     file.write(str(pool['generation']) + "\n")
#     file.write(str(pool['maxFitness']) + "\n")
#     file.write(str(pool['species']) + "\n")
#
#     for n in pool['species']:
#         file.write(str(n['topFitness']) + "\n")
#         file.write(str(n['staleness']) + "\n")
#         file.write(str(n['genomes']) + "\n")
#         species = n
#         for m in species['genomes']:
#             file.write(str(m['fitness']) + "\n")
#             file.write(str(m['maxneuron']) + "\n")
#
#     file.close()


# random_pad()

def calculateFitness(indexOfBoard: tuple[int, int]) -> float:
    getScore(indexOfBoard)

    scoreArray = Scores[indexOfBoard[0]][indexOfBoard[1]]

    new_score = scoreArray[0]
    numMerge = scoreArray[1]
    numMoves = scoreArray[2]

    TopScoresOfGeneration.append(new_score)


    largest_tile = Boards[indexOfBoard[0]][indexOfBoard[1]].getLargestTile()
    LargestTileOfGeneration.append(largest_tile)
    tile_bonus = {
        8: 3,
        16: 8,
        32: 20,
        64: 50,
        128: 280,
        256: 800,
        512: 3200,
        1024: 12800,
        2048: 50000
    }

    if numMoves == 0:
        return -16
        #
    fitness = new_score**2 + numMerge - numMoves
    # fitness = new_score
    # fitness = new_score ** 1.2 + ((largest_tile**1.2) * numMerge / numMoves)
    # for tile, bonus in tile_bonus.items():
    #     if largest_tile >= tile:
    #         fitness += bonus
    # fitness = new_score ** 2 + numMerge

    tiles = getInputs(indexOfBoard)
    empty = 0

    for num in tiles:
        for tile, bonus in tile_bonus.items():
            if num >= tile:
                fitness += bonus

        if num == -1:
            empty += 1

    fitness -= (empty * 3)

    # return new_score + numMerge * 25 + (Board[index].getLargestTile()*1.8)
    FitnessOfGeneration.append(fitness)
    return fitness


def evaluateParentThread(currentSpecies: int, currentGenome: int,
                         indexOfThread: int):
    global pool
    childThreads = THREADS[indexOfThread + 1]

    while currentGenome > -1:

        indexOfChildThread = 0

        while (currentGenome > -1 and indexOfChildThread < len(childThreads)):
            childThreads[indexOfChildThread] = threading.Thread(
                target=evaluateGenome, args=(currentSpecies, currentGenome,
                                             (indexOfThread // 2,
                                              indexOfChildThread),))

            childThreads[indexOfChildThread].start()
            currentSpecies, currentGenome = nextGenome(currentSpecies,
                                                       currentGenome)
            indexOfChildThread += 1

        joinChildThreads(childThreads)

    joinChildThreads(childThreads)


def joinChildThreads(childThreads: list):
    for i in range(len(childThreads)):
        if childThreads[i].is_alive():
            childThreads[i].join()


def joinParentThreads():
    for i in range(0, len(THREADS), 2):
        if THREADS[i].is_alive():
            THREADS[i].join()


initializePool()
initializeBoards()

if DrawProgressGraph:
    drawProgressGraph([], [], Population)

if DrawAverageFitness:
    drawAverageFitnessGraph([], [], Population)

if DrawLargestTileGraph:
    viz.drawLargestTileGraph([], [], population=Population)

while True:
    global pool

    species = 0
    genome = 0

    while species < len(pool['species']):
        indexOfParentThread = 0

        while (indexOfParentThread < NumParentThreads
               and species < len(pool['species'])):
            THREADS[indexOfParentThread] = (
                threading.Thread(target=evaluateParentThread,
                                 args=(species, 0, indexOfParentThread,)))

            THREADS[indexOfParentThread].start()

            indexOfParentThread += 2

            species += 1

        joinParentThreads()

    joinParentThreads()

    pool['topScore'] = max(max(TopScoresOfGeneration), pool['topScore'])
    pool['maxFitness'] = max(max(FitnessOfGeneration), pool['maxFitness'])

    print("\nTop Score: " + str(pool['topScore']) + "\nMax Fitness: " +
          str(pool['maxFitness']) + "\n")

    newGeneration()
