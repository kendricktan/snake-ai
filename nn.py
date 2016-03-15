from pygame.locals import *
import pygame, constants, math, random, pickle

def savePool(filename):
    with open(filename, 'wb') as output:
        pickle.dump(constants.pool, output, pickle.HIGHEST_PROTOCOL)

def loadPool(filename):
    with open(filename, 'rb') as input:
        constants.pool = pickle.load(input)

def sigmoid(x):
    try:
        return 2/(1+math.exp(-4.9*x))-1
    except OverflowError:
        return 1 / (1 + math.exp(-x))

class Pool:
    def __init__(self):
        self.newPool()

    def newPool(self):
        self.species = []
        #self.species.append(Species())
        self.generation = 0
        self.innovation = constants.Outputs
        self.currentSpecies = 0
        self.currentGenome = 0
        self.maxFitness = 0

class Species:
    def __init__(self):
        self.newSpecies()

    def newSpecies(self):
        self.topFitness = 0
        self.staleness = 0
        self.genomes = []
        self.genomes.append(Genome())
        self.averageFitness = 0

class Genome:
    def __init__(self):
        self.newGenome()

    def newGenome(self):
        self.genes = []
        self.genes.append(Genes())
        self.fitness = 0
        self.adjustedFitness = 0
        self.network = None
        self.maxneuron = 0
        self.globalRank = 0
        self.mutationRates = {}
        self.mutationRates['connections'] = constants.MutateConnectionsChance
        self.mutationRates['link'] = constants.LinkMutationChance
        self.mutationRates['bias'] = constants.BiasMutationChance
        self.mutationRates['node'] = constants.NodeMutationChance
        self.mutationRates['enable'] = constants.EnableMutationChance
        self.mutationRates['disable'] = constants.DisableMutationChance
        self.mutationRates['step'] = constants.StepSize

class Genes:
    def __init__(self):
        self.newGene()

    def newGene(self):
        self.into = 0
        self.out = random.randint(constants.Inputs+2, constants.MaxNodes-1)
        self.weight = 0.0
        self.enabled = True
        self.innovation = 0

class Network:
    def __init__(self):
        self.neurons = {}

class Neuron:
    def __init__(self):
        self.newNeuron()

    def newNeuron(self):
        self.incoming = []
        self.value = 0.0

class Cell:
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value

def copyGenome(genome):
    genome_copy = Genome()
    genome_copy.genes.pop()
    for gene in genome.genes:
        genome_copy.genes.append(gene)
    genome_copy.maxneuron = genome.maxneuron
    genome_copy.mutationRates = genome.mutationRates
    return genome_copy

def copyGene(gene):
    gene2 = Genes()
    gene2.into = gene.into
    gene2.out = gene.out
    gene2.weight = gene.weight
    gene2.enabled = gene.enabled
    gene2.innovation = gene.innovation
    return gene2

def basicGenome():
    genome = Genome()
    genome.maxneuron = constants.Inputs

    mutate(genome)
    return genome

def generateNetwork(genome):
    network = Network()

    for i in range(0, constants.Inputs+1):
        network.neurons[i] = Neuron()

    for i in range(0, constants.Outputs):
        network.neurons[constants.MaxNodes+i] = Neuron()

    genome.genes = sorted(genome.genes, key=lambda gene: gene.out)

    for gene in genome.genes:
        if gene.enabled:
            if gene.out not in network.neurons:
                network.neurons[gene.out] = Neuron()
            neuron = network.neurons[gene.out]
            neuron.incoming.append(gene)
            if gene.into not in network.neurons:
                network.neurons[gene.into] = Neuron()

    genome.network = network

def evaluateNetwork(network, inputs):
    inputs.append(1)
    if len(inputs) != constants.Inputs+1:
        print("Incorrect number of neural network inputs")
        return

    for i in range(0, len(inputs)):
        network.neurons[i].value = inputs[i]

    for key in network.neurons:
        neuron = network.neurons[key]
        sum = 0
        for j in range(0, len(neuron.incoming)):
            incoming = neuron.incoming[j]
            sum += incoming.weight

            if incoming.into in network.neurons:
                other = network.neurons[incoming.into]
                sum += other.value

        if len(neuron.incoming) > 0:
            neuron.value = sigmoid(round(sum, 8))

    outputs = {}
    for i in range(0, constants.Outputs):
        direction = constants.Output_Names[i]
        if network.neurons[constants.MaxNodes+i].value > 0:
            outputs[direction] = True
        else:
            outputs[direction] = False

    return outputs

def crossover(g1, g2):
    if g2.fitness > g1.fitness:
        tempg = g1
        g1 = g2
        g2 = tempg

    child = Genome()

    innovations2 = {}
    for gene in g2.genes:
        innovations2[gene.innovation] = gene

    for gene1 in g1.genes:
        if gene1.innovation in innovations2 and random.randint(1, 2) == 1:
            gene2 = innovations2[gene1.innovation]
            if gene2.enabled:
                child.genes.append(copyGene(gene2))
        else:
            child.genes.append(copyGene(gene1))

    child.maxneuron = max(g1.maxneuron, g2.maxneuron)

    for key in g1.mutationRates:
        child.mutationRates[key] = g1.mutationRates[key]

    return child

def randomNeuron(genes, nonInput):
    neurons = {}
    if not nonInput:
        for i in range(0, constants.Inputs):
            neurons[i] = True

    for i in range(0, constants.Outputs):
        neurons[constants.MaxNodes+i] = True

    for gene in genes:
        if (not nonInput) or gene.into > constants.Inputs:
            neurons[gene.into] = True
        if (not nonInput) or gene.out > constants.Inputs:
            neurons[gene.out] = True

    count = 0
    for key in neurons:
        count += 1

    n = random.randint(0, count)

    for key in neurons:
        n -= 1
        if n == 0:
            return key

    return 0

def containsLink(genes, link):
    for gene in genes:
        if gene.into == link.into and gene.out == link.out:
            return True
    return False

def pointMutate(genome):
    step = genome.mutationRates['step']

    for gene in genome.genes:
        if random.random() < constants.PerturbChance:
            gene.weight = gene.weight + random.random() * step * 2 - step
        else:
            gene.weight = random.random()*4-2

def linkMutate(genome, forceBias):
    neuron1 = randomNeuron(genome.genes, False)
    neuron2 = randomNeuron(genome.genes, True)

    newLink = Genes()

    if neuron1 < constants.Inputs and neuron2 < constants.Inputs:
        return

    if neuron2 < constants.Inputs:
        temp = neuron1
        neuron1 = neuron2
        neuron2 = temp

    newLink.into = neuron1
    newLink.out = neuron2

    if forceBias:
        newLink.into = constants.Inputs

    if containsLink(genome.genes, newLink):
        return

    newLink.innovation = newInnovation()
    newLink.weight = random.random()*4-2

    genome.genes.append(newLink)

def nodeMutate(genome):
    if len(genome.genes) == 0:
        return

    genome.maxneuron += 1

    gene = genome.genes[random.randint(0, len(genome.genes)-1)]

    if not gene.enabled:
        return

    gene.enabled = False

    gene1 = copyGene(gene)
    gene1.out = genome.maxneuron
    gene1.weight = 1.0
    gene1.innovation = newInnovation()
    gene1.enabled = True
    genome.genes.append(gene1)

    gene2 = copyGene(gene)
    gene2.into = genome.maxneuron
    gene2.innovation = newInnovation()
    gene2.enabled = True
    genome.genes.append(gene2)

def enableDisableMutate(genome, enable):
    candidates = []

    for gene in genome.genes:
        if gene.enabled is not enable:
            candidates.append(gene)

    if len(candidates) == 0:
        return

    gene = candidates[random.randint(0, len(candidates)-1)]
    gene.enabled = not gene.enabled

def mutate(genome):
    for key in genome.mutationRates:
        rate = genome.mutationRates[key]
        if random.randint(1, 2) == 1:
            genome.mutationRates[key] = 0.95*rate
        else:
            genome.mutationRates[key] = 1.05263*rate

    if random.random() < genome.mutationRates['connections']:
        pointMutate(genome)

    p = genome.mutationRates['link']
    while p > 0:
        if random.random() < p:
            linkMutate(genome, False)
        p -= 1

    p = genome.mutationRates['bias']
    while p > 0:
        if random.random() < p:
            linkMutate(genome, True)
        p -= 1

    p = genome.mutationRates['node']
    while p > 0:
        if random.random() < p:
            nodeMutate(genome)
        p -= 1

    p = genome.mutationRates['enable']
    while p > 0:
        if random.random() < p:
            enableDisableMutate(genome, True)
        p -= 1

    p = genome.mutationRates['disable']
    while p > 0:
        if random.random() < p:
            enableDisableMutate(genome, False)
        p -= 1

def disjoint(genes1, genes2):
    i1 = {}
    for gene in genes1:
        i1[gene.innovation] = True

    i2 = {}
    for gene in genes2:
        i2[gene.innovation] = True

    disjointGenes = 0
    for gene in genes1:
        if gene.innovation not in i2:
            disjointGenes += 1

    for gene in genes2:
        if gene.innovation not in i1:
            disjointGenes += 1

    n = max(len(genes1), len(genes2))

    try:
        return disjointGenes/n
    except ZeroDivisionError:
        return float('nan')

def weight(genes1, genes2):
    i2 = {}
    for gene in genes2:
        i2[gene.innovation] = gene

    sum = 0
    coincident = 0

    for gene in genes1:
        if gene.innovation in i2:
            gene2 = i2[gene.innovation]
            sum = sum + abs(gene.weight - gene2.weight)
            coincident += 1
    try:
        return sum/coincident
    except ZeroDivisionError:
        return float('nan')

def rankGlobally():
    _global = []

    for species in constants.pool.species:
        for genome in species.genomes:
            _global.append(genome)

    _global = sorted(_global, key=lambda g: g.fitness)

    for i in range(0, len(_global)):
        _global[i].globalRank = i

def calculateAverageFitness(species):
    total = 0

    for genome in species.genomes:
        total += genome.globalRank

    species.averageFitness = total/len(species.genomes)

def totalAverageFitness():
    total = 0

    for species in constants.pool.species:
        total += species.averageFitness

    return total

def cullSpecies(cutToOne):
    for species in constants.pool.species:
        species.genomes = sorted(species.genomes, key=lambda g: g.fitness, reverse=True)

        remaining = int(math.ceil(len(species.genomes)/2))

        if cutToOne:
            remaining = 1

        while len(species.genomes) > remaining:
            species.genomes.pop()

def removeStaleSpecies():
    survived = []

    for species in constants.pool.species:
        species.genomes = sorted(species.genomes, key=lambda g: g.fitness, reverse=True)

        if species.genomes[0].fitness > species.topFitness:
            species.topFitness = species.genomes[0].fitness
            species.staleness = 0
        else:
            species.staleness = species.staleness + 1

        if species.staleness < constants.StaleSpecies or species.topFitness >= constants.pool.maxFitness:
            survived.append(species)

    if len(survived) > 0:
        constants.pool.species = survived
    else:
        constants.pool.species.pop()

def removeWeakSpecies():
    survived = []
    sum = totalAverageFitness()
    for species in constants.pool.species:
        breed = math.floor(species.averageFitness/sum*constants.Population)
        if breed >= 1:
            survived.append(species)

    if len(survived) > 0:
        constants.pool.species = survived
    else:
        if not constants.pool.species:
            constants.pool.species.pop()

def sameSpecies(genome1, genome2):
    dd = constants.DeltaDisjoint*disjoint(genome1.genes, genome2.genes)
    dw = constants.DeltaWeights*weight(genome1.genes, genome2.genes)
    return dd + dw < constants.DeltaThreshold

def addToSpecies(child):
    foundSpecies = False

    for species in constants.pool.species:
        if not foundSpecies and sameSpecies(child, species.genomes[0]):
            species.genomes.append(child)
            foundSpecies = True
            break

    if not foundSpecies:
        childSpecies = Species()
        childSpecies.genomes.append(child)
        constants.pool.species.append(childSpecies)

def breedChild(species):
    child = None

    if random.random() < constants.CrossoverChance:
        g1 = species.genomes[random.randint(0, len(species.genomes)-1)]
        g2 = species.genomes[random.randint(0, len(species.genomes)-1)]
        child = crossover(g1, g2)

    else:
        g = species.genomes[random.randint(0, len(species.genomes)-1)]
        child = copyGenome(g)

    mutate(child)
    return child

def newGeneration():
    cullSpecies(False)
    rankGlobally()
    removeStaleSpecies()
    rankGlobally()

    for species in constants.pool.species:
        calculateAverageFitness(species)

    removeWeakSpecies()

    sum = totalAverageFitness()

    children = []

    for species in constants.pool.species:
        breed = math.floor(species.averageFitness/sum*constants.Population)-1
        for i in range(0, int(breed)):
            children.append(breedChild(species))

    cullSpecies(True)

    while len(children) + len(constants.pool.species) < constants.Population:
        species = constants.pool.species[random.randint(0, len(constants.pool.species)-1)]
        children.append(breedChild(species))

    for child in children:
        addToSpecies(child)

    constants.pool.generation += 1

def evaluateCurrent():
    species = constants.pool.species[constants.pool.currentSpecies]
    genome = species.genomes[constants.pool.currentGenome]

    inputs = constants.snakeWindow.getInputs()
    controller = evaluateNetwork(genome.network, inputs)

    return controller

def initializeRun():
    species = constants.pool.species[constants.pool.currentSpecies]
    genome = species.genomes[constants.pool.currentGenome]
    generateNetwork(genome)

    evaluateCurrent()

def initializePool():
    constants.pool = Pool()

    for i in range(0, constants.Population):
        addToSpecies(basicGenome())

    initializeRun()

def nextGenome():
    constants.pool.currentGenome += 1
    if constants.pool.currentGenome >= len(constants.pool.species[constants.pool.currentSpecies].genomes):
        constants.pool.currentGenome = 0
        constants.pool.currentSpecies += 1
        if constants.pool.currentSpecies >= len(constants.pool.species):
            newGeneration()
            constants.pool.currentSpecies = 0


def fitnessAlreadyMeasured():
    species = constants.pool.species[constants.pool.currentSpecies]
    genome = species.genomes[constants.pool.currentGenome]

    return genome.fitness != 0


def newInnovation():
    constants.pool.innovation += 1
    return constants.pool.innovation

def displayNN(genome):
    network = genome.network
    cells = {}

    # Display our inputs
    i = 0
    for x in range(0, 2):
        cells[i] = Cell(50, 25+(i*20), network.neurons[i].value)
        constants.snakeWindow.renderNNVisText(str(math.ceil(cells[i].value)), cells[i].x-40, cells[i].y-7, (0, 0, 0))
        i += 1

    constants.snakeWindow.renderNNVisText('Left', 95, constants.PADDING-20, (0, 0, 0))
    for dx in range(0, constants.LEFT_DIMENSION_INPUTS):
        for dy in range(0, constants.LEFT_DIMENSION_INPUTS):
            cells[i] = Cell(95+(dx*constants.NN_VISUALIZE_SIZE), constants.PADDING+(dy*constants.NN_VISUALIZE_SIZE), network.neurons[i].value)
            i += 1

    constants.snakeWindow.renderNNVisText('Right', 145, constants.PADDING-20, (0, 0, 0))
    for dx in range(0, constants.RIGHT_DIMENSION_INPUTS):
        for dy in range(0, constants.RIGHT_DIMENSION_INPUTS):
            cells[i] = Cell(155+(dx*constants.NN_VISUALIZE_SIZE), constants.PADDING+(dy*constants.NN_VISUALIZE_SIZE), network.neurons[i].value)
            i += 1

    constants.snakeWindow.renderNNVisText('Top', 125, 25, (0, 0, 0))
    for j in range(0, constants.FRONT_DIMENSION_INPUTS):
        cells[i] = Cell(135, 40+(j*constants.NN_VISUALIZE_SIZE), network.neurons[i].value)
        i += 1

    # Out bias cell
    biasCell = Cell(15, 80, network.neurons[constants.Inputs].value)
    cells[constants.Inputs] = biasCell

    # Displays our output
    for i in range(0, constants.Outputs):
        cells[constants.MaxNodes+i] = Cell(220, 35+20*i, network.neurons[constants.MaxNodes+i].value)

        if cells[constants.MaxNodes+i].value > 0:
            color = (0, 0, 0)
        else:
            color = (180, 180, 180)

        constants.snakeWindow.renderNNVisText(constants.Output_Names[i], 230, 25+20*i, color)

    for key in network.neurons:
        neuron = network.neurons[key]
        if key > constants.Inputs and key < constants.MaxNodes:
            cells[key] = Cell(150, 80, neuron.value)

    #Randomizing neuron positions
    for gene in genome.genes:
        if gene.enabled:
            if gene.into in cells and gene.out in cells:
                c1 = cells[gene.into]
                c2 = cells[gene.out]

                if gene.into > constants.Inputs and gene.out < constants.MaxNodes:
                    c1.x = 1.05*c1.x

                    if c1.x >= c2.x:
                        c1.x = c1.x - 4

                    if c1.x < 9:
                        c1.x = 9

                    if c1.x > 22:
                        c1.x = 22

                    c1.y = 1.1*c1.y

                if gene.out > constants.Inputs and gene.out < constants.MaxNodes:
                    c2.x = 1.1*c1.x

                    if c2.x >= c2.x:
                        c1.x = c2.x + 4

                    if c2.x < 9:
                        c2.x = 9

                    if c2.x > 22:
                        c2.x = 22

                    c2.y = 1.1*c1.y

    for key in cells:
        cell = cells[key]
        if key > constants.Inputs and key <= constants.MaxNodes:
            color = math.floor((cell.value+1)/2*256)
            if color > 255:
                if cell.value > 0:
                    color = (0, 255, 0)
                else:
                    color = (0, 150, 0)

            else:
                if cell.value > 0:
                    color = (255, 0, 0)
                else:
                    color = (150, 0 ,0)

            constants.snakeWindow.renderCustomColorBox(cell.x, cell.y, color)

        else:
            if cell.value > 0:
                constants.snakeWindow.renderCustomColorBox(cell.x, cell.y, (0, 0, 255))
            elif cell.value == -1:
                constants.snakeWindow.renderGrayBox(cell.x, cell.y)
            else:
                constants.snakeWindow.renderCustomColorBox(cell.x, cell.y, (0, 0, 0))

    for gene in genome.genes:
        if gene.enabled and gene.into in cells and gene.out in cells:
            c1 = cells[gene.into]
            c2 = cells[gene.out]

            # white = activated connection
            # Green = activate on stepping on it
            # Red = negative on stepping on it

            color = (0, 255, 0) # we know theres a connection there
            if c1.value < 0:
                if gene.weight < 0:
                    color = (255, 0, 0) # negative connection
            else:
                if gene.weight > 0:
                    color = (255, 255, 255) # positive connection


            constants.snakeWindow.drawLine((c1.x,c1.y+constants.GAME_WIDTH_HEIGHT*constants.BLOCK_SIZE),(c2.x,c2.y+constants.GAME_WIDTH_HEIGHT*constants.BLOCK_SIZE), color)

    pygame.display.update()
