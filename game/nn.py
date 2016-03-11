import math, random
from constants import *
from snake import *

def sigmoid(x):
    return 1/(1+math.exp(-x))

# TODO
def snakeControl(output):
    pass

class Pool:
    def __init__(self):
        self.newPool()

    def newPool(self):
        self.species = []
        self.generation = 0
        self.innovatoin = Outputs
        self.currentSpecies = 0
        self.currentGenome = 0
        self.currentFrame = 0
        self.maxFitness = 0

class Species:

    def __init__(self):
        self.newSpecies()

    def newSpecies(self):
        self.topFitness = 0
        self.staleness = 0
        self.genomes = []
        self.averageFitness = 0

    def calculateAverageFitness(self):
        total = 0

        for genome in self.genomes:
            total += genome.globalRank

        self.averageFitness = total/len(self.genomes)

class Genome:
    def __init__(self):
        self.newGenome()

    def newGenome(self):
        self.genes = []
        self.fitness = 0
        self.adjustedFitness = 0
        self.network = []
        self.maxneuron = 0
        self.globalRank = 0
        self.mutationRates = {}
        self.mutationRates['connections'] = MutateConnectionsChance
        self.mutationRates['link'] = LinkMutationChance
        self.mutationRates['bias'] = BiasMutationChance
        self.mutationRates['node'] = NodeMutationChance
        self.mutationRates['enable'] = EnableMutationChance
        self.mutationRates['disable'] = DisableMutationChance
        self.mutationRates['step'] = StepSize

    def basicGenome(self):
        genome = self.newGenome()
        innovation = 1

        genome.maxneuron = Inputs
        self.mutate()

    def genomeCopy(self):
        genome_copy = Genome()

    def randomNeuron(self, nonInput):
        neurons = {}

        if not nonInput:
            for i in range(0, Inputs):
                neurons[i] = True

        for i in range(0, Outputs):
            neurons[MaxNodes+i] = True

        for gene in self.genes:
            if (not nonInput) or gene.into > Inputs:
                neurons[gene.into] = True
            if (not nonInput) or gene.out > Inputs:
                neurons[gene.out] = True

        count = 0

        for key in neurons:
            count = count + 1

        n = random.random(0, count)

        for key in neurons:
            n = n-1
            if n == 0:
                return key

        return 0

    def containsLink(self, link):
        for gene in self.genes:
            if gene.into == link.gene and gene.out == link.out:
                return True
        return False

    def pointMutate(self):
        step = self.mutationRates['step']

        for gene in self.genes:
            if random.random() < PerturbChance:
                gene.weight = gene.weight + random.random() * step*2 - step
            else:
                gene.weight = math.random()*4-2

    def linkMutate(self, forceBias):
        neuron1 = self.randomNeuron(False)
        neuron2 = self.randomNeuron(True)

        newLink = Gene()

        # Both are input nodes if their key value is
        # less than the number of inputs
        if neuron1 <= Inputs and neuron2 <= Inputs:
            return

        if neuron2 <= Inputs:
            # Swap output and input
            temp = neuron1
            neuron1 = neuron2
            neuron2 = temp

        newLink.into = neuron1
        newLink.out = neuron2

        if forceBias:
            newLink.into = Inputs

        if self.containsLink(newLink):
            return

        newLink.innovation = newInnovation()
        newLink.weight = random.random()*4-2

        self.genes.append(newLink)

    def nodeMutate(self):
        if len(self.genes) == 0:
            return

        self.maxneuron += 1

        gene = self.genes[random.randint(0, len(self.genes))]

        if not gene.enabled:
            return

        gene.enabled = False

        gene1 = gene.copyGene()
        gene1.out = self.maxneuron
        gene1.weight = 1.0
        gene1.innovation = newInnovation()
        gene1.enabled = True

        self.genes.append(gene1)

        gene2 = gene.copyGene()
        gene2.into = self.maxneuron
        gene2.innovation = newInnovation()
        gene2.enabled = True

        self.genes.append(gene2)

    def enableDisableMutate(self, enable):
        candidates = []
        for gene in self.genes:
            if gene.enabled is not enable:
                candidates.append(gene)

        if len(candidates) == 0:
            return

        gene = candidates[random.randint(0, len(candidates))]
        gene.enabled = not gene.enabled

    def mutate(self):
        for key in self.mutationRates:
            if random.random(1,2) == 1:
                self.mutationRates[key] *= 0.95
            else:
                self.mutationRates[key] *= 1.05263

        if random.random() < self.mutationRates['connections']:
            self.pointMutate()

        p = self.mutationRates['link']
        while p > 0:
            if random.random() < p:
                self.linkMutate(False)
            p -= 1

        p = self.mutationRates['bias']
        while p > 0:
            if random.random() < p:
                self.linkMutate(True)
            p -= 1

        p = self.mutationRates['node']
        while p > 0:
            if random.random() < p:
                self.nodeMutate()
            p -= 1

        p = self.mutationRates['enable']
        while p > 0:
            if random.random() < p:
                self.enableDisableMutate(True)
            p -= 1

        p = self.mutationRates['disable']
        while p > 0:
            if random.random() < p:
                self.enableDisableMutate(False)
            p -= 1



class Gene:
    def __init__(self):
        self.newGene()

    def newGene(self):
        self.into = 0
        self.out = 0
        self.weight = 0
        self.enabled = 0.0
        self.innovation = 0

    def copyGene(self):
        gene2 = Gene()
        gene2.into = self.into
        gene2.out = self.out
        gene2.weight = self.weight
        gene2.enabled = self.enabled
        gene2.innovation = self.innovation

        return gene2


class Neuron:
    def __init__(self):
        pass

    def newNeuron(self):
        self.incoming = []
        self.value = 0.0


class Network:
    def __init__(self):
        self.neurons = {}

    def generateNetwork(self, genome):

        for i in range(0, Inputs):
            self.neurons[i] = Neuron()

        for i in range(0, Outputs):
            self.neurons[MaxNodes+i] = Neuron()

        genome.genes = sorted(genome.genes, key=lambda gene: gene.out)

        for gene in genome.genes:
            if gene.enabled:
                if gene.out not in self.neurons:
                    self.neurons[gene.out] = Neuron()
                neuron = self.neurons[gene.out]
                neuron.incoming.append(gene)

                if gene.into not in self.neurons:
                    self.neurons[gene.into] = Neuron()

        genome.network = self

    def evaluateNetwork(self, inputs):
        if len(inputs) != Inputs:
            print('Incorrect number of neural network inputs')
            return

        for i in range(0, len(inputs)):
            self.neurons[i].value = inputs[i]

        for neuron in self.neurons:
            sum = 0

            for incoming in neuron.incoming:
                other = neuron[incoming.into]
                sum = sum + incoming.weight + other.value

            if len(neuron.incoming) > 0:
                neuron.value = sigmoid(sum)

        outputs = {}
        for i in range(0, Outputs):
            direction = Output_Names[i]
            if self.neurons[MaxNodes+i].value > 0:
                outputs[direction] = True
            else:
                outputs[direction] = False

        return outputs

    def getSortedNeuronsIndex(self):
        return sorted(self, key=lambda x: self.neurons[x].value)


def newInnovation():
    global pool
    pool.innovation += 1
    return pool.innovation

def crossover(g1, g2):
    if g2.fitness > g2.fitness:
        tempg = g1
        g1 = g2
        g2 = tempg

    child = Genome()

    innovations2 = {}
    for gene in g2.genes:
        innovations2[gene.innovation] = gene

    for gene in g1.genes:
        gene2 = None

        try:
            gene2 = innovations2[gene.innovation]
        except:
            pass

        if gene2 is not None and random.randint(2) == 1 and gene2.enabled:
            child.genes.append(gene2.copyGene())
        else:
            child.genes.append(gene.copyGene())

    child.maxneuron = max(g1.maxneurons, g2.maxneurons)

    for key in g1.mutationRates:
        child.mutationRates[key] = g1.mutationRates[key]

    return child

def disjoint(genes1, genes2):
    i1 = {}

    for gene in genes1:
        i1[gene.innovation] = True

    i2  = {}
    for gene in genes2:
        i2[gene.innovation] = True

    disjointGenes = 0
    for gene in genes1:
        if not i2[gene.innovation]:
            disjointGenes += 1

    for gene in genes2:
        if not i1[gene.innovation]:
            disjointGenes += 1

    n = max(len(genes1), len(genes2))

    return disjointGenes/n

def weights(genes1, genes2):
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

    return sum/coincident

def sameSpecies(genome1, genome2):
    dd = DeltaDisjoint*disjoint(genome1.genes, genome2.genes)
    dw = DeltaWeights*weights(genome1.genes, genome2.genes)
    return dd+dw < DeltaThreshold

def rankGlobally(poolVar):
    rGlobal = []

    for species in poolVar.species:
        for genome in species.genomes:
            rGlobal.append(genome)

    rGlobal = sorted(rGlobal, key=lambda genome: genome.fitness)

    for i in range(0, len(rGlobal)):
        rGlobal[i].globalRank = i

def cullSpecies(poolVar, cutToOne):
    for species in poolVar.species:
        species.genomes = sorted(species.genomes, key=lambda genome: genome.fitness)

        remaining = math.ceil(len(species.genomes)/2)

        if cutToOne:
            remaining = 1

        species.genomes = species.genomes[0:remaining]

def breedChild(species):
    child = []

    if random.random() < CrossoverChance:
        g1 = species.genomes[random.randint(0, len(species.genomes))]
        g2 = species.genomes[random.randint(0, len(species.genomes))]
        child = crossover(g1, g2)
    else:
        g = species.genomes[random.randint(0, len(species.genomes))]
        child = g.genomeCopy()

    child.mutate()

    return child

def removeStaleSpecies(poolVar):
    survived = []

    for species in poolVar.species:
        species.genomes = sorted(species.genomes, key=lambda genome: genome.fitness)

        if species.genomes[0].fitness > species.topFitness:
            species.topFitness = species.genomes[0].fitness
            species.staleness = 0
        else:
            species.staleness += 1

        if species.staleness < StaleSpecies or species.topFitness >= poolVar.maxFitness:
            survived.append(species)

    poolVar.species = survived

def totalAverageFitness(poolVar):
    total = 0

    for species in poolVar.species:
        total += species.averageFitness

    return total

def removeWeakSpecies(poolVar):
    survived = []

    sum = totalAverageFitness(poolVar)

    for species in poolVar.species:
        breed = math.floor(species.averageFitness/sum*Population)
        if breed >= 1:
            survived.append(species)

    poolVar.species = survived

def addToSpecies(poolVar, child):
    foundSpecies = False

    for species in poolVar.species:
        if not foundSpecies and sameSpecies(child, species.genomes[0]):
            species.genomes.append(child)
            foundSpecies = True
            break

    if not foundSpecies:
        childSpecies = Species()
        childSpecies.genomes.append(child)
        poolVar.species.append(childSpecies)

def newGeneration(poolVar):
    cullSpecies(poolVar, False) # Cull bottom half of each species
    rankGlobally(poolVar)
    removeStaleSpecies(poolVar)
    rankGlobally(poolVar)

    for species in poolVar.species:
        species.calculateAverageFitness()

    removeWeakSpecies(poolVar)

    sum = totalAverageFitness(poolVar)

    children = []

    for species in poolVar.species:
        breed = math.floor(species.averageFitness/sum*Population)-1
        for i in range(0, breed):
            children.append(breedChild(species))

    cullSpecies(True) # Cull all but top member of each species

    while len(children) + len(poolVar.species) < Population:
        species = poolVar.species[random.randint(0, len(poolVar.species))]
        children.append(species)

    for child in children:
        addToSpecies(poolVar, child)

    poolVar.generation += 1

    #TODO
    # Save generation to a file

def nextGenome(poolVar):
    poolVar.currentGenome += 1

    if poolVar.currentGenome > len(poolVar.species[poolVar.currentGenome].genomes):
        poolVar.currentGenome = 1
        poolVar.currentSpecies += 1
        if poolVar.currentSpecies > len(poolVar.species):
            newGeneration(poolVar)
            poolVar.currentSpecies = 1

def fitnessAlreadyMeasured(poolVar):
    species = poolVar.species[poolVar.currentSpecies]
    genome = species.genomes[poolVar.currentGenome]

    return genome.fitness != 0

def evaluateCurrent(poolVar):
    species = poolVar.species[poolVar.currentSpecies]
    genome = species.genomes[poolVar.currentGenome]

    inputs = snakeWindow.getInputs()
    controller = genome.network.evaluateNetwork(inputs)

    if controller['Left'] and controller['Right']:
        controller['Left'] = False
        controller['Right'] = False

    if controller['Up'] and controller['Down']:
        controller['Up'] = False
        controller['Down'] = False

    snakeControl(controller)

def initializeRun(poolVar):
    # TODO
    # savestate.load(Filename)
    poolVar.currentFrame = 0
    timeout = TimeoutConstant

    species = poolVar.species[poolVar.currentSpecies]
    genome = species.genomes[poolVar.currentGenome]
    genome.network.generateNetwork(genome)

    evaluateCurrent(poolVar)

def initializePool():
    global pool
    pool = Pool()

    for i in range(0, Population):
        basic = Genome()
        basic.basicGenome()

        addToSpecies(pool, basic)

    initializeRun()

# TODO
def displayNN(genomeVar, snakeWindowVar):
    network = genomeVar.network
    cells = {}
    i = 0

    for dy in range(-NN_VISUALIZE_BLOCK_SIZE, NN_VISUALIZE_BLOCK_SIZE):
        for dx in range(-NN_VISUALIZE_BLOCK_SIZE, NN_VISUALIZE_BLOCK_SIZE):
            cell = Cell(dx, dy, network.neurons[i].value)
            cells[i] = cell
            i = i+1

    biasCell = Cell(8, 11, network.neurons[Inputs].value)
    cells[Inputs] = biasCell

    for i in range(0, Outputs):
        cell = Cell(22, 30+8*i, network.neurons[MaxNodes+i].value)
        cells[MaxNodes+i] = cell

        color = (245, 0, 245) if cell.value > 0 else (245, 0, 0)

        snakeWindowVar.renderNNVisText(Output_Names[i], DIR_TEXT_LOC[0], DIR_TEXT_LOC[1]+i*20, color)

    for keys in network.neurons:
        neuron = network.neurons[keys]

        if keys > Inputs and keys <= MaxNodes:
            cell = Cell(14, 4, neuron.value)
            cells[keys] = cell

    for n in range(0, 4):
        for gene in genomeVar.genes:
            if gene.enabled:
                c1 = cells[gene.into]
                c2 = cells[gene.out]

                if gene.into > Inputs and gene.into <= MaxNodes:
                    c1.x = 0.75*c1.x + 0.25*c2.x
                    if c1.x >= c2.x:
                        c1.x -= 4

                    if c1.x < 9:
                        c1.x = 9

                    if c1.x > 22:
                        c1.x = 22

                    c1.y = 0.75*c1.y + 0.25*c2.y

                if gene.out > Inputs and gene.out <= MaxNodes:
                    c2.x = 0.25*c1.x + 0.75*c2.x
                    if c1.x >= c2.x:
                        c2.x = c2.x + 4
                    if c2.x < 9:
                        c2.x = 9
                    if c2.x > 22:
                        c2.x = 22
                    c2.y = 0.25*c1.y+0.75*c2.y


    for key in cells:
        cell = cells[key]
        if key > Inputs or cell.value != 0:
            color = math.floor((cell.value+1)/2*256)
            if color > 255:
                snakeWindowVar.renderWhiteBox(cell.x-2, cell.y-2)

            if color < 0:
                snakeWindowVar.renderGrayBox(cell.x-2, cell.y-2)


    for gene in genomeVar.genes:
        if gene.enabled:
            c1 = cells[gene.into]
            c2 = cells[gene.out]

            color = (0, 0, 0)

            if gene.weight > 0:
                color = (255, 255, 255)

            snakeWindowVar.drawLine((c1.x+1,c1.y), (c2.x-3, c2.y), color)
