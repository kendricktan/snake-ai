################################################ Neural Network and Genetic Algorithm ##################################################

# Parameters for the network
pool = None # Our pool variable
Inputs = GAME_WIDTH*GAME_WIDTH # How many inputs are we supplying to the neural network
Outputs = 4 # How many outputs do we have (in our case we have 4 outputs: up, down, left, or right)
Output_Names = {0: 'Left', 1: 'Down', 2: 'Up', 3:'Right'}
Population = 300
DeltaDisjoint = 2.0
DeltaWeights = 0.4
DeltaThreshold = 1.0

StaleSpecies = 15

# Paramters for the genome
MutateConnectionsChance = 0.25
PerturbChance = 0.90
CrossoverChance = 0.75
LinkMutationChance = 2.0
NodeMutationChance = 0.50
BiasMutationChance = 0.40
StepSize = 0.1
DisableMutationChance = 0.4
EnableMutationChance = 0.2

TimeoutConstant = 20

MaxNodes = 2**32

# TODO
# Gets inputs for our neural network
def getInputs():
    return None

# Neural Network Classes
class Neuron:
    def __init__(self):
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

        genome.genes = sorted(genome.genes, key = lambda x: genome.genes[x].out) # Sort our genome.gene list

        # Iterate through each gene
        for gene in genome.genes:

            if gene.enabled:
                # Create a gene if it doesn't exist in our network
                if gene.out not in self.neurons:
                    self.neurons[gene.out] = Neuron()

                # Append current neuron to our list of incoming neurons
                neuron = self.neurons[gene.out]
                gene.append(neuron.incoming)

                if gene.into not in self.neurons:
                    self.neurons[gene.into] = Neuron()

    def evaluateNetwork(self, inputs):
        for i in range(0, Inputs):
            self.neurons[i].value = inputs[i]

        for k in self.neurons:
            neuron = self.neurons[k]
            sum = 0

            for j in range(0, neuron.incoming):
                incoming_ = neuron.incoming[j]
                other_ = self.neurons[incoming_]

                sum = incoming_.weight * other_.value

            if len(neuron.incoming) > 0:
                neuron.value = sigmoid(sum)

        outputs = {}
        for i in range(0, Outputs):
            if self.neurons[MaxNodes+i].value > 0:
                outputs[Output_Names[i]] = True
            else:
                outputs[Output_Names[i]] = False

        return outputs


class Genes:
    def __init__(self):
        self.into = 0
        self.out = 0
        self.weight = 0.0
        self.enabled = True
        self.innovation = 0

    # TODO
    def __deepcopy__(self, memo):
        pass

class Genome:
    def __init__(self):
        self.newGenome()

    def newGenome(self):
        self.genes = []
        self.fitness = 0
        self.adjustedFitness = 0
        self.network = Network()
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
        genome = Genome()
        genome.maxneuron = Inputs
        genome.mutate()
        return genome

    def randomNeuron(self, notInput):
        neurons = {}
        if not notInput:
            for i in range(0, Inputs):
                neurons[i] = True

        for i in range(0, Outputs):
            neurons[MaxNodes+i] = True

        for i in range(0, len(self.genes)):
            if (not notInput) or self.genes[i].into > Inputs:
                neurons[self.genes[i].into] = True
            if (not notInput) or self.genes[i].out > Inputs:
                neurons[self.genes[i].out] = True

        count = 0

        for i in neurons:
            count += 1

        n = random.randint(1, count)

        for k in neurons:
            n -= 1
            if n == 0:
                return k

        return 0

    def containsLink(self, link):
        for i in range(0, len(self.genes)):
            gene = self.genes[i]
            if gene.into == link.into and gene.out == link.out:
                return True
        return False

    def pointMutate(self):
        step = self.mutationRates['step']

        for i in range(0, len(self.genes)):
            gene = self.genes[i]

            if random.random() < PerturbChance:
                gene.weight = gene.weight + random.random() * step*2 - step
            else:
                gene.weight = random.random()*4-2

    def linkMutate(self, forceBias):
        neuron1 = self.randomNeuron(False)
        neuron2 = self.randomNeuron(True)

        newLink = Genes()

        if neuron1 <= Inputs and neuron2 <= Inputs:
            # Both input nodes
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


    def nodeMutate(self):
        if len(self.genes) == 0:
            return

        self.maxneuron += 1

        gene = self.genes[random.randint(0, len(self.genes))]

        if not gene.enabled:
            return

        gene.enabled = False

        gene1 = copy.deepcopy(gene)

        gene1.out = self.maxneuron
        gene1.weight = 1.0
        gene1.innovation = newInnovation()
        gene1.enabled = True

        self.genes.append(gene1)

        gene2 = copy.deepcopy(gene)
        gene2.into = self.maxneuron
        gene2.innovation = newInnovation()
        gene2.enabled = True

        self.genes.append(gene2)

    def enableDisableMutate(self, enable):
        candidates = []
        for k in self.genes:
            gene = self.genes[k]
            if gene.enabled is not enable:
                candidates.append(gene)

        if len(candidates) == 0:
            return

        gene = candidates[random.random(0, len(candidates))]
        gene.enabled = not gene.enabled

    def mutate(self):
        for m in self.mutationRates:
            if random.randint(1,2) == 1:
                self.mutationRates[m] *= 0.95
            else:
                self.mutationRates[m] *= 1.05263

        if random.random() < self.mutationRates['connections']:
            self.pointMutate()

        p = self.mutationRates['link']
        while p > 0:
            if random.random() < p:
                self.linkMutate(True)
            p -= 1

        p = self.mutationRates['bias']
        while p > 0:
            if random.random():
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


class Species:
    def __init__(self):
        self.topFitness = 0
        self.staleness = 0
        self.genomes = []
        self.averageFitness = 0

    def breedChild(self):
        child = []
        if random.random() < CrossoverChance:
            g1 = self.genomes[random.randint(1, len(self.genomes))]
            g2 = self.genomes[random.randint(1, len(self.genomes))]

            child = crossover(g1, g2)
        else:
            g = self.genomes[random.randint(1, len(self.genomes))]
            child = copy.deepcopy(g)

        child.mutate()

        return child

    def calculateAverageFitness(self):
        total = 0
        for g in range(0, len(self.genomes)):
            genome = self.genomes[g]
            total = total + genome.globalRank

        self.averageFitness = total/len(self.genomes)

class Pool:
    def __init__(self):
        self.species = []
        self.generation = 0
        self.innovations = Outputs
        self.currentSpecies = 1
        self.currentGenome = 1
        self.currentFrame = 0
        self.maxFitness = 0

def newInnovation():
    global pool
    pool.innovations += 1
    return pool.innovations

def crossover(g1, g2):
    if g2.fitness > g1.fitness:
        tempg = g1
        g1 = g2
        g2 = tempg

    child = Genome()
    innovations2 = {}

    for i in range(0, len(g2.genes)):
        gene = g1.genes[i]
        innovations2[gene.innovation] = gene

    for i in range(0, len(g1.genes)):
        gene1 = g1.genes[i]
        gene2 = innovations2[gene1.innovation]

        if gene2 is not None and random.randint(2) == 1 and gene2.enabled:
            child.genes.append(gene2.copy())
        else:
            child.genes.append(gene1.copy())

    child.maxneuron = math.max(g1.maxneuron, g2.maxneuron)

    for m in g1.mutationRates:
        child.mutationRates[m] = g1.mutationRates[m]

    return child

def disjoin(genes1, genes2):
    i1 = {}
    for i in range(0, len(genes1)):
        gene = genes1[i]
        i1[gene.innovation] = True

    i2 = {}
    for i in range(0, len(genes2)):
        gene = genes2[i]
        i2[gene.innovation] = True

    disjointGenes = 0
    for i in range(0, len(genes1)):
        gene = genes1[i]
        if not i2[gene.innovation]:
            disjointGenes += 1

    n = max(len(genes1), len(genes2))

    return disjointGenes/n

def weights(genes1, genes2):
    i2 = {}
    for i in range(0, len(genes2)):
        gene = genes2[i]
        i2[gene.innovation] = gene

    sum = 0
    conincident = 0

    for i in range(0, len(genes1)):
        gene = genes1[i]
        if i2[gene.innovation] != None:
            gene2 = i2[gene.innovation]
            sum = sum + abs(gene.weight - gene2.weight)
            conincident += 1

    return sum/conincident

def sameSpecies(genome1, genome2):
    dd = DeltaDisjoint*disjoin(genome1, genome2)
    dw = DeltaWeights*weights(genome1.genes, genome2.genes)
    return dd+dw < DeltaThreshold

def rankGlobally():
    global pool
    _global = []
    for i in range(0, len(pool.species)):
        species = pool.species[i]

        for g in range(0, len(species.genomes)):
            _global.append(species.genomes[g])

    _global.sort(key=lambda x: x.fitness)

    for i in range(0, len(_global)):
        _global[i].globalRank = i

def totalAverageFitness():
    global pool
    total = 0
    for i in range(0, len(pool.species)):
        species = pool.species[i]
        total += species.averageFitness
    return total

def cullSpecies(cutToOne):
    global pool
    for i in range(0, len(pool.species)):
        species = pool.species[i]

        species.genomes = sorted(species.genomes, key = lambda x: species.genomes[x].fitness)

        remaining = math.ceil(len(species.genomes/2))

        if cutToOne:
            remaining = 1

        while len(species.genomes) > remaining:
            species.genomes.pop()

def removeStaleSpecies():
    global pool

    survived = []

    for i in range(0, len(pool.species)):
        species = pool.species[i]

        species.genomes = sorted(species.genomes, key = lambda x: species.genomes[x].fitness)

        if species.genomes[0].fitness > species.topFitness:
            species.topFitness = species.genomes[0].fitness
            species.staleness = 0
        else:
            species.staleness = species.staleness + 1

        if species.staleness < StaleSpecies or species.topFitness >= pool.maxFitness:
            survived.append(species)

    pool.species = survived

def removeWeakSpecies():
    global pool
    survived = []

    sum = totalAverageFitness()

    for s in range(0, len(pool.species)):
        species = pool.species[s]

        breed = math.floor((species.averageFitness/sum*Population))

        if breed >= 1:
            survived.append(species)

    pool.species = survived

def addToSpecies(child):
    global pool
    foundSpecies = False
    for i in range(0, len(pool.species)):
        species = pool.species[i]

        if not foundSpecies and sameSpecies(child, species.genomes[0]):
            species.genomes.append(child)
            foundSpecies = True
            break

    if not foundSpecies:
        childSpecies = Species()
        childSpecies.genomes.append(child)
        pool.species.append(childSpecies)

def newGeneration():
    global pool

    cullSpecies(False) # Cull the bottom half of each species
    rankGlobally()
    removeStaleSpecies()
    rankGlobally()

    for i in range(0, len(pool.species)):
        species = pool.species[s]
        species.calculateAverageFitness()

    removeWeakSpecies()

    sum = totalAverageFitness()
    children = []

    for i in range(0, len(pool.species)):
        species = pool.species[i]
        breed = math.floor(species.averageFitness / sum * Population) - 1

        for i in range(0, breed):
            children.index(species.breedChild())

    cullSpecies(True) # Cut all but the top member of each species

    while len(children) + len(pool.species) < Population:
        species = pool.species[random.randint(0, len(pool.species))]
        children.append(species)

    for i in range(0, len(children)):
        child = children[i]
        addToSpecies(child)

    pool.generation = pool.generation + 1

def evaluateCurrent():
    global pool
    species = pool.species[pool.currentSpecies]
    genome = species.genomes[pool.currentGenome]

    # TODO
    inputs = None
    # inputs = getInputs()
    controller = genome.network.evaluateNetwork(inputs)

    if controller['Left'] and controller['Right']:
        controller['Left'] = False
        controller['Right'] = False
    if controller['Up'] and controller['Down']:
        controller['Up'] = False
        controller['Down'] = False

    # TODO
    # Joypad.set(controller)

def initializeRun():
    global pool

    pool.currentFrame = 0
    timeout = TimeoutConstant

    # TODO
    # clearjoypad()

    species = pool.species[pool.currentSpecies]
    genome = species.genomes[pool.currentGenome]

    genome.network.generateNetwork(genome)

def initializePool():
    global pool
    pool = Pool()

    for i in range(0, Population):
        basic = Genome()
        basic.basicGenome()

        addToSpecies(basic)

    initializeRun()

def nextGenome():
    global pool
    pool.currentGenome = pool.currentGenome + 1
    if pool.currentGenome > len(pool.species[pool.currentSpecies].genomes):
        pool.currentGenome = 1
        pool.currentSpecies = pool.currentSpecies + 1
        if pool.currentSpecies > len(pool.species):
            newGeneration()
            pool.currentSpecies =

def fitnessAlreadyMeasured():
    global pool
    species = pool.species[pool.currentSpecies]
    genome = species.genomes[pool.currentGenome]

    return genome.fitness != 0

def sigmoid(x):
    return 1/(1+math.exp(-x))

# Initializes our network pool
if pool == None:
    initializePool()

################################################ End Neural Network and Genetic Algorithm ##############################################