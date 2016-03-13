import math, random, constants

def sigmoid(x):
    return 1/(1+math.exp(-x))

def snakeControl(output, snakeVar):
    if snakeVar.dir == constants.Directions.Up or snakeVar.dir == constants.Directions.Down:
        if output['Left']:
            snakeVar.setDirection(constants.Directions.Left)
        elif output['Right']:
            snakeVar.setDirection(constants.Directions.Right)
    else:
        if output['Down']:
            snakeVar.setDirection(constants.Directions.Down)
        elif output['Up']:
            snakeVar.setDirection(constants.Directions.Up)


class Pool:
    def __init__(self):
        self.newPool()

    def newPool(self):
        self.species = []
        self.species.append(Species())
        self.generation = 0
        self.innovation = constants.Outputs
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
        self.genomes.append(Genome())
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
        self.genes.append(Gene())
        self.fitness = 0
        self.adjustedFitness = 0
        self.network = Network()
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

    def basicGenome(self):
        genome = Genome()
        genome.maxneuron = constants.Inputs
        genome.mutate()
        return genome

    def genomeCopy(self):
        genome_copy = Genome()
        genome_copy.genes.pop()

        for gene in self.genes:
            genome_copy.genes.append(gene)

        genome_copy.maxneuron = self.maxneuron
        genome_copy.mutationRates = self.mutationRates

        return genome_copy

    def randomNeuron(self, nonInput):
        neurons = {}

        if not nonInput:
            for i in range(0, constants.Inputs):
                neurons[i] = True

        for i in range(0, constants.Outputs):
            neurons[constants.MaxNodes+i] = True

        for gene in self.genes:
            if (not nonInput) or gene.into > constants.Inputs:
                neurons[gene.into] = True
            if (not nonInput) or gene.out > constants.Inputs:
                neurons[gene.out] = True

        count = 0

        for key in neurons:
            count = count + 1

        n = random.randint(0, count)

        for key in neurons:
            n = n-1
            if n == 0:
                return key

        return 0

    def containsLink(self, link):
        for gene in self.genes:
            if gene.into == link.into and gene.out == link.out:
                return True
        return False

    def pointMutate(self):
        step = self.mutationRates['step']

        for gene in self.genes:
            if random.random() < constants.PerturbChance:
                gene.weight = gene.weight + random.random() * step*2 - step
            else:
                gene.weight = random.random()*4-2

    def linkMutate(self, forceBias):
        neuron1 = self.randomNeuron(False)
        neuron2 = self.randomNeuron(True)

        newLink = Gene()

        # Both are input nodes if their key value is
        # less than the number of inputs
        if neuron1 <= constants.Inputs and neuron2 <= constants.Inputs:
            return

        if neuron2 <= constants.Inputs:
            # Swap output and input
            temp = neuron1
            neuron1 = neuron2
            neuron2 = temp

        newLink.into = neuron1
        newLink.out = neuron2

        if forceBias:
            newLink.into = constants.Inputs

        if self.containsLink(newLink):
            return

        newLink.innovation = newInnovation()
        newLink.weight = random.random()*4-2

        self.genes.append(newLink)

    def nodeMutate(self):
        if len(self.genes) == 0:
            return

        self.maxneuron += 1

        gene = self.genes[random.randint(0, len(self.genes)-1)]

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

        gene = candidates[random.randint(0, len(candidates)-1)]
        gene.enabled = not gene.enabled

    def mutate(self):
        for key in self.mutationRates:
            if random.randint(1,2) == 1:
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
        self.newNeuron()

    def newNeuron(self):
        self.incoming = []
        self.value = 0.0


class Network:
    def __init__(self):
        self.neurons = {}

    def generateNetwork(self, genome):

        for i in range(0, constants.Inputs+1):
            self.neurons[i] = Neuron()

        for i in range(0, constants.Outputs):
            self.neurons[constants.MaxNodes+i] = Neuron()

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
        inputs.append(1)

        if len(inputs) != constants.Inputs + 1:
            print('Incorrect number of neural network inputs')
            return

        for i in range(0, len(inputs)):
            self.neurons[i].value = inputs[i]

        for key in self.neurons:
            neuron = self.neurons[key]
            sum = 0

            for incoming in neuron.incoming:
                other = self.neurons[incoming.into]
                sum = sum + incoming.weight + other.value

            if len(neuron.incoming) > 0:
                neuron.value = sigmoid(sum)

        outputs = {}
        for i in range(0, constants.Outputs):
            direction = constants.Output_Names[i]
            if self.neurons[constants.MaxNodes+i].value > 0:
                outputs[direction] = True
            else:
                outputs[direction] = False

        return outputs

def newInnovation():
    constants.pool.innovation += 1
    return constants.pool.innovation

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

        if gene.innovation in innovations2:
            gene2 = innovations2[gene.innovation]

            if gene2 is not None and random.randint(1, 2) == 1 and gene2.enabled:
                child.genes.append(gene2.copyGene())
            else:
                child.genes.append(gene.copyGene())

    child.maxneuron = max(g1.maxneuron, g2.maxneuron)

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
        if not gene.innovation in i2:
            disjointGenes += 1

    for gene in genes2:
        if not gene.innovation in i2:
            disjointGenes += 1

    n = max(len(genes1), len(genes2))

    try:
        return disjointGenes/n
    except ZeroDivisionError:
        return float('Inf')

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

    if sum == 0 or coincident == 0:
        return 0
    return sum/coincident

def sameSpecies(genome1, genome2):
    dd = constants.DeltaDisjoint*disjoint(genome1.genes, genome2.genes)
    dw = constants.DeltaWeights*weights(genome1.genes, genome2.genes)
    return dd+dw < constants.DeltaThreshold

def rankGlobally(poolVar):
    rGlobal = []

    for species in poolVar.species:
        for genome in species.genomes:
            rGlobal.append(genome)

    rGlobal = sorted(rGlobal, key=lambda genome: genome.fitness, reverse=True)

    for i in range(0, len(rGlobal)):
        rGlobal[i].globalRank = i

def cullSpecies(poolVar, cutToOne):
    for species in poolVar.species:
        species.genomes = sorted(species.genomes, key=lambda genome: genome.fitness, reverse=True)

        remaining = math.ceil(len(species.genomes)/2)

        if cutToOne:
            remaining = 1

        species.genomes = species.genomes[0:int(remaining)]

def breedChild(species):
    child = Gene()

    if random.random() < constants.CrossoverChance:
        g1 = species.genomes[random.randint(0, len(species.genomes)-1)]
        g2 = species.genomes[random.randint(0, len(species.genomes)-1)]
        child = crossover(g1, g2)
    else:
        g = species.genomes[random.randint(0, len(species.genomes)-1)]
        child = g.genomeCopy()

    child.mutate()

    return child

def removeStaleSpecies(poolVar):
    survived = []

    for species in poolVar.species:
        species.genomes = sorted(species.genomes, key=lambda genome: genome.fitness, reverse=True)

        try:

            if species.genomes[0].fitness > species.topFitness:
                species.topFitness = species.genomes[0].fitness
                species.staleness = 0
            else:
                species.staleness += 1

            if species.staleness < constants.StaleSpecies or species.topFitness >= poolVar.maxFitness:
                survived.append(species)

        except:
            pass

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
        breed = math.floor(species.averageFitness/sum*constants.Population)
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
        breed = math.floor(species.averageFitness/sum*constants.Population)-1
        for i in range(0, int(breed)):
            children.append(breedChild(species))

    cullSpecies(poolVar, True) # Cull all but top member of each species

    while len(children) + len(poolVar.species) < constants.Population:
        species = poolVar.species[random.randint(0, len(poolVar.species)-1)]
        children.append(species)

    for child in children:
        addToSpecies(poolVar, child)

    poolVar.generation += 1

    #TODO
    # Save generation to a file

def nextGenome(poolVar):
    poolVar.currentGenome += 1

    if poolVar.currentGenome >= len(poolVar.species[poolVar.currentSpecies].genomes):
        poolVar.currentGenome = 0
        poolVar.currentSpecies += 1
        if poolVar.currentSpecies >= len(poolVar.species):
            newGeneration(poolVar)
            poolVar.currentSpecies = 0

def fitnessAlreadyMeasured(poolVar):
    species = poolVar.species[poolVar.currentSpecies]
    genome = species.genomes[poolVar.currentGenome]


    return genome.fitness != 0

def evaluateCurrent(poolVar, snakeVar):
    species = poolVar.species[poolVar.currentSpecies]
    genome = species.genomes[poolVar.currentGenome]

    inputs = constants.snakeWindow.getInputs()

    #if constants.NNObjects.SnakeBody in inputs: print('yay')

    controller = genome.network.evaluateNetwork(inputs)

    if controller['Left'] and controller['Right']:
        controller['Left'] = False
        controller['Right'] = False

    if controller['Up'] and controller['Down']:
        controller['Up'] = False
        controller['Down'] = False

    snakeControl(controller, snakeVar)

def initializeRun(poolVar, snakeVar):
    # TODO
    # savestate.load(Filename)
    poolVar.currentFrame = 0
    timeout = constants.TimeoutConstant

    species = poolVar.species[poolVar.currentSpecies]
    genome = species.genomes[poolVar.currentGenome]
    genome.network.generateNetwork(genome)

    evaluateCurrent(poolVar, snakeVar)

def initializePool(snakeVar):
    constants.pool = Pool()

    for i in range(0, constants.Population):
        addToSpecies(constants.pool, Genome().basicGenome())

    initializeRun(constants.pool, snakeVar)

def displayNN(genome, snakeWindowVar, pygameVar):
    network = genome.network
    cells = {}

    i = 0
    for dx in range(0, constants.GAME_WIDTH_HEIGHT):
        for dy in range(0, constants.GAME_WIDTH_HEIGHT):
            cell = constants.Cell(constants.PADDING+constants.NN_VISUALIZE_BLOCK_SIZE*dx, constants.PADDING+constants.NN_VISUALIZE_BLOCK_SIZE*dy, network.neurons[i].value)
            cells[i] = cell
            i = i + 1


    biasCell = constants.Cell(75, 85, network.neurons[constants.Inputs].value)
    cells[constants.Inputs] = biasCell

    for i in range(0, constants.Outputs):
        cell = constants.Cell(185, 35+10*i, network.neurons[constants.MaxNodes+i].value)
        cells[constants.MaxNodes+i] = cell

        if cell.value > 0:
            color = (0, 0, 0)

        else:
            color = (180, 180, 180)

        snakeWindowVar.renderNNVisText(constants.Output_Names[i], 230, 10+20*i, color)

    for key in network.neurons:
        if key > constants.Inputs and key <= constants.MaxNodes:
            cell = constants.Cell(95, 4+constants.PADDING+constants.NN_VISUALIZE_BLOCK_SIZE, network.neurons[key].value)
            cells[key] = cell


    for i in range(0, 5):
        for gene in genome.genes:
            if gene.enabled and gene.into in cells and gene.out in cells:
                c1 = cells[gene.into]
                c2 = cells[gene.out]

                if gene.into > constants.Inputs and gene.into < constants.MaxNodes:
                    c1.x = 0.75*c1.x + 0.25*c2.x

                    if c1.x >= c2.x:
                        c1.x = c1.x - 4

                    if c1.x < 9:
                        c1.x = 9

                    if c1.x > 22:
                        c1.x = 22

                    c1.y = 0.75*c1.y+0.25*c2.y

                if gene.out >= constants.Inputs and gene.out < constants.MaxNodes:
                    c2.x = 0.25*c1.x + 0.75*c2.x

                    if c2.x >= c2.x:
                        c1.x = c2.x + 4

                    if c2.x < 9:
                        c2.x = 9

                    if c2.x > 22:
                        c2.x = 22

                    c2.y = 0.25*c1.y + 0.75*c2.y

    for key in cells:
        cell = cells[key]
        if key >= constants.Inputs or cell.value is not 0:
            color = math.floor((cell.value+1)/2*256)

            if color > 255:
                snakeWindowVar.renderWhiteBox(cell.x, cell.y)
            else:
                snakeWindowVar.renderGrayBox(cell.x, cell.y)


    for gene in genome.genes:
        if gene.enabled:
            c1 = cells[gene.into]
            c2 = cells[gene.out]

            color = math.floor(abs(sigmoid(gene.weight)))

            #print(color)

            snakeWindowVar.drawLine((c1.x+1,c1.y+constants.GAME_WIDTH_HEIGHT*constants.BLOCK_SIZE),(c2.x-3,c2.y+constants.GAME_WIDTH_HEIGHT*constants.BLOCK_SIZE), (0, 0, 0,))

    pos = 35
    #for key in genome.mutationRates:
    #    snakeWindowVar.renderText(str(key) + ': ' + str(genome.mutationRates[key]), 25, pos)
    #    pos = pos + 20


    pygameVar.display.update()
