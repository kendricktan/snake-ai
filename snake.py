import pygame, random, sys, constants, nn, time, math
from pygame.locals import *

class Apple:
    def __init__(self):
        self.respawn()

    def respawn(self):
        self.x = random.randint(0, constants.GAME_WIDTH_HEIGHT - 1)
        self.y = random.randint(0, constants.GAME_WIDTH_HEIGHT - 1)


class Snake:
    global GAME_WIDTH_HEIGHT, BLOCK_SIZE

    def __init__(self, Apple):
        self._apple = Apple
        self.reset()

    def collide(self, x, y):
        cur_x = self.xs[0]
        cur_y = self.ys[0]

        if cur_x >= x and cur_x <= x and cur_y >= y and cur_y <= y:
            return True
        return False

    def getSnakeXY(self):
        outlist = []
        for i in range(0, len(self.xs)):
            outlist.append((self.xs[i], self.ys[i]))
        return outlist

    def setDirection(self, d):
        if d.value != self.dir.value * -1:
            self.dir = d

    def setApple(self, apple):
        self._apple = apple

    def update(self):
        # Updates snake body
        for i in range(len(self.xs) - 1, 0, -1):
            self.xs[i] = self.xs[i - 1]
            self.ys[i] = self.ys[i - 1]

        # Updates snake head
        if self.dir is constants.Directions.Down:
            self.ys[0] += 1
        elif self.dir is constants.Directions.Up:
            self.ys[0] -= 1
        elif self.dir is constants.Directions.Right:
            self.xs[0] += 1
        elif self.dir is constants.Directions.Left:
            self.xs[0] -= 1

        if self.collideSelf():
            return False

        if self.exceedBoundaries():
            return False

        if self.collide(self._apple.x, self._apple.y):
            self._apple.respawn()
            if self.speed <= 100:
                self.speed += 1
            self.score += 1
            self.xs.append(999)
            self.ys.append(999)

        self.moves += 1

        return True

    def collideSelf(self):
        for i in range(len(self.xs) - 1, 1, -1):
            if self.collide(self.xs[i], self.ys[i]):
                return True
        return False

    def exceedBoundaries(self):
        if self.xs[0] < 0 or self.xs[0] >= constants.GAME_WIDTH_HEIGHT or self.ys[0] < 0 or self.ys[0] >= constants.GAME_WIDTH_HEIGHT:
            return True
        return False

    def reset(self):
        # Current Direction
        self.dir = constants.Directions.Down

        # Snake speed
        self.speed = constants.GLOBAL_SPEED

        # Snake coordinates
        CENTER_POINT = int(math.ceil(constants.GAME_WIDTH_HEIGHT/2))
        self.xs = [CENTER_POINT, CENTER_POINT, CENTER_POINT, CENTER_POINT]
        self.ys = [0, 0, 0, 0]

        # etc
        self.score = 0

        # How many moves did we do
        self.moves = 0

        if self._apple:
            self._apple.respawn()


class SnakeWindow:

    def __init__(self):
        pygame.init()

        # Creates our window and names it
        self.window = pygame.display.set_mode((constants.GAME_WIDTH_HEIGHT * constants.BLOCK_SIZE, constants.GAME_WIDTH_HEIGHT * constants.BLOCK_SIZE + constants.NN_VISUALIZE_WIDTH_HEIGHT * constants.NN_VISUALIZE_SIZE))
        pygame.display.set_caption('snake-ai')

        # Renderer for snake and apple
        self.snake_img = pygame.Surface((constants.BLOCK_SIZE, constants.BLOCK_SIZE))
        self.snake_img.fill((255, 0, 0))
        self.apple_img = pygame.Surface((constants.BLOCK_SIZE, constants.BLOCK_SIZE))
        self.apple_img.fill((0, 255, 0))

        # Our font
        self.font = pygame.font.SysFont('Arial', 15)

        # Our game clock
        self.clock = pygame.time.Clock()

        # Our grey (but transparent) box
        self.grey_box = pygame.Surface((constants.NN_VISUALIZE_BLOCK_SIZE, constants.NN_VISUALIZE_BLOCK_SIZE))
        #self.grey_box.set_alpha(100)
        self.grey_box.fill((189, 195, 199))

        # Our white box
        self.white_box = pygame.Surface((constants.NN_VISUALIZE_BLOCK_SIZE, constants.NN_VISUALIZE_BLOCK_SIZE))
        #self.white_box.set_alpha(100)
        #self.grey_box.fill((255, 255, 255))

    # Sets our snake object
    def setSnake(self, snake):
        self._snake = snake

    def renderText(self, s, x, y):
        self.window.blit(self.font.render(s, True, (0, 0, 0,)),(x, y))

    def renderGrayBox(self, x, y):
        self.window.blit(self.grey_box, (x, constants.GAME_WIDTH_HEIGHT*constants.BLOCK_SIZE+y))

    def renderWhiteBox(self, x, y):
        self.window.blit(self.white_box, (x, constants.GAME_WIDTH_HEIGHT*constants.BLOCK_SIZE+y))

    def renderCustomColorBox(self, x, y, color):
        custombox = pygame.Surface((constants.NN_VISUALIZE_BLOCK_SIZE, constants.NN_VISUALIZE_BLOCK_SIZE))
        custombox.fill(color)
        self.window.blit(custombox, (x, constants.GAME_WIDTH_HEIGHT*constants.BLOCK_SIZE+y))

    def renderNNVisText(self, s, x, y, color):
        self.window.blit(self.font.render(s, True, color), (x, y+constants.GAME_WIDTH_HEIGHT*constants.BLOCK_SIZE))

    def drawLine(self, xy1, xy2, color):
        pygame.draw.lines(self.window, color, False, [xy1, xy2], 2)

    def update(self):
        # Key presses
        for e in pygame.event.get():
            if e.type == QUIT:
                sys.exit(0)
            elif e.type == KEYDOWN:
                if e.key == K_UP or e.key == K_w:
                    self._snake.setDirection(constants.Directions.Up)
                elif e.key == K_DOWN or e.key == K_s:
                    self._snake.setDirection(constants.Directions.Down)
                elif e.key == K_LEFT or e.key == K_a:
                    self._snake.setDirection(constants.Directions.Left)
                elif e.key == K_RIGHT or e.key == K_d:
                    self._snake.setDirection(constants.Directions.Right)
                elif e.key == K_o:
                    self._snake.speed += 5
                elif e.key == K_p:
                    self._snake.speed -= 5
                elif e.key == K_l:
                    self._snake.xs.append(999)
                    self._snake.ys.append(999)
                elif e.key == K_i:
                    print(self.getInputs())
                elif e.key == K_s:
                    nn.savePool('TEMP_POOL.dat')

        # Renders game and neural network visualize section
        self.window.fill((255, 255, 255), (0, 0, constants.GAME_WIDTH_HEIGHT*constants.BLOCK_SIZE, constants.GAME_WIDTH_HEIGHT*constants.BLOCK_SIZE))
        self.window.fill((218, 223, 225), (0, constants.GAME_WIDTH_HEIGHT*constants.BLOCK_SIZE, constants.GAME_WIDTH_HEIGHT*constants.BLOCK_SIZE, constants.NN_VISUALIZE_WIDTH_HEIGHT * constants.NN_VISUALIZE_SIZE))

        # Renders snake
        xy_list = self._snake.getSnakeXY()
        for xy in xy_list:
            self.window.blit(self.snake_img, (xy[0] * constants.BLOCK_SIZE, xy[1] * constants.BLOCK_SIZE))

        # Renders apple
        self.window.blit(self.apple_img, (self._snake._apple.x*constants.BLOCK_SIZE, self._snake._apple.y*constants.BLOCK_SIZE))

        # Renders score
        self.renderText('Score: ' + str(self._snake.score), 230, 5)

        # Information on our neural network


        pygame.display.update()


    # Generates a list of inputs for
    def getInputs(self):
        ret_list = []
        temp_list = []
        # Background
        for x in range(0, constants.GAME_WIDTH_HEIGHT):
            for y in range(0, constants.GAME_WIDTH_HEIGHT):
                temp_list.append(constants.NNObjects.Background.value)
            ret_list.append(temp_list)
            temp_list = []

        # Snake Body, and head
        xy_list = self._snake.getSnakeXY()

        for xy in xy_list:
            try:
                ret_list[xy[0]][xy[1]] = constants.NNObjects.SnakeBody.value
            except IndexError:
                pass
        #try:
        #    ret_list[xy_list[0][0]][xy_list[0][1]] = constants.NNObjects.SnakeHead.value
        #except IndexError:
        #    pass



        # Apple
        #ret_list[self._snake._apple.x][self._snake._apple.y] = constants.NNObjects.Apple

        # Final GAME_WIDTH_HEIGHT^2 list
        final_list = []

        for y_list in ret_list:
            for y_item in y_list:
                final_list.append(y_item)

        return final_list


# Class instances
constants.snake = Snake(Apple())
constants.snakeWindow = SnakeWindow()
constants.snakeWindow.setSnake(constants.snake)

if constants.pool == None:
    try:
        nn.loadPool('data/136_fitness_pool.dat')
        print('Loaded saved state')
    except:
        nn.initializePool()


while True:
    # Tick-tock
    constants.snakeWindow.clock.tick(constants.snake.speed)

    # Update snake window
    constants.snakeWindow.update()

    # Update snake
    still_alive = constants.snake.update()

    fitness = constants.snake.moves

    ## Neural Network ##
    if still_alive:
        species = constants.pool.species[constants.pool.currentSpecies]
        genome = species.genomes[constants.pool.currentGenome]

        nn.evaluateCurrent()
        nn.displayNN(genome)

        measured = 0
        total = 0

        for species in constants.pool.species:
            for genome in species.genomes:
                total += 1
                if genome.fitness != 0:
                    measured = measured + 1

        constants.snakeWindow.renderText('Gen: ' + str(constants.pool.generation), 5, 5)
        constants.snakeWindow.renderText('species: ' + str(constants.pool.currentSpecies), 5, 25)
        constants.snakeWindow.renderText('genome: ' + str(constants.pool.currentGenome) + '(' + str(measured) + ')', 5, 45)
        constants.snakeWindow.renderText('fitness: ' + str(fitness), 5, 65)

        pygame.display.update()

    else:

        species = constants.pool.species[constants.pool.currentSpecies]
        genome = species.genomes[constants.pool.currentGenome]
        genome.fitness = fitness

        if fitness > constants.pool.maxFitness:
            constants.pool.maxFitness = fitness
            nn.savePool(str(fitness) + '_fitness_pool.dat')

        constants.pool.currentSpecies = 0
        constants.pool.currentGenome = 0

        while nn.fitnessAlreadyMeasured():
            nn.nextGenome()
        nn.initializeRun()

        constants.snake.reset()
