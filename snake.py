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

    def moveDir(self, d):
        if d['Left']:
            if self.dir == constants.Directions.Up:
                #self.xs[0] -= 1
                self.dir = constants.Directions.Left
            elif self.dir == constants.Directions.Down:
                #self.xs[0] += 1
                self.dir = constants.Directions.Right
            elif self.dir == constants.Directions.Left:
                #self.ys[0] += 1
                self.dir = constants.Directions.Down
            elif self.dir == constants.Directions.Right:
                #self.ys[0] -= 1
                self.dir = constants.Directions.Up
        elif d['Right']:
            if self.dir == constants.Directions.Up:
                #self.xs[0] += 1
                self.dir = constants.Directions.Right
            elif self.dir == constants.Directions.Down:
                #self.xs[0] -= 1
                self.dir = constants.Directions.Left
            elif self.dir == constants.Directions.Left:
                #self.ys[0] -= 1
                self.dir = constants.Directions.Up
            elif self.dir == constants.Directions.Right:
                #self.ys[0] += 1
                self.dir = constants.Directions.Down
        elif d['Front']:
            if self.dir == constants.Directions.Down:
                #self.ys[0] += 1
                self.dir = constants.Directions.Down
            elif self.dir == constants.Directions.Up:
                #self.ys[0] -= 1
                self.dir = constants.Directions.Up
            elif self.dir == constants.Directions.Right:
                #self.xs[0] += 1
                self.dir = constants.Directions.Right
            elif self.dir == constants.Directions.Left:
                #self.xs[0] -= 1
                self.dir = constants.Directions.Left

        return self.update()

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

        '''
        # Snake with portals
        if self.xs[0] < 0:
            self.xs[0] = constants.GAME_WIDTH_HEIGHT-1
        elif self.xs[0]>= constants.GAME_WIDTH_HEIGHT:
            self.xs[0] = 0
        elif self.ys[0] < 0:
            self.ys[0] = constants.GAME_WIDTH_HEIGHT-1
        elif self.ys[0] >= constants.GAME_WIDTH_HEIGHT:
            self.ys[0] = 0
        '''

        if self.collide(self._apple.x, self._apple.y):
            self._apple.respawn()
            if self.speed <= 100:
                self.speed += 1
            self.score += 1
            self.move_no = 0
            self.updates_no = 0
            self.xs.append(999)
            self.ys.append(999)

        self.moves += 1
        self.move_no += 1
        self.updates_no += 1

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
        self.xs = [CENTER_POINT, CENTER_POINT, CENTER_POINT]
        self.ys = [3, 2, 1]

        # etc
        self.score = 0

        # How many moves did we do
        self.moves = 0

        # Used to count maximum moves we do eat time we eat an apple
        self.move_no = 0

        # How many updates have we done
        self.updates_no = 0

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

        pygame.display.update()


    # Generates a list of inputs for
    def getInputs(self):
        # Our list to get out from
        temp_list = []
        game_list = []
        # Background
        for x in range(0, constants.GAME_WIDTH_HEIGHT):
            for y in range(0, constants.GAME_WIDTH_HEIGHT):
                temp_list.append(constants.NNObjects.Background.value)
            game_list.append(temp_list)
            temp_list = []

        # Snake Body, and head
        xy_list = self._snake.getSnakeXY()

        for xy in xy_list:
            try:
                game_list[xy[0]][xy[1]] = constants.NNObjects.SnakeBody.value
            except IndexError:
                pass

        game_list[self._snake._apple.x][self._snake._apple.y] = constants.NNObjects.Apple.value

        # Snake variables
        snake_x = self._snake.xs[0]
        snake_y = self._snake.ys[0]

        # Our output list
        outlist = []
        outlist.append(snake_x- self._snake._apple.x) # Distance from snake to apple @ x axis
        outlist.append(snake_y- self._snake._apple.y) # Distance from snake to apple @ y axis

        # CLOCKWISE FASHION @@
        # Left to rightgetInputs

        # Left dimension
        # -3 because we want to position the snake in the center
        for dy in range(1, constants.LEFT_DIMENSION_INPUTS+1):
            for dx in range(-constants.LEFT_DIMENSION_INPUTS, 0):
                try:
                    temp_x = 999
                    temp_y = 999
                    if self._snake.dir == constants.Directions.Up:
                        temp_x = snake_x + dx
                        temp_y = snake_y - dy

                    elif self._snake.dir == constants.Directions.Down:
                        temp_x = snake_x - dx
                        temp_y = snake_y + dy

                    elif self._snake.dir == constants.Directions.Left:
                        temp_x = snake_x - dy
                        temp_y = snake_y - dx

                    elif self._snake.dir == constants.Directions.Right:
                        temp_x = snake_x + dy
                        temp_y = snake_y + dx

                    if temp_x < 0 or temp_y < 0:
                        raise IndexError

                    outlist.append(game_list[temp_x][temp_y])
                except IndexError: # If theres an index error then it'll be a dead end

                    outlist.append(constants.NNObjects.DeadEnd.value)

        # Right dimension
        for dy in range(1, constants.RIGHT_DIMENSION_INPUTS+1):
            for dx in range(-constants.RIGHT_DIMENSION_INPUTS, 0):
                try:
                    temp_x = 999
                    temp_y = 999
                    if self._snake.dir == constants.Directions.Up:
                        temp_x = snake_x - dx
                        temp_y = snake_y - dy

                    elif self._snake.dir == constants.Directions.Down:
                        temp_x = snake_x + dx
                        temp_y = snake_y + dy

                    elif self._snake.dir == constants.Directions.Left:
                        temp_x = snake_x - dy
                        temp_y = snake_y + dx

                    elif self._snake.dir == constants.Directions.Right:
                        temp_x = snake_x + dy
                        temp_y = snake_y - dx

                    if temp_x < 0 or temp_y < 0:
                        raise IndexError

                    outlist.append(game_list[temp_x][temp_y])
                except IndexError: # If theres an index error then it'll be a dead end
                    outlist.append(constants.NNObjects.DeadEnd.value)

        # Front dimension
        for i in range(1, constants.FRONT_DIMENSION_INPUTS+1):
            try:
                temp_x = snake_x
                temp_y = snake_y

                if self._snake.dir == constants.Directions.Up:
                    temp_y = snake_y - i

                elif self._snake.dir == constants.Directions.Down:
                    temp_y = snake_y + i

                elif self._snake.dir == constants.Directions.Right:
                    temp_x = snake_x + i

                elif self._snake.dir == constants.Directions.Left:
                    temp_x = snake_x - i

                if temp_x < 0 or temp_y < 0:
                        raise IndexError

                outlist.append(game_list[temp_x][temp_y])

            except IndexError: # If theres an index error then it'll be a dead end
                outlist.append(constants.NNObjects.DeadEnd.value)

        return outlist


# Class instances
constants.snake = Snake(Apple())
constants.snakeWindow = SnakeWindow()
constants.snakeWindow.setSnake(constants.snake)

if constants.pool == None:
    try:
        nn.loadPool('data/example_network.dat')
        print('Loaded saved state')
    except:
        nn.initializePool()


while True:
    # Tick-tock
    constants.snakeWindow.clock.tick(constants.snake.speed)

    # Update snake window
    constants.snakeWindow.update()

    fitness = constants.snake.score*5+(constants.snake.moves)*0.001

    ## Neural Network ##
    species = constants.pool.species[constants.pool.currentSpecies]
    genome = species.genomes[constants.pool.currentGenome]

    # Update snake
    still_alive = constants.snake.moveDir(nn.evaluateCurrent())
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

    # Time out, don't wanna run in an infinite loop
    if still_alive:
        if constants.snake.updates_no >= constants.MaxUpdateConstants:
            still_alive = False
        if constants.snake.move_no >= constants.MaxMoveConstants:
            still_alive = False

    if not still_alive:

        species = constants.pool.species[constants.pool.currentSpecies]
        genome = species.genomes[constants.pool.currentGenome]
        genome.fitness = fitness
        nn.displayNN(genome)

        if fitness > constants.pool.maxFitness:
            constants.pool.maxFitness = fitness
            nn.savePool(str(fitness) + '_fitness_pool.dat')

        constants.pool.currentSpecies = 0
        constants.pool.currentGenome = 0

        while nn.fitnessAlreadyMeasured():
            nn.nextGenome()
        nn.initializeRun()

        constants.snake.reset()
