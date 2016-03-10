import pygame, random, sys, math, copy, time
from pygame.locals import *
from enum import Enum

# I like to play snake in a symmetrical window
GAME_WIDTH_HEIGHT = 60

# Our blocksizes for the snake and apple
BLOCK_SIZE = 10


class Directions(Enum):
    Left = -2
    Up = -1
    Down = 1
    Right = 2


class Apple:
    def __init__(self):
        self.respawn()

    def respawn(self):
        self.x = random.randint(0, GAME_WIDTH_HEIGHT - 1)
        self.y = random.randint(0, GAME_WIDTH_HEIGHT - 1)


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
        if self.dir is Directions.Down:
            self.ys[0] += 1
        elif self.dir is Directions.Up:
            self.ys[0] -= 1
        elif self.dir is Directions.Right:
            self.xs[0] += 1
        elif self.dir is Directions.Left:
            self.xs[0] -= 1

        if self.collideSelf():
            return False

        if self.exceedBoundaries():
            return False

        if self.collide(self._apple.x, self._apple.y):
            self._apple.respawn()
            if self.score <= 100:
                self.speed += 1
            self.score += 1
            self.xs.append(999)
            self.ys.append(999)

        return True

    def collideSelf(self):
        for i in range(len(self.xs) - 1, 1, -1):
            if self.collide(self.xs[i], self.ys[i]):
                return True
        return False

    def exceedBoundaries(self):
        if self.xs[0] < 0 or self.xs[0] >= GAME_WIDTH_HEIGHT or self.ys[0] < 0 or self.ys[0] >= GAME_WIDTH_HEIGHT:
            return True
        return False

    def reset(self):
        # Current Direction
        self.dir = Directions.Down

        # Snake speed
        self.speed = 10

        # Snake coordinates
        self.xs = [5, 5, 5, 5]
        self.ys = [9, 8, 7, 6]

        # etc
        self.score = 0

        if self._apple:
            self._apple.respawn()


class SnakeWindow:
    global GAME_WIDTH_HEIGHT, BLOCK_SIZE

    def __init__(self):
        pygame.init()

        # Creates our window and names it
        self.window = pygame.display.set_mode((GAME_WIDTH_HEIGHT * BLOCK_SIZE, GAME_WIDTH_HEIGHT * BLOCK_SIZE))
        pygame.display.set_caption('snake-ai')

        # Renderer for snake and apple
        self.snake_img = pygame.Surface((BLOCK_SIZE, BLOCK_SIZE))
        self.snake_img.fill((255, 0, 0))
        self.apple_img = pygame.Surface((BLOCK_SIZE, BLOCK_SIZE))
        self.apple_img.fill((0, 255, 0))

        # Our font
        self.font = pygame.font.SysFont('Arial', 20)

        # Our game clock
        self.clock = pygame.time.Clock()

    # Sets our snake object
    def setSnake(self, snake):
        self._snake = snake

    def initPyGame(self):
        pass

    def renderText(self, s, x, y):
        self.window.blit(self.font.render(s, True, (0, 0, 0,)),(x, y));

    def update(self):
        # Key presses
        for e in pygame.event.get():
            if e.type == QUIT:
                sys.exit(0)
            elif e.type == KEYDOWN:
                if e.key == K_UP:
                    self._snake.setDirection(Directions.Up)
                elif e.key == K_DOWN:
                    self._snake.setDirection(Directions.Down)
                elif e.key == K_LEFT:
                    self._snake.setDirection(Directions.Left)
                elif e.key == K_RIGHT:
                    self._snake.setDirection(Directions.Right)
                elif e.key == K_o:
                    self._snake.speed += 5
                elif e.key == K_p:
                    self._snake.speed -= 5
                elif e.key == K_l:
                    self._snake.xs.append(999)
                    self._snake.ys.append(999)

        # Renders window white
        self.window.fill((255, 255, 255))

        # Renders snake
        xy_list = self._snake.getSnakeXY()
        for xy in xy_list:
            self.window.blit(self.snake_img, (xy[0] * BLOCK_SIZE, xy[1] * BLOCK_SIZE))

        # Renders apple
        self.window.blit(self.apple_img, (self._snake._apple.x*BLOCK_SIZE, self._snake._apple.y*BLOCK_SIZE))

        # Renders score
        self.renderText(str(self._snake.score), 5, 5)

        pygame.display.update()


# Class instances
snake = Snake(Apple())
snakeWindow = SnakeWindow()
snakeWindow.setSnake(snake)

while True:
    # Tick-tock
    snakeWindow.clock.tick(snake.speed)

    # Update snake
    still_alive = snake.update()

    if not still_alive:
        snake.reset()

    # Update snake window
    snakeWindow.update()
