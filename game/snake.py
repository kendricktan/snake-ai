import pygame, random, sys
from pygame.locals import *

# Snake parameters
xs = [290, 290, 290, 290, 290]
ys = [290, 280, 270, 260, 250]
block_size = 20
dirs = 0
score = 0

# Apple parameters
applepos = (random.randint(0, 29)*20, random.randint(0, 29)*20)

# Game window
pygame.init()
s=pygame.display.set_mode((600, 600))
pygame.display.set_caption('Snake')

# Rendering snake and apple objects
appleimage = pygame.Surface((10, 10))
appleimage.fill((0, 255, 0))
img = pygame.Surface((10, 10))
img.fill((255, 0, 0))
f = pygame.font.SysFont('Arial', 20)

# Our game clock
clock = pygame.time.Clock()

# Check if snake collides
def collide(x1, x2, y1, y2, w1, w2, h1, h2):
    if x1+w1>x2 and x1<x2+w2 and y1+h1>y2 and y1<y2+h2:
        return True
    else:
        return False

# Dies
def die(screen, score):
    f=pygame.font.SysFont('Arial', 30)
    t=f.render('Your score was: '+str(score), True, (0, 0, 0))
    screen.blit(t, (10, 270))
    pygame.display.update()
    pygame.time.wait(2000)
    sys.exit(0)

# Resets screen
def reset():
    global xs, ys, dirs, score, applepos
    xs = [290, 290, 290, 290, 290]
    ys = [290, 280, 270, 260, 250]
    dirs = 0
    score = 0
    applepos = (random.randint(0, 590), random.randint(0, 590))

# Game loop
while True:
    clock.tick(10)
    # Key handler event
    for e in pygame.event.get():
        if e.type == QUIT:
            sys.exit(0)
        elif e.type == KEYDOWN:
            if e.key == K_UP and dirs != 0:dirs = 2
            elif e.key == K_DOWN and dirs != 2:dirs = 0
            elif e.key == K_LEFT and dirs != 1:dirs = 3
            elif e.key == K_RIGHT and dirs != 3:dirs = 1

    # Check if snake collides with itself
    i = len(xs)-1
    while i >= 2:
        if collide(xs[0], xs[i], ys[0], ys[i], 10, 10, 10, 10):
            reset()
        i-= 1

    # Check if snake eats apple
    if collide(xs[0], applepos[0], ys[0], applepos[1], 10, 10, 10, 10):
            score+=1
            xs.append(700)
            ys.append(700)
            applepos=(random.randint(0,29)*20,random.randint(0,29)*20)

    # Exceed boundaries
    if xs[0] < 0 or xs[0] > 580 or ys[0] < 0 or ys[0] > 580: reset()

    # Move snake
    i = len(xs)-1
    while i >= 1:
        xs[i] = xs[i-1];ys[i] = ys[i-1];i -= 1
    if dirs==0:ys[0] += 10
    elif dirs==1:xs[0] += 10
    elif dirs==2:ys[0] -= 10
    elif dirs==3:xs[0] -= 10	

    # Renders images
    s.fill((255, 255, 255))	
    for i in range(0, len(xs)):
        s.blit(img, (xs[i], ys[i]))
    s.blit(appleimage, applepos);t=f.render(str(score), True, (0, 0, 0));s.blit(t, (10, 10));pygame.display.update()

