from collections import namedtuple

import pygame
from colors import Colors

DIMENSIONS = (640, 480)

pygame.init()
pygame.display.set_mode(DIMENSIONS)
pygame.display.set_caption("Snake PyGame")
# font = pygame.font.Font('aria.ttf', 25)

blackColor = Colors.Black.value



running = True


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()

pygame.quit()