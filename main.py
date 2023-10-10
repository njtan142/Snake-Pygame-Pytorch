import pygame
from colors import Colors
from game import Game
from direction import Direction
from agent import Agent, train

DIMENSIONS = (240 * 2*2, 240 * 2*2)

pygame.init()
display = pygame.display.set_mode(DIMENSIONS)
pygame.display.set_caption("Snake PyGame")
font = pygame.font.Font('arial.ttf', 25)

blackColor = Colors.Black.value

running = True

game = Game(display, pygame, font, DIMENSIONS)
train(game)
