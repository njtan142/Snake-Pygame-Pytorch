import random

import pygame.time

from direction import Direction
from collections import namedtuple

BLOCKSIZE = 20
SPEED = 20

Point = namedtuple("Point", "x, y")


class Game():
    def __init__(self):
        self.food = None
        self.score = None
        self.snake = None
        self.pygame = None
        self.direction = None

    def __int__(self, pygame, Dimensions: tuple):
        self.pygame = pygame
        self.w = Dimensions[0]
        self.h = Dimensions[1]
        self.direction = Direction.Right

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [  # The snake with a size of three (3) starting from the head
            self.head,
            Point(self.head.x - (BLOCKSIZE), self.head.y),  # 1st Tail
            Point(self.head.x - (BLOCKSIZE * 2), self.head.y),  # 2nd Tail
        ]

        self.place_food()

        self.clock = pygame.time.Clock

    def place_food(self):
        # floor division (//) prevents the point from going out of bounds with the screen
        randx = random.randint(0, (self.w - BLOCKSIZE) // BLOCKSIZE) * BLOCKSIZE
        randy = random.randint(0, (self.w - BLOCKSIZE) // BLOCKSIZE) * BLOCKSIZE

        self.food = Point(randx, randy)
        if self.food in self.snake:
            self.place_food()

    def game_step(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.Left
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.Right
                elif event.key == pygame.K_UP:
                    self.direction = Direction.Up
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.Down

        self._move()
        self.snake.insert(0, self.head)

        gameover = False


        return gameover, self.score



    def _move(self):
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.Right:
            x += BLOCKSIZE
        elif self.direction == Direction.Left:
            x -= BLOCKSIZE

        # y-axis of pygame starts from the top to bottom e.g. (0 - 255) so to move up, we reduce the y-axis coordinate
        elif self.direction == Direction.Down:
            y += BLOCKSIZE
        elif self.direction == Direction.Up:
            y -= BLOCKSIZE

        self.head = Point(x, y)


    def _is_collission(self):
        if self.head.x > self.w - BLOCKSIZE or self.head.x < 0 or self.head.y < 0 or self.head.y < self.h - BLOCKSIZE:
            return True
        if self.head in self.snake[1:]:
            return True

        return False
