import random
import pygame
from direction import Direction
from collections import namedtuple
from colors import Colors
import numpy
import math

BLOCK_SIZE = 20
SPEED = 20000
LIMIT = 100

Point = namedtuple("Point", "x, y")


class Game:
    def __init__(self, display: pygame.display, pg: pygame, font: pygame.font, Dimensions: tuple):
        self.pg = pg
        self.display = display
        self.font = font
        self.clock = pygame.time.Clock()
        self.w = Dimensions[0]
        self.h = Dimensions[1]

        self.reset()

    def reset(self):
        self.direction = Direction.Right
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [  # The snake with a size of three (3) starting from the head
            self.head,
            Point(self.head.x - (BLOCK_SIZE), self.head.y),  # 1st Tail
            Point(self.head.x - (BLOCK_SIZE * 2), self.head.y),  # 2nd Tail
        ]
        self.score = 0
        self.food = None
        self.place_food()
        self.frame_iteration = 0

    def place_food(self):
        # floor division (//) prevents the point from going out of bounds with the screen
        randx = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        randy = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE

        self.food = Point(randx, randy)
        if self.food in self.snake:
            self.place_food()

    def game_step(self, action):
        self.frame_iteration += 1
        self._move(action)
        self.snake.insert(0, self.head)

        game_over = False
        if self.is_collision() or self.frame_iteration > (LIMIT * len(self.snake)):
            game_over = True
            return game_over, self.score, -100


        reward = 0
        if self.head == self.food:
            #self.frame_iteration = 0
            self.score += 1
            reward += 15 + self.score
            self.place_food()
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(SPEED)

        return game_over, self.score, reward

    def _move(self, action=None):
        straight_turn = [1, 0, 0]
        right_turn = [0, 1, 0]
        left_turn = [0, 0, 1]
        if action is not None:
            if numpy.array_equal(action, right_turn):
                if self.direction == Direction.Right:
                    self.direction = Direction.Down
                elif self.direction == Direction.Left:
                    self.direction = Direction.Up
                elif self.direction == Direction.Down:
                    self.direction = Direction.Left
                elif self.direction == Direction.Up:
                    self.direction = Direction.Right
            elif numpy.array_equal(action, left_turn):
                if self.direction == Direction.Right:
                    self.direction = Direction.Up
                elif self.direction == Direction.Left:
                    self.direction = Direction.Down
                elif self.direction == Direction.Down:
                    self.direction = Direction.Right
                elif self.direction == Direction.Up:
                    self.direction = Direction.Left

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.Right:
            x += BLOCK_SIZE
        elif self.direction == Direction.Left:
            x -= BLOCK_SIZE

        # y-axis of pygame starts from the top to bottom e.g. (0 - 255) so to move up, we reduce the y-axis coordinate
        elif self.direction == Direction.Down:
            y += BLOCK_SIZE
        elif self.direction == Direction.Up:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y < 0 or pt.y > self.h - BLOCK_SIZE:
            return True
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(Colors.White.value)

        for pt in self.snake:
            pygame.draw.rect(self.display, Colors.Blue1.value, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, Colors.Blue2.value,
                             pygame.Rect(pt.x + 4, pt.y + 4, BLOCK_SIZE - (4 * 2), BLOCK_SIZE - (4 * 2)))

        pygame.draw.rect(self.display, Colors.White.value,
                         pygame.Rect(self.head.x + 6, self.head.y + 6, BLOCK_SIZE - (6 * 2), BLOCK_SIZE - (6 * 2)))
        pygame.draw.rect(self.display, Colors.Red.value, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = self.font.render("Score: " + str(self.score) + " | Frame: " +
                                str(self.frame_iteration) + "/" + str(LIMIT * len(self.snake)+LIMIT),
                                True, Colors.Black.value)
        self.display.blit(text, [0, 0])
        self.pg.display.flip()
