import math

import torch
import random
import numpy
import pygame
from direction import Direction
from game import Game, Point, BLOCK_SIZE
from model import LinearQnet, QTrainer

from collections import deque

MAX_MEMORY = 100_000
BATCH_SIZE = 100_000
LR = 0.002


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model: LinearQnet = LinearQnet(11, 256, 3)
        self.trainer: QTrainer = QTrainer(self.model, learning_rate=LR, gamma=self.gamma)

    def get_state(self, game: Game):
        head = game.head
        pt_left = Point(head.x - BLOCK_SIZE, head.y)
        pt_right = Point(head.x + BLOCK_SIZE, head.y)
        pt_up = Point(head.x, head.y - BLOCK_SIZE)
        pt_down = Point(head.x, head.y + BLOCK_SIZE)

        dxleft = game.direction == Direction.Left
        dxright = game.direction == Direction.Right
        dxup = game.direction == Direction.Up
        dxdown = game.direction == Direction.Down

        state = [
            # danger straight
            (dxright and game.is_collision(pt_right)) or
            (dxleft and game.is_collision(pt_left)) or
            (dxup and game.is_collision(pt_up)) or
            (dxdown and game.is_collision(pt_down)),

            # danger right
            (dxright and game.is_collision(pt_down)) or
            (dxdown and game.is_collision(pt_left)) or
            (dxleft and game.is_collision(pt_up)) or
            (dxup and game.is_collision(pt_right)),

            # danger left
            (dxright and game.is_collision(pt_up)) or
            (dxup and game.is_collision(pt_left)) or
            (dxleft and game.is_collision(pt_down)) or
            (dxdown and game.is_collision(pt_right)),

            # Move Direction
            dxleft,
            dxright,
            dxup,
            dxdown,

            # Food Location
            game.food.x < game.head.x,  # left
            game.food.x > game.head.x,  # right
            game.food.y < game.head.y,  # up
            game.food.y > game.head.y,  # down
        ]

        return numpy.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            nums = [0,0,1,2]
            move = random.randint(0, 3)
            final_move[nums[move]] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return  final_move

def train(game: Game):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    agent = Agent()
    record = 45
    highest = 90
    agent.n_games = 100

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and game.direction is not Direction.Right:
                    game.direction = Direction.Left
                elif event.key == pygame.K_RIGHT and game.direction is not Direction.Left:
                    game.direction = Direction.Right
                elif event.key == pygame.K_UP and game.direction is not Direction.Down:
                    game.direction = Direction.Up
                elif event.key == pygame.K_DOWN and game.direction is not Direction.Up:
                    game.direction = Direction.Down

        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        game_over, score, reward = game.game_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        #agent.train_short_memory(state_old, final_move, reward, state_new, game_over)

        # remember
        agent.remember(state_old, final_move, reward, state_new, game_over)
        # agent.train_long_memory()

        if game_over:
            iterations = game.frame_iteration
            game.reset()
            agent.n_games += 1



            if agent.n_games > 100 and (score == 0 or score == 1):
                print("skip training")
                agent.n_games /= 3
            else:
                skipped = False
                if score == 0:
                    score = 1
                    skipped = True
                agent.n_games -= ((record/score) // math.sqrt(agent.n_games)) - 20
                # agent.train_long_memory()
                if skipped:
                    score = 0

            if score > record:
                record = score
            if score > record / 2:
                agent.train_long_memory()
            if(record > highest):
                for i in range(len(agent.memory) // 2):
                    agent.memory.pop()
                highest = record
                agent.model.save()
                print("Model Saved: ", highest)

            if score <= record/3:
                for i in range(iterations//2):
                    agent.memory.pop()

            print(" Game", agent.n_games, " Score", score, " Record", record, "Highest", highest, record / 3, score<= record/3, len(agent.memory))
