import numpy as np

from imports import *

FPS = 20


class SnakeEnv:
    def __init__(self, frame_x=256, frame_y=256, render=False):
        self.change_to = None
        self.direction = None
        self.score = None
        self.snake_body = None
        self.fps_controller = None
        self.snake_pos = None
        self.food_pos = None
        self.game_window = None
        self.frame_X = frame_x
        self.frame_Y = frame_y
        self.block_size = 16
        self.render = render
        self.action_space_size = 4
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.action_dic = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3}
        self.no_eat = 0
        self.resize = (50, 50)
        self.reset()

    def _generate_food_pos(self):
        while True:
            food_pos = [np.random.randint(0, self.frame_X // self.block_size) * self.block_size,
                        np.random.randint(0, self.frame_Y // self.block_size) * self.block_size]

            if food_pos not in self.snake_body:
                return food_pos

    def _end_game(self):
        if self.render:
            pygame.quit()
        return self.score

    def reset(self):
        self.snake_pos = [self.frame_X // 2, self.frame_Y // 2]
        self.snake_body = [[self.frame_X // 2, self.frame_Y // 2 - i * self.block_size] for i in range(3)]
        self.score = 0

        self.change_to = None
        self.direction = 'DOWN'
        self.food_pos = self._generate_food_pos()

        if self.render:
            pygame.init()
            self.fps_controller = pygame.time.Clock()
            pygame.display.set_caption('SNAKE')
            self.game_window = pygame.display.set_mode((self.frame_X, self.frame_Y))

        return self.get_image()

    def step(self, action):
        before = (self.snake_pos[0], self.snake_pos[1])
        self._move(action)
        self.no_eat += 1

        self.snake_body.insert(0, list(self.snake_pos))
        reward = 0

        if self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]:
            self.score += 1
            self.no_eat = 0
            reward = 30
            self.food_pos = self._generate_food_pos()
        else:
            self.snake_body.pop()

        if self.snake_pos[0] < 0 or self.snake_pos[0] > self.frame_X - self.block_size:
            return True, self._end_game(), -15, self.get_image()

        if self.snake_pos[1] < 0 or self.snake_pos[1] > self.frame_Y - self.block_size:
            return True, self._end_game(), -15, self.get_image()

        for block in self.snake_body[1:]:
            if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                return True, self._end_game(), -15, self.get_image()

        if self.render:
            self.game_window.fill(pygame.Color(0, 0, 0))
            for pos in self.snake_body:
                pygame.draw.rect(self.game_window, pygame.Color(255, 255, 255),
                                 pygame.Rect(pos[0], pos[1], self.block_size, self.block_size))

            pygame.draw.rect(self.game_window, pygame.Color(255, 0, 0),
                             pygame.Rect(self.food_pos[0], self.food_pos[1], self.block_size, self.block_size))

            pygame.event.get()
            pygame.display.update()
            self.fps_controller.tick(FPS)

        if reward == 0:
            reward = 30 * np.log(len(self.snake_body) + np.sqrt(
                (before[0] - self.food_pos[0]) ** 2 + (before[1] - self.food_pos[1]) ** 2)) / np.log(
                len(self.snake_body) + np.sqrt(
                    (self.snake_pos[0] - self.food_pos[0]) ** 2 + (self.snake_pos[1] - self.food_pos[1]) ** 2))

            if self.no_eat > 10:
                reward += - 20 / len(self.snake_body)

        return False, self.score, reward, self.get_image()

    def _move(self, action):
        if action == 'UP' and self.direction != 'DOWN':
            self.direction = 'UP'
        if action == 'DOWN' and self.direction != 'UP':
            self.direction = 'DOWN'
        if action == 'LEFT' and self.direction != 'RIGHT':
            self.direction = 'LEFT'
        if action == 'RIGHT' and self.direction != 'LEFT':
            self.direction = 'RIGHT'

        if self.direction == 'UP':
            self.snake_pos[1] -= self.block_size
        if self.direction == 'DOWN':
            self.snake_pos[1] += self.block_size
        if self.direction == 'LEFT':
            self.snake_pos[0] -= self.block_size
        if self.direction == 'RIGHT':
            self.snake_pos[0] += self.block_size

    def get_image(self):
        img = np.zeros((self.frame_Y, self.frame_X, 3), dtype=np.uint8)
        for pos in self.snake_body:
            img[pos[1]:pos[1] + self.block_size, pos[0]:pos[0] + self.block_size, :] = 255

        img[self.food_pos[1]:self.food_pos[1] + self.block_size, self.food_pos[0]:self.food_pos[0] + self.block_size,
        :] = np.array([0, 0, 255])

        img = cv.resize(img, self.resize, interpolation=cv.INTER_AREA)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img

    def sample_action(self):
        return np.random.choice(self.actions)
