import numpy as np

from imports import *

FPS = 15


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
        self.action_space_size = 3
        self.actions = ['LEFT', 'RIGHT', 'FORWARD']
        self.action_dic = {'LEFT': 0, 'RIGHT': 1, 'FORWARD': 2}

        self.move_left = {'UP':'LEFT', 'LEFT':'DOWN', 'DOWN':'RIGHT', 'RIGHT':'UP'}
        self.move_right = {'UP':'RIGHT', 'RIGHT':'DOWN', 'DOWN':'LEFT', 'LEFT':'UP'}

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
            reward = 1
            self.food_pos = self._generate_food_pos()
        else:
            self.snake_body.pop()

        if self.snake_pos[0] < 0 or self.snake_pos[0] > self.frame_X - self.block_size:
            return True, self._end_game(), -1, self.get_image()

        if self.snake_pos[1] < 0 or self.snake_pos[1] > self.frame_Y - self.block_size:
            return True, self._end_game(), -1, self.get_image()

        for block in self.snake_body[1:]:
            if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                return True, self._end_game(), -1.5, self.get_image()

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
            dist_before = np.linalg.norm([before[0] - self.food_pos[0], before[1] - self.food_pos[1]])
            dist_now = np.linalg.norm([self.snake_pos[0] - self.food_pos[0], self.snake_pos[1] - self.food_pos[1]])
            if dist_now > dist_before:
                reward = - 1 / max(1, dist_now)
            elif dist_now < dist_before:
                reward = 1 / max(1, dist_now)
            # else:
            #     reward = -0.01
            # if self.no_eat > 20:
            #     reward = -1 / (10 + len(self.snake_body))


        return False, self.score, reward, self.get_image()

    def _move(self, action):
        if action == 'LEFT':
            self.direction = self.move_left[self.direction]
        if action == 'RIGHT':
            self.direction = self.move_right[self.direction]
        if action == 'FORWARD':
            pass

        if self.direction == 'UP':
            self.snake_pos[1] -= self.block_size
        if self.direction == 'DOWN':
            self.snake_pos[1] += self.block_size
        if self.direction == 'LEFT':
            self.snake_pos[0] -= self.block_size
        if self.direction == 'RIGHT':
            self.snake_pos[0] += self.block_size

    def _check_n_danger(self, n, direction):
        if direction == 'UP':
            pos = (self.snake_pos[1] - n * self.block_size, self.snake_pos[0])
            return pos[0] < 0 or any([pos[0] == block[0] and pos[1] == block[1] for block in self.snake_body])
        if direction == 'DOWN':
            pos = (self.snake_pos[1] + n * self.block_size, self.snake_pos[0])
            return pos[0] > self.frame_X // self.block_size or any([pos[0] == block[0] and pos[1] == block[1] for block in self.snake_body])
        if direction == 'LEFT':
            pos = (self.snake_pos[1], self.snake_pos[0] - n * self.block_size)
            return pos[1] < 0 or any([pos[0] == block[0] and pos[1] == block[1] for block in self.snake_body])
        if direction == 'RIGHT':
            pos = (self.snake_pos[1], self.snake_pos[0] + n * self.block_size)
            return pos[1] > self.frame_X // self.block_size or any([pos[0] == block[0] and pos[1] == block[1] for block in self.snake_body])

    def _check_n_danger_front(self, n):
        return self._check_n_danger(n, self.direction)

    def _check_n_danger_right(self, n):
        return self._check_n_danger(n, self.move_right[self.direction])

    def _check_n_danger_left(self, n):
        return self._check_n_danger(n, self.move_left[self.direction])

    def get_image(self):
        # img = np.zeros((self.frame_Y // self.block_size + 1, self.frame_X // self.block_size, 3), dtype=np.uint8)
        # for pos in self.snake_body:
        #     if 0 <= pos[1] // self.block_size < self.frame_X // self.block_size and 0 <= pos[0] // self.block_size < self.frame_X // self.block_size:
        #         img[pos[1] // self.block_size, pos[0] // self.block_size, :] = 255
        #
        # img[self.food_pos[1] // self.block_size, self.food_pos[0] // self.block_size, :] = np.array([0, 0, 255])
        #
        #
        # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #
        # img[self.frame_Y // self.block_size, 0] = self.snake_pos[0] // self.block_size
        # img[self.frame_Y // self.block_size, 1] = self.snake_pos[1] // self.block_size
        # return img

        output_state = [
            self._check_n_danger_front(1),
            self._check_n_danger_right(1),
            self._check_n_danger_left(1),
            self._check_n_danger_front(2),
            self._check_n_danger_right(2),
            self._check_n_danger_left(2),
            self._check_n_danger_front(3),
            self._check_n_danger_right(3),
            self._check_n_danger_left(3),
            self.direction == 'UP',
            self.direction == 'DOWN',
            self.direction == 'LEFT',
            self.direction == 'RIGHT',
            (self.food_pos[0] <= self.snake_pos[0]),
            (self.food_pos[0] > self.snake_pos[0]),
            (self.food_pos[1] <= self.snake_pos[1]),
            (self.food_pos[1] > self.snake_pos[1])
        ]
        return output_state

    def _has_space(self):
        if not (0 <= self.snake_pos[1] < self.block_size < self.frame_X // self.block_size and 0 <= self.snake_pos[
            0] < self.block_size < self.frame_X // self.block_size):
            return 0
        board = self.create_board()
        board[self.snake_pos[0] // self.block_size, self.snake_pos[1] // self.block_size] = 0
        def DFS(pos):
            if not (0 <= pos[1] < self.block_size < self.frame_X // self.block_size and 0 <= pos[0] < self.block_size < self.frame_X // self.block_size):
                return 0
            if board[pos[0], pos[1]] == 1:
                return 0
            sum = board[pos[0], pos[1]]
            return sum + DFS((pos[0]+1, pos[1])) + DFS((pos[0]-1, pos[1])) + DFS((pos[0], pos[1] + 1)) + DFS((pos[0], pos[1] - 1))

        return DFS((self.snake_pos[0] // self.block_size, self.snake_pos[1] // self.block_size)) - 1 > len(self.snake_body)


    def create_board(self):
        img = np.zeros((self.frame_Y // self.block_size, self.frame_X // self.block_size), dtype=np.uint8)
        for pos in self.snake_body:
            if 0 <= pos[1] // self.block_size < self.frame_X // self.block_size and 0 <= pos[0] // self.block_size < self.frame_X // self.block_size:
                img[pos[1] // self.block_size, pos[0] // self.block_size] = 1

        img[self.food_pos[1] // self.block_size, self.food_pos[0] // self.block_size] = 0

        return img

    def get_image_gif(self):
        img = np.zeros((self.frame_Y, self.frame_X, 3), dtype=np.uint8)
        for pos in self.snake_body:
            if 0 <= pos[1] // self.block_size < self.frame_X // self.block_size and 0 <= pos[0] // self.block_size < self.frame_X // self.block_size:
                img[pos[1]: pos[1] + self.block_size, pos[0]:pos[0] + self.block_size, :] = 255

        img[self.food_pos[1]:self.food_pos[1] + self.block_size, self.food_pos[0]: self.food_pos[0] + self.block_size, :] = np.array([255, 0, 0])

        return img

    def sample_action(self):
        return np.random.choice(self.actions)
