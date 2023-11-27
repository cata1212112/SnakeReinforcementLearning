from imports import *


class SnakeEnv:
    def __init__(self, frame_x=720, frame_y=480, render=False):
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
        self.block_size = 10
        self.render = render
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

    def step(self, action):
        self._move(action)

        self.snake_body.insert(0, list(self.snake_pos))

        reward = -1

        if self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]:
            self.score += 1
            reward = 10
            self.food_pos = self._generate_food_pos()
        else:
            self.snake_body.pop()

        if self.render:
            self.game_window.fill(pygame.Color(0, 0, 0))
            for pos in self.snake_body:
                pygame.draw.rect(self.game_window, pygame.Color(255, 255, 255),
                                 pygame.Rect(pos[0], pos[1], self.block_size, self.block_size))

            pygame.draw.rect(self.game_window, pygame.Color(255, 0, 0),
                             pygame.Rect(self.food_pos[0], self.food_pos[1], self.block_size, self.block_size))

            pygame.display.update()
            self.fps_controller.tick(1)

        if self.snake_pos[0] < 0 or self.snake_pos[0] > self.frame_X - self.block_size:
            return True, self._end_game(), -10

        if self.snake_pos[1] < 0 or self.snake_pos[1] > self.frame_Y - self.block_size:
            return True, self._end_game(), -10

        for block in self.snake_body[1:]:
            if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                return True, self._end_game(), -10

        return False, self.score, reward

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
