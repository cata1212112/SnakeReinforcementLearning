class Episodes:

    def __init__(self, steps = 2):
        self.steps = steps
        self.states = []

        self.state = None

    def append(self, state, action, reward, done, next_state):
        self.states.append([state, action, reward, done, next_state])

    def rool_out_reward(self, gamma):
        total_reward = 0
        for step in reversed(self.states):
            total_reward += step[2] + total_reward * gamma

        return total_reward

    def rollout(self, gamma):
        reward = self.rool_out_reward(gamma)
        self.collapse_steps(reward)

    def completed(self):
        if self.states[-1][3]:
            return True

        return len(self.states) == self.states

    def collapse_steps(self, reward):
        self.state = (self.states[0][0], self.states[0][1], reward, self.states[-1][3], self.states[-1][4])

    def __len__(self):
        return len(self.states)

