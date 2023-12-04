import torch

from imports import *
from DQN import *
from nStepEpisodes import *


class Agent:
    STATE_IMAGE_REPREZENTATION = 0
    GAMMA = 0.99
    BATCH_SIZE = 512
    REPLAY_SIZE = 300000
    REPLAY_START_SIZE = 75000
    LEARNING_RATE = 0.001
    SYNC_TARGET = 10000
    EPSILON_START = 1.0
    EPSILON_FINAL = 0
    EPSILON_DECAY = 300000
    ACTUAL_EPSILON_DECAY = 0
    STEPS = 1000000
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env):
        self.replay_memory = collections.deque(maxlen=Agent.REPLAY_SIZE)
        self.env = env
        self.network = DQN(env.resize).to(Agent.DEVICE)

        print("Numar de parametrii: ", sum(p.numel() for p in self.network.parameters() if p.requires_grad))

        self.target_network = DQN(env.resize).to(Agent.DEVICE)

        # self.network.load_state_dict(torch.load("best/best.dat"))
        # self.target_network.load_state_dict(torch.load("best/best.dat"))
        self.episode_steps = Episodes()

        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, epsilon=0.0):
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.sample_action()
        else:
            if Agent.STATE_IMAGE_REPREZENTATION:
                state_tensor = transforms.ToTensor()(self.state).to(Agent.DEVICE).unsqueeze(0)
            else:
                state_tensor = torch.tensor(self.state, dtype=torch.float32).to(Agent.DEVICE).unsqueeze(0)

            q_vals = self.network(state_tensor)
            _, action_max = torch.max(q_vals, dim=1)
            action = int(action_max.item())
            action = self.env.actions[action]

        is_done, score, reward, new_state = self.env.step(action)
        self.total_reward += reward

        # self.episode_steps.append(self.state, action, reward, is_done, new_state)
        # if self.episode_steps.completed():
        #     self.episode_steps.rollout(Agent.GAMMA)
        #     self.replay_memory.append([self.episode_steps.state[0], self.env.action_dic[self.episode_steps.state[1]]
        #                                , self.episode_steps.state[2], self.episode_steps.state[3],
        #                                self.episode_steps.state[4]])
        #     self.episode_steps = Episodes()

        self.replay_memory.append([self.state, self.env.action_dic[action], reward, is_done, new_state])
        self.state = new_state

        if is_done:
            done_reward = self.total_reward
            self._reset()

        return done_reward, score

    def loss(self, batch):
        states, actions, rewards, dones, next_states = batch

        if Agent.STATE_IMAGE_REPREZENTATION:
            states_tensor = torch.stack([transforms.ToTensor()(img) for img in states]).to(Agent.DEVICE)
        else:
            states_tensor = torch.stack([torch.tensor(img, dtype=torch.float32) for img in states]).to(Agent.DEVICE)

        actions_tensor = torch.tensor(actions, dtype=torch.int64).to(Agent.DEVICE)
        rewards_tensor = torch.tensor(rewards).to(Agent.DEVICE)
        dones_tensor = torch.ByteTensor(dones).to(Agent.DEVICE)
        dones_tensor = dones_tensor.bool()

        if Agent.STATE_IMAGE_REPREZENTATION:
            next_states_tensor = torch.stack([transforms.ToTensor()(img) for img in next_states]).to(Agent.DEVICE)
        else:
            next_states_tensor = torch.stack([torch.tensor(img, dtype=torch.float32) for img in next_states]).to(Agent.DEVICE)


        state_action_values = self.network(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)

        next_state_values = self.target_network(next_states_tensor)
        next_state_values = next_state_values.max(1)[0]

        # next_state_actions = self.network(next_states_tensor).max(1)[1]
        # next_state_values = self.target_network(next_states_tensor).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
        #
        # next_state_actions.detach()
        # state_action_values = self.network(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)

        next_state_values.masked_fill_(dones_tensor, -1)
        next_state_values = next_state_values.detach()

        expected_state_action_values = rewards_tensor + Agent.GAMMA * next_state_values

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def _generate_random_alphanumeric_tag(self, length=10):
        characters = string.ascii_letters + string.digits
        tag = ''.join(secrets.choice(characters) for _ in range(length))
        return tag

    def _sample_batch(self):
        indeces = np.random.choice(len(self.replay_memory), Agent.BATCH_SIZE, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.replay_memory[i] for i in indeces])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones,
                                                                                                  dtype=np.uint8), np.array(
            next_states)

    def learn(self):
        epsilon = Agent.EPSILON_START
        optimizer = torch.optim.Adam(self.network.parameters(), lr=Agent.LEARNING_RATE)

        total_rewards = []
        frames = 0
        best_mean_reward = None
        writer = SummaryWriter()
        scores = []

        progress_bar = tqdm(total=Agent.STEPS, desc='Training Progress')

        while True:
            progress_bar.update(1)
            if frames > Agent.STEPS:
                return
            frames += 1
            epsilon = max(Agent.EPSILON_FINAL, Agent.EPSILON_START - frames / Agent.EPSILON_DECAY)

            reward, score = self.play_step(epsilon)
            scores.append(score)

            if reward is not None:
                total_rewards.append(reward)
                mewn_reward = np.mean(total_rewards[-100:])
                writer.add_scalar("epsilon", epsilon, frames)
                writer.add_scalar("reward_last_100", mewn_reward, frames)
                writer.add_scalar("reward", reward, frames)
                writer.add_scalar("score", score, frames)

                if best_mean_reward is None or best_mean_reward < mewn_reward:
                    torch.save(self.network.state_dict(), f"best/best.dat")
                    best_mean_reward = mewn_reward

            if len(self.replay_memory) < Agent.REPLAY_START_SIZE:
                continue

            if frames % Agent.SYNC_TARGET == 0:
                self.target_network.load_state_dict(self.network.state_dict())

            optimizer.zero_grad()
            batch = self._sample_batch()
            loss_ = self.loss(batch)
            loss_.backward()
            optimizer.step()

        progress_bar.close()

    def play(self):
        recording = []
        with torch.no_grad():
            self.network.load_state_dict(torch.load("best/best.dat"))
            is_done = True
            while is_done:
                recording.append(self.env.get_image_gif())

                if Agent.STATE_IMAGE_REPREZENTATION:
                    state_tensor = transforms.ToTensor()(self.state).to(Agent.DEVICE).unsqueeze(0)
                else:
                    state_tensor = torch.tensor(self.state, dtype=torch.float32).to(Agent.DEVICE).unsqueeze(0)

                q_vals = self.network(state_tensor)
                _, action_max = torch.max(q_vals, dim=1)
                action = int(action_max.item())
                action = self.env.actions[action]
                is_done, score, reward, new_state = self.env.step(action)
                is_done = not is_done
                self.state = new_state
        imageio.mimsave("game.gif", recording)
