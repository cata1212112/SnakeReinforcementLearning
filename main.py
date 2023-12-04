from SnakeEnvironment import *
from Agent import *

# torch.autograd.set_detect_anomaly(True)
# a = SnakeEnv(render=False)
#
# agent = Agent(a)
# agent.learn()

a = SnakeEnv(render=True)

agent = Agent(a)
agent.play()