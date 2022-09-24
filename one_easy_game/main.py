import gym
import torch
from torch import nn, optim
import numpy as np
import random
import matplotlib.pyplot as plt

# show a demo
def test():
    env = gym.make("CartPole-v1")
    print("Observation shape:", env.observation_space.shape)
    print("Number of actions:", env.action_space.n)
    for _ in range(20):
        observation = env.reset()  # init observation(state)
        for t in range(500):
            env.render()  # show screen of the game
            action = env.action_space.sample()  # choose a action randomly from actions
            observation, reward, done, info = env.step(action)
            print(observation, reward, done, info)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()


class QNet(nn.Sequential):
    def __init__(self):
        super(QNet, self).__init__(
            # nn.Linear(4, 256),
            nn.Linear(4, 64),
            nn.ReLU(),
            # nn.Linear(256, 128),
            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.Linear(128, 2)
            nn.Linear(32, 2)
        )

class CartPole_Game:
    def __init__(self, exp_pool_size, explore):
        self.env = gym.make("CartPole-v1")  # type: gym.Env
        self.exp_pool = []  # trajectory
        self.exp_pool_size = exp_pool_size
        self.q_net = QNet()
        self.explore = explore
        self.loss_function = nn.MSELoss()
        self.opt = optim.Adam(self.q_net.parameters())

    def __call__(self):
        is_render = False
        is_history = False
        count = 0
        R_history = []
        avg = 0
        while True:
            # collect train data
            observation = self.env.reset()
            R = 0  # total reward
            while True:
                if is_render:
                    self.env.render()
                if len(self.exp_pool) >= self.exp_pool_size:
                    self.exp_pool.pop(0)
                    self.explore += 1e-7   # 探索值，当此值较小时，将会去收集新鲜样本，当我们的样本越来越多时， 我们就应该逐渐减少探索。
                    if torch.rand(1) > self.explore:
                        action = self.env.action_space.sample()
                    else:
                        _observation = torch.tensor(observation, dtype=torch.float32)
                        Qs = self.q_net(_observation[None, ...])  # [None, ...] -> add a batch dimensionality
                        action = torch.argmax(Qs, 1)[0].item()

                else:
                    action = self.env.action_space.sample()

                next_observation, reward, done, _ = self.env.step(action)
                R += reward
                self.exp_pool.append([observation, reward, action, next_observation, done])  # save information
                observation = next_observation

                if done:
                    avg = 0.95 * avg + 0.05 * R  # moving averages
                    print(avg, R)
                    if avg > 400:
                        is_render = True
                        is_history = True
                    if is_history:
                        R_history.append(R)
                        if R == 500 or R < 300:
                            count += 1
                    break

            if count == 20:
                return R_history
                break

            # train the model
            if len(self.exp_pool) >= self.exp_pool_size:
                exps = random.choices(self.exp_pool, k=100)
                _observation = torch.tensor([exp[0] for exp in exps], dtype=torch.float32)
                _reward = torch.tensor([[exp[1]] for exp in exps])
                _action = torch.tensor([[exp[2]] for exp in exps])
                _next_observation = torch.tensor([exp[3] for exp in exps], dtype=torch.float32)
                _done = torch.tensor([[exp[4]] for exp in exps]).int()

                # predict
                _Qs = self.q_net(_observation)
                _Q = torch.gather(_Qs, 1, _action)

                # target
                _next_Qs = self.q_net(_next_observation)
                _max_Q = torch.max(_next_Qs, dim=1, keepdim=True)[0]
                _target_Q = _reward + (1 - _done) * 0.9 * _max_Q

                loss = self.loss_function(_Q, _target_Q.detach())
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()


if __name__ == "__main__":
    # test()
    g = CartPole_Game(10000, 0.9)
    R_history = g()
    fig = plt.figure()
    plt.plot(R_history)
    plt.show()












