import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from agent_dir.agent import Agent
from environment import Environment

#import matplotlib.pyplot as plt

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        action_prob = F.softmax(x, dim=1)
        return action_prob

class AgentPG(Agent):
    def __init__(self, env, args):
        self.env = env
        self.model = PolicyNet(state_dim = self.env.observation_space.shape[0],
                               action_num= self.env.action_space.n,
                               hidden_dim=64)
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        if args.test_pg:
            self.load('pg.cpt')

        # discounted reward
        self.gamma = 0.99

        # training hyperparameters
        self.num_episodes = 10000#100000 # total training episodes (actually too large...)
        self.display_freq = 10 # frequency to display training progress

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)

        # saved rewards and actions
        self.rewards, self.saved_actions = [], []
        self.action_log_prob = []
        self.model.to(self.device)


    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        print('load model from', load_path)
        self.model.load_state_dict(torch.load(load_path, map_location=torch.device(self.device)))

    def init_game_setting(self):
        self.rewards, self.saved_actions, self.action_log_prob = [], [], []

    def make_action(self, state, test=False):
        #action = self.env.action_space.sample() # TODO: Replace this line!
        # Use your model to output distribution over actions and sample from it.
        # HINT: torch.distributions.Categorical
        #if test:
        #    self.model.train()
        #else:
        self.model.eval()
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        prob = torch.distributions.Categorical(self.model(state))
        action = prob.sample()
        self.action_log_prob.append(prob.log_prob(action))
        return action.item()

    def update(self):
        # TODO:
        # discount reward
        # R_i = r_i + GAMMA * R_{i+1}
        self.model.train()
        total = len(self.rewards)
		#print(self.rewards)
        R = [0 for i in range(total+1)]
        for i, r in enumerate(reversed(self.rewards)):
            R[total-i-1] = r + self.gamma * R[total-i]
        R.pop()
        R = torch.tensor(R).to(self.device)
        R = (R-R.mean())/(R.std())
        # TODO:
        # compute PG loss
        # loss = sum(-R_i * log(action_prob))
        self.optimizer.zero_grad()
        prob = torch.cat(self.action_log_prob).to(self.device)
        loss = (-R*prob).sum()
        loss.backward()
        self.optimizer.step()

    def train(self):
        avg_reward = None
        ever_best = 0
        total_r = 0
        plt_reward = []
        plt_epoch = []
        for epoch in range(self.num_episodes):
            state = self.env.reset()
            self.init_game_setting()
            done = False
            while(not done):
                action = self.make_action(state)
                state, reward, done, _ = self.env.step(action)

                self.saved_actions.append(action)
                self.rewards.append(reward)
                total_r += reward

            # update model
            self.update()
            # for logging
            last_reward = np.sum(self.rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1
            if epoch > 0 and epoch % 10 == 0:
                plt_reward.append(total_r/10)
                plt_epoch.append(epoch)
                total_r = 0
            if epoch % self.display_freq == 0:
                print('Epochs: %d/%d | Avg reward: %f '%
                       (epoch, self.num_episodes, avg_reward))
            #if epoch > 0 and epoch % 100 == 0:
                #plt.plot(plt_epoch, plt_reward)
                #plt.savefig("pg.png")
            if avg_reward > ever_best:
                ever_best = avg_reward
                print("Epoch %d, best reward ever %f, saving model"%(epoch, avg_reward))
				#self.save('pg.cpt')
        #plt.plot(plt_epoch, plt_reward)
        #plt.savefig("pg.png")
			#if avg_reward > 50: # to pass baseline, avg. reward > 50 is enough.
			#    self.save('pg.cpt')
			#    break
