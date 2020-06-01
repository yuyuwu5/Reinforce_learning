import random
import math
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from agent_dir.agent import Agent
from environment import Environment

#import matplotlib.pyplot as 

use_cuda = torch.cuda.is_available()


class DQN(nn.Module):
    '''
    This architecture is the one from OpenAI Baseline, with small modification.
    '''
    def __init__(self, channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Linear(512, num_actions)

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.lrelu(self.fc(x.view(x.size(0), -1)))
        q = self.head(x)
        return q


class AgentDQN(Agent):
    def __init__(self, env, args):
        self.env = env
        self.input_channels = 4
        self.num_actions = self.env.action_space.n

        # build target, online network
        self.target_net = DQN(self.input_channels, self.num_actions)
        self.target_net = self.target_net.cuda() if use_cuda else self.target_net
        self.online_net = DQN(self.input_channels, self.num_actions)
        self.online_net = self.online_net.cuda() if use_cuda else self.online_net

        if args.test_dqn:
            self.load('dqn')

        # discounted reward
        self.GAMMA = 0.99

        # training hyperparameters
        self.train_freq = 4 # frequency to train the online network
        self.learning_start = 10000 # before we start to update our network, we wait a few steps first to fill the replay.
        self.batch_size = 64
        self.num_timesteps = 3000000 # total training steps
        self.display_freq = 10 # frequency to display training progress
        self.save_freq = 20000 # frequency to save the model
        self.target_update_freq = 100 # frequency to update target network
        self.buffer_size = 12800 # max size of replay buffer

        # optimizer
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = optim.RMSprop(self.online_net.parameters(), lr=1e-4)

        self.steps = 0 # num. of passed steps
        self.epsilon = 0.9
        self.epsilon_decay = 100000
        self.epsilon_end = 0.05
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TODO: initialize your replay buffer
        self.buffer = []
        self.buffer_ptr = 0


    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.online_net.state_dict(), save_path + '_online.cpt')
        torch.save(self.target_net.state_dict(), save_path + '_target.cpt')

    def load(self, load_path):
        print('load model from', load_path)
        if use_cuda:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt'))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt'))
        else:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt', map_location=lambda storage, loc: storage))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt', map_location=lambda storage, loc: storage))

    def init_game_setting(self):
        # we don't need init_game_setting in DQN
        pass

    def make_action(self, state, test=False):
        # TODO:
        # Implement epsilon-greedy to decide whether you want to randomly select
        # an action or not.
        # HINT: You may need to use and self.steps
        if test:
            with torch.no_grad():
                state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0).to(self.device)
                bestAct = self.online_net(state).max(1)
                return bestAct[1].item()
        threshold = self.epsilon_end + (self.epsilon - self.epsilon_end) * math.exp(-1 * self.steps / self.epsilon_decay)
		#print("step %s threshold: %s" %(self.steps, threshold))
        if random.random() > threshold:
            with torch.no_grad():
                bestAct = self.online_net(state).max(1)
                return bestAct[1].item()
        else:
            return random.randrange(self.num_actions)

    def update(self):
        # TODO:
        if len(self.buffer) < self.batch_size:
            return
        # step 1: Sample some stored experiences as training examples.
        sample = random.sample(self.buffer, self.batch_size)
        aggregate_batch = list(zip(*sample))
        now_state = aggregate_batch[0]
        action = aggregate_batch[1]
        reward = aggregate_batch[2]
        next_state = aggregate_batch[3]
        not_finish = aggregate_batch[4]
        
        batch_now_state = torch.cat(now_state)
        batch_action = torch.cat(action)
        batch_reward = torch.cat(reward)
        batch_next_state = torch.cat(next_state)
        batch_not_finish = torch.cat(not_finish)
        # step 2: Compute Q(s_t, a) with your model.
		#print(batch_reward)
        Q_now = self.online_net(batch_now_state)
        Q_now = Q_now.gather(1, batch_action).squeeze(1)
        # step 3: Compute Q(s_{t+1}, a) with target model.
        Q_next = self.target_net(batch_next_state).detach().max(1)[0]
		#print(Q_next)
        # step 4: Compute the expected Q values: rewards + gamma * max(Q(s_{t+1}, a))
        ExpectQ = batch_reward + self.GAMMA * Q_next * batch_not_finish
		#print(ExpectQ)
        # step 5: Compute temporal difference loss
		#loss = (ExpectQ - Q_now)**2
		#loss = loss.mean()
        loss = F.smooth_l1_loss(Q_now, ExpectQ)
		#print("loss1", loss)
		#print(loss)
		#exit(0)
        # HINT:
        # 1. You should not backprop to the target model in step 3 (Use torch.no_grad)
        # 2. You should carefully deal with gamma * max(Q(s_{t+1}, a)) when it
        #    is the terminal state.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self):
        episodes_done_num = 0 # passed episodes
        total_reward = 0 # compute average reward
        loss = 0
        best_reward = 0
        plt_reward = []
        plt_epi = []
        while(True):
            state = self.env.reset()
            # State: (80,80,4) --> (1,4,80,80)
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0).to(self.device)

            done = False
            while(not done):
                # select and perform action
                action = self.make_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                # process new state
                next_state = torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0).to(self.device)
              
                # TODO: store the transition in memory
                action = torch.tensor([[action]], device=self.device)
                reward = torch.tensor([reward], device=self.device)
                if done:
                    not_finish = torch.tensor([0], device=self.device)
                else:
                    not_finish = torch.tensor([1], device=self.device)
                if len(self.buffer) < self.buffer_size:
                    self.buffer.append((state, action, reward, next_state, not_finish))
                self.buffer[self.buffer_ptr] = (state, action, reward, next_state, not_finish)
                self.buffer_ptr = (self.buffer_ptr+1)%self.buffer_size

                # move to the next state
                state = next_state

                # Perform one step of the optimization
                if self.steps > self.learning_start and self.steps % self.train_freq == 0:
                    loss = self.update()

                # TODO: update target network
                if self.steps > self.learning_start and self.steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())


                # save the model
                if self.steps % self.save_freq == 0:
                    self.save('dqn')
                self.steps += 1

            if episodes_done_num % self.display_freq == 0:
                plt_reward.append(total_reward/self.display_freq)
                plt_epi.append(episodes_done_num)
                if episodes_done_num % 100 == 0:
                    #plt.plot(plt_epi, plt_reward)
                    #plt.savefig('show.png')
                    with open('plt_reward_origin', 'wb') as f:
                        pickle.dump(plt_reward, f)
                    with open('plt_epi_origin', 'wb') as f:
                        pickle.dump(plt_epi, f)
                print('Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f '%
                        (episodes_done_num, self.steps, self.num_timesteps, total_reward / self.display_freq, loss))
				#if total_reward/self.display_freq > best_reward:
				#    best_reward = total_reward/self.display_freq
				#    print("Ever best reward %s" %(total_reward/self.display_freq))
				#    self.save('dqn')
                total_reward = 0

            episodes_done_num += 1
            if self.steps > self.num_timesteps:
                break
        self.save('dqn')
