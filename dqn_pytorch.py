#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import gym
import math
import queue
import random
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import os


# In[2]:


import wandb


# In[3]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[4]:


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()


# In[48]:


parameter = {
    'batch_size': 32,
    'gamma' : 0.99,
    'eps_start': 1,
    'eps_end': 0.01,
    'eps_decay': 1000000,
    'lr': 0.00025,
    'seed': 42,
    'target_update': 10000,
    'max_episodes': 500000,
    'max_memory': 100000,
}


# In[6]:


wandb.init(project='RL-Breakout', entity='lmingming')


# In[7]:


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.determinstic =True
#     torch.backends.cudnn.benchmark = True


# In[8]:


set_seed(parameter['seed'])


# In[9]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[10]:


device


# ## 定义模型，与deepmind的模型一致

# In[11]:


resize = T.Compose([T.ToPILImage(),
                    T.Resize((84, 84)),
                    T.Grayscale(),
                    T.ToTensor()])


# In[12]:


class DQN(nn.Module):
    def __init__(self, h=84, w=84, c=4, outputs=4):
        super(DQN, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()
        )
        self.linear1 = nn.Linear(7*7*64, 512)
        self.linear2 = nn.Linear(512, outputs)
        self.dl_linear1 = nn.Linear(7*7*64, 512)
        self.dl_linear2 = nn.Linear(512, 1)
    
    def forward(self, x):
        output = self.seq(x)
        dl_out = self.dl_linear1(output)
        dl_out = self.dl_linear2(dl_out)
        output = self.linear1(output)
        output = self.linear2(output)
        output = output + dl_out
        
        return output


# In[13]:


class FireResetEnv(gym.Wrapper):
    """reset时游戏循环随机动作直到fire"""
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
    
    def reset(self):
        obs, info = self.env.reset()
        while True:
            action = self.env.unwrapped.action_space.sample()
            obs, resward, done, _, _ = self.env.step(action)
            if action == 1:
                break
        return obs, info


# In[14]:


class OneLiveEnv(gym.Wrapper):
    """游戏只有一条命，结束重开"""
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
    
    def step(self, action):
        obs, reward, done, _, _ = self.env.step(action)
        if self.env.unwrapped.ale.lives() < 5:
            done = True
            reward = -1.0
        if reward > 0:
            reward = 1.0
        return obs, reward, done, _, _


# In[15]:


class SkipFrameEnv(gym.Wrapper):
    """跳过4帧画面，将4帧画面堆叠作为一个状态，4帧执行相同的动作，reset时，前n步执行随机动作，n在一个区间内随机取值
    """
    def __init__(self, env, max_head_step=5):
        gym.Wrapper.__init__(self, env)
        self.max_head_step = max_head_step # 默认最大5步内执行随机动作
        
    def reset(self):
        obs, info = self.env.reset()
        head_step = np.random.randint(1, self.max_head_step)
        for _ in range(head_step):
            obs, _, done, _, _ = self.step(self.env.action_space.sample())
            if done:
                return self.reset()
        
        return obs, done
        
    def step(self, action):
        frames = [] # 存储4帧画面
        for _ in range(4):
            frames.append(resize(self.env.render()))
            obs, reward, done, _, _ = self.env.step(action)
            if done:
                break

        for _ in range(4 - len(frames)):
            frames.append(resize(env.render()))
        four_frame = torch.cat(frames).float()
        return four_frame.unsqueeze(0), reward, done, _, _


# In[16]:


class AnimationsRecordEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.animations = []
    
    def reset(self):
        obs, info = self.env.reset()
        self.animations.append(obs)
        return obs, info
    
    def step(self, action):
        obs, reward, done, info1, info2 = self.env.step(action)
        self.animations.append(obs)
        return obs, reward, done, info1, info2
    
    def get_animations(self):
        return self.animations


# In[17]:


class LastNFrameEnv(gym.Wrapper):
    """保存过去三帧画面作为一个状态"""
    def __init__(self, env, n):
        gym.Wrapper.__init__(self, env)
        self.n_frame = deque([], maxlen=n)
        self.n = n
        
    def reset(self):
        obs, info = self.env.reset()
        for i in range(self.n):
            self.n_frame.append(resize(obs))
        state = torch.cat(tuple(self.n_frame)).float()
        return state.unsqueeze(0), info
    
    def step(self, action):
        obs, reward, done, info1, info2 = self.env.step(action)
        self.n_frame.append(resize(obs))
        state = torch.cat(tuple(self.n_frame)).float()
        return state.unsqueeze(0), reward, done, info1, info2
    
    def get_n_frame(self):
        return self.n_frame


# In[18]:


class RandomStateResetEnv(gym.Wrapper):
    """游戏开始前n步执行随机动作"""
    def __init__(self, env, n=10):
        gym.Wrapper.__init__(self, env)
        self.n = n
    
    def reset(self):
        obs, info = self.env.reset()
        for i in range(self.n):
            obs, reward, done, info1, info2 = self.step(action)
            if done:
                return self.reset()
        return obs, done
    
    def step(self, action):
        obs, reward, done, info1, info2 = self.env.step(action)
        return obs, reward, done, info1, info2


# In[19]:


eps = 1
def start_end(step):
    global eps
    eps -= (1 - 0.1) / 1000000
    return max(eps, 0.1)


# In[20]:


x = np.arange(1,1000000, 1)


# In[21]:


y = [start_end(i) for i in x]


# In[22]:


plt.plot(x, y)


# In[31]:


class ActionSelector:
    def __init__(self, eps_start, eps_end, eps_decay):
        self.eps = eps_start
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        
    def select_action(self, state, policy_net, env, device, is_train=True):
        if is_train:
            self.eps -= (self.eps_start - self.eps_end) / self.eps_decay
            self.eps = max(self.eps, self.eps_end)
        else:
            self.eps = 0.1
            
        sample = random.random()
        if sample > self.eps:
            with torch.no_grad():
                a = policy_net(state).max(1)[1].view(1, 1)
        else:
            a = torch.tensor([[env.action_space.sample()]], device=device)
        return a


# In[32]:


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


# In[33]:


class ReplayMemory:
    def __init__(self, max_memory):
        self.memory = deque([], maxlen=max_memory)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
        
    def sampel(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


# In[34]:


def play_animation(animations):
    fig, ax = plt.subplots(1,1)
    fig.show()
    for pic in animations:
        ax.clear()
        ax.imshow(pic)
        fig.canvas.draw()


# In[35]:


env = gym.make('ALE/Breakout-v5', render_mode='rgb_array').unwrapped
env = FireResetEnv(env)
env = OneLiveEnv(env)
env = LastNFrameEnv(env, 3)


# In[36]:


n_action = env.action_space.n
policy_net = DQN(outputs=n_action, c=3).to(device)
target_net = DQN(outputs=n_action, c=3).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()


# ## 训练

# In[37]:


optimizer = optim.RMSprop(policy_net.parameters(), lr=parameter['lr'])


# In[38]:


criterion = nn.SmoothL1Loss()


# In[39]:


actorSelector = ActionSelector(parameter['eps_start'], parameter['eps_end'], parameter['eps_decay'])


# In[40]:


memory = ReplayMemory(parameter['max_memory'])


# In[41]:


def optimize_model():
    if len(memory) < 3000:
        return
    
    batch = memory.sampel(parameter['batch_size'])
    batch = Transition(*zip(*batch))
    
    non_final_mask = torch.tensor(batch.done, device=device, dtype=torch.bool) # 标记那个状态是最终状态
    non_final_next_state = torch.cat(batch.next_state).to(device)
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    state_action_values = policy_net(state_batch).gather(1, action_batch) 
#     next_state_values = torch.zeros(parameter['batch_size'], device=device)
#     next_state_actions = policy_net(non_final_next_state).max(1)[1].unsqueeze(1).detach()
    next_state_values = target_net(non_final_next_state).max(1)[0].detach()
    expected_state_action_values = (next_state_values * parameter['gamma']) + reward_batch
    next_state_values[non_final_mask] = 0.0 # 将最终状态的值置为0
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.item()


# In[42]:


total_reward = 0
total_loss = 0
total_t = 0
total_step = 0
for i_episodes in range(parameter['max_episodes']):
    obs, info = env.reset()
    wandb_show_reward = 0
    wandb_show_loss = 0
    wandb_t = 0
    for t in count():
        action = actorSelector.select_action(obs.to(device), policy_net, env, device)
        next_obs, reward, done, _, _ = env.step(action.item())
        total_reward += reward
        wandb_show_reward += reward
        reward = torch.tensor([reward], device=device)
        # 'state', 'action', 'reward', 'next_state', 'done'
        memory.push(obs, action, reward, next_obs, done)
        obs = next_obs
        
        loss = optimize_model()
        if loss:
            total_loss += loss
            wandb_show_loss += loss
            total_t += 1
        wandb_t += 1
        
        total_step += 1
        if total_step % parameter['target_update'] == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        if done:
            break
    
    wandb.log({"1 episode reward": wandb_show_reward, "1 episode loss": wandb_show_loss / wandb_t})
    
    if i_episodes % 100 == 0:
        if total_t == 0:
            total_t = 1
        print(f"episodes:{i_episodes}, loss: {total_loss / total_t}, reward: {total_reward / 100}")
        total_loss = 0
        total_reward = 0
        total_t = 0
    
    if i_episodes % 1000 and i_episodes != 0:
        torch.save(policy_net, './policy_net1.pkl')
        torch.save(target_net, './target_net1.pkl')


# In[85]:


torch.save(policy_net, './policy_net1.pkl')
torch.save(target_net, './target_net1.pkl')


# In[119]:


env_test = gym.make('ALE/Breakout-v5', render_mode='rgb_array').unwrapped
env_test = LastNFrameEnv(env_test, 3)
# env_test = AnimationsRecordEnv(env_test)
# env_test = FireResetEnv(env_test)
# env_test = OneLiveEnv(env_test)
# env_test = SkipFrameEnv(env_test, 5)


# In[64]:


env_test = gym.make('ALE/Breakout-v5', render_mode='rgb_array').unwrapped
env_test = AnimationsRecordEnv(env_test)
env_test = FireResetEnv(env_test)
env_test = OneLiveEnv(env_test)
env_test = LastNFrameEnv(env_test, 3)


# In[65]:


obs, info = env_test.reset()


# In[66]:


for _ in range(1000):
    action = actorSelector.select_action(obs.to(device), policy_net, env_test, device)
    obs, reward, done, _, _ = env_test.step(action.item())
    if done:
        obs, info = env_test.reset()


# In[67]:


env_test.unwrapped.ale.lives()


# In[70]:


play_animation(env_test.get_animations())


# In[ ]:




