#!/usr/bin/env python
# coding: utf-8

# ### input data prerpoccessing

# In[ ]:


# 初始化報告數據字典
report_data = {1: 0, 2: 0}  # 假設初始值為0，您需要根據實際情況進行調整

# 假設 total_collected_data_ue1 和 total_collected_data_ue2 是已知的變量
total_collected_data_ue1 = 1000  # 這裡使用示例數據，您需要使用實際數據替換
total_collected_data_ue2 = 1000

# 計算封包損失
Ue1PacketLoss = total_collected_data_ue1 - report_data[1]
Ue2PacketLoss = total_collected_data_ue2 - report_data[2]

# 定義狀態空間和動作空間
Ue1SliceSize = 64  # 示例大小，請根據需求調整
Ue2SliceSize = 128
StateSpace = [(total_collected_data_ue1, Ue1SliceSize), (total_collected_data_ue2, Ue2SliceSize)]
ActionSpace = [(64, 128, 256, 512, 1024), (64, 128, 256, 512, 1024)]

# 計算獎勵
reward = Ue1PacketLoss + Ue2PacketLoss


# ### 另一種方法(Q-Learning with SVC)

# Environment

# In[30]:


import gym
import numpy as np
from gym import spaces
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import random
import time

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.MultiDiscrete([5, 5])  # 定義兩個動作空間各有5種選擇
        self.observation_space = spaces.Box(low=0, high=1, shape=(2, 2), dtype=np.float32)
        self.state = np.zeros((2, 2), dtype=np.float32)

    def reset(self):
        self.state = np.zeros((2, 2), dtype=np.float32)
        return self.state

    def step(self, action):
        # 將action索引轉換為實際值
        action_values = [64, 128, 256, 512, 1024]
        action1 = action_values[action[0]]
        action2 = action_values[action[1]]
        self.state = np.random.rand(2, 2).astype(np.float32)
        reward = -np.sum(np.abs(self.state - 0.5))  # 假設的reward計算
        done = np.random.rand() > 0.95
        return self.state, reward, done, {}

    def render(self, mode='human', close=False):
        print(f"State: {self.state}")


# Agent

# In[31]:


class QLearningSVC:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.alpha = 0.8
        self.svc = SVC(probability=True)
        self.scaler = StandardScaler()
        self.svc_trained = False

    def preprocess_state(self, state):
        return self.scaler.transform([state])

    def remember(self, state, action, reward, next_state, done):
        old_value = self.q_table[tuple(state)][action]
        next_max = np.max(self.q_table[tuple(next_state)])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[tuple(state)][action] = new_value

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return [random.randint(0, 4), random.randint(0, 4)]  # 隨機選擇動作
        if self.svc_trained:
            state = self.preprocess_state(state)
            action_index = self.svc.predict(state)[0]
            print(f"SVC select action = {action_index}")
            return [action_index // 5, action_index % 5]  # 轉換為二維動作索引
        flat_index = np.argmax(self.q_table[tuple(state)])
        return [flat_index // 5, flat_index % 5]

    def train_svc(self):
        X, y = [], []
        for state, actions in self.q_table.items():
            for action, value in enumerate(actions):
                X.append(state)
                y.append(action)
        X = np.array(X)
        y = np.array(y)
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        self.svc.fit(X, y)
        self.svc_trained = True

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# 開始訓練

# In[ ]:


if __name__ == "__main__":
    env = CustomEnv()
    state_size = env.observation_space.shape
    action_size = env.action_space.nvec.prod()

    agent = QLearningSVC(state_size, action_size)
    EPISODES = 30
    Time = 60

    for e in range(EPISODES):
        state = env.reset()
        state = state.flatten()  # 展平狀態
        total_reward = 0

        for times in range(Time):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()
            reward = reward if not done else -10
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            time.sleep(5)
            print(f"action = {action}")

            if done:
                agent.decay_epsilon()
                if e % 10 == 0:
                    agent.train_svc()
                print(f"Episode: {e}/{EPISODES}, Score: {times}, Epsilon: {agent.epsilon:.2}")
                break

    env.close()


# In[ ]:


# def output_action()

