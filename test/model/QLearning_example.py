import numpy as np
import random

class Environment:
    def __init__(self):
        self.state = 0
        self.end_states = [0, 6]

    def reset(self):
        self.state = 3  # 初始狀態
        return self.state

    def step(self, action):
        """執行動作，更新狀態，回傳獎勵和終止條件。
        动作 0: 左移一格
        动作 1: 右移一格
        """
        if action == 0 and self.state > 0:
            self.state -= 1
        elif action == 1 and self.state < 6:
            self.state += 1
        
        reward = 1 if self.state == 6 else 0  # 到達終點獲得獎勵
        done = self.state in self.end_states
        return self.state, reward, done

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros((7, 2))  # 7個狀態，2個可能的動作
        self.alpha = alpha  # 學習率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索機率

    def choose_action(self, state):
        """選擇動作：探索或利用"""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice([0, 1])  # 探索：隨機選擇動作
        else:
            return np.argmax(self.q_table[state])  # 利用：選擇當前最佳動作

    def update_q_table(self, state, action, reward, next_state):
        """更新 Q 表"""
        best_next_action = np.argmax(self.q_table[next_state])  # 下一狀態的最佳動作
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def train(self, env, episodes=1000):
        for _ in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state

if __name__ == "__main__":
    env = Environment()
    agent = QLearningAgent()

    agent.train(env, episodes=1000)
    print("Trained Q-Table:")
    print(agent.q_table)
