import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 建立Q網絡
def build_model(state_size, action_size):
    model = Sequential()
    model.add(Dense(24, input_dim=state_size, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# 選擇行動
def choose_action(state, model, action_size, epsilon):
    if np.random.rand() <= epsilon:
        return np.random.choice(action_size)
    q_values = model.predict(state)
    return np.argmax(q_values[0])

# 訓練DQN
def train_dqn(episodes=1000, batch_size=64):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    model = build_model(state_size, action_size)
    target_model = build_model(state_size, action_size)
    target_model.set_weights(model.get_weights())

    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    discount_factor = 0.99

    memory = []

    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        total_reward = 0

        while not done:
            action = choose_action(state, model, action_size, epsilon)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            total_reward += reward
            memory.append((state, action, reward, next_state, done))
            state = next_state

            if len(memory) > batch_size:
                minibatch = np.random.choice(memory, batch_size, replace=False)
                for state, action, reward, next_state, done in minibatch:
                    target = reward
                    if not done:
                        target = reward + discount_factor * np.amax(target_model.predict(next_state)[0])
                    target_f = model.predict(state)
                    target_f[0][action] = target
                    model.fit(state, target_f, epochs=1, verbose=0)

            if done:
                target_model.set_weights(model.get_weights())
                epsilon = max(epsilon_min, epsilon * epsilon_decay)
                print(f"Episode: {episode+1}/{episodes}, Total Reward: {total_reward}")

train_dqn()