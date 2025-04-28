import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import gym
from gym import spaces
import random

def load_data():
    data = pd.read_csv('KDDTrain+.csv')
    X = data.drop(columns=['label'])
    y = data['label']
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = load_data()

def train_supervised_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

supervised_model = train_supervised_model(X_train, y_train)

y_pred = supervised_model.predict(X_test)
print("Accuracy of Supervised Model:", accuracy_score(y_test, y_pred))

class IntrusionDetectionEnv(gym.Env):
    def __init__(self, supervised_model, X_test, y_test):
        super(IntrusionDetectionEnv, self).__init__()
        self.supervised_model = supervised_model
        self.X_test = X_test
        self.y_test = y_test
        self.current_step = 0
        self.action_space = spaces.Discrete(2)  
        self.observation_space = spaces.Box(low=0, high=1, shape=(X_test.shape[1],), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self.X_test.iloc[self.current_step].values

    def step(self, action):
        true_label = self.y_test.iloc[self.current_step]
        predicted_label = self.supervised_model.predict([self.X_test.iloc[self.current_step]])[0]

        reward = 0
        if action == 1 and predicted_label == true_label:
            reward = 1  
        elif action == 0 and predicted_label != true_label:
            reward = 1 
        else:
            reward = -1 

        self.current_step += 1
        done = self.current_step >= len(self.X_test) - 1
        next_state = self.X_test.iloc[self.current_step].values if not done else None

        return next_state, reward, done, {}

from stable_baselines3 import PPO

env = IntrusionDetectionEnv(supervised_model, X_test, y_test)
model_rl = PPO("MlpPolicy", env, verbose=1)
model_rl.learn(total_timesteps=10000)

def evaluate_combined_system(env, model_rl, X_test, y_test):
    state = env.reset()
    total_reward = 0
    correct_predictions = 0
    for i in range(len(X_test)):
        action, _ = model_rl.predict(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        true_label = y_test.iloc[i]
        predicted_label = env.supervised_model.predict([X_test.iloc[i]])[0]
        if action == 1 and predicted_label == true_label:
            correct_predictions += 1
        if done:
            break
    accuracy = correct_predictions / len(X_test)
    print(f"Combined System Accuracy: {accuracy}")
    print(f"Total Reward: {total_reward}")

evaluate_combined_system(env, model_rl, X_test, y_test)