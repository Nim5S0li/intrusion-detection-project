import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def generate_synthetic_data(num_samples=10000):
    duration = np.random.randint(0, 1000, num_samples)
    src_bytes = np.random.randint(0, 1e6, num_samples)
    dst_bytes = np.random.randint(0, 1e6, num_samples)
    wrong_fragment = np.random.randint(0, 5, num_samples)
    hot = np.random.randint(0, 10, num_samples)

    protocol_type = np.random.choice([0, 1, 2], num_samples, p=[0.7, 0.2, 0.1])
    flag = np.random.choice([0, 1, 2, 3], num_samples)
    service = np.random.choice([0, 1, 2, 3, 4], num_samples)

    logged_in = np.random.randint(0, 2, num_samples)
    root_shell = np.random.randint(0, 2, num_samples)
    other_features = np.random.rand(num_samples, 31)

    data = np.column_stack([
        duration, protocol_type, service, flag, src_bytes, dst_bytes,
        logged_in, wrong_fragment, hot, root_shell, *other_features.T
    ])

    labels = np.random.choice([0, 1], num_samples, p=[0.8, 0.2])

    feature_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'logged_in', 'wrong_fragment', 'hot', 'root_shell'
    ] + [f'feature_{i}' for i in range(10, 41)]

    X = pd.DataFrame(data, columns=feature_names)
    y = pd.Series(labels, name='label')

    return X, y


def load_data():
    X, y = generate_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


X_train, X_test, y_train, y_test = load_data()


def train_supervised_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=50, random_state=42) 
    model.fit(X_train, y_train)
    return model


supervised_model = train_supervised_model(X_train, y_train)
y_pred = supervised_model.predict(X_test)
print("Accuracy of Supervised Model:", accuracy_score(y_test, y_pred))

all_probas = supervised_model.predict_proba(X_test)[:, 1]



class IntrusionDetectionEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, supervised_model, X_test, y_test, all_probas):
        super(IntrusionDetectionEnv, self).__init__()
        self.supervised_model = supervised_model
        self.X_test = X_test
        self.y_test = y_test
        self.all_probas = all_probas
        self.current_step = 0

        obs_dim = self.X_test.shape[1] + 1 
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def _get_observation(self):
        raw_obs = self.X_test[self.current_step]
        proba = self.all_probas[self.current_step]
        return np.append(raw_obs, proba).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_observation(), {}  

    def step(self, action):
        true_label = self.y_test.iloc[self.current_step]
        predicted_label = int(self.all_probas[self.current_step] > 0.5)

        if action == 1 and predicted_label == true_label:
            reward = 1
        elif action == 0 and predicted_label != true_label:
            reward = 3  
        elif action == 0 and predicted_label == true_label:
            reward = -2  
        else:
            reward = -1 

        self.current_step += 1
        terminated = self.current_step >= len(self.X_test)
        truncated = False

        next_state = self._get_observation() if not terminated else None

        return next_state, reward, terminated, truncated, {}


env_raw = IntrusionDetectionEnv(supervised_model, X_test, y_test, all_probas)
env = DummyVecEnv([lambda: env_raw]) 


policy_kwargs = dict(
    net_arch=dict(pi=[256, 256], vf=[256, 256]),
)

model_rl = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, ent_coef=0.3, n_steps=64, verbose=1)
model_rl.learn(total_timesteps=300000) 


model_rl.save("ppo_intrusion_detection")



def evaluate_combined_system(env, model_rl, X_test, y_test, all_probas):
    state = env.reset()
    total_reward = 0
    correct_decisions = 0

    for i in range(len(X_test)):  
        action, _states = model_rl.predict(state, deterministic=False)
        action = int(action.item()) 

        obs, reward, terminated, truncated = env.step([action]) 

        predicted_label = int(all_probas[i] > 0.5)
        true_label = y_test.iloc[i]

        if (action == 1 and predicted_label == true_label) or \
           (action == 0 and predicted_label != true_label):
            correct_decisions += 1

        total_reward += reward.item()  
        state = obs

        print(f"Step {i}: Action={action}, Predicted={predicted_label}, True={true_label}, Reward={reward.item()}")


    accuracy = correct_decisions / len(X_test)
    print(f"\n✅ Combined System Accuracy: {accuracy}")
    print(f"🎯 Total Reward: {total_reward}")


evaluate_combined_system(env, model_rl, X_test, y_test, all_probas)