from cambrian.envs.env import MjCambrianEnv
from cambrian import MjCambrianConfig, MjCambrianTrainer


# Initialize the environment
env = MjCambrianEnv(MjCambrianConfig())

# Define actions for both agents
action = {
    "agent": [0.1, 0.2],
    "agent2": [-0.1, 0.3],
}

# Step the environment
obs, reward, terminated, truncated, info = env.step(action)
print(obs, reward, terminated, truncated, info)