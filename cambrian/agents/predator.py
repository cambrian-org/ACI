from typing import Optional
import numpy as np
from cambrian.agents.agent import MjCambrianAgentConfig
from cambrian.envs.maze_env import MjCambrianMazeEnv
from cambrian.utils import get_logger
from cambrian.utils.types import ActionType, ObsType
from .point import MjCambrianAgentPoint

class MjCambrianAgentPredator(MjCambrianAgentPoint):

    def __init__(
        self,
        config: MjCambrianAgentConfig,
        name: str,
        *,
        prey: str,
        speed: float = 0.01,
        capture_threshold: float = 1.0,
    ):
        super().__init__(config, name)

        self._prey = prey
        self._speed = speed
        self._capture_threshold = capture_threshold

    def reset(self, *args) -> ObsType:
        return super().reset(*args)

    def get_action_privileged(self, env: MjCambrianMazeEnv) -> ActionType:
        assert self._prey in env.agents, f"Prey {self._prey} not found in env"
        prey_pos = env.agents[self._prey].pos

        target_vector = prey_pos[:2] - self.pos[:2]
        distance = np.linalg.norm(target_vector)

        if distance < self._capture_threshold:
            # get_logger().info(f"{self.name} captured {self._prey}!")
            return [0.0, 0.0]  

        target_theta = np.arctan2(target_vector[1], target_vector[0])
        theta_action = np.interp(target_theta, [-np.pi, np.pi], [-1, 1])

        return [self._speed, theta_action]