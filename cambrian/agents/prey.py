from typing import Optional
import numpy as np
from cambrian.agents.agent import MjCambrianAgentConfig
from cambrian.envs.maze_env import MjCambrianMazeEnv
from cambrian.utils import get_logger
from cambrian.utils.types import ActionType, ObsType
from .point import MjCambrianAgentPoint


class MjCambrianAgentPrey(MjCambrianAgentPoint):

    def __init__(
        self,
        config: MjCambrianAgentConfig,
        name: str,
        *,
        predator: str,
        speed: float = 0.75,
        safe_distance: float = 5.0,
    ):
        super().__init__(config, name)

        self._predator = predator
        self._speed = speed
        self._safe_distance = safe_distance

    def reset(self, *args) -> ObsType:
        return super().reset(*args)

    def get_action_privileged(self, env: MjCambrianMazeEnv) -> ActionType:
        assert self._predator in env.agents, f"Predator {self._predator} not found in env"
        predator_pos = env.agents[self._predator].pos

        escape_vector = self.pos[:2] - predator_pos[:2]
        distance = np.linalg.norm(escape_vector)

        if distance > self._safe_distance:
            # get_logger().info(f"{self.name} is safe from {self._predator}.")
            return [-1.0, 1.0]

        escape_theta = np.arctan2(escape_vector[1], escape_vector[0])
        theta_action = np.interp(escape_theta, [-np.pi, np.pi], [-1, 1])

        return [-1.0, -1.0]