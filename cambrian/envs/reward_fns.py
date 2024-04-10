from typing import Dict, Any, List, Optional

import numpy as np
import mujoco as mj

from cambrian.envs.env import MjCambrianEnv
from cambrian.envs.object_env import MjCambrianObjectEnv
from cambrian.animals.animal import MjCambrianAnimal

# =====================
# Common reward logic


def apply_termination_reward(
    reward: float, terminated: bool, termination_reward: float = 1.0
) -> float:
    """Terminated indicates that the episode was ended early in a success.
    Returns termination_reward if terminated, else reward."""
    return termination_reward if terminated else reward


def apply_truncation_reward(
    reward: float, truncated: bool, *, truncation_reward: float = -1.0
) -> float:
    """Truncated indicates that the episode was ended early in a failure.
    Returns truncation_reward if truncated, else reward."""
    return truncation_reward if truncated else reward


def postprocess_reward(
    reward: float,
    terminated: bool,
    truncated: bool,
    *,
    termination_reward: float = 1.0,
    truncation_reward: float = -1.0,
) -> float:
    """Applies termination and truncation rewards to the reward."""
    reward = apply_termination_reward(
        reward, terminated, termination_reward=termination_reward
    )
    reward = apply_truncation_reward(
        reward, truncated, truncation_reward=truncation_reward
    )
    return reward


# =====================
# Reward functions


def termination_and_truncation_only(
    env: MjCambrianEnv,
    animal: MjCambrianAnimal,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    termination_reward: float = 1.0,
    truncation_reward: float = -1.0,
) -> float:
    """Rewards the animal for reaching the target."""
    return postprocess_reward(
        0,
        terminated,
        truncated,
        termination_reward=termination_reward,
        truncation_reward=truncation_reward,
    )


def euclidean_delta_from_init(
    env: MjCambrianEnv,
    animal: MjCambrianAnimal,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    factor: float = 1.0,
) -> float:
    """
    Rewards the change in distance over the previous step scaled by the timestep.
    """
    return calc_delta(animal, info, animal.init_pos) * factor


def reward_if_close_to_object(
    env: MjCambrianObjectEnv,
    animal: MjCambrianAnimal,
    info: Dict[str, Any],
) -> float:
    """Terminates the episode if the animal is close to an object. Terminate is only
    true if the object is set to terminate_if_close = True."""
    reward = 0
    for obj in env.objects.values():
        if obj.is_close(animal.pos):
            reward += obj.config.reward_if_close
    return reward


def penalize_if_has_contacts(
    env: MjCambrianEnv,
    animal: MjCambrianAnimal,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    penalty: float = -1.0,
) -> float:
    """Penalizes the animal if it has contacts with the ground."""
    return penalty if info["has_contacts"] else 0.0


def reward_if_animals_in_view(
    env: MjCambrianEnv,
    animal: MjCambrianAnimal,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    reward: float,
    from_animals: Optional[List[str]] = None,
    to_animals: Optional[List[str]] = None,
    hfov: float = 45,
) -> float:
    """This reward function rewards the animal if it is in the view of other animals.

    Keyword Args:
        reward (float): The reward to give the animal if it is in view of another animal.
        from_animals (Optional[List[str]]): The names of the animals that the reward
            should be calculated from. If None, the reward will be calculated from all
            animals.
        to_animals (Optional[List[str]]): The names of the animals that the reward
            should be calculated to. If None, the reward will be calculated to all
            animals.
        hfov (float): The horizontal fov to check whether the to animal is within view
            of the from animal. Default is 45. This is in degrees.
    """
    # Early exit if the animal is not in the from_animals list
    if from_animals is not None and animal.name not in from_animals:
        return 0

    accumulated_reward = 0
    to_animals = to_animals or list(env.animals.keys())
    for to_animal in [env.animals[name] for name in to_animals]:
        if animal.name == to_animal.name:
            continue

        # Extract the position and rotation matrices for both animals
        # We'll use these to calculate the direction the from animal is facing
        # relative to the to animal. This will determine if the to animal is in
        # view of the from animal.
        vec = to_animal.pos - animal.pos
        yaw = np.arctan2(animal.mat[1, 0], animal.mat[0, 0])
        relative_yaw = np.arctan2(vec[1], vec[0]) - yaw

        # Early exit if the animal isn't within view of the from animal
        if np.abs(relative_yaw) > np.deg2rad(hfov) / 2:
            continue

        # Now we'll trace a ray between the two animals to make sure the animal is
        # within view
        geomid = np.zeros(1, np.int32)
        _ = mj.mj_ray(
            env.model,
            env.data,
            animal.pos,
            vec,
            animal.geomgroup_mask,  # mask out this animal to avoid self-collision
            1,  # include static geometries
            -1,  # include all bodies
            geomid,
        )

        # Early exit again if the animal is occluded
        if geomid != to_animal.geom.id:
            continue

        accumulated_reward += reward
    return accumulated_reward


def combined_reward(
    env: MjCambrianEnv,
    animal: MjCambrianAnimal,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    **reward_fns,
) -> float:
    """Combines multiple reward functions into one."""
    reward = 0
    for fn in reward_fns.values():
        reward += fn(env, animal, terminated, truncated, info)
    return reward


# =====================
# Utility functions


def calc_delta(
    animal: MjCambrianAnimal, info: Dict[str, Any], point: np.ndarray = np.array([0, 0])
) -> np.ndarray:
    """Calculates the delta position of the animal from a point.

    NOTE: returns delta position of current pos from the previous pos to the point
    (i.e. current - prev)
    """

    current_distance = np.linalg.norm(animal.pos[:2] - point[:2])
    prev_distance = np.linalg.norm(info["prev_pos"][:2] - point[:2])
    return current_distance - prev_distance
