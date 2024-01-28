import argparse
from typing import Any, List, Tuple, TYPE_CHECKING, Optional, Callable, Dict
from pathlib import Path
from dataclasses import dataclass
import contextlib

import gymnasium as gym
import mujoco as mj
import numpy as np

if TYPE_CHECKING:
    from cambrian.ml.model import MjCambrianModel


def safe_index(list_to_index: List[Any], value: Any) -> int:
    """Safely get the index of a value in a list, or -1 if not found. Normally,
    list.index() throws an exception if the value is not found."""
    try:
        return list_to_index.index(value)
    except ValueError:
        return -1


def get_include_path(
    model_path: str | Path, *, throw_error: bool = True
) -> Path | None:
    """Tries to find the model path. `model_path` can either be relative to the
    execution file, absolute, or relative to cambrian.evolution_envs.three_d.mujoco. The
    latter is the typical method, where `assets/<model>.xml` specifies the model path
    located in cambrian/evolution_envs/three_d/mujoco/assets/<model>.xml.

    If the file can't be found, a FileNotFoundError is raised if throw_error is True. If
    throw_error is False, None is returned.
    """
    path = Path(model_path)
    if path.exists():
        pass
    elif (rel_path := Path(__file__).parent / path).exists():
        path = rel_path
    else:
        if throw_error:
            raise FileNotFoundError(f"Could not find path `{model_path}`.")
        else:
            return None

    return path


# ============


def evaluate_policy(
    env: gym.Env,
    model: "MjCambrianModel",
    num_runs: int,
    *,
    record_path: Optional[Path] = None,
    step_callback: Optional[Callable] = None,
):
    """Evaluate a policy.

    Args:
        env (gym.Env): The environment to evaluate the policy on. Assumed to be a
            VecEnv wrapper around a MjCambrianEnv.
        model (MjCambrianModel): The model to evaluate.
        num_runs (int): The number of runs to evaluate the policy on.

    Keyword Args:
        record_path (Optional[Path]): The path to save the video to. If None, the video
            is not saved. This is passed directly to MjCambrianEnv.renderer.save(), so
            see that method for more details.
    """
    # To avoid circular imports
    from cambrian.evolution_envs.three_d.mujoco.env import MjCambrianEnv

    cambrian_env: MjCambrianEnv = env.envs[0].unwrapped
    if record_path is not None:
        # don't set to record_path is not None directly cause this will delete overlays
        cambrian_env.record = True

    prev_init_goal_pos = None
    if (eval_goal_pos := cambrian_env.maze.config.eval_goal_pos) is not None:
        prev_init_goal_pos = cambrian_env.maze.config.init_goal_pos
        cambrian_env.maze.config.init_goal_pos = eval_goal_pos

    run = 0
    cumulative_reward = 0

    print(f"Starting {num_runs} evaluation run(s)...")
    obs = env.reset()
    while run < num_runs:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        cumulative_reward += reward[0]

        if done:
            print(f"Run {run} done. Cumulative reward: {cumulative_reward}")
            cumulative_reward = 0
            run += 1

        if cambrian_env.config.env_config.add_overlays:
            cambrian_env.overlays["Exp"] = cambrian_env.config.training_config.exp_name
            cambrian_env.overlays["Cumulative Reward"] = f"{cumulative_reward:.2f}"
        env.render()

        if step_callback is not None:
            step_callback()

    if record_path is not None:
        cambrian_env.save(record_path)
        cambrian_env.record = False

    if prev_init_goal_pos is not None:
        cambrian_env.maze.config.init_goal_pos = prev_init_goal_pos


# =============


class MjCambrianArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_argument("config", type=str, help="Path to config file")
        self.add_argument(
            "-o",
            "--override",
            "--overrides",
            dest="overrides",
            action="extend",
            nargs="+",
            type=str,
            help="Override config values. Do <config>.<key>=<value>",
            default=[],
        )

    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)

        return args


# =============


def generate_sequence_from_range(range: Tuple[float, float], num: int) -> List[float]:
    """"""
    return [np.average(range)] if num == 1 else np.linspace(*range, num)


def merge_dicts(d1: dict, d2: dict) -> dict:
    """Merge two dictionaries. d2 takes precedence over d1."""
    return {**d1, **d2}


@contextlib.contextmanager
def setattrs_temporary(*args: Tuple[Any, Dict[str, Any]]) -> None:
    """Temporarily set attributes of an object."""
    prev_values = []
    for obj, kwargs in args:
        prev_values.append({})
        for attr, value in kwargs.items():
            if isinstance(obj, dict):
                prev_values[-1][attr] = obj[attr]
                obj[attr] = value
            else:
                prev_values[-1][attr] = getattr(obj, attr)
                setattr(obj, attr, value)
    yield
    for (obj, _), kwargs in zip(args, prev_values):
        for attr, value in kwargs.items():
            if isinstance(obj, dict):
                obj[attr] = value
            else:
                setattr(obj, attr, value)


# =============
# Mujoco utils


def get_body_id(model: mj.MjModel, body_name: str) -> int:
    """Get the ID of a Mujoco body."""
    return mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)


def get_body_name(model: mj.MjModel, bodyadr: int) -> str:
    """Get the name of a Mujoco body."""
    return mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bodyadr)


def get_geom_id(model: mj.MjModel, geom_name: str) -> int:
    """Get the ID of a Mujoco geometry."""
    return mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, geom_name)


def get_geom_name(model: mj.MjModel, geomadr: int) -> str:
    """Get the name of a Mujoco geometry."""
    return mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geomadr)


def get_site_id(model: mj.MjModel, site_name: str) -> int:
    """Get the ID of a Mujoco geometry."""
    return mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)


def get_site_name(model: mj.MjModel, siteadr: int) -> str:
    """Get the name of a Mujoco geometry."""
    return mj.mj_id2name(model, mj.mjtObj.mjOBJ_SITE, siteadr)


def get_joint_id(model: mj.MjModel, joint_name: str) -> int:
    """Get the ID of a Mujoco geometry."""
    return mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)


def get_joint_name(model: mj.MjModel, jointadr: int) -> str:
    """Get the name of a Mujoco geometry."""
    return mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jointadr)


def get_camera_id(model: mj.MjModel, camera_name: str) -> int:
    """Get the ID of a Mujoco camera."""
    return mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, camera_name)


def get_camera_name(model: mj.MjModel, cameraadr: int) -> str:
    """Get the name of a Mujoco camera."""
    return mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, cameraadr)


def get_light_id(model: mj.MjModel, light_name: str) -> int:
    """Get the ID of a Mujoco light."""
    return mj.mj_name2id(model, mj.mjtObj.mjOBJ_LIGHT, light_name)


def get_light_name(model: mj.MjModel, lightadr: int) -> str:
    """Get the name of a Mujoco light."""
    return mj.mj_id2name(model, mj.mjtObj.mjOBJ_LIGHT, lightadr)


@dataclass
class MjCambrianJoint:
    """Helper class which stores information about a Mujoco joint.

    Attributes:
        adr (int): The Mujoco joint ID (index into model.jnt_* arrays).
        qposadr (int): The index of the joint's position in the qpos array.
        numqpos (int): The number of positions in the joint.
        qveladr (int): The index of the joint's velocity in the qvel array.
        numqvel (int): The number of velocities in the joint.
    """

    adr: int
    qposadr: int
    numqpos: int
    qveladr: int
    numqvel: int

    @staticmethod
    def create(model: mj.MjModel, jntadr: int) -> "MjCambrianJoint":
        """Create a Joint object from a Mujoco model and joint body ID."""
        qposadr = model.jnt_qposadr[jntadr]
        qveladr = model.jnt_dofadr[jntadr]

        jnt_type = model.jnt_type[jntadr]
        if jnt_type == mj.mjtJoint.mjJNT_FREE:
            numqpos = 7
            numqvel = 6
        elif jnt_type == mj.mjtJoint.mjJNT_BALL:
            numqpos = 4
            numqvel = 3
        else:  # mj.mjtJoint.mjJNT_HINGE or mj.mjtJoint.mjJNT_SLIDE
            numqpos = 1
            numqvel = 1

        return MjCambrianJoint(jntadr, qposadr, numqpos, qveladr, numqvel)


@dataclass
class MjCambrianActuator:
    """Helper class which stores information about a Mujoco actuator.

    Attributes:
        adr (int): The Mujoco actuator ID (index into model.actuator_* arrays).
        low (float): The lower bound of the actuator's range.
        high (float): The upper bound of the actuator's range.
    """

    adr: int
    low: float
    high: float


@dataclass
class MjCambrianGeometry:
    """Helper class which stores information about a Mujoco geometry

    Attributes:
        adr (int): The Mujoco geometry ID (index into model.geom_* arrays).
        rbound (float): The radius of the geometry's bounding sphere.
        pos (np.ndarray): The position of the geometry relative to the body.
    """

    adr: int
    rbound: float
    pos: np.ndarray
