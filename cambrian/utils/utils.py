import argparse
from typing import Any, List, Tuple, TYPE_CHECKING, Optional, Callable, Dict, Generator
from pathlib import Path
from dataclasses import dataclass
import contextlib
import ast

import gymnasium as gym
import mujoco as mj
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv

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
    execution file, absolute, or relative to the path of the cambrian folder. The
    latter is the typical method, where `assets/<model>.xml` specifies the model path
    located in REPO_PATH/models/assets/<model>.xml.

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
    env: VecEnv,
    model: "MjCambrianModel",
    num_runs: int,
    *,
    record_kwargs: Optional[Dict[str, Any]] = None,
    step_callback: Optional[Callable[[], bool]] = lambda: True,
    done_callback: Optional[Callable[[int], bool]] = lambda _: True,
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
    from cambrian.envs.env import MjCambrianEnv
    from cambrian.utils.logger import get_logger

    cambrian_env: MjCambrianEnv = env.envs[0].unwrapped
    if record_kwargs is not None:
        # don't set to `record_path is not None` directly bc this will delete overlays
        cambrian_env.record = True

    run = 0
    obs = env.reset()
    get_logger().info(f"Starting {num_runs} evaluation run(s)...")
    while run < num_runs:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)

        if done:
            get_logger().info(
                f"Run {run} done. Cumulative reward: {cambrian_env.cumulative_reward}"
            )

            if not done_callback(run):
                break

            run += 1

        env.render()

        if not step_callback():
            break

    if record_kwargs is not None:
        cambrian_env.save(**record_kwargs)
        cambrian_env.record = False


def calculate_fitness(evaluations_npz: Path) -> float:
    """Calculate the fitness of the animal. This is done by taking the 3rd quartile of
    the evaluation rewards."""
    # Return negative infinity if the evaluations file doesn't exist
    if not evaluations_npz.exists():
        return -float("inf")

    def top_25_excluding_outliers(data: np.ndarray) -> float:
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        filtered_data = data[(data > q1 - 1.5 * iqr) & (data < q3 + 1.5 * iqr)]
        return float(np.mean(np.sort(filtered_data)[-len(filtered_data) // 4 :]))

    data = np.load(evaluations_npz)
    rewards = data["results"]
    return top_25_excluding_outliers(rewards)


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

        self._args = None

    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)
        self._args = args

        return args


# =============


def generate_sequence_from_range(range: Tuple[float, float], num: int) -> List[float]:
    """"""
    return [np.average(range)] if num == 1 else np.linspace(*range, num)


def merge_dicts(d1: dict, d2: dict) -> dict:
    """Merge two dictionaries. d2 takes precedence over d1."""
    return {**d1, **d2}


@contextlib.contextmanager
def setattrs_temporary(
    *args: Tuple[Any, Dict[str, Any]]
) -> Generator[None, None, None]:
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
    try:
        yield
    finally:
        for (obj, _), kwargs in zip(args, prev_values):
            for attr, value in kwargs.items():
                if isinstance(obj, dict):
                    obj[attr] = value
                else:
                    setattr(obj, attr, value)


def get_gpu_memory_usage(return_total_memory: bool = True) -> Tuple[float, float]:
    """Get's the total and used memory of the GPU in GB."""
    assert torch.cuda.is_available(), "No CUDA device available"
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
    free_memory = torch.cuda.mem_get_info()[0] / 1024**3
    used_memory = total_memory - free_memory

    if return_total_memory:
        return used_memory, total_memory
    else:
        return used_memory


def get_observation_space_size(observation_space: gym.spaces.Space) -> int:
    """Get the size of an observation space. Returns size in GB."""
    if isinstance(observation_space, gym.spaces.Box):
        return np.prod(observation_space.shape) / 1024**3
    elif isinstance(observation_space, gym.spaces.Discrete):
        return observation_space.n / 1024**3
    elif isinstance(observation_space, gym.spaces.Tuple):
        return sum(
            get_observation_space_size(space) for space in observation_space.spaces
        )
    elif isinstance(observation_space, gym.spaces.Dict):
        return sum(
            get_observation_space_size(space)
            for space in observation_space.spaces.values()
        )
    else:
        raise ValueError(
            f"Unsupported observation space type: {type(observation_space)}"
        )


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


def get_sensor_id(model: mj.MjModel, sensor_name: str) -> int:
    """Get the ID of a Mujoco sensor."""
    return mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, sensor_name)


def get_sensor_name(model: mj.MjModel, sensoradr: int) -> str:
    """Get the name of a Mujoco sensor."""
    return mj.mj_id2name(model, mj.mjtObj.mjOBJ_SENSOR, sensoradr)


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

    @property
    def ctrlrange(self) -> float:
        return self.high - self.low


@dataclass
class MjCambrianGeometry:
    """Helper class which stores information about a Mujoco geometry

    Attributes:
        id (int): The Mujoco geometry ID (index into model.geom_* arrays).
        rbound (float): The radius of the geometry's bounding sphere.
        pos (np.ndarray): The position of the geometry relative to the body.
        group (int): The geometry group the geometry belongs to.
    """

    id: int
    rbound: float
    pos: np.ndarray
    group: int


# =============
# Misc utils


def literal_eval_with_callables(
    node_or_string, safe_callables: Dict[str, Callable] = {}
):
    """
    Safely evaluate an expression node or a string containing a Python expression.
    The expression can contain literals, lists, tuples, dicts, unary and binary
    operators. Calls to functions specified in 'safe_callables' dictionary are allowed.

    Args:
        node_or_string (ast.Node or str): The expression node or string to evaluate.
        safe_callables (Dict[str, Callable]): A dictionary mapping function names to
            callable Python objects. Only these functions can be called within the
            expression.

    Returns:
        The result of the evaluated expression.

    Raises:
        ValueError: If the expression contains unsupported or malformed nodes or tries
            to execute unsupported operations.

    Examples:
        >>> literal_eval_with_callables("1 + 2")
        3
        >>> literal_eval_with_callables("sqrt(4)", {'sqrt': math.sqrt})
        2.0
    """
    if isinstance(node_or_string, str):
        string = node_or_string
        node = ast.parse(node_or_string, mode="eval").body
    else:
        node = node_or_string
        string = ast.dump(node)

    op_map = {
        ast.Add: lambda x, y: x + y,
        ast.Sub: lambda x, y: x - y,
        ast.Mult: lambda x, y: x * y,
        ast.Div: lambda x, y: x / y,
        ast.Mod: lambda x, y: x % y,
        ast.Pow: lambda x, y: x**y,
        ast.FloorDiv: lambda x, y: x // y,
        ast.And: lambda x, y: x and y,
        ast.Or: lambda x, y: x or y,
        ast.Eq: lambda x, y: x == y,
        ast.NotEq: lambda x, y: x != y,
        ast.Lt: lambda x, y: x < y,
        ast.LtE: lambda x, y: x <= y,
        ast.Gt: lambda x, y: x > y,
        ast.GtE: lambda x, y: x >= y,
        ast.Is: lambda x, y: x is y,
        ast.IsNot: lambda x, y: x is not y,
        ast.In: lambda x, y: x in y,
        ast.NotIn: lambda x, y: x not in y,
        ast.BitAnd: lambda x, y: x & y,
        ast.BitOr: lambda x, y: x | y,
        ast.BitXor: lambda x, y: x ^ y,
        ast.LShift: lambda x, y: x << y,
        ast.RShift: lambda x, y: x >> y,
        ast.Invert: lambda x: ~x,
    }

    def _convert(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, (ast.Tuple, ast.List)):
            return type(node.elts)(map(_convert, node.elts))
        elif isinstance(node, ast.Dict):
            return {_convert(k): _convert(v) for k, v in zip(node.keys, node.values)}
        elif isinstance(node, ast.UnaryOp):
            operand = _convert(node.operand)
            return operand if isinstance(node.op, ast.UAdd) else -operand
        elif isinstance(node, ast.BinOp) and type(node.op) in op_map:
            left = _convert(node.left)
            right = _convert(node.right)
            return op_map[type(node.op)](left, right)
        elif isinstance(node, ast.BoolOp) and type(node.op) in op_map:
            return op_map[type(node.op)](
                _convert(node.values[0]), _convert(node.values[1])
            )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Name
            ):
                obj = safe_callables.get(node.func.value.id)
                if obj is not None and hasattr(obj, node.func.attr):
                    method = getattr(obj, node.func.attr)
                    if callable(method):
                        return method(
                            *map(_convert, node.args),
                            **{kw.arg: _convert(kw.value) for kw in node.keywords},
                        )
            elif isinstance(node.func, ast.Name) and node.func.id in safe_callables:
                func = safe_callables[node.func.id]
                if callable(func):
                    return func(
                        *map(_convert, node.args),
                        **{kw.arg: _convert(kw.value) for kw in node.keywords},
                    )

        raise ValueError(f"Unsupported node type: {type(node)}")

    try:
        return _convert(node)
    except ValueError as e:
        raise ValueError(f"Error evaluating expression: {string}") from e
