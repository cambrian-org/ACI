from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
import numpy as np
from enum import Enum
import pickle

import gymnasium as gym
import mujoco as mj
from gymnasium import spaces
from stable_baselines3.common.utils import set_random_seed

from cambrian.evolution_envs.three_d.mujoco.animal import MjCambrianAnimal
from cambrian.evolution_envs.three_d.mujoco.maze import MjCambrianMaze
from cambrian.evolution_envs.three_d.mujoco.cambrian_xml import MjCambrianXML
from cambrian.evolution_envs.three_d.mujoco.config import MjCambrianConfig
from cambrian.evolution_envs.three_d.mujoco.utils import get_model_path
from cambrian.evolution_envs.three_d.mujoco.renderer import (
    MjCambrianRenderer,
    MjCambrianViewerOverlay,
    MjCambrianTextViewerOverlay,
    MjCambrianImageViewerOverlay,
    MjCambrianCursor,
    resize_with_aspect_fill,
    TEXT_HEIGHT,
    TEXT_MARGIN,
)


class MjCambrianEnv(gym.Env):
    """A MjCambrianEnv defines a gymnasium environment that's based off mujoco.

    In our context, a MjCambrianEnv contains a maze and at least one animal.

    Initialization progression goes as follows:
    - create each animal and for each
        - load the base xml to MjModel
        - parse the geometry and place eyes at the appropriate locations
        - load the actuators/joints
        - create the action/observation spaces
        - return the a new xml which includes adjustments (e.g. eyes/cameras, etc.)
    - create the environment xml (maze + animals + etc.)
    - create the main MjModel/MjData (through MujocoEnv constructor)

    NOTES:
    - This is an overridden version of the MujocoEnv class. The two main differences is
    that we allow for multiple agents and use our own custom renderer. It also reduces
    the need to create temporary xml files which MujocoEnv had to load. It's essentially
    a copy of MujocoEnv with the two aforementioned major changes.

    Args:
        config_path (str | Path | MjCambrianConfig): The path to the config file or the
            config object itself.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self, config: str | Path | MjCambrianConfig
    ):
        self._setup_config(config)

        self.animals: Dict[str, MjCambrianAnimal] = {}
        self._create_animals()

        self.xml = self.generate_xml()

        self.model = mj.MjModel.from_xml_string(self.xml.to_string())
        self.data = mj.MjData(self.model)

        self.renderer: MjCambrianRenderer = None
        self.render_mode = (
            "human" if "human" in self.renderer_config.render_modes else "rgb_array"
        )
        if self.env_config.use_renderer:
            self.renderer = MjCambrianRenderer(self.renderer_config)

        self._episode_step = 0
        self._max_episode_steps = self.config.training_config.max_episode_steps
        self._num_resets = 0

        self._record: bool = False
        self._rollout: Dict[str, Any] = {}
        self._overlays: Dict[str, Any] = {}

        self._reward_fn = self._get_reward_fn(self.env_config.reward_fn_type)

        # Used to store the optimal path for each animal in the maze
        # Each animal has a different start position so optimal path is different
        # Tuple[List, List] = [path, accumulated_path_lengths]
        self._optimal_animal_paths: Dict[str, Tuple[List, List]] = {}

    def _setup_config(self, config: str | Path | MjCambrianConfig):
        """Helper method to setup the config. This is called by the constructor."""
        self.config = MjCambrianConfig.load(config)
        self.env_config = self.config.env_config
        self.renderer_config = self.env_config.renderer_config

    def _create_animals(self):
        """Helper method to create the animals.

        Under the hood, the `create` method does the following:
            - load the base xml to MjModel
            - parse the geometry and place eyes at the appropriate locations
            - create the action/observation spaces
        """
        default_animal_config = self.config.animal_config
        for i in range(self.env_config.num_animals):
            animal_config = default_animal_config.copy()
            animal_config.idx = i
            if animal_config.name is None:
                animal_config.name = f"animal_{i}"
            assert animal_config.name not in self.animals
            self.animals[animal_config.name] = MjCambrianAnimal(animal_config)

    def generate_xml(self) -> MjCambrianXML:
        """Generates the xml for the environment."""
        xml = MjCambrianXML(get_model_path(self.env_config.scene_path))

        # Create the directional light, if desired
        if self.env_config.use_directional_light:
            xml.add(
                xml.find(".//worldbody"),
                "light",
                directional="true",
                cutoff="100",
                exponent="1",
                diffuse="1 1 1",
                specular=".1 .1 .1",
                pos="0 0 1.3",
                dir="-0 0 -1.3",
            )
        if self.env_config.maze_config.use_target_light_source is None:
            self.env_config.maze_config.use_target_light_source = (
                not self.env_config.use_directional_light
            )

        # Add the animals to the xml
        for animal in self.animals.values():
            xml += animal.generate_xml()

        # Create the maze and add it to the xml
        self.maze, maze_xml = MjCambrianMaze.make_maze(self.env_config.maze_config)
        xml += maze_xml

        # Disable the headlight
        if not self.env_config.use_headlight:
            xml.add(xml.add(xml.root, "visual"), "headlight", active="0")

        # Update the assert path to point to the fully resolved path
        compiler = xml.find(".//compiler")
        assert compiler is not None
        if (texturedir := compiler.attrib.get("texturedir")) is not None:
            texturedir = str(get_model_path(xml.base_dir / texturedir))
            compiler.attrib["texturedir"] = texturedir
        if (meshdir := compiler.attrib.get("meshdir")) is not None:
            meshdir = str(get_model_path(xml.base_dir / meshdir))
            compiler.attrib["meshdir"] = meshdir
        if (assetdir := compiler.attrib.get("assetdir")) is not None:
            assetdir = str(get_model_path(xml.base_dir / assetdir))
            compiler.attrib["assetdir"] = assetdir

        return xml

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[Any, Any]] = None
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """Reset the environment.

        Will reset all underlying components (the maze, the animals, etc.). The
        simulation will then be stepped once to ensure that the observations are
        up-to-date.

        Returns:
            Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]: The observations for each
                animal and the info dict for each animal.
        """
        super().reset(seed=seed, options=options)
        if seed is not None:
            set_random_seed(seed)

        mj.mj_resetData(self.model, self.data)

        self.maze.reset(self.model, self.data)

        info: Dict[str, Any] = {a: {} for a in self.animals}
        obs: Dict[str, Dict[str, Any]] = {}
        for name, animal in self.animals.items():
            init_qpos = (
                self.maze.cell_rowcol_to_xy(*animal.config.init_pos)
                if animal.config.init_pos is not None
                else self.maze.generate_reset_pos()
            )
            obs[name] = animal.reset(self.model, self.data, init_qpos)
            if self.env_config.use_goal_obs:
                obs[name]["goal"] = self.maze.goal.copy()

            path = self.maze.compute_optimal_path(animal.pos, self.maze.goal)
            accum_path_len = np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))
            self._optimal_animal_paths[name] = (path, accum_path_len)

            info[name]["pos"] = animal.pos

        info["maze"] = {}
        info["maze"]["goal"] = self.maze.goal

        self._step_mujoco_simulation(1)

        if self.renderer is not None:
            extent = self.model.stat.extent
            self.renderer.config.camera_config.setdefault("distance", extent)
            self.renderer.reset(self.model, self.data)

        self._episode_step = 0
        self._num_resets += 1
        if not self.record:
            self._overlays.clear()
            self._rollout.clear()

        return obs, info

    def step(
        self, action: Dict[str, Any]
    ) -> Tuple[
        Dict[str, Any],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """Step the environment.

        The dynamics is updated through the `do_simulation` method.

        Args:
            action (Dict[str, Any]): The action to take for each animal. The keys
            define the animal name, and the values define the action for that animal.

        Returns:
            Dict[str, Any]: The observations for each animal.
            Dict[str, float]: The reward for each animal.
            Dict[str, bool]: Whether each animal has terminated.
            Dict[str, bool]: Whether each animal has truncated.
            Dict[str, Dict[str, Any]]: The info dict for each animal.
        """
        obs: Dict[str, Any] = {}
        info: Dict[str, Any] = {a: {} for a in self.animals}
        for name, animal in self.animals.items():
            info[name]["prev_pos"] = animal.pos

            obs[name] = animal.step(action[name])
            if self.env_config.use_goal_obs:
                obs[name]["goal"] = self.maze.goal.copy()

            if not self.config.animal_config.disable_intensity_sensor:
                info[name]["intensity"] = animal.intensity_sensor.last_obs
            info[name]["action"] = action[name]

        info["maze"] = {}
        info["maze"]["goal"] = self.maze.goal

        self._step_mujoco_simulation(self.env_config.frame_skip)

        terminated = self.compute_terminated()
        truncated = self.compute_truncated()
        reward = self.compute_reward(terminated, truncated, info)

        self._episode_step += 1

        if not self.config.env_config.add_overlays:
            self._overlays["Step"] = self._episode_step

        if self.record:
            self._rollout.setdefault("actions", []).append(list(action.values()))
            self._rollout.setdefault("positions", []).append(
                [a.pos for a in self.animals.values()]
            )

        return obs, reward, terminated, truncated, info

    def _step_mujoco_simulation(self, n_frames):
        """Sets the mujoco simulation. Will step the simulation `n_frames` times, each
        time checking if the animal has contacts. If so, will break early (if the
        environment is configured to truncate on contact)."""
        # Check contacts at _every_ step.
        # NOTE: Doesn't process whether hits are terminal or not
        for _ in range(n_frames):
            mj.mj_step(self.model, self.data)

            # TODO: don't break here since it will effect the other animals. Instead,
            # have a `should_terminate` flag or something.
            if self.env_config.truncate_on_contact and self.data.ncon > 0:
                if any(animal.has_contacts for animal in self.animals.values()):
                    break

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mj.mj_rnePostConstraint(self.model, self.data)

    def compute_reward(
        self,
        terminated: Dict[str, bool],
        truncated: Dict[str, bool],
        info: Dict[str, bool],
    ) -> Dict[str, float]:
        """Computes the reward for the environment.

        Args:
            terminated (Dict[str, bool]): Whether each animal has terminated.
                Termination indicates success (agent has reached the goal).
            truncated (Dict[str, bool]): Whether each animal has truncated.
                Truncation indicates failure (agent has hit the wall or something).
            info (Dict[str, bool]): The info dict for each animal.
        """

        rewards: Dict[str, float] = {}
        for name, animal in self.animals.items():
            # Early exits
            if terminated[name]:
                rewards[name] = 1
                continue
            elif truncated[name]:
                rewards[name] = -1
                continue

            # Call reward_fn
            rewards[name] = self._reward_fn(animal, info[name])

            # Add a -1 to the reward if the animal has contacts and truncate_on_contact
            # is False
            if not self.env_config.truncate_on_contact and animal.has_contacts:
                rewards[name] -= 1

        return rewards

    def compute_terminated(self) -> Dict[str, bool]:
        """Compute whether the env has terminated. Termination indicates success,
        whereas truncated indicates failure."""

        terminated: Dict[str, bool] = {}
        for name, animal in self.animals.items():
            if self.env_config.terminate_at_goal:
                terminated[name] = self._is_at_goal(animal)
            else:
                terminated[name] = False

        return terminated

    def compute_truncated(self) -> bool:
        """Compute whether the env has terminated. Termination indicates success,
        whereas truncated indicates failure. Failure, for now, indicates that the
        animal has touched the wall."""

        truncated: Dict[str, bool] = {}
        for name, animal in self.animals.items():
            over_max_steps = self._episode_step >= self._max_episode_steps
            if self.env_config.truncate_on_contact:
                truncated[name] = animal.has_contacts or over_max_steps
            else:
                truncated[name] = over_max_steps

        return truncated

    def render(self) -> Dict[str, np.ndarray]:
        """Renders the environment.

        Returns:
            Dict[str, np.ndarray]: The rendered image for each render mode mapped to
                its str.

        TODO:
            - Make the cursor stuff clearer
        """

        assert self.renderer is not None, "Renderer has not been initialized! "
        "Ensure `use_renderer` is set to True in the constructor."

        renderer = self.renderer
        renderer_width = renderer.width
        renderer_height = renderer.height

        overlays: List[MjCambrianViewerOverlay] = []
        overlay_width = int(renderer_width * self.env_config.overlay_width)
        overlay_height = int(renderer_height * self.env_config.overlay_height)
        overlay_size = (overlay_width, overlay_height)

        cursor = MjCambrianCursor(x=0, y=renderer_height - TEXT_MARGIN * 2)
        for key, value in self._overlays.items():
            if issubclass(type(value), MjCambrianViewerOverlay):
                overlays.append(value)
            else:
                cursor.y -= TEXT_HEIGHT + TEXT_MARGIN
                overlays.append(MjCambrianTextViewerOverlay(f"{key}: {value}", cursor))

        if not self.config.env_config.add_overlays:
            return self.renderer.render(overlays=overlays)

        cursor = MjCambrianCursor(0, 0)
        for i, (name, animal) in enumerate(self.animals.items()):
            cursor.x += 2 * i * overlay_width
            cursor.y = 0
            if cursor.x + overlay_width * 2 > renderer_width:
                print("WARNING: Renderer width is too small!!")
                continue

            composite = animal.create_composite_image()
            if composite is None:
                # Make the composite image black so we can still render other overlays
                composite = np.zeros((*overlay_size, 3), dtype=np.uint8)
            if self.config.animal_config.disable_intensity_sensor:
                # Make the intensity image black so we can still render other overlays
                intensity = np.zeros(composite.shape, dtype=np.uint8)
            else:
                intensity = animal.intensity_sensor.last_obs

            # NOTE: flipud here since we always flipud when copying buffer from gpu,
            # and when reading the buffer again after drawing the overlay, it will be
            # flipped again. Flipping here means it will be the right side up.
            new_composite = np.flipud(resize_with_aspect_fill(composite, *overlay_size))
            new_intensity = np.flipud(resize_with_aspect_fill(intensity, *overlay_size))

            overlays.append(MjCambrianImageViewerOverlay(new_composite, cursor))

            cursor.x -= TEXT_MARGIN
            cursor.y -= TEXT_MARGIN
            lon_eyes = animal.config.num_eyes_lon
            lat_eyes = animal.config.num_eyes_lat
            overlay_text = f"LonxLat: {lon_eyes}x{lat_eyes}"
            overlays.append(MjCambrianTextViewerOverlay(overlay_text, cursor))
            cursor.y += TEXT_HEIGHT
            eye0 = next(iter(animal.eyes.values()))
            overlay_text = f"Res: {tuple(eye0.resolution)}"
            overlays.append(MjCambrianTextViewerOverlay(overlay_text, cursor))
            cursor.y += TEXT_HEIGHT
            overlay_text = f"FOV: {tuple(eye0.fov)}"
            overlays.append(MjCambrianTextViewerOverlay(overlay_text, cursor))
            cursor.y = overlay_height - TEXT_HEIGHT * 2 + TEXT_MARGIN * 2
            overlay_text = f"Animal: {name}"
            overlays.append(MjCambrianTextViewerOverlay(overlay_text, cursor))

            cursor.x += overlay_width
            cursor.y = 0

            overlays.append(MjCambrianImageViewerOverlay(new_intensity, cursor))
            overlay_text = str(intensity.shape[:2])
            overlays.append(MjCambrianTextViewerOverlay(overlay_text, cursor))
            cursor.y = overlay_height - TEXT_HEIGHT * 2 + TEXT_MARGIN * 2
            overlay_text = animal.intensity_sensor.name
            overlays.append(MjCambrianTextViewerOverlay(overlay_text, cursor))

        return renderer.render(overlays=overlays)

    @property
    def episode_step(self) -> int:
        """Returns the current episode step."""
        return self._episode_step

    @property
    def num_resets(self) -> int:
        """Returns the number of resets."""
        return self._num_resets

    @property
    def max_episode_steps(self) -> int:
        """Returns the max episode steps."""
        return self._max_episode_steps

    @property
    def overlays(self) -> Dict[str, Any]:
        """Returns the overlays."""
        return self._overlays

    @property
    def agents(self) -> List[str]:
        """Returns the agents in the environment.

        This is part of the PettingZoo API.
        """
        return list(self.agents.keys())

    @property
    def possible_agents(self) -> List[str]:
        """Returns the possible agents in the environment.

        This is part of the PettingZoo API.

        Assumes that the possible agents are the same as the agents.
        """
        return self.agents

    @property
    def observation_spaces(self) -> spaces.Dict:
        """Creates the observation spaces.

        This is part of the PettingZoo API.

        By default, this environment will support multi-animal
        observationsa/actions/etc. This method will create _all_ the obeservation
        spaces for the environment. But note that stable baselines3 only suppots single
        agent environments (i.e. non-nested spaces.Dict), so ensure you wrap this env
        with a `wrappers.MjCambrianSingleAnimalEnvWrapper` if you want to use stable
        baselines3.
        """

        # Create the observation_spaces
        observation_spaces: Dict[str, spaces.Space] = {}
        for name, animal in self.animals.items():
            observation_space: spaces.Dict = animal.observation_space
            if self.env_config.use_goal_obs:
                observation_space.spaces["goal"] = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
                )
            observation_spaces[name] = observation_space
        return spaces.Dict(observation_spaces)

    @property
    def action_spaces(self) -> spaces.Dict:
        """Creates the action spaces.

        This is part of the PettingZoo API.

        By default, this environment will support multi-animal
        observationsa/actions/etc. This method will create _all_ the action
        spaces for the environment. But note that stable baselines3 only suppots single
        agent environments (i.e. non-nested spaces.Dict), so ensure you wrap this env
        with a `wrappers.MjCambrianSingleAnimalEnvWrapper` if you want to use stable
        baselines3.
        """

        # Create the action_spaces
        action_spaces: Dict[str, spaces.Space] = {}
        for name, animal in self.animals.items():
            action_spaces[name] = animal.action_space
        return spaces.Dict(action_spaces)

    @property
    def record(self):
        """Returns whether the environment is recording."""
        return self._record

    @record.setter
    def record(self, value: bool):
        """Sets whether the environment is recording."""
        self._record = value
        self.renderer.record = value

        if not self.record:
            self._rollout.clear()

    def save(self, path: str | Path):
        """Saves the simulation output to the given path."""
        self.renderer.save(path)

        print(f"Saving rollout to {path.with_suffix('.pkl')}")
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(self._rollout, f)
        print(f"Saved rollout to {path.with_suffix('.pkl')}")

    def _is_at_goal(self, animal: MjCambrianAnimal) -> bool:
        """Returns whether the animal is at the goal."""
        return (
            np.linalg.norm(animal.pos - self.maze.goal)
            < self.env_config.distance_to_goal_threshold
        )

    # ================
    # Reward Functions

    def _get_reward_fn(self, reward_fn_type: str):
        assert reward_fn_type is not None, "reward_fn_type must be set"
        fn_name = f"_reward_fn_{reward_fn_type}"
        assert hasattr(self, fn_name), f"Unrecognized reward_fn_type {reward_fn_type}"
        return getattr(self, fn_name)

    def _reward_fn_euclidean(
        self,
        animal: MjCambrianAnimal,
        info: bool,
    ) -> float:
        """Rewards the euclidean distance to the goal."""
        current_distance_to_goal = np.linalg.norm(animal.pos - self.maze.goal)
        initial_distance_to_goal = np.linalg.norm(animal.init_pos - self.maze.goal)
        return 1 - current_distance_to_goal / initial_distance_to_goal

    def _reward_fn_euclidean_and_at_goal(
        self,
        animal: MjCambrianAnimal,
        info: bool,
    ) -> float:
        """This reward combines `reward_fn_euclidean` and `reward_fn_sparse`."""
        euclidean_reward = self._reward_fn_euclidean(animal, info)
        return 1 if self._is_at_goal(animal) else euclidean_reward

    def _reward_fn_delta_euclidean(
        self,
        animal: MjCambrianAnimal,
        info: bool,
    ) -> float:
        """Rewards the change in distance to the goal from the previous step."""
        current_distance_to_goal = np.linalg.norm(animal.pos - self.maze.goal)
        previous_distance_to_goal = np.linalg.norm(info["prev_pos"] - self.maze.goal)
        return np.clip(current_distance_to_goal - previous_distance_to_goal, 0, 1)

    def _reward_fn_delta_euclidean_w_movement(
        self,
        animal: MjCambrianAnimal,
        info: bool,
    ) -> float:
        """Same as delta_euclidean, but also rewards movement away from the initial
        position"""
        current_distance_to_goal = np.linalg.norm(animal.pos - self.maze.goal)
        previous_distance_to_goal = np.linalg.norm(info["prev_pos"] - self.maze.goal)
        delta_distance_to_goal = current_distance_to_goal - previous_distance_to_goal
        delta_distance_from_init = np.linalg.norm(animal.init_pos - animal.pos)
        return np.clip(delta_distance_to_goal + delta_distance_from_init, 0, 1)

    def _reward_fn_distance_along_path(
        self,
        animal: MjCambrianAnimal,
        info: bool,
    ) -> float:
        """Rewards the distance along the optimal path to the goal."""
        path, accum_path_len = self._optimal_animal_paths[animal.name]
        idx = np.argmin(np.linalg.norm(path[:-1] - animal.pos, axis=1))
        return accum_path_len[idx] / accum_path_len[-1]

    def _reward_fn_delta_distance_along_path(
        self,
        animal: MjCambrianAnimal,
        info: bool,
    ) -> float:
        """Rewards the distance along the optimal path to the goal."""
        path, accum_path_len = self._optimal_animal_paths[animal.name]
        idx = np.argmin(np.linalg.norm(path[:-1] - animal.pos, axis=1))
        prev_idx = np.argmin(np.linalg.norm(path[:-1] - info["prev_pos"], axis=1))
        return (accum_path_len[idx] - accum_path_len[prev_idx]) / accum_path_len[-1]

    def _reward_fn_intensity_sensor(
        self,
        animal: MjCambrianAnimal,
        info: bool,
        *,
        gamma: float = 4.0,
    ) -> float:
        """The reward is the grayscaled intensity of the a intensity sensor taken to
        the power of some gamma value multiplied by a
        scale factor (1 / max_episode_steps).
        """
        assert "intensity" in info
        intensity = np.mean(info["intensity"]) / 255.0
        return intensity ** gamma

    def _reward_fn_intensity_and_velocity(
        self,
        animal: MjCambrianAnimal,
        info: bool,
    ) -> float:
        """This reward combines `reward_fn_intensity_sensor` and
        `reward_fn_delta_euclidean`."""
        intensity_reward = self._reward_fn_intensity_sensor(animal, info)
        velocity_reward = self._reward_fn_delta_euclidean(animal, info)
        return (intensity_reward + velocity_reward) / 2

    def _reward_fn_intensity_euclidean_and_at_goal(
        self,
        animal: MjCambrianAnimal,
        info: bool,
    ) -> float:
        """This reward combines `reward_fn_intensity_sensor`,
        `reward_fn_euclidean`, and `reward_fn_sparse`."""
        intensity_reward = self._reward_fn_intensity_sensor(animal, info)
        euclidean_reward = self._reward_fn_delta_euclidean(animal, info)
        reward = (intensity_reward + euclidean_reward) / 2
        return 1 if self._is_at_goal(animal) else reward

    def _reward_fn_intensity_and_at_goal(
        self,
        animal: MjCambrianAnimal,
        info: bool,
    ) -> float:
        """The reward is the intensity whenever the animal is outside some threshold
        in terms of euclidean distance to the goal. But if it's within this threshold,
        then the reward is 1."""
        intensity_reward = self._reward_fn_intensity_sensor(animal, info)
        return 3 if self._is_at_goal(animal) else intensity_reward

    def _reward_fn_intensity_and_euclidean(
        self,
        animal: MjCambrianAnimal,
        info: bool,
    ) -> float:
        """This reward combines `reward_fn_intensity_sensor` and
        `reward_fn_euclidean`."""
        intensity_reward = self._reward_fn_intensity_sensor(animal, info)
        euclidean_reward = self._reward_fn_euclidean(animal, info)
        return (intensity_reward + euclidean_reward) / 2

    def _reward_fn_sparse(
        self,
        animal: MjCambrianAnimal,
        info: bool,
    ) -> float:
        """This reward is 1 if the animal is at the goal, 0 otherwise."""
        return 1 if self._is_at_goal(animal) else 0


def make_single_env(
    config: Path | str | MjCambrianConfig, seed: int, **kwargs
) -> MjCambrianEnv:
    """Utility function for multiprocessed MjCambrianEnv."""

    def _init():
        env = MjCambrianEnv(config, **kwargs)
        env.reset(seed=seed)
        return env

    return _init


if __name__ == "__main__":
    from utils import MjCambrianArgumentParser

    parser = MjCambrianArgumentParser()

    parser.add_argument(
        "--mj-viewer",
        action="store_true",
        help="Whether to use the mujoco viewer.",
        default=False,
    )

    parser.add_argument(
        "-t",
        "--total-timesteps",
        type=int,
        help="The number of timesteps to run the environment for.",
        default=np.inf,
    )
    parser.add_argument(
        "--record-path",
        type=str,
        help="The path to save the video to. It will save a gif and mp4. "
        "Don't specify an extension. If not specified, will not record.",
        default=None,
    )
    parser.add_argument(
        "--record-composites",
        action="store_true",
        help="Whether to record the composite image in addition to the full rendered "
        "image. Only used if `--record-path` is specified.",
    )

    args = parser.parse_args()

    config = MjCambrianConfig.load(args.config, overrides=args.overrides)
    config.use_renderer = not args.mj_viewer
    env = MjCambrianEnv(config)
    print(env.xml.write("test.xml"))
    env.reset()

    print("Running...")
    if args.mj_viewer:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(
            env.model, env.data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            while viewer.is_running():
                mj.mj_step(env.model, env.data)
                viewer.sync()
    else:
        record_composites = False
        if args.record_path is not None:
            assert (
                args.total_timesteps < np.inf
            ), "Must specify `-t\--total-timesteps` if recording."
            env.renderer.record = True
            if args.record_composites:
                record_composites = True
                composites = {k: [] for k in env.animals}

        while env.renderer.is_running() and env._episode_step < args.total_timesteps:
            env.step(env.action_spaces.sample())
            env.render()
            if record_composites:
                for name, animal in env.animals.items():
                    composite = animal.create_composite_image()
                    resized_composite = resize_with_aspect_fill(
                        composite, composite.shape[0] * 20, composite.shape[1] * 20
                    )
                    composites[name].append(resized_composite)
        env.close()

        if args.record_path is not None:
            env.renderer.save(args.record_path)
            print(f"Saved video to {args.record_path}")
            if record_composites:
                import imageio

                for name, composite in composites.items():
                    path = f"{args.record_path}_{name}_composites"
                    imageio.mimwrite(
                        f"{path}.gif",
                        composite,
                        duration=1000 * 1 / 30,
                    )
                    imageio.imwrite(f"{path}.png", composite[-1])

    print("Exiting...")
