"""Defines the MjCambrianEnv class."""

import pickle
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Self, Tuple

import mujoco as mj
import numpy as np
from gymnasium import Env, spaces
from hydra_config import HydraContainerConfig, config_wrapper
from pettingzoo import ParallelEnv

from cambrian.agents.agent import MjCambrianAgent, MjCambrianAgentConfig
from cambrian.renderer import (
    MjCambrianRenderer,
    MjCambrianRendererConfig,
    MjCambrianRendererSaveMode,
)
from cambrian.renderer.overlays import MjCambrianCursor, MjCambrianViewerOverlay
from cambrian.utils.cambrian_xml import MjCambrianXML, MjCambrianXMLConfig
from cambrian.utils.logger import get_logger
from cambrian.utils.spec import MjCambrianSpec, spec_from_xml
from cambrian.utils.types import (
    ActionType,
    InfoType,
    MjCambrianRewardFn,
    MjCambrianStepFn,
    MjCambrianTerminationFn,
    MjCambrianTruncationFn,
    ObsType,
    RenderFrame,
    RewardType,
    TerminatedType,
    TruncatedType,
)

# ======================


@config_wrapper
class MjCambrianEnvConfig(HydraContainerConfig):
    """Defines a config for the cambrian environment.

    Attributes:
        instance (Callable[[Self], "MjCambrianEnv"]): The class method to use to
            instantiate the environment.

        xml (MjCambrianXMLConfig): The xml for the scene. This is the xml that will be
            used to create the environment. See `MjCambrianXML` for more info.

        step_fn (MjCambrianStepFn): The step function to use. See the `MjCambrianStepFn`
            for more info. The step fn is called before the termination, truncation, and
            reward fns, and after the action has been applied to the agents. It takes
            the environment, the observations, the info dict, and any additional kwargs.
            Returns the updated observations and info dict.
        termination_fn (MjCambrianTerminationFn): The termination function to use. See
            the :class:`MjCambrianTerminationFn` for more info.
        truncation_fn (MjCambrianTruncationFn): The truncation function to use. See the
            :class:`MjCambrianTruncationFn` for more info.
        reward_fn (MjCambrianRewardFn): The reward function type to use. See the
            :class:`MjCambrianRewardFn` for more info.

        frame_skip (int): The number of mujoco simulation steps per `gym.step()` call.
        max_episode_steps (int): The maximum number of steps per episode.
        n_eval_episodes (int): The number of episodes to evaluate for.

        add_overlays (bool): Whether to add overlays or not.
        clear_overlays_on_reset (bool): Whether to clear the overlays on reset or not.
            Consequence of setting to False is that when drawing position overlays
            and when mazes change between evaluations, the sites will be drawn on top
            of each other which may not be desired. When record is False, the overlays
            are always be cleared.
        debug_overlays_size (float): The size of the debug overlays. This is a
            percentage of the total renderer size. If 0, debug overlays are disabled.
        renderer (Optional[MjCambrianViewerConfig]): The default viewer config to
            use for the mujoco viewer. If unset, no renderer will be used. Should
            set to None if `render` will never be called. This may be useful to
            reduce the amount of vram consumed by non-rendering environments.

        save_filename (Optional[str]): The filename to save recordings to. This is more
            of a placeholder for external scripts to use, if desired.

        agents (List[MjCambrianAgentConfig]): The configs for the agents.
            The key will be used as the default name for the agent, unless explicitly
            set in the agent config.
    """

    instance: Callable[[Self], "MjCambrianEnv"]

    xml: MjCambrianXMLConfig

    step_fn: MjCambrianStepFn
    termination_fn: MjCambrianTerminationFn
    truncation_fn: MjCambrianTruncationFn
    reward_fn: MjCambrianRewardFn

    frame_skip: int
    max_episode_steps: int
    n_eval_episodes: int

    add_overlays: bool
    clear_overlays_on_reset: bool
    debug_overlays_size: float
    renderer: Optional[MjCambrianRendererConfig] = None

    save_filename: Optional[str] = None

    agents: Dict[str, MjCambrianAgentConfig | Any]


class MjCambrianEnv(ParallelEnv, Env):
    """A MjCambrianEnv defines a gymnasium environment that's based off mujoco.

    NOTES:
    - This is an overridden version of the MujocoEnv class. The two main differences is
    that we allow for /reset multiple agents and use our own custom renderer. It also
    reduces the need to create temporary xml files which MujocoEnv had to load. It's
    essentially a copy of MujocoEnv with the two aforementioned major changes.

    Args:
        config (MjCambrianEnvConfig): The config object.
        name (Optional[str]): The name of the environment. This is added as an overlay
            to the renderer.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: MjCambrianEnvConfig, name: Optional[str] = None):
        self._config = config
        self._name = name or self.__class__.__name__

        self._agents: Dict[str, MjCambrianAgent] = {}
        self._create_agents()

        self._xml = self.generate_xml()
        try:
            self._spec = spec_from_xml(self._xml)
            self._spec.compile()
        except Exception:
            get_logger().error(f"Error creating model\n{self._xml.to_string()}")
            raise

        self._spec.env = self

        self.render_mode = "rgb_array"
        self._renderer: MjCambrianRenderer = None
        if renderer_config := self._config.renderer:
            if "human" in self._config.renderer.render_modes:
                self.render_mode = "human"
            self._renderer = MjCambrianRenderer(renderer_config)

        self._episode_step = 0
        self._max_episode_steps = self._config.max_episode_steps
        self._num_resets = 0
        self._num_timesteps = 0
        self._stashed_cumulative_reward = 0
        self._cumulative_reward = 0
        self._timings = deque(maxlen=25)

        self._record: bool = False
        self._rollout: Dict[str, Any] = {}
        self._overlays: Dict[str, Any] = {}

        # We'll store the info dict as a state within this class so that the truncation,
        # termination, and reward functions can use it for keeping a state. Like passing
        # the info dict to these functions allows them to edit them and keep around
        # information that is helpful for subsequent calls. It will always be reset
        # during the reset method and will only be maintained during an episode length.
        # Because the info dict is treated as stateful, take care in not adding new keys
        # on each step, as this will cause the info dict to grow until the end of the
        # episode.
        self._info: Dict[str, Dict[str, Any]]

    def _create_agents(self):
        """Helper method to create the agents."""
        for name, agent_config in self._config.agents.items():
            assert name not in self._agents, f"Agent {name} already exists."
            self._agents[name] = agent_config.instance(agent_config, name)

    def generate_xml(self) -> MjCambrianXML:
        """Generates the xml for the environment.

        .. todo::

            Can we update to use MjSpec?
        """
        xml = MjCambrianXML.from_string(self._config.xml)

        # Add the agents to the xml
        for agent in self._agents.values():
            xml += agent.generate_xml()

        return xml

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[Any, Any]] = None
    ) -> Tuple[ObsType, InfoType]:
        """Reset the environment.

        Will reset all underlying components (the maze, the agents, etc.). The
        simulation will then be stepped once to ensure that the observations are
        up-to-date.

        Returns:
            Tuple[ObsType, InfoType]: The observations for each
                agent and the info dict for each agent.
        """
        if seed is not None and self._num_resets == 0:
            self.set_random_seed(seed)

        # First, reset the mujoco simulation
        mj.mj_resetData(self._spec.model, self._spec.data)

        # Reset the info dict. We'll update the stateful info dict here, as well.
        info: Dict[str, Dict[str, Any]] = {a: {} for a in self._agents}
        self._info = info

        # Then, reset the agents
        obs: Dict[str, Dict[str, Any]] = {}
        for name, agent in self._agents.items():
            obs[name] = agent.reset(self._spec)

        # Recompile the model/data
        self._spec.recompile()

        # 'll step the simulation once to allow for states to propagate
        self._step_mujoco_simulation(1, info)

        # Now update the info dict
        if self._renderer is not None:
            self._renderer.reset(self._spec)

        # Update metadata variables
        self._episode_step = 0
        self._stashed_cumulative_reward = self._cumulative_reward
        self._cumulative_reward = 0
        self._num_resets += 1
        self._timings.clear()
        self._timings.append(time.time())

        # Reset the caches
        if not self._record:
            self._rollout.clear()
            self._overlays.clear()
        elif self._config.clear_overlays_on_reset:
            self._overlays.clear()

        if self._record:
            # TODO make this cleaner
            self._rollout.setdefault("actions", [])
            self._rollout["actions"].append(
                [np.zeros_like(a.action_space.sample()) for a in self._agents.values()]
            )
            self._rollout.setdefault("positions", [])
            self._rollout["positions"].append([a.qpos for a in self._agents.values()])

        return self._config.step_fn(self, obs, info)

    def step(
        self, action: ActionType
    ) -> Tuple[ObsType, RewardType, TerminatedType, TruncatedType, InfoType]:
        """Step the environment.

        The dynamics is updated through the `_step_mujoco_simulation` method.

        Args:
            action (Dict[str, Any]): The action to take for each agent.
                The keys define the agent name, and the values define the action for
                that agent.

        Returns:
            Dict[str, Any]: The observations for each agent.
            Dict[str, float]: The reward for each agent.
            Dict[str, bool]: Whether each agent has terminated.
            Dict[str, bool]: Whether each agent has truncated.
            Dict[str, Dict[str, Any]]: The info dict for each agent.
        """
        info = self._info

        # First, apply the actions to the agents and step the simulation
        # print(self.observation_spaces)
        for name, agent in self._agents.items():
            # if name in action:
            #     print(f"Agent action pair: {name}:{action[name].shape}")
            # else:
            #     print("Action for ", name, "not found in action dict. Available actions:", action.keys())
            if not agent.trainable or agent.config.use_privileged_action:
                if not agent.trainable and name in action:
                    get_logger().warning(
                        f"Action for {name} found in action dict. "
                        "This will be overridden by the agent.",
                        extra={"once": True},
                    )
                action[name] = agent.get_action_privileged(self)

            assert name in action, f"Action for {name} not found in action dict."
            agent.apply_action(action[name])
            info[name]["prev_pos"] = agent.pos.copy()
            info[name]["action"] = action[name]

        # Then, step the mujoco simulation
        self._step_mujoco_simulation(self._config.frame_skip, info)

        # We'll then step each agent to render it's current state and get the obs
        obs: Dict[str, Any] = {}
        for name, agent in self._agents.items():
            obs[name] = agent.step()
        # Call helper methods to update the observations, rewards, terminated, and info
        obs, info = self._config.step_fn(self, obs, info)
        terminated = self._compute_terminated(info)
        truncated = self._compute_truncated(info)
        reward = self._compute_reward(terminated, truncated, info)

        self._episode_step += 1
        self._num_timesteps += 1
        self._cumulative_reward += sum(reward.values())

        if self._record:
            self._rollout["actions"].append(list(action.values()))
            self._rollout["positions"].append([a.pos for a in self._agents.values()])

        if (
            self._config.debug_overlays_size > 0
            and self._record
            or "human" in self._config.renderer.render_modes
        ):
            self._overlays["Name"] = self._name
            self._overlays["Total Timesteps"] = self.num_timesteps
            self._overlays["Step"] = self._episode_step
            self._overlays["Cumulative Reward"] = round(self._cumulative_reward, 2)

            self._timings.append(time.time())
            fps = (len(self._timings) - 1) / (self._timings[-1] - self._timings[0])
            self._overlays["FPS"] = round(fps, 2)

        return obs, reward, terminated, truncated, info

    def _step_mujoco_simulation(self, n_frames: int, info: InfoType):
        """Sets the mujoco simulation. Will step the simulation `n_frames` times, each
        time checking if the agent has contacts."""
        # Initially set has_contacts to False for all agents
        for name in self._agents:
            info[name]["has_contacts"] = False

        # Check contacts at _every_ step.
        # NOTE: Doesn't process whether hits are terminal or not
        for _ in range(n_frames):
            mj.mj_step(self._spec.model, self._spec.data)

            # Check for contacts. We won't break here, but we'll store whether an
            # agent has contacts or not. If we didn't store during the simulation
            # step, contact checking would only occur after the frame skip, meaning
            # that, if during the course of the frame skip, the agent hits an object
            # and then moves away, the contact would not be detected.
            if self._spec.data.ncon > 0:
                for name, agent in self._agents.items():
                    if not info[name]["has_contacts"]:
                        # Only check for has contacts if it hasn't been set to True
                        # This reduces redundant checks
                        info[name]["has_contacts"] = agent.has_contacts

    def _compute_terminated(self, info: InfoType) -> TerminatedType:
        """Compute whether the env has terminated. Termination indicates success,
        whereas truncated indicates failure.

        The default implementation will always return False for all agents. This can
        be overridden in subclasses to provide custom termination conditions.
        """

        terminated: Dict[str, bool] = {}
        for name, agent in self._agents.items():
            terminated[name] = self._config.termination_fn(self, agent, info[name])

        return terminated

    def _compute_truncated(self, info: InfoType) -> TruncatedType:
        """Compute whether the env has terminated. Termination indicates success,
        whereas truncated indicates failure.

        The default implementation will always return False for all agents. This can
        be overridden in subclasses to provide custom termination conditions.
        """

        truncated: Dict[str, bool] = {}
        for name, agent in self._agents.items():
            truncated[name] = self._config.truncation_fn(self, agent, info[name])

        return truncated

    def _compute_reward(
        self,
        terminated: TerminatedType,
        truncated: TruncatedType,
        info: InfoType,
    ) -> RewardType:
        """Computes the reward for the environment.

        Args:
            terminated (TerminatedType): Whether each agent has terminated.
                Termination indicates success (agent has reached the goal).
            truncated (TruncatedType): Whether each agent has truncated.
                Truncation indicates failure (agent has hit the wall or something).
            info (InfoType): The info dict for each agent.
        """

        rewards: Dict[str, float] = {}
        for name, agent in self._agents.items():
            rewards[name] = self._config.reward_fn(
                self, agent, terminated[name], truncated[name], info[name]
            )

        return rewards

    def render(self) -> RenderFrame:
        """Renders the environment.

        Returns:
            RenderFrame: The rendered frame.
        """

        assert self._renderer is not None, "Renderer has not been initialized! "
        "Ensure `use_renderer` is set to True in the constructor."

        overlays = []
        if self._config.add_overlays:
            cursor = MjCambrianCursor(
                self._renderer.width,
                self._renderer.height,
                position=MjCambrianCursor.Position.TOP_LEFT,
            )
            for key, value in self._overlays.items():
                if issubclass(type(value), MjCambrianViewerOverlay):
                    overlay = value
                else:
                    overlay = MjCambrianViewerOverlay.create_text_overlay(
                        f"{key}: {value}"
                    )
                cursor = overlay.place(cursor)
                overlays.append(overlay)

            if self._config.debug_overlays_size > 0:
                overlays.extend(self._generate_overlays())

        return self._renderer.render(overlays=overlays)

    def _generate_overlays(self) -> List[MjCambrianViewerOverlay]:
        overlays: List[MjCambrianViewerOverlay] = []

        renderer_width = self._renderer.width
        renderer_height = self._renderer.height

        trainable_agents = {n: a for n, a in self._agents.items() if a.trainable}
        num_agents = len(trainable_agents)
        overlay_width = int(renderer_width // num_agents) if num_agents > 0 else 0
        overlay_height = int(renderer_height * self._config.debug_overlays_size)
        cursor = MjCambrianCursor(
            overlay_width, overlay_height, position=MjCambrianCursor.Position.TOP_LEFT
        )
        for agent in self._agents.values():
            agent_overlays = agent.render()
            for overlay in agent_overlays:
                cursor = overlay.place(cursor)
                overlays.append(overlay)
        return overlays

    @property
    def name(self) -> str:
        """Returns the name of the environment."""
        return self._name

    @property
    def xml(self) -> MjCambrianXML:
        """Returns the xml for the environment."""
        return self._xml

    @property
    def agents(self) -> Dict[str, MjCambrianAgent]:
        """Returns the agents in the environment."""
        return self._agents

    @property
    def renderer(self) -> MjCambrianRenderer:
        """Returns the renderer for the environment."""
        return self._renderer

    @property
    def spec(self) -> MjCambrianSpec:
        """Returns the mujoco spec for the environment."""
        return self._spec

    @property
    def model(self) -> mj.MjModel:
        """Returns the mujoco model for the environment."""
        return self._spec.model

    @property
    def data(self) -> mj.MjData:
        """Returns the mujoco data for the environment."""
        return self._spec.data

    @property
    def episode_step(self) -> int:
        """Returns the current episode step."""
        return self._episode_step

    @property
    def num_timesteps(self) -> int:
        """Returns the number of timesteps."""
        return self._num_timesteps

    @property
    def max_episode_steps(self) -> int:
        """Returns the max episode steps."""
        return self._max_episode_steps

    @property
    def overlays(self) -> Dict[str, Any]:
        """Returns the overlays."""
        return self._overlays

    @property
    def cumulative_reward(self) -> float:
        """Returns the cumulative reward."""
        return self._cumulative_reward

    @property
    def stashed_cumulative_reward(self) -> float:
        """Returns the previous cumulative reward."""
        return self._stashed_cumulative_reward

    @property
    def num_agents(self) -> int:
        """Returns the number of agents in the environment.

        This is part of the PettingZoo API.
        """
        return len(self.agents)

    @property
    def possible_agents(self) -> List[str]:
        """Returns the possible agents in the environment.

        This is part of the PettingZoo API.

        Assumes that the possible agents are the same as the agents.
        """
        return list(self._agents.keys())

    @property
    def observation_spaces(self) -> spaces.Dict:
        """Creates the observation spaces.

        This is part of the PettingZoo API.

        By default, this environment will support multi-agent
        observations/actions/etc. This method will create _all_ the observation
        spaces for the environment. But note that stable baselines3 only supports single
        agent environments (i.e. non-nested spaces.Dict), so ensure you wrap this env
        with a `wrappers.MjCambrianSingleagentEnvWrapper` if you want to use stable
        baselines3.
        """

        # Create the observation_spaces
        observation_spaces: Dict[str, spaces.Space] = {}
        for name, agent in self._agents.items():
            if agent.trainable:
                # print("name, agent.observation_space: ", name, agent.observation_space)
                observation_spaces[name] = agent.observation_space

        # print(f"#################Observation spaces: {observation_spaces}")
        return spaces.Dict(observation_spaces)

    @property
    def action_spaces(self) -> spaces.Dict:
        """Creates the action spaces.

        This is part of the PettingZoo API.

        By default, this environment will support multi-agent
        observations/actions/etc. This method will create _all_ the action
        spaces for the environment. But note that stable baselines3 only supports single
        agent environments (i.e. non-nested spaces.Dict), so ensure you wrap this env
        with a `wrappers.MjCambrianSingleagentEnvWrapper` if you want to use stable
        baselines3.
        """

        # Create the action_spaces
        action_spaces: Dict[str, spaces.Space] = {}
        for name, agent in self._agents.items():
            if agent.trainable:
                action_spaces[name] = agent.action_space
        return spaces.Dict(action_spaces)

    def observation_space(self, agent: str) -> spaces.Space:
        """Returns the observation space for the given agent.

        This is part of the PettingZoo API.
        """
        assert agent in list(
            self.observation_spaces.keys()
        ), f"Agent {agent} not found. Available: {list(self.observation_spaces.keys())}"
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        """Returns the action space for the given agent.

        This is part of the PettingZoo API.
        """
        assert agent in list(
            self.action_spaces.keys()
        ), f"Agent {agent} not found. Available: {list(self.action_spaces.keys())}"
        return self.action_spaces[agent]

    def state(self) -> np.ndarray:
        """Returns the state of the environment.

        This is part of the PettingZoo API.
        """
        raise NotImplementedError("Not implemented yet.")

    def set_random_seed(self, seed: int | float | None):
        """Sets the seed for the environment."""
        from stable_baselines3.common.utils import set_random_seed

        if seed is None:
            return

        get_logger().info(f"Setting random seed to {seed}")
        set_random_seed(seed)

    def record(self, record: bool = True, *, path: Optional[Path] = None) -> bool:
        """Sets whether the environment is recording."""
        self._record = record
        self._renderer.record(record, path=path)

        if not self._record:
            self._rollout.clear()

    def save(self, path: str | Path, *, save_pkl: bool = False, **kwargs):
        """Saves the simulation output to the given path."""
        self._renderer.save(path, **kwargs)

        if save_pkl:
            get_logger().info(f"Saving rollout to {path.with_suffix('.pkl')}")
            with open(path.with_suffix(".pkl"), "wb") as f:
                pickle.dump(self._rollout, f)
            get_logger().debug(f"Saved rollout to {path.with_suffix('.pkl')}")

    def close(self):
        """Closes the environment."""
        pass


if __name__ == "__main__":
    import argparse

    from cambrian import MjCambrianConfig, run_hydra

    REGISTRY = {}

    def register_fn(fn: Callable):
        REGISTRY[fn.__name__] = fn
        return fn

    @register_fn
    def run_mj_viewer(config: MjCambrianConfig, **__):
        import mujoco.viewer

        env = config.env.instance(config.env)
        env.reset(seed=config.seed)
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            while viewer.is_running():
                # env.step(env.action_spaces.sample())
                viewer.sync()

    @register_fn
    def run_renderer(config: MjCambrianConfig, *, record: bool, no_step: bool, **__):
        config.save(config.expdir / "config.yaml")

        env = config.env.instance(config.eval_env)
        env.record(record, path=config.expdir)

        env.reset(seed=config.seed)
        env.spec.save(config.expdir / "env.xml")

        action = {name: [-1.0, -0.0] for name, a in env.agents.items() if a.trainable}
        # print("Action in run_renderer: ", action)
        env.step(action.copy())

        if "human" in config.env.renderer.render_modes:
            import glfw

            def custom_key_callback(_, key, *args, **__):
                if key == glfw.KEY_R:
                    env.reset()
                elif key == glfw.KEY_UP:
                    name = next(iter(action.keys()))
                    action[name][0] += 0.001
                    action[name][0] = min(1.0, action[name][0])
                elif key == glfw.KEY_DOWN:
                    name = next(iter(action.keys()))
                    action[name][0] -= 0.001
                    action[name][0] = max(-1.0, action[name][0])
                elif key == glfw.KEY_LEFT:
                    name = next(iter(action.keys()))
                    action[name][1] -= 0.01
                    action[name][1] = max(-1.0, action[name][1])
                elif key == glfw.KEY_RIGHT:
                    name = next(iter(action.keys()))
                    action[name][1] += 0.01
                    action[name][1] = min(1.0, action[name][1])
                elif key == glfw.KEY_S:
                    get_logger().info(f"Saving env to {config.expdir / 'env.xml'}")
                    mj.mj_saveLastXML(str(config.expdir / "env.xml"), env.model)

            env.renderer.viewer.custom_key_callback = custom_key_callback

        while env.renderer.is_running():
            if env.episode_step > env.max_episode_steps:
                break

            if not no_step:
                env.step(action.copy())

            env.render()

            if record:
                for name, agent in env.agents.items():
                    if not agent.trainable:
                        continue

        if record:
            env.save(
                config.expdir / "eval",
                save_pkl=False,
                save_mode=MjCambrianRendererSaveMode.MP4
                | MjCambrianRendererSaveMode.GIF
                | MjCambrianRendererSaveMode.PNG,
            )

    def main(config: MjCambrianConfig, *, fn: str, **kwargs):
        if fn not in REGISTRY:
            raise ValueError(f"Unknown function {fn}")
        REGISTRY[fn](config, **kwargs)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fn", type=str, help="The method to run.", choices=REGISTRY.keys()
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record the simulation.",
        default=False,
    )
    parser.add_argument(
        "--no-step",
        action="store_true",
        help="Don't step the environment. Useful for debugging.",
        default=False,
    )
    run_hydra(main, parser=parser)
