"""This module contains the trainer class for training and evaluating agents."""

from pathlib import Path
from typing import Callable, Concatenate, Dict, Optional, Type

from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecMonitor,
)

from cambrian.envs.env import MjCambrianEnv, MjCambrianEnvConfig
from cambrian.ml.model import MjCambrianModel
from cambrian.utils import evaluate_policy
from cambrian.utils.config.config import (
    MjCambrianBaseConfig,
    MjCambrianConfig,
    config_wrapper,
)
from cambrian.utils.logger import get_logger
from cambrian.utils.wrappers import make_wrapped_env


@config_wrapper
class MjCambrianTrainerConfig(MjCambrianBaseConfig):
    """Settings for the training process. Used for type hinting.

    Attributes:
        total_timesteps (int): The total number of timesteps to train for.
        max_episode_steps (int): The maximum number of steps per episode.
        n_envs (int): The number of parallel environments to use for training.

        model (Type[MjCambrianModel]): The model to use for training.
        callbacks (Dict[str, BaseCallback]): The callbacks to use for training.
        wrappers (Dict[str, Callable[[VecEnv], VecEnv]] | None): The wrappers to use for
            training. If None, will ignore.

        prune_fn (Optional[Callable[[MjCambrianConfig], bool]]): The function to use to
            determine if an experiment should be pruned. If None, will ignore. If set,
            this function will be called prior to training to check whether the config
            is valid for training. This is the get around the fact that some sweepers
            will evaluate configs that are invalid for training, which is a waste
            computationally. The train method will return -inf if this function returns
            True. NOTE: for nevergrad, it is recommended to use cheap_constraints.
        fitness_fn (Callable[[MjCambrianConfig, float]]): The function to use to
            calculate the fitness of the agent after training.
    """

    total_timesteps: int
    max_episode_steps: int
    n_envs: int

    model: Type[MjCambrianModel]
    callbacks: Dict[str, BaseCallback | Callable[[VecEnv], BaseCallback]]
    wrappers: Dict[str, Callable[[VecEnv], VecEnv] | None]

    prune_fn: Optional[Callable[[Concatenate[MjCambrianConfig, ...]], bool]] = None
    fitness_fn: Callable[Concatenate[MjCambrianConfig, ...], float]


class MjCambrianTrainer:
    """This is the trainer class for running training and evaluation.

    Args:
        config (MjCambrianConfig): The config to use for training and evaluation.
    """

    def __init__(self, config: MjCambrianConfig):
        self._config = config

        self._config.expdir.mkdir(parents=True, exist_ok=True)

        self._logger = get_logger()
        self._logger.info(f"Logging to {self._config.expdir / 'logs'}...")

    def train(self) -> float:
        """Train the agent."""

        # Set to warn so we have something output to the error log
        self._logger.warning(f"Training the agent in {self._config.expdir}...")

        self._config.save(self._config.expdir / "config.yaml")
        self._config.pickle(self._config.expdir / "config.pkl")

        # Delete an existing finished file, if it exists
        if (finished := self._config.expdir / "finished").exists():
            finished.unlink()

        # Prune the experiment, if necessary
        if (prune_fn := self._config.trainer.prune_fn) and prune_fn(self._config):
            Path(self._config.expdir / "pruned").touch()
            return -float("inf")

        # Setup the environment, model, and callbacks
        env = self._make_env(self._config.env, self._config.trainer.n_envs)
        eval_env = self._make_env(self._config.eval_env, 1, monitor="eval_monitor.csv")
        callback = self._make_callback(eval_env)
        model = self._make_model(env)

        # Save the eval environments xml
        # All xml's _should_ be the same
        xml_path = self._config.expdir / "env.xml"
        cambrian_env: MjCambrianEnv = eval_env.envs[0].unwrapped
        cambrian_env.xml.write(xml_path)

        # Start training
        total_timesteps = self._config.trainer.total_timesteps
        model.learn(total_timesteps=total_timesteps, callback=callback)
        self._logger.info("Finished training the agent...")

        # Save the policy
        self._logger.info(f"Saving model to {self._config.expdir}...")
        model.save_policy(self._config.expdir)
        self._logger.debug(f"Saved model to {self._config.expdir}...")

        # The finished file indicates to the evo script that the agent is done
        Path(self._config.expdir / "finished").touch()

        # Calculate fitness
        fitness = self._config.trainer.fitness_fn(self._config)
        self._logger.info(f"Final Fitness: {fitness}")

        # Save the final fitness to a file
        with open(self._config.expdir / "train_fitness.txt", "w") as f:
            f.write(str(fitness))

        return fitness

    def eval(self, *, filename: Optional[Path | str] = None) -> float:
        self._config.save(self._config.expdir / "eval_config.yaml")

        eval_env = self._make_env(self._config.eval_env, 1, monitor="eval_monitor.csv")
        model = self._make_model(eval_env)
        if (self._config.expdir / "best_model.zip").exists():
            self._logger.info("Loading best model...")
            model = model.load(self._config.expdir / "best_model")

        # Save the eval environments xml
        xml_path = self._config.expdir / "eval_env.xml"
        cambrian_env: MjCambrianEnv = eval_env.envs[0].unwrapped
        cambrian_env.xml.write(xml_path)

        n_runs = self._config.eval_env.n_eval_episodes
        filename = self._config.eval_env.save_filename
        record_kwargs = dict(
            path=self._config.expdir / filename,
            save_mode=self._config.eval_env.renderer.save_mode,
        )
        evaluate_policy(eval_env, model, n_runs, record_kwargs=record_kwargs)

        # Calculate fitness
        fitness = self._config.trainer.fitness_fn(self._config)
        self._logger.info(f"Final Fitness: {fitness}")

        # Save the final fitness to a file
        with open(self._config.expdir / f"{filename}_fitness.txt", "w") as f:
            f.write(str(fitness))

        return fitness

    def test(self) -> float:
        """This is a test method which tests the evolutionary loop. It will return a
        fitness consistent with the expected performance of the agent given the config.
        This can be used to test the optimizer."""

        self._config.save(self._config.expdir / "test_config.yaml")

        # Calculate fitness
        fitness = self._config.trainer.fitness_fn(self._config)
        self._logger.info(f"Final Fitness: {fitness}")

        # Save the final fitness to a file
        with open(self._config.expdir / "test_fitness.txt", "w") as f:
            f.write(str(fitness))

        return fitness

    # ========

    def _calc_seed(self, i: int) -> int:
        return self._config.seed + i

    def _make_env(
        self,
        config: MjCambrianEnvConfig,
        n_envs: int,
        *,
        monitor: str | None = "monitor.csv",
    ) -> VecEnv:
        assert n_envs > 0, f"n_envs must be > 0, got {n_envs}."

        # Create the environments
        envs = []
        for i in range(n_envs):
            wrappers = [w for w in self._config.trainer.wrappers.values() if w]
            wrapped_env = make_wrapped_env(
                config=config.copy(),
                name=self._config.expname,
                wrappers=wrappers,
                seed=self._calc_seed(i),
            )
            envs.append(wrapped_env)

        # Wrap the environments
        vec_env = DummyVecEnv(envs) if n_envs == 1 else SubprocVecEnv(envs)
        if monitor is not None:
            vec_env = VecMonitor(vec_env, str(self._config.expdir / monitor))

        # Do an initial reset
        vec_env.reset()
        return vec_env

    def _make_callback(self, env: VecEnv) -> CallbackList:
        """Makes the callbacks."""
        from functools import partial

        callbacks = []
        for callback in self._config.trainer.callbacks.values():
            # TODO: is this a good assumption? is there a better way to do this?
            if isinstance(callback, partial):
                callback = callback(env)
            callbacks.append(callback)

        return CallbackList(callbacks)

    def _make_model(self, env: VecEnv) -> MjCambrianModel:
        """This method creates the model."""
        return self._config.trainer.model(env=env)


if __name__ == "__main__":
    import argparse

    from cambrian.utils.config import run_hydra

    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--train", action="store_true", help="Train the model")
    action.add_argument("--eval", action="store_true", help="Evaluate the model")
    action.add_argument("--test", action="store_true", help="Test the evo loop")

    def main(config: MjCambrianConfig, *, train: bool, eval: bool, test: bool) -> float:
        """This method will return a float if training. The float represents the
        "fitness" of the agent that was trained. This can be used by hydra to
        determine the best hyperparameters during sweeps."""
        runner = MjCambrianTrainer(config)

        if train:
            return runner.train()
        elif eval:
            return runner.eval()
        elif test:
            return runner.test()

    run_hydra(main, parser=parser)
