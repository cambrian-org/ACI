"""Callbacks used during training and/or evaluation."""

import csv
import glob
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.experimental.callbacks import Callback as HydraCallback
from omegaconf import DictConfig
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
    ProgressBarCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

from cambrian.envs import MjCambrianEnv
from cambrian.ml.model import MjCambrianModel
from cambrian.utils.logger import get_logger


class MjCambrianPlotMonitorCallback(BaseCallback):
    """Should be used with an EvalCallback to plot the evaluation results.

    This callback will take the monitor.csv file produced by the VecMonitor and
    plot the results and save it as an image. Should be passed as the
    `callback_after_eval` for the EvalCallback.

    Args:
        logdir (Path | str): The directory where the evaluation results are stored. The
            evaluations.npz file is expected to be at `<logdir>/<filename>.csv`. The
            resulting plot is going to be stored at
            `<logdir>/evaluations/<filename>.png`.
        filename (Path | str): The filename of the monitor file. The saved file will be
            `<logdir>/<filename>.csv`. And the resulting plot will be saved as
            `<logdir>/evaluations/<filename>.png`.
    """

    parent: EvalCallback

    def __init__(self, logdir: Path | str, filename: Path | str, n_episodes: int = 1):
        self.logdir = Path(logdir)
        self.filename = Path(filename)
        self.filename_csv = self.filename.with_suffix(".csv")
        self.filename_png = self.filename.with_suffix(".png")
        self.evaldir = self.logdir / "evaluations"
        self.evaldir.mkdir(parents=True, exist_ok=True)

        self.n_episodes = n_episodes
        self.n_calls = 0

    def _on_step(self) -> bool:
        if not (self.logdir / self.filename_csv).exists():
            get_logger().warning(f"No {self.filename_csv} file found.")
            return

        # Temporarily set the monitor ext so that the right file is read
        old_ext = Monitor.EXT
        Monitor.EXT = str(self.filename_csv)
        x, y = ts2xy(load_results(self.logdir), "timesteps")
        Monitor.EXT = old_ext
        if len(x) <= 20 or len(y) <= 20:
            get_logger().warning(f"Not enough {self.filename} data to plot.")
            return True
        original_x, original_y = x.copy(), y.copy()

        get_logger().info(f"Plotting {self.filename} results at {self.evaldir}")

        def moving_average(data, window=1):
            return np.convolve(data, np.ones(window), "valid") / window

        n = min(len(y) // 10, 1000)
        y = y.astype(float)

        if self.n_episodes > 1:
            assert len(y) % self.n_episodes == 0, (
                "n_episodes must be a common factor of the"
                f" number of episodes in the {self.filename} data."
            )
            y = y.reshape(-1, self.n_episodes).mean(axis=1)
        else:
            y = moving_average(y, window=n)

        x = moving_average(x, window=n).astype(int)

        # Make sure the x, y are of the same length
        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]

        plt.plot(x, y)
        plt.plot(original_x, original_y, color="grey", alpha=0.3)

        plt.xlabel("Number of Timesteps")
        plt.ylabel("Rewards")
        plt.savefig(self.evaldir / self.filename.with_suffix(".png"))
        plt.cla()

        return True


class MjCambrianEvalCallback(EvalCallback):
    """Overwrites the default EvalCallback to support saving visualizations at the same
    time as the evaluation.

    Note:
        Only the first environment is visualized
    """

    def _init_callback(self):
        self.log_path = Path(self.log_path)
        self.n_evals = 0

        # Delete all the existing renders
        for f in glob.glob(str(self.log_path / "vis_*")):
            get_logger().info(f"Deleting {f}")
            Path(f).unlink()

        super()._init_callback()

    def _on_step(self) -> bool:
        # Early exit
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        env: MjCambrianEnv = self.eval_env.envs[0].unwrapped

        # Add some overlays
        # env.overlays["Exp"] = env.config.expname # TODO
        env.overlays["Best Mean Reward"] = f"{self.best_mean_reward:.2f}"
        env.overlays["Total Timesteps"] = f"{self.num_timesteps}"

        # Run the evaluation
        get_logger().info(f"Starting {self.n_eval_episodes} evaluation run(s)...")
        env.record(self.render)
        continue_training = super()._on_step()

        if self.render:
            # Save the visualization
            filename = Path(f"vis_{self.n_evals}")
            env.save(self.log_path / filename)
            env.record(False)

        if self.render:
            # Copy the most recent gif to latest.gif so that we can just watch this file
            for f in self.log_path.glob(str(filename.with_suffix(".*"))):
                shutil.copy(f, f.with_stem("latest"))

        self.n_evals += 1
        return continue_training

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]):
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        env: MjCambrianEnv = self.eval_env.envs[0].unwrapped

        # If done, do some logging
        if locals_["done"]:
            run = locals_["episode_counts"][locals_["i"]]
            cumulative_reward = env.stashed_cumulative_reward
            get_logger().info(f"Run {run} done. Cumulative reward: {cumulative_reward}")

        super()._log_success_callback(locals_, globals_)


class MjCambrianGPUUsageCallback(BaseCallback):
    """This callback will log the GPU usage at the end of each evaluation.
    We'll log to a csv."""

    parent: EvalCallback

    def __init__(
        self,
        logdir: Path | str,
        logfile: Path | str = "gpu_usage.csv",
        *,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)

        self.logfile = self.logdir / logfile
        with open(self.logfile, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timesteps",
                    "memory_reserved",
                    "max_memory_reserved",
                    "memory_available",
                ]
            )

    def _on_step(self) -> bool:
        if torch.cuda.is_available():
            # Get the GPU usage, log it and save it to the file
            device = torch.cuda.current_device()
            memory_reserved = torch.cuda.memory_reserved(device)
            max_memory_reserved = torch.cuda.max_memory_reserved(device)
            memory_available = torch.cuda.get_device_properties(device).total_memory

            # Log to the output file
            with open(self.logfile, "a") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        self.num_timesteps,
                        memory_reserved,
                        max_memory_reserved,
                        memory_available,
                    ]
                )

            # Log to stdout
            if self.verbose > 0:
                get_logger().debug(subprocess.getoutput("nvidia-smi"))
                get_logger().debug(torch.cuda.memory_summary())

        return True


class MjCambrianSavePolicyCallback(BaseCallback):
    """Should be used with an EvalCallback to save the policy.

    This callback will save the policy at the end of each evaluation. Should be passed
    as the `callback_after_eval` for the EvalCallback.

    Args:
        logdir (Path | str): The directory to store the generated visualizations. The
            resulting visualizations are going to be stored at
            `<logdir>/evaluations/visualization.gif`.
    """

    parent: EvalCallback

    def __init__(
        self,
        logdir: Path | str,
        *,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)

        self.model: MjCambrianModel = None

    def _on_step(self) -> bool:
        self.model.save_policy(self.logdir)

        return True


class MjCambrianProgressBarCallback(ProgressBarCallback):
    """Overwrite the default progress bar callback to flush the pbar on deconstruct."""

    def __del__(self):
        """This string will restore the terminal back to its original state."""
        if hasattr(self, "pbar"):
            print("\x1b[?25h")


class MjCambrianCallbackListWithSharedParent(CallbackList):
    def __init__(self, callbacks: Iterable[BaseCallback] | Dict[str, BaseCallback]):
        if isinstance(callbacks, dict):
            callbacks = callbacks.values()

        self.callbacks = []
        super().__init__(list(callbacks))

    @property
    def parent(self):
        return getattr(self.callbacks[0], "parent", None)

    @parent.setter
    def parent(self, parent):
        for cb in self.callbacks:
            cb.parent = parent


# ==================


class MjCambrianSaveConfigCallback(HydraCallback):
    """This callback will save the resolved hydra config to the logdir."""

    def on_run_start(self, config: DictConfig, **kwargs):
        self._save_config(config)

    def on_multirun_start(self, config: DictConfig, **kwargs):
        self._save_config(config)

    def _save_config(self, config: DictConfig):
        from omegaconf import OmegaConf

        config.logdir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, config.logdir / "full.yaml")


class WandbCallback(BaseCallback):
    """Callback for logging metrics to Weights & Biases.
    
    This callback will log:
    - Training metrics (episode length, reward, etc.)
    - Evaluation metrics
    - Model checkpoints
    - Environment videos
    """
    
    def __init__(
        self,
        project_name: str,
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.project_name = project_name
        self.run_name = run_name
        self.config = config or {}
        
    def _on_training_start(self) -> None:
        """Initialize wandb at the start of training."""
        import wandb
        
        wandb.init(
            project=self.project_name,
            name=self.run_name,
            config=self.config,
            sync_tensorboard=True,
        )
        
    def _on_step(self) -> bool:
        """Log metrics at each step."""
        import wandb
        
        # Log training metrics
        if self.locals.get("dones", None) is not None:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    episode_info = self.locals["infos"][i]
                    wandb.log({
                        "train/episode_length": episode_info.get("episode", {}).get("l", 0),
                        "train/episode_reward": episode_info.get("episode", {}).get("r", 0),
                        "train/global_step": self.num_timesteps,
                    })
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Log rollout metrics."""
        import wandb
        
        # Log rollout metrics
        wandb.log({
            "train/rollout_ep_len_mean": self.locals.get("ep_info_buffer", {}).get("l", 0),
            "train/rollout_ep_rew_mean": self.locals.get("ep_info_buffer", {}).get("r", 0),
            "train/rollout_ep_len_std": self.locals.get("ep_info_buffer", {}).get("l_std", 0),
            "train/rollout_ep_rew_std": self.locals.get("ep_info_buffer", {}).get("r_std", 0),
        })
    
    def _on_training_end(self) -> None:
        """Close wandb at the end of training."""
        import wandb
        wandb.finish()
