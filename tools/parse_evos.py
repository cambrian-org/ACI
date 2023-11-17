"""This script will a saved evolution folder."""

import argparse
from typing import Dict, Union, Optional, Any
from pathlib import Path
import pickle
import os
import yaml
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from collections import defaultdict
from functools import partial
import shutil
import glob
from dataclasses import dataclass, field

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from cambrian.evolution_envs.three_d.mujoco.config import MjCambrianConfig

from plot import moving_average

mpl.rcParams["image.cmap"] = "jet"


def np_float32_representer(dumper: yaml.Dumper, value: np.float32) -> yaml.Node:
    return dumper.represent_float(float(value))


yaml.add_representer(np.float32, np_float32_representer)


def np_float64_representer(dumper: yaml.Dumper, value: np.float64) -> yaml.Node:
    return dumper.represent_float(float(value))


yaml.add_representer(np.float32, np_float32_representer)
yaml.add_representer(np.float64, np_float64_representer)

dataclass = dataclass(kw_only=True)


@dataclass
class Rank:
    path: Path

    rank: int

    config: MjCambrianConfig = None
    evaluations: Dict[str, Any] = None
    monitor: Dict[str, Any] = None


@dataclass
class Generation:
    path: Path

    generation: int
    ranks: Dict[int, Rank] = field(default_factory=dict)


@dataclass
class Data:
    path: Path

    generations: Dict[int, Generation] = field(default_factory=dict)

    accumulated_data: Dict[str, Any] = field(default_factory=dict)


def save_data(generations: Dict, folder: Path):
    """Save the parsed data to a pickle file."""
    pickle_file = folder / "parse_evos" / "data.pkl"
    pickle_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pickle_file, "wb") as f:
        pickle.dump(generations, f)
    print(f"Saved parsed data to {pickle_file}.")


def try_load_pickle_data(folder: Path) -> Union[None, Dict]:
    """Try to load the data from the pickle file."""
    pickle_file = folder / "parse_evos" / "data.pkl"
    if pickle_file.exists():
        with open(pickle_file, "rb") as f:
            generations = pickle.load(f)
        print(f"Loaded parsed data from {pickle_file}.")
        return generations

    print(f"Could not load {pickle_file}.")
    return None


def get_generation_file_paths(folder: Path) -> Data:
    """Create the initial storage dict for parsing and get the paths to all the
    generation/rank folders."""
    generations = dict()
    for root, dirs, files in os.walk(folder):
        root = Path(root)
        if not root.stem.startswith("generation_"):
            continue

        generation = int(root.stem.split("generation_", 1)[1])

        ranks = dict()
        for dir in dirs:
            dir = Path(dir)
            if not dir.stem.startswith("rank_"):
                continue

            rank = int(dir.stem.split("rank_", 1)[1])
            ranks[rank] = Rank(path=root / dir, rank=rank)

        ranks = dict(sorted(ranks.items()))
        generations[generation] = Generation(
            path=root, generation=generation, ranks=ranks
        )

    generations = dict(sorted(generations.items()))
    return Data(path=folder, generations=generations)


def load_data(folder: Path) -> Data:
    print(f"Loading data from {folder}...")
    data = get_generation_file_paths(folder)

    for generation, generation_data in data.generations.items():
        print(f"Loading generation {generation}...")

        for rank, rank_data in generation_data.ranks.items():
            print(f"\tLoading rank {rank}...")

            # Get the config file
            if (config_file := rank_data.path / "config.yaml").exists():
                print(f"\tLoading config from {config_file}...")
                rank_data.config = MjCambrianConfig.load(config_file)

            # Get the evaluations file
            if (evaluations_file := rank_data.path / "evaluations.npz").exists():
                with np.load(evaluations_file) as evaluations_data:
                    evaluations = {
                        k: evaluations_data[k] for k in evaluations_data.files
                    }
                rank_data.evaluations = evaluations

            # Get the monitor file
            if (rank_data.path / "monitor.csv").exists():
                monitor = load_results(rank_data.path)
                rank_data.monitor = monitor

    return data


def plot(
    data: Data,
    output_folder: Path,
    *,
    rank_to_use: Optional[int] = None,
    generation_to_use: Optional[int] = None,
    verbose: int = 0,
    dry_run: bool = False,
):
    print("Parsing data...")

    RANK_FORMAT_MAP = {
        0: ".C0",
        1: ".C1",
        2: "sC2",
        3: "^C3",
        4: "xC4",
        5: "vC5",
        6: "8C6",
    }

    def _plot(
        *args,
        title,
        xlabel,
        ylabel,
        label=None,
        locator=tkr.MultipleLocator(),
        **kwargs,
    ):
        fig = plt.figure(title)

        plt.plot(*args, **kwargs)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.suptitle(title)

        if label is not None:
            if not hasattr(fig, "labels"):
                fig.labels = set()
            fig.labels.add(label)
            plt.legend(sorted(fig.labels))

        ax = fig.gca()
        ax.xaxis.set_major_locator(locator)

        return fig

    output_folder.mkdir(parents=True, exist_ok=True)
    for generation, generation_data in data.generations.items():
        if generation_to_use is not None and generation != generation_to_use:
            continue

        print(f"Parsing generation {generation}...")

        for rank, rank_data in generation_data.ranks.items():
            if rank_to_use is not None and rank != rank_to_use:
                continue

            print(f"\tParsing rank {rank}...")

            # ======
            # CONFIG
            # ======
            if (config := rank_data.config) is not None:
                animal_config = config.animal_config
                default_eye_config = animal_config.default_eye_config
                current_generation = config.evo_config.generation
                parent_generation = config.evo_config.parent_generation

                # Plot the rank-over-rank config data
                def _config_plot(attr, *args, **kwargs):
                    y = config.copy()
                    for k in attr.split("."):
                        if not hasattr(y, k):
                            raise ValueError(f"Could not find {attr} in config.")
                        y = getattr(y, k)
                    y = np.average(np.abs(y)) if isinstance(y, list) else y
                    if not dry_run:
                        _plot(
                            generation,
                            y,
                            f"{RANK_FORMAT_MAP[rank]}",
                            *args,
                            label=f"Rank {rank}",
                            title=attr,
                            xlabel="generation",
                            ylabel=k,
                            **kwargs,
                        )
                    data.accumulated_data[attr] = y

                _config_plot("animal_config.num_eyes_lat")
                _config_plot("animal_config.num_eyes_lon")
                _config_plot("animal_config.default_eye_config.fov")
                _config_plot("animal_config.default_eye_config.resolution")
                _config_plot("evo_config.generation.generation")
                _config_plot("evo_config.generation.rank")
                _config_plot("evo_config.parent_generation.rank")
                _config_plot("evo_config.parent_generation.generation")

            # ======
            # EVALS
            # ======
            # TODO: we should try to see how performance improves over the course of training
            if (evaluations := rank_data.evaluations) is not None:
                if not dry_run:
                    _plot(
                        generation,
                        np.average(evaluations["results"]),
                        f"{RANK_FORMAT_MAP[rank]}",
                        label=f"Rank {rank}",
                        title="average_eval_rewards",
                        xlabel="generation",
                        ylabel="rewards",
                    )

            # =======
            # MONITOR
            # =======

            # Get the data and delete it from the dict. We'll write the parsed data back
            # under the same key.
            if (monitor := rank_data.monitor) is not None:
                x, y = ts2xy(monitor, "timesteps")
                y = moving_average(y.astype(float), window=min(len(y) // 10, 1000))
                x = x[len(x) - len(y) :]

                # TODO: this looks terrible
                if not dry_run:
                    _plot(
                        x,
                        y,
                        f"-{RANK_FORMAT_MAP[rank][-2:]}",
                        label=f"Rank {rank}",
                        title="monitor",
                        xlabel="timesteps",
                        ylabel="rewards",
                        locator=tkr.AutoLocator(),
                    )

                # Accumulate the data from all ranks
                data.accumulated_data.setdefault("monitor", dict())
                data.accumulated_data["monitor"].setdefault(rank, [])
                data.accumulated_data["monitor"][rank].append((x, y))

    # All generations plots
    if "monitor" in data.accumulated_data:
        x_prev = 0
        for rank in data.accumulated_data["monitor"]:
            for x, y in data.accumulated_data["monitor"][rank]:
                x += x_prev
                if not dry_run:
                    fig = _plot(
                        x,
                        y,
                        "-C7",
                        label="All generations",
                        title=f"monitor_all_generations_{rank}",
                        xlabel="timesteps",
                        ylabel="rewards",
                        locator=tkr.AutoLocator(),
                    )
                    # plot vertical bar
                    plt.axvline(x=x[-1], color="C3", linestyle="--")
                    fig.labels.add("Agent Evolution")
                    plt.legend(sorted(fig.labels))
                    x_prev = x[-1]

    for fig in plt.get_fignums():
        fig = plt.figure(fig)
        plt.savefig(output_folder / f"{fig._suptitle.get_text()}.png", dpi=300)


def eval(
    generations: Dict,
    output_folder: Path,
    *,
    rank_to_use: Optional[int] = None,
    generation_to_use: Optional[int] = None,
    verbose: int = 0,
    dry_run: bool = False,
):
    print("Evaluating model...")

    def _run_eval(logdir: Path, tmpdir: Path, filename: Path, config: Prodict):
        env = DummyVecEnv([make_env(0, 0, config, 0, force_set_env_rendering=True)])
        env.render_mode = "human"
        env.get_attr("sim")[0].render = partial(
            env.get_attr("sim")[0].render,
            current_canvas=True,
            render_video=False,
            logdir=tmpdir,
        )
        model = PPO.load(logdir / "best_model.zip", print_system_info=False)
        episode_rewards, episode_lengths = evaluate_policy(
            model,
            env,
            render=True,
            n_eval_episodes=1,
            return_episode_rewards=True,
            warn=True,
        )
        make_video(tmpdir, sorted(glob.glob((tmpdir / "sim_*.jpg").as_posix())))
        (tmpdir / "vid.gif").rename(filename.with_suffix(".gif"))
        (tmpdir / "vid.mp4").rename(filename.with_suffix(".mp4"))
        shutil.rmtree(tmpdir, ignore_errors=True)

    output_folder.mkdir(parents=True, exist_ok=True)
    for generation, data in generations.items():
        if generation_to_use is not None and generation != generation_to_use:
            continue

        print(f"Evaluating generation {generation}...")

        ranks = data["ranks"]
        for rank, rank_data in ranks.items():
            if rank_to_use is not None and rank != rank_to_use:
                continue

            print(f"\tEvaluating rank {rank}...")

            if verbose > 1:
                print(
                    yaml.dump(
                        rank_data["config"].to_dict(is_recursive=True), sort_keys=False
                    )
                )

            if not dry_run:
                _run_eval(
                    rank_data["path"],
                    output_folder / "temp",
                    output_folder / f"generation_{generation}_rank_{rank}",
                    rank_data["config"],
                )

            print(f"\tDone.")


def main(args):
    folder = Path(args.folder)
    plots_folder = (
        folder / "parse_evos" / "plots" if args.output is None else Path(args.output)
    )
    plots_folder.mkdir(parents=True, exist_ok=True)
    evals_folder = (
        folder / "parse_evos" / "evals" if args.output is None else Path(args.output)
    )
    evals_folder.mkdir(parents=True, exist_ok=True)

    if args.force or (data := try_load_pickle_data(folder)) is None:
        data = load_data(folder)

        if not args.no_save:
            save_data(data, folder)

    kwargs = dict(
        data=data,
        output_folder=plots_folder,
        rank_to_use=args.rank,
        generation_to_use=args.generation,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )
    if args.plot:
        plot(**kwargs)
    if args.eval:
        eval(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse the evolution folder.")

    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("--dry-run", action="store_true", help="Dry run.")

    parser.add_argument("folder", type=str, help="The folder to parse.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="The output folder. Defaults to <folder>/parse_evos/",
        default=None,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force loading of the data. If not passed, this script will try to find a parse_evos.pkl file and load that instead.",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save the parsed data."
    )
    parser.add_argument(
        "--rank",
        type=int,
        help="The rank to plot. If not passed, all ranks are plotted.",
        default=None,
    )
    parser.add_argument(
        "--generation",
        type=int,
        help="The generation to plot. If not passed, all generations are plotted.",
        default=None,
    )
    parser.add_argument("--eval", action="store_true", help="Evaluate the data.")
    parser.add_argument("--plot", action="store_true", help="Plot the data.")

    args = parser.parse_args()

    main(args)
