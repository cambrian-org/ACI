from pathlib import Path

from stable_baselines3.common.vec_env import (
    VecEnv,
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
)
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    CallbackList,
)
from stable_baselines3.common.utils import set_random_seed

from cambrian.evolution_envs.three_d.mujoco.feature_extractors import (
    MjCambrianCombinedExtractor,
)
from cambrian.evolution_envs.three_d.mujoco.config import MjCambrianConfig
from cambrian.evolution_envs.three_d.mujoco.wrappers import make_single_env
from cambrian.evolution_envs.three_d.mujoco.callbacks import (
    PlotEvaluationCallback,
    SaveVideoCallback,
    CallbackListWithSharedParent,
    MjCambrianProgressBarCallback,
)
from cambrian.evolution_envs.three_d.mujoco.model import MjCambrianModel
from cambrian.evolution_envs.three_d.mujoco.utils import evaluate_policy


class MjCambrianTrainer:
    """This is the trainer class for running training and evaluation.

    Args:
        config (MjCambrianConfig): The config to use for training and evaluation.
    """

    def __init__(self, config: MjCambrianConfig):
        self.config = config
        self.seed = self.config.training_config.seed

        self.verbose = self.config.training_config.verbose

        self.logdir = Path(
            Path(self.config.training_config.logdir)
            / self.config.training_config.exp_name
        )
        self.logdir.mkdir(parents=True, exist_ok=True)

        self.config.write_to_yaml(self.logdir / "config.yaml")

        set_random_seed(self._calc_seed(0))

    def train(self):
        """Train the animal."""
        env = self._make_env(self.config.training_config.n_envs)
        eval_env = self._make_env(1, use_monitor=False)
        callback = self._make_callback(env, eval_env)
        model = self._make_model(env)

        total_timesteps = self.config.training_config.total_timesteps
        model.learn(total_timesteps=total_timesteps, callback=callback)
        if self.verbose > 1:
            print("Finished training the animal...")

        if self.verbose > 1:
            print(f"Saving model to {self.logdir}...")
        model.save_policy(self.logdir)
        if self.verbose > 1:
            print(f"Saved model to {self.logdir}...")

    def eval(self, num_runs: int, record: bool = False):
        env = self._make_env(1)
        model = self._make_model(env)

        record_path = self.logdir / "eval" if record else None
        evaluate_policy(env, model, num_runs, record_path=record_path)

    # ========

    def _calc_seed(self, i: int) -> int:
        return self.config.training_config.seed + i * self.seed

    def _make_env(self, n_envs: int, *, use_monitor: bool = True) -> VecEnv:
        assert n_envs > 0, f"n_envs must be > 0, got {n_envs}."

        envs = [make_single_env(self.config, self._calc_seed(i)) for i in range(n_envs)]

        if n_envs == 1:
            vec_env = DummyVecEnv(envs)
        else:
            vec_env = SubprocVecEnv(envs)
        if use_monitor:
            vec_env = VecMonitor(vec_env, str(self.logdir / "monitor.csv"))
        return MjCambrianModel._wrap_env(vec_env)

    def _make_callback(self, env: VecEnv, eval_env: VecEnv) -> BaseCallback:
        """Makes the callbacks.

        Current callbacks:
            - SaveVideoCallback: Saves a video of an evaluation episode when a new best
                model is found.
            - StopTrainingOnNoModelImprovement: Stops training when no new best model
                has been found for a certain number of evaluations. See config for
                settings.
            - MjCambrianAnimalPoolCallback: Writes the best model to the animal pool
                when a new best model is found.
            - PlotEvaluationCallback: Plots the evaluation performance over time to a
                `monitor.png` file.
            - MjCambrianProgressBarCallback: Prints a progress bar to the console for
                indicating the training progress.
            - EvalCallback: Evaluates the model every `eval_freq` steps. See config for
                settings. This is provided by Stable Baselines.
        """
        callbacks_on_new_best = []
        # callbacks_on_new_best.append(
        #     SaveVideoCallback(
        #         eval_env,
        #         self.logdir,
        #         self.config.training_config.max_episode_steps,
        #         verbose=self.verbose,
        #     )
        # )
        callbacks_on_new_best.append(
            StopTrainingOnNoModelImprovement(
                self.config.training_config.max_no_improvement_evals,
                self.config.training_config.min_no_improvement_evals,
                verbose=self.verbose,
            )
        )
        callbacks_on_new_best = CallbackListWithSharedParent(callbacks_on_new_best)

        callbacks_after_eval = []
        callbacks_after_eval.append(
            PlotEvaluationCallback(self.logdir, verbose=self.verbose)
        )
        callbacks_after_eval.append(
            SaveVideoCallback(
                eval_env,
                self.logdir,
                self.config.training_config.max_episode_steps,
                verbose=self.verbose,
            )
        )
        callbacks_after_eval = CallbackListWithSharedParent(callbacks_after_eval)

        eval_cb = EvalCallback(
            env,
            best_model_save_path=self.logdir,
            log_path=self.logdir,
            eval_freq=self.config.training_config.eval_freq,
            deterministic=True,
            render=False,
            callback_on_new_best=callbacks_on_new_best,
            callback_after_eval=callbacks_after_eval,
        )

        progress_bar_callback = MjCambrianProgressBarCallback()
        return CallbackList([eval_cb, progress_bar_callback])

    def _make_model(self, env: VecEnv) -> MjCambrianModel:
        """This method creates the model.

        If available, the weights of a previously trained model are loaded into the new
        model. See `MjCambrianModel` for more details, but because the shape of the
        output may be different between animals, the weights with different shapes
        are ignored.
        """
        policy_kwargs = dict(
            features_extractor_class=MjCambrianCombinedExtractor,
        )

        if (checkpoint_path := self.config.training_config.checkpoint_path) is not None:
            model = MjCambrianModel.load(checkpoint_path, env=env, verbose=self.verbose)
        else:
            model = MjCambrianModel(
                "MultiInputPolicy",
                env,
                n_steps=self.config.training_config.n_steps,
                batch_size=self.config.training_config.batch_size,
                learning_rate=self.config.training_config.learning_rate,
                policy_kwargs=policy_kwargs,
                verbose=self.verbose,
            )

        if (policy_path := self.config.training_config.policy_path) is not None:
            policy_path = Path(policy_path)
            assert policy_path.exists(), f"Checkpoint path {policy_path} does not exist."
            print(f"Loading model weights from {policy_path}...")
            model.load_policy(policy_path.parent)
        return model


if __name__ == "__main__":
    from utils import MjCambrianArgumentParser

    parser = MjCambrianArgumentParser()

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--train", action="store_true", help="Train the model")
    action.add_argument("--eval", action="store_true", help="Evaluate the model")

    parser.add_argument(
        "--record",
        action="store_true",
        help="Record the evaluation. Only used if `--eval` is passed.",
    )
    parser.add_argument(
        "-n",
        "--num-runs",
        type=int,
        help="Number of runs to evaluate. Only used if `--eval` is passed.",
        default=1,
    )

    args = parser.parse_args()

    config = MjCambrianConfig.load(args.config, overrides=args.overrides)
    config.training_config.setdefault("exp_name", Path(args.config).stem)

    runner = MjCambrianTrainer(config)

    if args.train:
        runner.train()
    elif args.eval:
        runner.eval(args.num_runs, args.record)
