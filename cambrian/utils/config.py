from typing import Dict, Any, Optional, Callable, Concatenate
import argparse
from pathlib import Path
import os

from cambrian.ml.trainer import MjCambrianTrainerConfig
from cambrian.envs.env import MjCambrianEnvConfig
from cambrian.utils.base_config import config_wrapper, MjCambrianBaseConfig


@config_wrapper
class MjCambrianConfig(MjCambrianBaseConfig):
    """The base config for the mujoco cambrian environment. Used for type hinting.

    Attributes:
        logdir (Path): The directory to log training data to.
        expname (str): The name of the experiment. Used to name the logging
            subdirectory. If unset, will set to the name of the config file.

        seed (int): The base seed used when initializing the default thread/process.
            Launched processes should use this seed value to calculate their own seed
            values. This is used to ensure that each process has a unique seed.

        training (MjCambrianTrainingConfig): The config for the training process.
        env (MjCambrianEnvConfig): The config for the environment.
        eval_env (MjCambrianEnvConfig): The config for the evaluation environment.
        logging (Optional[Dict[str, Any]]): The config for the logging process.
            Passed to `logging.config.dictConfig`.
    """

    logdir: Path
    expname: str

    seed: int

    trainer: MjCambrianTrainerConfig
    env: MjCambrianEnvConfig
    eval_env: MjCambrianEnvConfig
    logging: Optional[Dict[str, Any]] = None


# =============


def run_hydra(
    main_fn: Optional[
        Callable[[Concatenate[MjCambrianBaseConfig, ...]], None]
    ] = lambda *_, **__: None,
    /,
    *,
    parser: argparse.ArgumentParser = argparse.ArgumentParser(),
    config_path: str = f"{os.getcwd()}/configs",
    config_name: str = "base",
    **instantiate_kwargs,
):
    """This function is the main entry point for the hydra application.

    The benefits of using this setup rather than the compose API is that we can
    use the sweeper and launcher APIs, which are not available in the compose API.

    Args:
        main_fn (Callable[[Concatenate[[MjCambrianConfig], ...], None]): The main
            function to be called after the hydra configuration is parsed. It should
            take the config as an argument and kwargs which correspond to the argument
            parser returns. We don't return the config directly because hydra allows
            multi-run sweeps and it doesn't make sense to return multiple configs in
            this case.

            Example:

            ```python
            def main(config: MjCambrianConfig, *, verbose: int):
                print(config, verbose)

            parser = argparse.ArgumentParser()
            parser.add_argument("--verbose", type=int, default=0)

            run_hydra(main_fn=main, parser=parser)
            ```

    Keyword Args:
        parser (argparse.ArgumentParser): The parser to use for the hydra
            application. If None, a new parser will be created.
        config_path (str): The path to the config directory. This should be the
            absolute path to the directory containing the config files. By default,
            this is set to the current working directory.
        config_name (str): The name of the config file to use. This should be the
            name of the file without the extension. By default, this is set to
            "base".
        instantiate_kwargs: Additional keyword arguments to pass to the instantiate function.
    """
    import hydra
    from omegaconf import DictConfig

    # Add one default argument for the --hydra-help message
    parser.add_argument(
        "--hydra-help", action="store_true", help="Print the hydra help message."
    )

    def hydra_argparse_override(fn: Callable, /):
        """This function allows us to add custom argparse parameters prior to hydra
        parsing the config.

        We want to set some defaults for the hydra config here. This is a workaround
        in a way such that we don't

        Note:
            Augmented from hydra discussion #2598.
        """
        import sys
        from functools import partial

        parsed_args, unparsed_args = parser.parse_known_args()

        # Move --hydra-help to unparsed_args if it's present
        # Hydra has a weird bug (I think) that doesn't allow overrides when
        # --hydra-help is passed, so remove all unparsed arguments if --hydra-help
        # is passed.
        if parsed_args.hydra_help:
            unparsed_args = ["--hydra-help"]
        del parsed_args.hydra_help

        # By default, argparse uses sys.argv[1:] to search for arguments, so update
        # sys.argv[1:] with the unparsed arguments for hydra to parse (which uses
        # argparse).
        sys.argv[1:] = unparsed_args

        return partial(fn, **vars(parsed_args))

    @hydra.main(version_base=None, config_path=config_path, config_name=config_name)
    @hydra_argparse_override
    def main(cfg: DictConfig, **kwargs):
        config = MjCambrianBaseConfig.instantiate(cfg, **instantiate_kwargs)
        return main_fn(config, **kwargs)

    main()


if __name__ == "__main__":

    def main(config: MjCambrianConfig):
        pass

    run_hydra(main)
