from pathlib import Path

from hydra_config import run_hydra

from cambrian import MjCambrianConfig, MjCambrianTrainer


def main(config: MjCambrianConfig, train: bool, eval: bool) -> float:
    assert train or eval, "Either train or eval must be set to True."

    runner = MjCambrianTrainer(config)
    if train:
        return runner.train()
    if eval:
        return runner.eval()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation.",
    )

    config_path = Path(__file__).parent / "configs"
    run_hydra(
        main, config_path=config_path, config_name=Path(__file__).stem, parser=parser
    )
