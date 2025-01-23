# Configuring an Experiment

This package uses [`hydra`](#hydra) extensively for configuration management. All config files are located within the {src}`configs/ <cambrian/configs>` directory. Each config is defined by a schema which is used to validate the config.

## Hydra

Hydra is a Python package designed to simplify the configuration management of complex applications. It enables users to compose, override, and manage configurations dynamically, making it especially useful for applications that need to handle multiple runtime scenarios, such as machine learning experiments or distributed systems.

### Features

Hydra provides a range of features to streamline configuration management:

- **Dynamic Composition**: Combine multiple configuration files or objects to define application behavior.
- **Overrides**: Modify configurations directly from the command line or programmatically.
- **Structured Configs**: Support for hierarchical and type-safe configurations using dataclasses or YAML.
- **Plugins**: Extend Hydra’s functionality with a rich ecosystem of plugins, including support for launching jobs on various platforms.
- **Experiment Tracking**: Log and organize experiment configurations automatically.

### Usage

We use Hydra to manage configurations for our experiments. The configuration files are stored in the {src}``configs/ <cambrian/configs>`` directory and each configuration file is composed of other configuration files via the `defaults` field. See [](#config-composition) for more information on how to use Hydra in this project.

## Config Schema

Each config schema is defined within the file which it's used; for instance, {class}`~cambrian.envs.env.MjCambrianEnvConfig` is defined in {src}``env.py <cambrian/envs/env.py>``. The schema is implemented as a dataclass and added to the [Hydra ConfigStore](https://hydra.cc/docs/tutorials/structured_config/config_store/) using the {func}`~hydra_config.config_wrapper`. Thus, the associated config file should implement this schema; for instance, {src}``env.yaml <cambrian/configs/env/env.yaml>`` implements the {class}`~cambrian.envs.env.MjCambrianEnvConfig` schema.

## Config Composition

To reduce redundancy, configs are composed using the [Default List](https://hydra.cc/docs/advanced/defaults_list/) in hydra. For instance, {class}`~cambrian.envs.maze_env.MjCambrianMazeEnvConfig` (which inherits from {class}`~cambrian.envs.env.MjCambrianEnvConfig`) is defined in {src}``maze_env.yaml <cambrian/configs/env/maze_env.yaml>`` and simply includes the default `env` like the following:

```yaml
defaults:
  # This inherits from the default env config
  - env

  # Tells hydra to use the MjCambrianMazeEnvConfig schema
  - /MjCambrianMazeEnvConfig

# Anything remaining in this file overrides or adds to the default env config
# ...
```

This same logic applies in many other scenarios throughout the config definitions.

## Config Introspection

You can introspect into the configs by running the following:

```bash
python <ANY SCRIPT> exp=<EXPERIMENT> -c all
```

This will print out the entire config. You can print out specific parts of the config
using `-p <dot.separated.path>`.

```bash
python <ANY SCRIPT> exp=<EXPERIMENT> -c all -p <dot.separated.path>
```

```{note}
This is hydra syntax. For more information, run `python <ANY SCRIPT> --hydra-help`.
```

## Config Overrides

All configs are parsed by [hydra](https://hydra.cc/docs/intro) and can be overridden by passing in
`<dot.separated.path>=<value>` to the script. Checkout hydra's documentation for more
details.
