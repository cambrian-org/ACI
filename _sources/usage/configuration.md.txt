# Configuring an Experiment

This package uses [`hydra`](#hydra) extensively for configuration management. All config files are located within the [`configs/`](https://github.com/camera-culture/ACI/tree/main/configs) directory. Each config is defined by a schema which is used to validate the config.

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

We use Hydra to manage configurations for our experiments. The configuration files are stored in the `configs/` directory and each configuration file is composed of other configuration files via the `defaults` field. See [](#config-composition) for more information on how to use Hydra in this project.

## Config Schema

Each config schema is defined within the file which it's used; for instance, [`MjCambrianEnvConfig`](https://camera-culture.github.io/ACI/reference/api/cambrian/envs/index.html#cambrian.envs.MjCambrianEnvConfig) is defined in [`env.py`](https://github.com/camera-culture/ACI/blob/02fa04cd38c3b4d73c9ff4cc8e67ec08114507be/cambrian/envs/env.py#L62). The schema is implemented as a dataclass and added to the [Hydra ConfigStore](https://hydra.cc/docs/tutorials/structured_config/config_store/) using the [`config_wrapper` decorator](https://AaronYoung5.github.io/hydra-config/reference/api/hydra_config/config/index.html#hydra_config.config.config_wrapper). Thus, the associated config file should implement this schema; for instance, [`configs/env/env.yaml`](https://github.com/camera-culture/ACI/blob/main/configs/env/env.yaml) implements the `MjCambrianEnvConfig` schema.

## Config Composition

To reduce redundancy, configs are composed using the [Default List](https://hydra.cc/docs/advanced/defaults_list/) in hydra. For instance, [`MjCambrianMazeEnvConfig`](https://camera-culture.github.io/ACI/reference/api/cambrian/envs/index.html#cambrian.envs.MjCambrianMazeEnvConfig) (which inherits from `MjCambrianEnvConfig`) is defined in [`configs/env/maze_env.yaml`](https://github.com/camera-culture/ACI/blob/main/configs/env/maze_env.yaml) and simply includes the default `env` like the following:

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
python <ANY SCRIPT> exp=<EXPERIMENT> -p <dot.separated.path>
```

```{note}
This is hydra syntax. For more information, run `python <ANY SCRIPT> --hydra-help`.
```

## Config Overrides

All configs should be put under `configs`. These are parsed by
[hydra](https://hydra.cc/docs/intro) and can be overridden by passing in
`<dot.separated.path>=<value>` to the script. Checkout hydra's documentation for more
details.
