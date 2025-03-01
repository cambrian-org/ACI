# Using Custom Configs

All the [previous examples](./index.md) have been based on configs which ship with the
repository. However, it's possible to specify custom tasks which are created outside
of the ACI repository. An example of a custom task is shown in the
[Optics example](./optics.md#using-a-random-mask). This page will go into detail about
how this is done. See the
[code](https://github.com/cambrian-org/ACI/tree/main/tools/optics) for specifics.

## Creating Custom Configs

Because ACI is based on [Hydra](https://hydra.cc/), you will need to specify any
configurations from YAML (or via the command line). In this example, we specify configs
via YAML files. To tell Hydra to look for config files within your directory, there are
two steps.

First, as seen in the
[Optics example](https://github.com/cambrian-org/ACI/tree/main/tools/optics), you have
to override `config_path` in :meth:`hydra_config.run_hydra`:

```python
if __name__ == "__main__":
    config_path = Path(__file__).parent / "configs"
    run_hydra(main, config_path=config_path, config_name="optics_sweep")
```

This snippet basically tells Hydra to look in the `configs` directory for a YAML
file named `optics_sweep.yaml`. This file will be the base configuration for your task.

The second step is to add an additional path to the Hydra search path which includes
the global ACI configs. This should be done inside `optics_sweep.yaml`:

```yaml
hydra:
  searchpath:
    - pkg://cambrian/configs
```

Additionally, although not required, it's recommend to only include configurations in
this file which overrides those that are set by default by `cambrian`. We should then
load the `base` configuration as the first default. Any configs that we set in the
YAML will then override the `base` configuration.

```yaml
defaults:
  - base
```

## Running the Example

Once you have your custom configs set up, you can run the example using the following
command:

```bash
python tools/optics/optics_sweep.py
```

You should then find the video output to look like the following:

```{video} assets/optics/circular_aperture_sweep.mp4
:align: center
:class: example-figure
:figwidth: 75%
:loop:
:autoplay:
:muted:
:caption: In this example, we sweep various aperture radii to demonstrate the relative blurring for different aperture size values. The radius is essentially a percentage that the aperture is open; a value of 0.1 corresponds to a 10% open aperture and 1.0 corresponds to a fully open aperture.
```
