# Training an Agent

[Hydra](https://hydra.cc/) is powerful in the sense that it provides different methods for sweeping parameters and launching jobs. Common use cases are doing a grid search over hyperparameters while launching these jobs on a cluster or running locally.

## Running a single training loop

To run locally, you can use the {src}`run.sh <scripts/run.sh>` script. For example, you can run:

```bash
bash scripts/run.sh cambrian/main.py --train exp=<EXPERIMENT>
```

{src}`run.sh <scripts/run.sh>` isn't actually necessary, it just sets some default environment variables and is helpful as it's the same entrypoint for running on a cluster. You can also run the above command directly with `python cambrian/main.py ...`.

```{tip}
When invoked with bash, `run.sh` script will default to setting `MUJOCO_GL=egl`. Training should always be done with `MUJOCO_GL=egl` cause that runs with a headless implementation of OpenGL and is significantly faster.
```

```{note}
You can also run `--eval` instead of `--train` to run evaluation to visualize the env. Set the `render_modes` to include `'human'` to visualize the env. It will also automatically save the video to `logs/<date>/<exp>/eval.*`. Use `-h` to see options.
```

### Running on a cluster

We have provided scripts to run on three clusters: [SuperCloud](https://supercloud.mit.edu), [OpenMind](https://mcgovern.mit.edu/tile/openmind-computing-cluster/), and [Euler](https://euler-cluster.readthedocs.io/en/latest/). Coupled with these scripts are the [Slurm-based launcher configuration](https://hydra.cc/docs/plugins/submitit_launcher/), located at {src}`configs/hydra/launcher/ <cambrian/configs/hydra/launcher>`. When running on a cluster, a daemon job will always be launched to monitor the training process; this is a requirement of hydra and will simply block until the training is complete.

To run, you can still use the `run.sh` script, which has some default Slurm configs set.

```{literalinclude} ../../scripts/run.sh
:language: bash
:caption: run.sh
```

For example, you can run:

```bash
# Note the `sbatch` command is used to submit the job to the cluster using Slurm.
# You can change `supercloud` to `openmind`, `euler`, or a custom launcher depending on the cluster.
sbatch scripts/run.sh cambrian/main.py --train hydra/launcher=supercloud exp=<EXPERIMENT>
```

You can then add additional Slurm configuration variables directly to the above command. For example, to set the partition and qos, you can run the following. Note that this only sets the qos/partition on the daemon job, not the training job. The second command demonstrates how to set the partition and qos for the training job within the launcher config.

```bash
# Set the partition and qos for the daemon job
sbatch scripts/run.sh --partition=<PARTITION> --qos=<QOS> cambrian/main.py --train hydra/launcher=supercloud exp=<EXPERIMENT>

# Set the partition and qos for the training job
sbatch scripts/run.sh cambrian/main.py --train hydra/launcher=supercloud exp=<EXPERIMENT> hydra.launcher.partition=<PARTITION> hydra.launcher.qos=<QOS>
```

```{todo}
We plan to add AWS support in the future.
```

### Continue training from a checkpoint

During each training, a `best_model.zip` is saved in the log directory. To load this model and continue training, you can override `trainer/model=loaded_model`. By default, the loader will look at `trainer.model.path` to find the model to load. This will default to `{expdir}/best_model`, so if the day has changed or you've moved the model, you may need to update this path.

```bash
bash scripts/run.sh cambrian/main.py --train exp=<EXPERIMENT> trainer/model=loaded_model trainer.model.path=<MODEL_PATH>
```

## Running evolution

In the above examples, we are simply running a single training loop. If you want to run an evolutionary experiment, you can set `evo=evo`. For example, you can run:

```bash
bash scripts/run.sh cambrian/main.py --train exp=<EXPERIMENT> evo=evo --multirun
```

The evolution loop utilizes the [`nevergrad` sweeper](https://hydra.cc/docs/plugins/nevergrad_sweeper/), and it's configs are located at {src}`configs/hydra/sweeper/ <cambrian/configs/hydra/sweeper>`. You can replace `bash` with `sbatch` to run on a cluster.

In total, you should see `min(hydra.sweeper.optim.num_workers, hydra.launcher.array_parallelism) + 1` jobs. The `+ 1` is for the daemon job that monitors the training process. See the {src}`evolution_nevergrad.yaml <cambrian/configs/hydra/sweeper/evolution_nevergrad.yaml>` and {src}`slurm.yaml <cambrian/configs/hydra/launcher/slurm.yaml>` for more info.

Running on a cluster can aid in parallelizing the training process. And note the
`--multirun` flag (can also be `-m`) is required to run a sweep.

### Adding parameters to optimize

Nevergrad requires there to be optimization parameters to sweep over; see [the docs](https://hydra.cc/docs/plugins/nevergrad_sweeper/#defining-the-parameters) for more details. In our framework, we've provided a few examples of how to set up the parameters. To use those, for example, you can run the following:

```bash
bash scripts/run.sh cambrian/main.py --train exp=<EXPERIMENT> evo=evo \
    +exp/mutations=[res,num_eyes,lon_range] -m
```

This will enable the resolution, number of eyes, and placement range mutations. Alternatively, you can just specify the grouping.

```bash
bash scripts/run.sh cambrian/main.py --train exp=<EXPERIMENT> evo=evo \
    +exp/mutations/groupings/numeyes1_res0_lon1 -m
```

```{note}
Note the `+` before the `exp/mutations` parameter. This is required to add to the existing list of mutations as hydra will complain that the key doesn't exist before adding.
```

### Resuming a failed evolution

To resume a failed sweep for the nevergrad sweeper, you simply need to rerun your
experiment. The nevergrad resuming logic is handled in hydra and enabled via the
`optim.load_if_exists` parameter in the config. This will load the optimizer state
from the last checkpoint and continue the optimization.

Keep in mind that the default path nevergrad will look for a optimizer pickle to load
is at `<logdir>/nevergrad.pkl`. If you're resuming an evolutionary run launched on a
different day, you may need to either:

1. Copy the failed evolutionary logdir to a new log directory matching the logdir
   pattern for that day (recommended).
2. Change the logdir to the previous day (i.e. `... logdir=logs/<old date>/${expname}`).
3. Change the optimizer path in the config to point to the correct path.

## Running a training sweep

By default, evolution uses nevergrad for optimization. But it may be useful to run a
simple grid search of different parameters. This can be done via Hydra's basic sweeper
functionality. To run a sweep, which trains each agent with a different set of
parameters, you can run the following:

```bash
# If running on a slurm-enabled cluster
sbatch scripts/run.sh cambrian/main.py --train hydra/launcher=<CLUSTER> exp=<EXPERIMENT> <SWEEP_PARAMS> --multirun
# If running locally
bash scripts/run.sh cambrian/main.py --train exp=<EXPERIMENT> <SWEEP_PARAMS> --multirun
```

Running on a cluster can aid in parallelizing the training process. And note the
`--multirun` flag (can also be `-m`) is required to run a sweep.

### Adding sweep parameters

The simplest way to specify sweep params is via a comma separated list, e.g.
`... param1=1,2,3 param2='[1,2],[3,4]'`. The previous example will run six total runs,
iterating through all the permutations of the grid search. Please refer to
[Hydra's sweeper documentation](https://hydra.cc/docs/1.0/tutorials/basic/running_your_app/multi-run/#internaldocs-banner)
for more detailed information and specific override syntax support. To use
interpolations within the sweep, you must specify each interpolation prefixed with `\$`,
as in `expname=<EXPNAME>_\${param1}_\${param2}`.

```{warning}
By default, sweeps will write to the same `logs` directory since they all share
the same experiment name. Ensure you set `expname` to a value which utilizes a sweep
parameter to differentiate between runs, e.g.
`expname=<EXPNAME>_\${param1}_\${param2}`.

Alternatively, you can add `overlay=sweep` to the command to automatically add the
sweep parameters to the experiment name.
```

```{note}
The basic sweeper will sweep through _all_ permutations of the parameters. This can
be very slow for large sweeps. For more advanced sweeps, consider using optimization
via `evo=evo` ([see above](#running-evolution)).
```
