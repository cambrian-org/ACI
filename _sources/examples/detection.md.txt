# Detection Task

The detection task is designed to train the agent to perform object discrimination. There are two objects in the environment, a goal object and an adversarial object, A screenshot of the agent's view is shown below:

```{figure} assets/detection/screenshot.png
:align: center
:class: example-figure
:width: 75%
**Screenshot of the detection task.** The detection task incorporates two objects, a goal object and an adversarial object. The goal (visualized with a green dot above in the birds eye view) has vertical stripes. Conversely, the adversarial object (visualized with a red dot) has horizontal stripes. The green and red dots are not visible to the agent. The first person view of the agent is shown at the bottom of the image. In this screenshot, the agent has three eyes, each with 50x50 pixels.
```

The goal object has vertical stripes, while the adversarial object has horizontal stripes. In order to succeed at the task, the agent must learn to discern between these two objects. The reward function is as follows:

```{math}
\quad R_{\text{Detection}} = -\lambda \left( \|\mathbf{x}_t - \mathbf{x}_f\| - \|\mathbf{x}_{t-1} - \mathbf{x}_f\| \right) + w_g + w_a + w_c\\
```

$R_{\text{Detection}}$ is the reward at time $t$ for each task, $\lambda$ is a scaling factor, $x_t$ and $x_{t-1}$ is the position of the agent at time $t$ and $t-1$ respectively, $x_0$ is the initial position of the agent, and $x_f$ is the position of the goal. The $w$ variables are non-zero when certain conditions are met. $w_g$ and $w_a$ indicates the reward/penalty given for reaching the goal and adversary, respectively. $w_c$ is the penalty for contacting a wall. In essence, the agent is incentivized to navigate to the goal as quickly as it can. During training, $\lambda = 0.25$, $w_g = 1$, $w_a = -1$, and $w_c = -1$. Additionally, when an agent reaches the goal or adversary, the episode terminates.

## Training an Agent

To train an agent on the detection task, you can use the following command:

```bash
bash scripts/run.sh cambrian/main.py --train example=detection
```

A successfully trained agent may look like the following:

```{video} assets/detection/trained_agent.mp4
:align: center
:class: example-figure
:figwidth: 75%
:loop:
:autoplay:
:muted:
:caption: This agent has been successfully trained to discern between the goal and adversary objects. The above command uses a default set of eye parameters; in this case, 3 eyes, each with a field-of-view of 45°, and a resolution of 20x20 pixels.
```

## Evaluating an Agent

You can evaluate a trained policy or the task itself in a few ways.

### Evaluating using an Privileged Policy

We provide a simple policy that demonstrates good performance in the detection task. The policy is privileged in the sense that it has access to all environment states. The logic is defined in the {class}`~cambrian.agents.point.MjCambrianAgentPointSeeker` class. To evaluate using this policy, you can use the following command:

```bash
bash scripts/run.sh cambrian/main.py --eval example=detection env/agents@env.agents.agent=point_seeker
```

This command will save the evaluation results in the log directory, which defaults to `logs/<today's date>/detection/`.

```{video} assets/detection/privileged_agent.mp4
:align: center
:figwidth: 75%
:loop:
:autoplay:
:muted:
:caption: The privileged policy simply tries to reduce the distance and angle error between itself in the goal. It doesn't use any visual stimuli.
```

```{tip}
You can also visualize the evaluation in a gui window by setting `env.renderer.render_modes='[human]'`. You may also need to set the environment variable `MUJOCO_GL=glfw` to use the window-based renderer.
```

### Evaluating using a Trained Policy

You can also evaluate a trained policy using the `trainer/model=loaded_model` argument.

```bash
bash scripts/run.sh cambrian/main.py --eval example=detection trainer/model=loaded_model
```

This command will save the evaluation results in the log directory, as well.

## Evolving an Agent

You can also evolve an agent using the following command:

```bash
sbatch scripts/run.sh cambrian/main.py --train task=detection evo=evo hydra/launcher=supercloud evo/mutations='[num_eyes,resolution,lon_range]' -m
```

This command will launch the individual scripts using [submitit](https://github.com/facebookincubator/submitit), with configurations defined in the `configs/hydra/launcher/supercloud.yaml` file. This file can be amended to work for other Slurm clusters.

Additionally, this command enables 3 types of mutations: `num_eyes`, `resolution`, and `lon_range`. Additional mutations can be added by modifying the `evo/mutations` argument.

The optimized configuration and its trained policy is shown below:

```{video} assets/detection/evolved_agent.mp4
:align: center
:class: example-figure
:figwidth: 75%
:loop:
:autoplay:
:muted:
:caption: The optimized policy has 3 eyes, with a longitudinal range of 60° and a resolution of 14x14 pixels.
```

## Task Configuration

```{literalinclude} ../../cambrian/configs/task/detection.yaml
:language: yaml
:caption: configs/task/detection.yaml
```
