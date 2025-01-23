# Navigation Task

The navigation task is designed to train an agent to orient itself in a map-like environment. The agent is spawned at one end of a map and is tasked to navigate as far as it can get through the corridor-like world to reach the other end.

```{figure} assets/navigation/screenshot.png
:align: center
:class: example-figure
:width: 75%
**Screenshot of the navigation task.** The agent is spawned at the right end of the map in two possible locations (with small permutations to remove deterministic behavior). The agent must then navigate to the left while avoiding contact with walls using only visual stimuli.
```

The reward function is as follows, where the agent is rewarded for moving towards the end and penalized for contacting walls:

```{math}
\quad R_{\text{Navigation}} = \phantom{-}\lambda \left( \|\mathbf{x}_t - \mathbf{x}_0\| - \|\mathbf{x}_{t-1} - \mathbf{x}_0\| \right) + w_g + w_c \\
```

$R_{\text{Navigation}}$ is the reward at time $t$ for each task, $\lambda$ is a scaling factor, $x_t$ and $x_{t-1}$ is the position of the agent at time $t$ and $t-1$ respectively, $x_0$ is the initial position of the agent, and $x_f$ is the position of the goal. The $w$ variables are non-zero when certain conditions are met. $w_g$ and $w_a$ indicates the reward/penalty given for reaching the goal and adversary, respectively. $w_c$ is the penalty for contacting a wall. In essence, the agent is incentivized to navigate to the goal as quickly as it can. During training, $\lambda = 0.25$, $w_g = 1$, $w_a = -1$, and $w_c = -1$. Additionally, when an agent reaches the goal or adversary, the episode terminates.

## Training/Evaluation/Evolving an Agent

The training, evaluation, and evolution of an agent for the navigation task is identical to the detection task, except the `task=navigation`. Please refer to the [Detection Task](./detection.md) for more information. To enable a privileged agent, you should specify the agent as `...agent=point_seeker_maze` instead of `...agent=point_seeker` so that the policy avoids the maze walls.

Below is an example of a successfully trained agent on the navigation task:

```{video} assets/navigation/trained_agent.mp4
:align: center
:class: example-figure
:figwidth: 75%
:loop:
:autoplay:
:muted:
:caption: This agent has been successfully trained to navigate from the right side of the map to the left. It uses a default set of eye parameters; in this case, 15 eyes, each with a field-of-view of 10°, and a resolution of 1x1 pixels.
```

Below is an example of a privileged agent evaluation on the navigation task:

```{video} assets/navigation/privileged_agent.mp4
:align: center
:class: example-figure
:figwidth: 75%
:loop:
:autoplay:
:muted:
:caption: The privileged policy simply tries to reduce the distance between itself and the end of the map. It uses a simple breadth-first search algorithm to select a trajectory through the maze without contacting the walls.
```

## Evolving an Agent

You can also evolve an agent using the following command (similar to the [Detection Task](./detection.md#evolving-an-agent))

```bash
sbatch scripts/run.sh cambrian/main.py --train task=navigation evo=evo hydra/launcher=supercloud evo/mutations='[num_eyes,resolution,lon_range]' -m
```

The optimized configuration and its trained policy is shown below:

```{video} assets/navigation/evolved_agent.mp4
:align: center
:class: example-figure
:figwidth: 75%
:loop:
:autoplay:
:muted:
:caption: The optimized policy has 10 eyes, with a longitudinal range of 120° and a resolution of 4x4 pixels.
```

## Task Configuration

```{literalinclude} ../../cambrian/configs/task/navigation.yaml
:language: yaml
:caption: configs/task/navigation.yaml
```
