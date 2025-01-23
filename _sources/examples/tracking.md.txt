# Tracking Task

The tracking task is very similar to the [Detection Task](./detection.md), but instead of all objects being static, they move around the environment.

```{figure} assets/tracking/screenshot.png
:align: center
:class: example-figure
:width: 75%
**Screenshot of the tracking task.** Similar to the [Detection Task](./detection.md), the tracking task implements a simple control for the objects. Each object will move to random positions within the map, their movement visualized by the green and red trails. The agent thus must learn to discern between the strips like in the detection task, but also track the objects as they move.
```

The reward function for the tracking task is identical to the detection task.

```{math}
\quad R_{\text{Detection}} = \quad R_{\text{Tracking}} = -\lambda \left( \|\mathbf{x}_t - \mathbf{x}_f\| - \|\mathbf{x}_{t-1} - \mathbf{x}_f\| \right) + w_g + w_a + w_c\\
```

$R_{\text{Tracking}}$ is the reward at time $t$ for each task, $\lambda$ is a scaling factor, $x_t$ and $x_{t-1}$ is the position of the agent at time $t$ and $t-1$ respectively, $x_0$ is the initial position of the agent, and $x_f$ is the position of the goal. The $w$ variables are non-zero when certain conditions are met. $w_g$ and $w_a$ indicates the reward/penalty given for reaching the goal and adversary, respectively. $w_c$ is the penalty for contacting a wall. In essence, the agent is incentivized to navigate to the goal as quickly as it can. During training, $\lambda = 0.25$, $w_g = 1$, $w_a = -1$, and $w_c = -1$. Additionally, when an agent reaches the goal or adversary, the episode terminates.

## Training/Evaluation/Evolving an Agent

The training, evaluation, and evolution of an agent for the tracking task is identical to the detection task. Please refer to the [Detection Task](./detection.md) for more information.

Below is an example of a successfully trained agent on the tracking task:

```{video} assets/tracking/trained_agent.mp4
:align: center
:class: example-figure
:figwidth: 75%
:loop:
:autoplay:
:muted:
:caption: This agent has been successfully trained to discern between moving goal and adversarial objects. The above command uses a default set of eye parameters; in this case, 3 eyes, each with a field-of-view of 45°, and a resolution of 20x20 pixels.
```

Below is an example of a privileged agent evaluation on the tracking task:

```{video} assets/tracking/privileged_agent.mp4
:align: center
:class: example-figure
:figwidth: 75%
:loop:
:autoplay:
:muted:
:caption: The privileged policy simply tries to reduce the distance and angle error between itself in the goal. It doesn't use any visual stimuli.
```

## Task Configuration

```{literalinclude} ../../cambrian/configs/task/tracking.yaml
:language: yaml
:caption: configs/task/tracking.yaml
```
